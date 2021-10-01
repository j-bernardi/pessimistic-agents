"""
Using code from:
https://github.com/cpark321/uncertainty-deep-learning/blob/master/01.%20Bayes-by-Backprop.ipynb
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
import numpy as np

from utils import set_gpu

device = set_gpu()


class Linear_BBB(nn.Module):
    """
        Layer of our BNN.
    """
    def __init__(self, input_features, output_features, prior_var=1.):
        """
            Initialization of our layer : our prior is a normal distribution
            centered in 0 and of variance 20.
        """
        # initialize layers
        super().__init__()
        # set input and output dimensions
        self.input_features = input_features
        self.output_features = output_features

        # initialize mu and rho parameters for the weights of the layer
        self.w_mu = nn.Parameter(torch.zeros(output_features, input_features))
        self.w_rho = nn.Parameter(torch.zeros(output_features, input_features))

        #initialize mu and rho parameters for the layer's bias
        self.b_mu = nn.Parameter(torch.zeros(output_features))
        self.b_rho = nn.Parameter(torch.zeros(output_features))

        #initialize weight samples (these will be calculated whenever the layer makes a prediction)
        self.w = None
        self.b = None

        # initialize prior distribution for all of the weights and biases
        self.prior = torch.distributions.Normal(0, prior_var)
        self.unit_normal = torch.distributions.Normal(0, 1)

    def forward(self, x):
        """
          Optimization process
        """
        # sample weights
        w_epsilon = self.unit_normal.sample(self.w_mu.shape).to(device)
        self.w = self.w_mu + torch.log(1 + torch.exp(self.w_rho)) * w_epsilon

        # sample bias
        b_epsilon = self.unit_normal.sample(self.b_mu.shape).to(device)
        self.b = self.b_mu + torch.log(1+torch.exp(self.b_rho)) * b_epsilon

        # record log prior by evaluating log pdf of prior at sampled weight and bias
        w_log_prior = self.prior.log_prob(self.w)
        b_log_prior = self.prior.log_prob(self.b)
        self.log_prior = torch.sum(w_log_prior) + torch.sum(b_log_prior)

        # record log variational posterior by evaluating log pdf of normal distribution defined by parameters with respect at the sampled values
        self.w_post = Normal(self.w_mu.data, torch.log(1+torch.exp(self.w_rho)))
        self.b_post = Normal(self.b_mu.data, torch.log(1+torch.exp(self.b_rho)))
        self.log_post = self.w_post.log_prob(self.w).sum() + self.b_post.log_prob(self.b).sum()
        
        return F.linear(x, self.w, self.b)


class MLP_BBB(nn.Module):
    def __init__(
            self, input_size, output_size, hidden_units, noise_tol=.1,
            prior_var=1.):

        # initialize the network like you would with a standard multilayer
        # perceptron, but using the BBB layer
        super().__init__()
        self.hidden1 = Linear_BBB(
            input_size, hidden_units, prior_var=prior_var)
        self.hidden2 = Linear_BBB(
            hidden_units, hidden_units, prior_var=prior_var)
        self.out = Linear_BBB(hidden_units, output_size, prior_var=prior_var)
        # we will use the noise tolerance to calculate our likelihood
        self.noise_tol = noise_tol

    def forward(self, x):
        # again, this is equivalent to a standard multilayer perceptron
        x = torch.tanh(self.hidden1(x))
        x = torch.tanh(self.hidden2(x))
        x = torch.sigmoid(self.out(x))
        return x

    def log_prior(self):
        # calculate the log prior over all the layers
        return self.hidden1.log_prior + self.hidden2.log_prior + self.out.log_prior

    def log_post(self):
        # calculate the log posterior over all the layers
        return self.hidden1.log_post + self.hidden2.log_post + self.out.log_post

    def sample_elbo(self, input, target, samples):

        # we calculate the negative elbo, which will be our loss function
        #initialize tensors
        outputs = torch.zeros(samples, target.shape[0])
        log_priors = torch.zeros(samples)
        log_posts = torch.zeros(samples)
        log_likes = torch.zeros(samples)
        # make predictions and calculate prior, posterior, and likelihood for a given number of samples
        for i in range(samples):
            outputs[i] = self(input).reshape(-1) # make predictions
            log_priors[i] = self.log_prior() # get log prior
            log_posts[i] = self.log_post() # get log variational posterior
            log_likes[i] = Normal(outputs[i], self.noise_tol).log_prob(target.reshape(-1)).sum() # calculate the log likelihood
        # calculate monte carlo estimate of prior posterior and likelihood
        log_prior = log_priors.mean()
        log_post = log_posts.mean()
        log_like = log_likes.mean()
        # calculate the negative elbo (which is our loss function)
        loss = log_post - log_prior - log_like
        return loss


class BBBNet: 
    """A lot of the variables won't actually be used
    """
    def __init__(
            self,
            layer_sizes,
            input_size,
            feat_mean=0.5,
            lr=1e-3,
            name="Unnamed_gln",
            batch_size=None,
            q=0.1,
            n_samples=20,
            **kwargs  # soak up unused kwargs
    ):

        self.samples = n_samples
        self.q = q

        assert layer_sizes[-1] == 1, "Final layer should have 1 neuron"
        self.layer_sizes = layer_sizes
        self.input_size = input_size
        self.feat_mean = feat_mean
        self.lr = lr
        self.name = name
        self.batch_size = batch_size

        def display(*args):
            p_string = f"\nCreating BayesByBackprop {self.name} with:"
            for v in args:
                p_string += f"\n{v}={getattr(self, v)}"
            print(p_string)

        self.net = MLP_BBB(self.input_size, 1, 32, prior_var=1)
        self.optimizer = optim.Adam(self.net.parameters(), lr=lr)
        display()

    def predict(self, inputs, target=None):
        if not isinstance(inputs, torch.Tensor):
            raise TypeError(f"Expected torch tensor, got {type(inputs)}")
        if target is not None and not isinstance(target, torch.Tensor):
            raise TypeError(f"Expected torch tensor, got {type(target)}")
        input_features = inputs.float()
        target = None if target is None else target.float()

        assert input_features.ndim == 2 and (
               target is None or (
                   target.ndim == 1
                   and target.shape[0] == input_features.shape[0])), (
            f"Incorrect dimensions for input: {input_features.shape}"
            + ("" if target is None else f", or targets: {target.shape}"))

        if target is None:
            self.net.eval()
            with torch.no_grad():
                y_samp = torch.zeros((self.samples, input_features.shape[0]))
                for s in range(self.samples):
                    y_tmp = self.net(input_features)
                    y_samp[s] = y_tmp.reshape(-1)
            # predictions = np.mean(y_samp, axis=0)
            prediction_values, _ = torch.median(y_samp, dim=0)
            return prediction_values
        else:
            self.net.train()
            self.optimizer.zero_grad()
            loss = self.net.sample_elbo(input_features, target, 1)
            loss.backward()
            self.optimizer.step()

    def uncertainty_estimate(
            self, states, x_batch, y_batch, quantile, max_est_scaling=None,
            converge_epochs=5, debug=False):

        states = torch.tensor(np.asarray(states))

        self.net.eval()
        with torch.no_grad():
            y_samp = torch.zeros((self.samples, len(states)))
            for s in range(self.samples):
                y_tmp = self.net(states)
                y_samp[s] = y_tmp.reshape(-1)
        return torch.quantile(y_samp, quantile, dim=0)

    def update_learning_rate(self, lr):
        self.lr = lr

    def copy_values(self, new_weights):
        self.net.load_state_dict(new_weights)
