import torch as tc
from torch._C import device
try:
    import torch_xla.core.xla_model as xm
except:
    xm = None

class DNNModel(tc.nn.Module):

    def __init__(
            self, input_size, output_size, dropout_rate, hidden_sizes,
            sigmoid_vals=True):
        super(DNNModel, self).__init__()
        self.dropout_rate = dropout_rate
        hidden_modules = [tc.nn.Linear(input_size, hidden_sizes[0])]
        for i in range(len(hidden_sizes)):
            hidden_modules.append(tc.nn.Sigmoid())
            if self.dropout_rate:
                hidden_modules.append(tc.nn.Dropout(p=self.dropout_rate))
            if (i + 1) < len(hidden_sizes):
                next_size = hidden_sizes[i + 1]
            else:
                next_size = output_size
            hidden_modules.append(tc.nn.Linear(hidden_sizes[i], next_size))
        if sigmoid_vals:
            hidden_modules.append(tc.nn.Sigmoid())

        self.f = tc.nn.Sequential(*hidden_modules)

    def forward(self, x):
        return self.f(x)


class DropoutNet:

    def __init__(
            self,
            input_size,
            num_actions,
            num_horizons,
            feat_mean=0.5,
            lr=1e-2,
            lr_gamma=0.95,
            lr_steps=75,
            name="Unnamed_mcd",
            n_samples=60,
            dropout_rate=0.5,
            hidden_sizes=(256, 256, 256),
            sigmoid_vals=True,
            baserate_breadth=0.01,
            weight_decay=None,
            use_gaussian=True,
            device='cpu',
            **kwargs
    ):
        self.samples = n_samples
        self.weight_decay = weight_decay
        self.l2=baserate_breadth

        self.gaussian = use_gaussian

        self.input_size = input_size
        self.num_actions = num_actions
        self.num_horizons = num_horizons
        self.output_size = num_actions * num_horizons
        self.feat_mean = feat_mean
        self.lr = lr
        self.lr_gamma = lr_gamma
        self.lr_steps = lr_steps
        # self.lr_decay_per_step = 0.998
        self.min_lr = 0.0001
        self.name = name + "_MCD"
        self.device = device
        self.net = DNNModel(
            self.input_size, self.output_size, dropout_rate=dropout_rate,
            hidden_sizes=hidden_sizes, sigmoid_vals=sigmoid_vals)
        self.net = self.net.train().to(self.device)
        print(f"{self.name}:")
        print(self.net)
        # self.loss_f = tc.nn.SmoothL1Loss()
        self.loss_f = tc.nn.MSELoss()
        self.optimizer = None
        self.lr_schedule = None
        self.make_optimizer()

    def make_optimizer(self):
        """Currently  implements a singleton, only producing schedule on
        2nd call (i.e. after burn in).

        Can be used to reset the optimizer if desired.
        """
        if self.optimizer is None:
            self.optimizer = tc.optim.Adam(
                self.net.parameters(), lr=self.lr,
                weight_decay=self.weight_decay)
            # self.optimizer = tc.optim.SGD(
            #     self.net.parameters(), lr=self.lr,
            #     weight_decay=self.weight_decay)
        elif self.lr_schedule is None and self.lr_steps is not None:
            self.lr_schedule = tc.optim.lr_scheduler.StepLR(
                self.optimizer, step_size=self.lr_steps, gamma=self.lr_gamma)

    def uncertainty_estimate(
            self, states, actions, quantile, horizon=None, debug=False):
        self.net.train()
        if horizon is None:
            horizon = self.num_horizons
        with tc.no_grad():
            y_samples = self.take_samples(states, actions, horizon)
        y_mean = tc.mean(y_samples, dim=0)
        # gaussian = False
        if actions is not None:
            actions = actions.to(self.device)

        if self.gaussian:
            if self.weight_decay:
                # l2 = 0.01  # a guess at a "prior length scale"
                tau = (
                        self.l2 * (1. - self.net.dropout_rate)
                        / (2. * states.shape[0] * self.weight_decay))
            else:
                tau = 20
            y_variance = tc.var(y_samples, dim=0)
            y_variance += (1. / tau)
            assert y_variance.shape == y_mean.shape
            y_std = tc.sqrt(y_variance)
            fit_gaussian = tc.distributions.Normal(y_mean, y_std)
            quantile_vals = fit_gaussian.icdf(tc.tensor(quantile, device=self.device))
        else:
            quantile_vals = tc.quantile(y_samples, quantile, dim=0)
            y_std = None

        if debug:
            print(f"Uncertainty estimate for {self.name}, h{horizon}")
            print(f"(Medians)\n{tc.median(y_samples, dim=0)[0]}")
            print(f"Means\n{y_mean}")
            print(f"Stds\n{y_std}")
            print(f"Quantiles_{quantile:.5f}\n{quantile_vals}")

        return quantile_vals

    def take_samples(self, input_x, actions=None, horizon=None):
        """Return a sample-wise array of net predictions for x

        Operates for a specific time horizon on the net. Optionally
        indexed at actions.
        """

        input_x = input_x.to(self.device)
        print(f'sampling input_x: {input_x}')
        print(f'sampling actions: {actions}')

        if actions is not None:
            actions = actions.to(self.device)

        out_size = [self.samples, input_x.shape[0]] + (
            [self.num_actions] if actions is None else [])
        y_samp = tc.zeros(*out_size)
        if horizon is not None:
            horizon_slice = slice(
                (horizon - 1) * self.num_actions, horizon * self.num_actions)
        else:
            horizon_slice = None
        for s in range(self.samples):
            sample_batch = self.net(input_x)
            if horizon_slice is not None:
                sample_batch = sample_batch[..., horizon_slice]
            if actions is not None:
                sample_batch = tc.squeeze(
                    tc.gather(sample_batch, 1, actions), 1)
            y_samp[s] = sample_batch
        return y_samp

    def copy_values(self, new_weights):
        self.net.load_state_dict(new_weights)

    def update_learning_rate(self, lr):
        assert lr <= self.lr, (
            f"Not expecting changes to LR self {self.lr} -> {lr}")
        self.lr = lr

    def predict(self, inputs, actions=None, target=None, horizon=None, debug=False):
        """

        inputs (tc.tensor): observations tensor
        actions (tc.tensor): actions took at that observation
        target (tc.tensor): the target value
        horizon (int): horizon to estimate value over
        debug (bool): whether to print extra stuff
        """
        if horizon is not None and horizon <= 0:
            raise ValueError(horizon)
        if not isinstance(inputs, tc.Tensor):
            raise TypeError(f"Expected torch tensor, got {type(inputs)}")
        if target is not None and not isinstance(target, tc.Tensor):
            raise TypeError(f"Expected torch tensor, got {type(target)}")
        # print(self.device)
        input_features = inputs.float().to(self.device)
        target = None if target is None else target.float().to(self.device)
        if actions is not None:
            actions = actions.to(self.device)
    
        assert (target is None) == (actions is None)
        assert input_features.ndim == 2 and (
                target is None
                or target.shape == (input_features.shape[0], 1)), (
                f"Incorrect dimensions for input: {input_features.shape}"
                + ("" if target is None else f", or targets: {target.shape}"))
        
        

        self.net.train()
        if target is None:
            assert actions is None
            with tc.no_grad():
                y_samp = self.take_samples(input_features, horizon=horizon)
            prediction_values = tc.mean(y_samp, dim=0)
            # prediction_values, _ = torch.median(y_samp, dim=0)
            return prediction_values
        else:
            assert actions is not None
            y_pred = self.net(input_features)
            if horizon is not None:
                horizon_slice = slice(
                    (horizon - 1) * self.num_actions,
                    horizon * self.num_actions)
                y_pred = y_pred[..., horizon_slice]
            if debug and y_pred.shape[0] > 1:
                print(f"Estimates\n{y_pred.squeeze()}")
            if y_pred.shape[-1] != 1:
                actions = actions.to(self.device)
                y_pred = tc.gather(y_pred, 1, actions)

            if debug and y_pred.shape[0] > 1:
                print(f"Gathered estimates\n{y_pred.squeeze()}")

            assert y_pred.shape == target.shape, f"{y_pred.shape}, {target.shape}"
            loss = self.loss_f(y_pred, target)
            if debug:
                print(f"Loss: {loss}")
            self.optimizer.zero_grad()
            loss.backward()

            # stability
            # for param in self.net.parameters():
            #     param.grad.data.clamp_(-1, 1)

            if xm is not None:
                xm.mark_step()
                xm.optimizer_step(self.optimizer)
            else:
                self.optimizer.step()

            if self.lr_schedule is not None\
                    and self.lr_schedule.get_last_lr()[0] > self.min_lr\
                    and (horizon is None or horizon == self.num_horizons):
                init_lr = self.lr_schedule.get_last_lr()
                self.lr_schedule.step()
                new_lr = self.lr_schedule.get_last_lr()
                if new_lr != init_lr:
                    print("Stepping learning rate", init_lr, "->", new_lr)
