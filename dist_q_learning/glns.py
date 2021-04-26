import abc
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline

from scipy.special import logit as slogit

import scipy.stats
NP_RANDOM_GEN = np.random.Generator(np.random.PCG64())

# import pygln 

from gated_linear_networks import gaussian
import haiku as hk
import jax
import jax.numpy as jnp
import tree

class GGLN():
    """Gaussian Gated Linear Network
    
    Uses the GGLN implementation from deepmind-research.
    Can update from new data, and make predictions.
    """

    def __init__(self, 
                 layer_sizes,
                 input_size,
                 context_dim,
                 bias_len=3,
                 lr=1e-3,
                 name='Unnamed_gln',
                 min_sigma_sq=0.5,
                 rng_key=None,
                 batch_size=None):

        """Set up the GGLN.

        Initialises all the variables, including the GGLN parameters

        Args:
          layer_sizes (list[int]): the number of neurons in each layer.
              the final layer should have 1 output neuron.
          input_size (int): the length of the input data
          context_dim (int): the number of hyperplanes for the halfspace
              gatings
          bias_len (int): the number of 'bias' gaussians that are appended 
              to the first input (not added to the side_info)
          lr (float): the learning rate
          name (str): the name of the gln.
          min_sigma_sq (float): the minimum allowed value for the variance
              of each of the gaussians
          rng_key (int): an int to seed the RNG for this GGLN

        """

        
        assert layer_sizes[-1]==1, "Final layer should have 1 neuron"
        self.layer_sizes = layer_sizes
        self.input_size = input_size
        self.context_dim = context_dim
        self.bias_len = bias_len
        self.lr = lr
        self.name = name
        self.min_sigma_sq=min_sigma_sq
        self.batch_size = batch_size
        # make rng for this GGLN
        if rng_key is None:
          rng_key = np.random.randint(low=0,high=int(2**30))
        
        self._rng = hk.PRNGSequence(jax.random.PRNGKey(rng_key))

        # make init, inference and update functions,
        # these are GPU compatible thanks to jax and haiku
        # self._init_fn, self.inference_fn_ = hk.without_apply_rng(
        #     hk.transform_with_state(self.inference_fn))
        # _, self.update_fn_ = hk.without_apply_rng(hk.transform_with_state(self.update_fn))

        # self._batch_init_fn, self.batch_inference_fn_ = hk.without_apply_rng(
        #     hk.transform_with_state(self.batch_inference_fn))
        # _, self.batch_update_fn_ = hk.without_apply_rng(
        #     hk.transform_with_state(self.batch_update_fn))

        # self._inference_fn = jax.jit(self.inference_fn_)
        # self._update_fn = jax.jit(self.update_fn_)
        # self._batch_inference_fn = jax.jit(self.batch_inference_fn_)
        # self._batch_update_fn = jax.jit(self.batch_update_fn_)


        # # self.init_fn = self._init_fn
        # # self.inference_fn = self._inference_fn
        # # self.update_fn = self._update_fn

        # if self.batch_size is None:
        #   self.init_fn = self._init_fn
        #   self.inference_fn = self._inference_fn
        #   self.update_fn = self._update_fn
        #   # make dummy variables used for initialising the GGLN
        #   dummy_inputs = jnp.ones([input_size, 2])
        #   dummy_side_info = jnp.ones([input_size])
        # else:
        #   self.init_fn = jax.jit(self._batch_init_fn)
        #   self.inference_fn = self._batch_inference_fn
        #   self.update_fn = self._batch_update_fn

        #   # make dummy variables used for initialising the GGLN
        #   dummy_inputs = jnp.ones([self.batch_size, input_size, 2])
        #   dummy_side_info = jnp.ones([self.batch_size, input_size])
        

        self._init_fn, self.inference_fn_ = hk.without_apply_rng(
            hk.transform_with_state(self.inference_fn))
        self._batch_init_fn, self.batch_inference_fn_ = hk.without_apply_rng(
            hk.transform_with_state(self.batch_inference_fn))
        _, self.update_fn_ = hk.without_apply_rng(hk.transform_with_state(self.update_fn))
        _, self.batch_update_fn_ = hk.without_apply_rng(
            hk.transform_with_state(self.batch_update_fn))

        self._inference_fn = jax.jit(self.inference_fn_)
        self._batch_inference_fn = jax.jit(self.batch_inference_fn_)
        self._update_fn = jax.jit(self.update_fn_)
        self._batch_update_fn = jax.jit(self.batch_update_fn_)

        if batch_size is None:
          self.init_fn = self._init_fn
          self.inference_fn = self._inference_fn
          self.update_fn = self._update_fn

          dummy_inputs = jnp.ones([input_size, 2])
          dummy_side_info = jnp.ones([input_size])

        else:
          self.init_fn = self._batch_init_fn
          self.inference_fn = self._batch_inference_fn
          self.update_fn = self._batch_update_fn

          dummy_inputs = jnp.ones([self.batch_size, input_size, 2])
          dummy_side_info = jnp.ones([self.batch_size, input_size])


        print(dummy_inputs.shape)
        print(dummy_side_info.shape)
        # initialise the GGLN
        self.gln_params, self.gln_state = self.init_fn(next(self._rng), dummy_inputs, dummy_side_info)


    def gln_factory(self):
      # makes the GGLN
      # This 'factory' method is just taken from deepmind's implementation
        return gaussian.GatedLinearNetwork(
            output_sizes=self.layer_sizes,
            context_dim=self.context_dim,
            bias_len=self.bias_len,
            name=self.name)
            
    def inference_fn(self, inputs, side_info):
      return self.gln_factory().inference(inputs, side_info, 0.5)

    def batch_inference_fn(self, inputs, side_info):
      return jax.vmap(self.inference_fn, in_axes=(0,0))(inputs, side_info)

    def update_fn(self, inputs, side_info, label, learning_rate):
      params, predictions, unused_loss = self.gln_factory().update(
          inputs, side_info, label, learning_rate, 0.5)
      return predictions, params

    def batch_update_fn(self, inputs, side_info, label, learning_rate):
      predictions, params = jax.vmap(
          self.update_fn, in_axes=(0, 0, 0, None))(
              inputs,
              side_info,
              label,
              learning_rate)
      avg_params = tree.map_structure(lambda x: jnp.mean(x, axis=0), params)
      return predictions, avg_params

    
    def predict(self, input, target=None):
        """ Performs predictions and updates for the GGLN.

        If no target is provided it does predictions,
        if a target is provided it updates the GGLN. 

        """

        input = jnp.array(input)

        if self.batch_size is None or len(input.shape) < 2:
          # make the input, which is the gaussians centered on the 
          # values of the data, with variance of 1
          input_with_sig_sq = jnp.vstack((input, jnp.ones(len(input)))).T   
          # the side_info is just the input data
          side_info = input


          if target is None:
              # if no target is provided do prediction
              predictions, _ = self.inference_fn(self.gln_params, self.gln_state, 
                                  input_with_sig_sq, side_info)
              return predictions[-1, 0]

          else:
              #if a target is provided, update the GLN parameters
              (_, gln_params), _ = self.update_fn(self.gln_params, self.gln_state,
                                      input_with_sig_sq, side_info, target[0],
                                      learning_rate=self.lr)

              self.gln_params = gln_params


        else:

            input_with_sig_sq = jnp.stack((input, jnp.ones(input.shape)),2)
            print(input_with_sig_sq)
            print(input_with_sig_sq.shape)
            # input_with_sig_sq = jnp.transpose(input_with_sig_sq,axes=(2,0,1))
            # print(input_with_sig_sq.shape)
            # side_info = input


            if target is None:
                # if no target is provided do prediction
                predictions, _ = self.inference_fn(self.gln_params, self.gln_state, 
                                    input_with_sig_sq, side_info)
                return predictions[:, -1, 0]

            else:
                #if a target is provided, update the GLN parameters
                (_, gln_params), _ = self.update_fn(self.gln_params, self.gln_state,
                                        input_with_sig_sq, side_info, target,
                                        learning_rate=self.lr)

                self.gln_params = gln_params

    def update_learning_rate(self, lr):
        # updates the learning rate to the new value

        self.lr =  lr

