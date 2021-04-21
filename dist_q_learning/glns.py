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

    # @jax.jit
    def __init__(self, 
                 layer_sizes,
                 input_size,
                 context_dim,
                 bias_len=3,
                 lr=1e-3,
                 name='Unnamed_gln',
                 min_sigma_sq=0.5,
                 dummy_init=None):
        
        self.layer_sizes = layer_sizes
        # self.layer_sizes.append(1)
        self.input_size = input_size
        self.context_dim = context_dim
        self.bias_len = bias_len
        self.lr = lr
        self.name = name
        self.min_sigma_sq=min_sigma_sq

        # self.gln = gaussian.GatedLinearNetwork(
        #         output_sizes=self.layer_sizes,
        #         context_dim=self.context_dim,
        #         bias_len=self.bias_len,
        #         name=self.name)

        self._rng = hk.PRNGSequence(jax.random.PRNGKey(np.random.randint(low=0,high=int(2**30))))

        self._init_fn, self.inference_fn_ = hk.without_apply_rng(
            hk.transform_with_state(self.inference_fn))
        _, self.update_fn_ = hk.without_apply_rng(hk.transform_with_state(self.update_fn))

        self._inference_fn = jax.jit(self.inference_fn_)
        self._update_fn = jax.jit(self.update_fn_)

        self.init_fn = self._init_fn
        self.inference_fn = self._inference_fn
        self.update_fn = self._update_fn

        dummy_inputs = jnp.ones([input_size, 2])
        dummy_side_info = np.ones([input_size])
        
        self.gln_params, self.gln_state = self._init_fn(next(self._rng), dummy_inputs, dummy_side_info)


    def gln_factory(self):
        return gaussian.GatedLinearNetwork(
            output_sizes=self.layer_sizes,
            context_dim=self.context_dim,
            bias_len=self.bias_len,
            name=self.name)
            
    def inference_fn(self, inputs, side_info):
      return self.gln_factory().inference(inputs, side_info, self.min_sigma_sq)

    def batch_inference_fn(self, inputs, side_info):
      return jax.vmap(self.inference_fn, in_axes=(0, 0))(inputs, side_info)

    def update_fn(self, inputs, side_info, label, learning_rate):
      params, predictions, unused_loss = self.gln_factory().update(
          inputs, side_info, label, learning_rate, self.min_sigma_sq)
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
        input = jnp.array(input)
        input_with_sig_sq = jnp.vstack((input, jnp.ones(len(input)))).T   
        side_info = input
        if target is None:
            
            predictions, _ = self.inference_fn(self.gln_params, self.gln_state, 
                                input_with_sig_sq, side_info)

            return predictions[-1,0]

        else:

            (_, gln_params), _ = self.update_fn(self.gln_params, self.gln_state,
                                    input_with_sig_sq, side_info, target[0],
                                    learning_rate=self.lr)

            self.gln_params = gln_params

    def update_learning_rate(self, lr):

        self.lr =  lr


# gln1= GGLN([2,2,2],4, 5)

# preds = gln1.predict(np.array([2.1, 4.3, -0.1, 3.2]))
# print(preds)

# preds = gln1.predict(np.array([2.1, 4.3, -0.1, 3.2]), target=[9])

# preds = gln1.predict(np.array([2.1, 4.3, -0.1, 3.2]))
# print(preds)
# print(gln1.gln_params)
# print(gln1.gln_state)
# print('hon')