import numpy as np
import haiku as hk
import jax
import jax.numpy as jnp
import tree

from gated_linear_networks import gaussian


def input_mean_transform(x):
    ''' Changes the range of x from  [-4.8, -5, -0.418, -2] to [4.8, 5, 0.418, 2]
    to each being in [0, 1]
    '''
    x = x / jnp.array([4.8, 5, 0.418, 2])

    # x = x / np.array([1,1])

    x = (x + 1) / 2

    return x

def sideinfo_transform(x):
    ''' Changes the range of x from  [-4.8, -5, -0.418, -2] to [4.8, 5, 0.418, 2]
    to each being in [-1, 1]
    '''
    x = x / jnp.array([4.8, 5, 0.418, 2])

    return x

class GGLN():
    """Gaussian Gated Linear Network
    
    Uses the GGLN implementation from deepmind-research.
    Can update from new data, and make predictions.
    """

    def __init__(
            self,
            layer_sizes,
            input_size,
            context_dim,
            bias_len=3,
            lr=1e-3,
            name='Unnamed_gln',
            min_sigma_sq=0.5,
            rng_key=None,
            batch_size=None,
            init_bias_weights=None,
            bias_std=0.05,
            bias_max_mu=1,
            input_mean_transform=input_mean_transform,
            sideinfo_transform=input_mean_transform):
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
        assert layer_sizes[-1] == 1, "Final layer should have 1 neuron"
        self.layer_sizes = layer_sizes
        self.input_size = input_size
        self.context_dim = context_dim
        self.bias_len = bias_len
        self.lr = lr
        self.name = name
        self.min_sigma_sq = min_sigma_sq
        self.batch_size = batch_size
        self.input_mean_transform = input_mean_transform
        self.sideinfo_transform = sideinfo_transform

        # make rng for this GGLN TODO - that's quite high. I guess it's 32-bit?
        if rng_key is None:
            rng_key = np.random.randint(low=0, high=int(2 ** 30))

        self._rng = hk.PRNGSequence(jax.random.PRNGKey(rng_key))

        # make init, inference and update functions,
        # these are GPU compatible thanks to jax and haiku
        def gln_factory():
            return gaussian.GatedLinearNetwork(
                output_sizes=self.layer_sizes,
                context_dim=self.context_dim,
                bias_len=self.bias_len,
                name=self.name,
                bias_std=bias_std,
                bias_max_mu=bias_max_mu
            )

        def inference_fn(inputs, side_info):
            return gln_factory().inference(inputs, side_info, 0.5)

        def batch_inference_fn(inputs, side_info):
            return jax.vmap(inference_fn, in_axes=(0, 0))(inputs, side_info)

        def update_fn(inputs, side_info, label, learning_rate):
            params, predictions, unused_loss = gln_factory().update(
                inputs, side_info, label, learning_rate, 0.5)
            return predictions, params

        def batch_update_fn(inputs, side_info, label, learning_rate):
            predictions, params = jax.vmap(
                update_fn, in_axes=(0, 0, 0, None))(
                    inputs,
                    side_info,
                    label,
                    learning_rate)
            avg_params = tree.map_structure(
                lambda x: jnp.mean(x, axis=0), params)
            return predictions, avg_params

        # Haiku transform functions.
        self._init_fn, inference_fn_ = hk.without_apply_rng(
            hk.transform_with_state(inference_fn))
        self._batch_init_fn, batch_inference_fn_ = hk.without_apply_rng(
            hk.transform_with_state(batch_inference_fn))
        _, update_fn_ = hk.without_apply_rng(
            hk.transform_with_state(update_fn))
        _, batch_update_fn_ = hk.without_apply_rng(
            hk.transform_with_state(batch_update_fn))

        self._inference_fn = jax.jit(inference_fn_)
        self._batch_inference_fn = jax.jit(batch_inference_fn_)
        self._update_fn = jax.jit(update_fn_)
        self._batch_update_fn = jax.jit(batch_update_fn_)

        if self.batch_size is None:
            self.init_fn = self._init_fn
            self.inference_fn = self._inference_fn
            self.update_fn = self._update_fn
            dummy_inputs = jnp.ones([input_size, 2])
            dummy_side_info = np.ones([input_size])
        else:
            self.init_fn = self._batch_init_fn
            self.inference_fn = self._batch_inference_fn
            self.update_fn = self._batch_update_fn
            dummy_inputs = jnp.ones([self.batch_size, input_size, 2])
            dummy_side_info = np.ones([self.batch_size, input_size])

        # initialise the GGLN
        self.gln_params, self.gln_state = self.init_fn(
            next(self._rng), dummy_inputs, dummy_side_info)

        if init_bias_weights is not None:
            self.set_bais_weights(init_bias_weights)

        self.update_nan_count = 0
        self.update_attempts = 0
        self.update_count = 0

    def predict(self, inputs, target=None):
        """Performs predictions and updates for the GGLN

        Args:
            inputs (np.ndarray): Shape (b, outputs)
            target (np.ndarray): has shape (b, outputs). If no target
                is provided, predictions are returned. Else, GGLN
                parameters are updated toward the target
        """
        # Sanitise inputs
        inputs = jnp.array(inputs)
        target = jnp.array(target) if target is not None else None
        assert inputs.ndim == 2 and (
               target is None
               or (target.ndim == 1 and target.shape[0] == inputs.shape[0])), (
            f"Currently only supports inputs 2d: {inputs.shape}, targets 1d: "
            + "(None)" if target is None else f"{target.shape}")

        # or len(inputs.shape) < 2:
        # make the inputs, which is the gaussians centered on the
        # values of the data, with variance of 1
        if self.batch_size is None:
            # the side_info is just the inputs data
            side_info = inputs.T

            # transform side_info and inputs
            if self.sideinfo_transform is not None:
                side_info = self.sideinfo_transform(side_info)
                if (side_info < -1).any() or (side_info > 1).any():
                    print('Side info out of range [-1, 1]')

            if self.input_mean_transform is not None:
                inputs = self.input_mean_transform(inputs)
                if (inputs < 0).any() or (inputs > 1).any():
                    print('Inputs out of range [0, 1]')

            inputs_with_sig_sq = jnp.vstack((inputs, jnp.ones(inputs.shape))).T


            if target is None:
                # if no target is provided do prediction
                predictions, _ = self.inference_fn(
                    self.gln_params, self.gln_state, inputs_with_sig_sq,
                    side_info)
                return predictions[-1, 0]

            else:
                # if a target is provided, update the GLN parameters
                (_, gln_params), _ = self.update_fn(
                    self.gln_params, self.gln_state, inputs_with_sig_sq,
                    side_info, target, learning_rate=self.lr)

                # self.gln_params = gln_params
                self.update_attempts += 1

                has_nans = False

                for v in gln_params.values():
                    if jnp.isnan(v['weights']).any():
                        self.update_nan_count += 1
                        has_nans = True
                        print('===NANS===')
                        break

                if not has_nans:
                    self.gln_params = gln_params
                    self.update_count += 1
                    # print(f'success, target: {target}')
                else:
                    raise RuntimeError("Weights have NaNs")
        else:
            side_info = inputs

            # transform side_info and inputs
            if self.sideinfo_transform is not None:
                side_info = self.sideinfo_transform(side_info)
                if (side_info < -1).any() or (side_info > 1).any():
                    print('Side info out of range [-1, 1]')

            if self.input_mean_transform is not None:
                inputs = self.input_mean_transform(inputs)
                if (inputs < 0).any() or (inputs > 1).any():
                    print('Inputs out of range [0, 1]')
            inputs_with_sig_sq = jnp.stack((inputs, jnp.ones_like(inputs)), 2)

            if target is None:
                # if no target is provided do prediction
                predictions, _ = self.inference_fn(
                    self.gln_params, self.gln_state, inputs_with_sig_sq,
                    side_info)
                return predictions[:, -1, 0]
            else:
                # if a target is provided, update the GLN parameters
                (_, gln_params), _ = self.update_fn(
                    self.gln_params, self.gln_state, inputs_with_sig_sq,
                    side_info, target, learning_rate=self.lr)

                self.update_attempts += 1

                has_nans = False
                for v in gln_params.values():
                    if jnp.isnan(v['weights']).any():
                        self.update_nan_count += 1
                        has_nans = True
                        print('===NANS===')
                        break

                if not has_nans:
                    self.gln_params = gln_params
                    self.update_count += 1
                    # print(f'success, target: {target}')
                else:
                    raise ValueError("Has Nans in weights")

    def predict_with_sigma(self, inputs, target=None):
        """ Performs predictions and updates for the GGLN.

        If no target is provided it does predictions,
        if a target is provided it updates the GGLN. 

        TODO - can it be a parameterised version of predict() ?
        """
        inputs = jnp.array(inputs)
        target = jnp.array(target) if target is not None else None

        if self.batch_size is None:  # or len(inputs.shape) < 2:
            # make the inputs, which is the gaussians centered on the
            # values of the data, with variance of 1
            inputs_with_sig_sq = jnp.vstack((inputs, jnp.ones(inputs.shape))).T
            # the side_info is just the inputs data
            side_info = inputs.T

            if target is None:
                # if no target is provided do prediction
                predictions, _ = self.inference_fn(
                    self.gln_params, self.gln_state, inputs_with_sig_sq,
                    side_info)
                return predictions[-1, 0], jnp.sqrt(predictions[-1, 1])

            else:
                # if a target is provided, update the GLN parameters
                (_, gln_params), _ = self.update_fn(
                    self.gln_params, self.gln_state, inputs_with_sig_sq,
                    side_info, target, learning_rate=self.lr)

                # self.gln_params = gln_params
                self.update_attempts += 1

                has_nans = False

                for v in gln_params.values():
                    if jnp.isnan(v['weights']).any():
                      self.update_nan_count += 1
                      has_nans = True
                      print('===NANS===')
                      break

                if not has_nans:
                    self.gln_params = gln_params
                    self.update_count += 1
                    # print(f'success, target: {target}')

                return not has_nans
                # if self.update_nan_count%100 == 0\
                #         and self.update_nan_count > 0:
                #     print(f'Nan count: {self.update_nan_count}')
                #     print(f'attempts: {self.update_attempts}')
                #     print(f'updates: {self.update_count}')
        else:
            inputs_with_sig_sq = jnp.stack((inputs, jnp.ones(inputs.shape)), 2)
            side_info = inputs
            if target is None:
                # if no target is provided do prediction
                predictions, _ = self.inference_fn(
                    self.gln_params, self.gln_state, inputs_with_sig_sq,
                    side_info)
                return predictions[:, -1, 0], jnp.sqrt(predictions[:, -1, 1])

            else:
                # if a target is provided, update the GLN parameters
                (_, gln_params), _ = self.update_fn(
                    self.gln_params, self.gln_state, inputs_with_sig_sq,
                    side_info, target, learning_rate=self.lr)
                self.update_attempts += 1
                has_nans = False
                for v in gln_params.values():
                    if jnp.isnan(v['weights']).any():
                        self.update_nan_count += 1
                        has_nans = True
                        print('===NANS===')
                        break

                if not has_nans:
                    self.gln_params = gln_params
                    self.update_count += 1

    def update_learning_rate(self, lr):
        # updates the learning rate to the new value
        self.lr = lr

    def set_bais_weights(self, bias_vals):
        """Sets the weights for the bias inputs

        args:
          bias_vals (List[Float]): the values to set each of the bais 
          weights to, in order of the bias mu. If one of these is None, 
          then don't update that bias weight
        """

        assert len(bias_vals) == self.bias_len

        # make a mutable dict so we can actually modify things
        gln_p_temp = hk.data_structures.to_mutable_dict(self.gln_params)
        for key, v in self.gln_params.items():
            # for each layer in the gln
            w_temp = v['weights']

            for i in range(self.bias_len):
                bias_val = bias_vals[i]

                if bias_val is not None:
                    # update the bias weight if we have a value for it
                    # the bias wgights are at the end of the weight arrays.
                    # eg. the first bias weight is at index -1*self.bias_len
                    w_temp = jax.ops.index_update(
                        w_temp,
                        jax.ops.index[:, :, - self.bias_len + i],
                        bias_val)

            gln_p_temp[key]['weights'] = w_temp # update the weights
        
        # update the gln_params which we actually use
        self.gln_params = hk.data_structures.to_immutable_dict(gln_p_temp)
