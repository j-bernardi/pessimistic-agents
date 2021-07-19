import haiku as hk
import jax
import jax.numpy as jnp
import tree

from gated_linear_networks import gaussian


JAX_RANDOM_KEY = jax.random.PRNGKey(0)


class GGLN:
    """Gaussian Gated Linear Network

    Uses the GGLN implementation from deepmind-research.
    Can update from new data, and make predictions.
    """

    def __init__(
            self,
            layer_sizes,
            input_size,
            context_dim,
            feat_mean=0.5,
            bias_len=3,
            lr=1e-3,
            name="Unnamed_gln",
            min_sigma_sq=0.5,
            rng_key=None,
            batch_size=None,
            init_bias_weights=None,
            bias_std=0.05,
            bias_max_mu=1):
        """Set up the GGLN.

        Initialises all the variables, including the GGLN parameters

        Args:
          layer_sizes (list[int]): the number of neurons in each layer.
              the final layer should have 1 output neuron.
          input_size (int): the length of the input data
          context_dim (int): the number of hyperplanes for the halfspace
              gatings
          feat_mean (float): the mean of the input features - usually
            0 or 0.5 depending on whether normalised [-1 or 0, 1],
            respectively
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
        self.feat_mean = feat_mean
        self.bias_len = bias_len
        self.lr = lr
        self.name = name
        self.min_sigma_sq = min_sigma_sq
        self.batch_size = batch_size

        print(f"Creating GLN {self.name} with mean={self.feat_mean}, "
              f"lr={self.lr}")

        if rng_key is not None:
            self._rng = hk.PRNGSequence(jax.random.PRNGKey(rng_key))
        else:
            self._rng = hk.PRNGSequence(JAX_RANDOM_KEY)

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
            dummy_side_info = jnp.ones([input_size])
        else:
            self.init_fn = self._batch_init_fn
            self.inference_fn = self._batch_inference_fn
            self.update_fn = self._batch_update_fn
            dummy_inputs = jnp.ones([self.batch_size, input_size, 2])
            dummy_side_info = jnp.ones([self.batch_size, input_size])

        # initialise the GGLN
        self.gln_params, self.gln_state = self.init_fn(
            next(self._rng), dummy_inputs, dummy_side_info)

        if init_bias_weights is not None:
            print("Setting bias weights")
            self.set_bias_weights(init_bias_weights)

    def predict(self, inputs, target=None):
        """Performs predictions and updates for the GGLN

        Args:
            inputs (jnp.ndarray): A (N, context_dim) array of input
                features
            target (Optional[jnp.ndarray]): A (N, outputs) array of
                targets. If provided, the GGLN parameters are updated
                toward the target. Else, predictions are returned.
        """
        # Sanitise inputs
        input_features = jnp.array(inputs)
        # Input mean usually the x values, but can just be a PDF
        # standard deviation is so that sigma_squared spans whole space
        initial_pdfs = (
            jnp.full_like(input_features, self.feat_mean),
            # input_features,
            jnp.full_like(input_features, 1.),
        )
        target = jnp.array(target) if target is not None else None

        assert input_features.ndim == 2 and (
               target is None or (
                   target.ndim == 1
                   and target.shape[0] == input_features.shape[0])), (
            f"Incorrect dimensions for input: {input_features.shape}"
            + ("" if target is None else f", or targets: {target.shape}"))

        # make the inputs, which is the Gaussians centered on the
        # values of the data, with variance of 1
        if self.batch_size is None:
            inputs_with_sig_sq = jnp.vstack(initial_pdfs).T
            side_info = input_features.T
        else:
            inputs_with_sig_sq = jnp.stack(initial_pdfs, 2)
            side_info = input_features

        if target is None:
            # if no target is provided do prediction
            predictions, _ = self.inference_fn(
                self.gln_params, self.gln_state, inputs_with_sig_sq, side_info)
            # print("PREDS")
            # print(predictions)
            return predictions[..., -1, 0]
        else:
            # if a target is provided, update the GLN parameters
            (_, new_gln_params), _ = self.update_fn(
                self.gln_params, self.gln_state, inputs_with_sig_sq, side_info,
                target, learning_rate=self.lr)
            self.gln_params = new_gln_params
            self.check_weights()

    def predict_with_sigma(self, inputs, target=None):
        """ Performs predictions and updates for the GGLN.

        If no target is provided it does predictions,
        if a target is provided it updates the GGLN. 

        TODO - can it be a parameterised version of predict() ?
        """
        raise NotImplementedError("Fallen out of usage")
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
                    if jnp.isnan(v["weights"]).any():
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
                    if jnp.isnan(v["weights"]).any():
                        self.update_nan_count += 1
                        has_nans = True
                        print('===NANS===')
                        break

                if not has_nans:
                    self.gln_params = gln_params
                    self.update_count += 1

    def uncertainty_estimate(
            self, states, x_batch, y_batch, max_est_scaling=None,
            converge_epochs=20, debug=False):
        """Get parameters to Beta distribution defining uncertainty

        Args:
            states (jnp.ndarray): states to estimate uncertainty for
            x_batch (jnp.ndarray): converge on this batch of data before
                making uncertainty estimates
            y_batch (jnp.ndarray): converge on this batch of targets
                before making uncertainty estimates
            max_est_scaling (Optional[float]): whether to scale-down the

        Returns:
            ns (jnp.ndarray):
            alphas (jnp.ndarray):
            betas (jnp.ndarray):
        """

        fake_targets = jnp.stack(
            (jnp.full(states.shape[0], 0.5),
             jnp.full(states.shape[0], 0.),
             jnp.full(states.shape[0], 1.)),
            axis=1)
        fake_means = jnp.empty((states.shape[0], 3))

        initial_lr = self.lr
        initial_params = hk.data_structures.to_immutable_dict(self.gln_params)

        # TODO - batch learning instead? Or sampled?
        self.update_learning_rate(
            initial_lr * (x_batch.shape[0] / self.batch_size))
        for convergence_epoch in range(converge_epochs):
            self.predict(x_batch, y_batch)
        self.update_learning_rate(initial_lr)

        converged_ws = hk.data_structures.to_immutable_dict(self.gln_params)
        for i, s in enumerate(states):
            for j, fake_target in enumerate(fake_targets[i]):
                # Update to fake target - single step
                self.update_learning_rate(initial_lr * (1. / self.batch_size))
                self.predict(
                    jnp.expand_dims(s, 0), jnp.expand_dims(fake_target, 0))
                # Collect the estimate of the mean
                new_est = jnp.squeeze(self.predict(jnp.expand_dims(s, 0)), 0)
                fake_means = jax.ops.index_update(fake_means, (i, j), new_est)
                # Clean up
                self.copy_values(converged_ws)
                self.update_learning_rate(initial_lr)

        if max_est_scaling is not None:
            fake_means /= max_est_scaling
        updated_to_current_est = fake_means[:, 0]
        biased_ests = fake_means[:, 1:]
        if debug:
            print(f"Post-scaling midpoints\n{updated_to_current_est}")
            print(f"Post-scaling fake zeros\n{biased_ests[:, 0]}")
            print(f"Post-scaling fake ones\n{biased_ests[:, 1]}")

        ns, alphas, betas = self.pseudocount(
            updated_to_current_est, biased_ests, mult_diff=2., debug=debug)

        # Definitely reset state
        self.copy_values(initial_params)
        self.lr = initial_lr

        return ns, alphas, betas

    @staticmethod
    def pseudocount(
            actual_estimates, fake_estimates, mult_diff=None, debug=False):
        """Return a pseudocount given biased values

        Recover count with the assumption that delta_mean = delta_sum_val / n
        I.e. reverse engineer so that n = delta_sum_val / delta_mean

        Args:
            actual_estimates: the estimate of the current mean about
                which to estimate uncertainty via pseudocount
            fake_estimates (np.ndarray): the estimates biased towards
                the GLN's min_val, max_val
            mult_diff (float): Optional factor to multiply the
                difference by (useful if estimate is not across full
                range
        Returns:
            ns, alphas, betas
        """
        assert actual_estimates.shape[0] == fake_estimates.shape[0]
        assert fake_estimates.ndim == 2 and fake_estimates.shape[1] == 2\
               and actual_estimates.ndim == 1
        in_range = [
            jnp.all(jnp.logical_and(x >= 0, x <= 1.))
            for x in (actual_estimates, fake_estimates)]
        if not all(in_range):
            print(f"WARN - some estimates out of range {in_range}")
        if jnp.any(actual_estimates[:, None] == fake_estimates):
            raise ValueError(f"\n{actual_estimates}\n{fake_estimates}")

        diff = (
            actual_estimates[:, None] - fake_estimates) * jnp.array([1., -1.])
        diff = jnp.where(diff == 0., 1e-8, diff)
        if mult_diff is not None:
            diff *= mult_diff

        n_ais0 = fake_estimates[:, 0] / diff[:, 0]
        n_ais1 = (1. - fake_estimates[:, 1]) / diff[:, 1]
        n_ais = jnp.dstack((n_ais0, n_ais1))

        ns = jnp.squeeze(jnp.min(n_ais, axis=-1), axis=0)
        alphas = actual_estimates * ns + 1.
        betas = (1. - actual_estimates) * ns + 1.

        if debug:
            print(f"ns={ns}\nalphas=\n{alphas}\nbetas=\n{betas}")

        assert jnp.all(ns > 0), f"\nns={ns}\nalphas=\n{alphas}\nbetas=\n{betas}"
        assert jnp.all(alphas > 0.) and jnp.all(betas > 0.), (
            f"\nalphas=\n{alphas}\nbetas=\n{betas}")

        return ns, alphas, betas

    def update_learning_rate(self, lr):
        # updates the learning rate to the new value
        self.lr = lr

    def set_bias_weights(self, bias_vals):
        """Sets the weights for the bias inputs

        args:
            bias_vals (List[Float]): the values to set each of the bias
                weights to, in order of the bias mu. If one of these is
                None, then don't update that bias weight
        """

        assert len(bias_vals) == self.bias_len

        biased_gln_params = {}
        for layer_key, flat_component in self.gln_params.items():
            w_temp = flat_component["weights"]
            for i, bias_val in enumerate(bias_vals):
                if bias_val is not None:
                    # Update the bias weight if we have a value for it.
                    # The bias weights are at the end of the weight arrays.
                    # E.g. the first bias weight is at index -1 * self.bias_len
                    w_temp = jax.ops.index_update(
                        w_temp,
                        jax.ops.index[..., -self.bias_len + i],
                        bias_val)

            biased_gln_params[layer_key] = {"weights": w_temp}

        # update the gln_params which we actually use
        self.copy_values(biased_gln_params)

    def check_weights(self):
        for v in self.gln_params.values():
            if jnp.isnan(v['weights']).any():
                raise ValueError("Has Nans in weights")

    def copy_values(self, new_gln_params, debug=False):
        """Copy only the values of some GLN parameters

        Requires GLN to have same data structure of weights - see
        assumptions in code
        """
        assert len(new_gln_params) == len(self.gln_params), (
            "GLN parameter structures don't match")
        if debug:
            print(f"Will copy new weights {new_gln_params.keys()} "
                  f"into current {self.gln_params.keys()}")
            print(f"Current param type {type(self.gln_params)}")
        gln_params = {}
        for layer_key, self_layer_key in zip(
                new_gln_params.keys(), self.gln_params.keys()):
            if debug:
                print(f"Copying {layer_key} into {self_layer_key}, "
                      f"current weights type: "
                      f"{type(self.gln_params[self_layer_key]['weights'])}"
                      f"new weights type: "
                      f"{type(new_gln_params[layer_key]['weights'])}, ")
            subkeys = list(new_gln_params[layer_key].keys())
            assert len(subkeys) == 1 and subkeys[0] == "weights", subkeys
            new_weights = new_gln_params[layer_key]["weights"].copy()
            gln_params[self_layer_key] = {"weights": new_weights}
        self.gln_params = gln_params
