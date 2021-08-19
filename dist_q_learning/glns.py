import haiku as hk
import jax
import jax.numpy as jnp
import tree

from gated_linear_networks import gaussian

# TEMP
import os
import numpy as np
######


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
            # this is hyperplane bias only; drawn from normal with this dev
            bias_std=0.05,
            bias_max_mu=1.):
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
            respectively. Side info always in [-1, 1], so this is
            features only
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
        self.bias_std = bias_std
        self.context_dim = context_dim
        self.feat_mean = feat_mean
        self.bias_len = bias_len
        self.bias_max_mu = bias_max_mu
        self.lr = lr
        self.name = name
        self.min_sigma_sq = min_sigma_sq
        self.batch_size = batch_size
        self.update_count = 0

        def display(*args):
            p_string = f"\nCreating GLN {self.name} with:"
            for v in args:
                p_string += f"\n{v}={getattr(self, v)}"
            print(p_string)

        display(
            "feat_mean", "lr", "min_sigma_sq", "bias_len", "bias_std",
            "bias_max_mu")

        if rng_key is not None:
            self._rng = hk.PRNGSequence(jax.random.PRNGKey(rng_key))
        else:
            self._rng = hk.PRNGSequence(jax.random.PRNGKey(0))

        # make init, inference and update functions,
        # these are GPU compatible thanks to jax and haiku
        def gln_factory():
            return gaussian.GatedLinearNetwork(
                output_sizes=self.layer_sizes,
                context_dim=self.context_dim,
                bias_len=self.bias_len,
                name=self.name,
                bias_std=self.bias_std,
                bias_max_mu=self.bias_max_mu
            )

        def inference_fn(inputs, side_info):
            return gln_factory().inference(
                inputs, side_info, self.min_sigma_sq)

        def batch_inference_fn(inputs, side_info):
            return jax.vmap(inference_fn, in_axes=(0, 0))(inputs, side_info)

        def update_fn(inputs, side_info, label, learning_rate):
            params, predictions, unused_loss = gln_factory().update(
                inputs, side_info, label, learning_rate, self.min_sigma_sq)
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

        def hessian_update_fn(inputs, side_info, label, learning_rate):
            # TODO should this 0.5 be min_sigma_squared? How to check?
            params, predictions, unused_loss = gln_factory().update(
                inputs, side_info, label, learning_rate, self.min_sigma_sq,
                use_newtons=True)

            return predictions, params

        def batch_hessian_update_fn(inputs, side_info, label, learning_rate):
            # Cancelled batching - so that we can sum hessians and take inverse
            predictions, updated_params = hessian_update_fn(
                    inputs,
                    side_info,
                    label,
                    learning_rate)
            raise NotImplementedError()
            return predictions, updated_params

        self.transform_to_positive = jax.jit(self._transform_to_positve)

        # Haiku transform functions.
        self._init_fn, inference_fn_ = hk.without_apply_rng(
            hk.transform_with_state(inference_fn))
        self._batch_init_fn, batch_inference_fn_ = hk.without_apply_rng(
            hk.transform_with_state(batch_inference_fn))
        _, update_fn_ = hk.without_apply_rng(
            hk.transform_with_state(update_fn))
        _, batch_update_fn_ = hk.without_apply_rng(
            hk.transform_with_state(batch_update_fn))
        _, hessian_update_fn_ = hk.without_apply_rng(
            hk.transform_with_state(hessian_update_fn))
        _, batch_hessian_update_fn_ = hk.without_apply_rng(
            hk.transform_with_state(batch_hessian_update_fn))

        self._inference_fn = jax.jit(inference_fn_)
        self._batch_inference_fn = jax.jit(batch_inference_fn_)
        self._update_fn = jax.jit(update_fn_)
        self._batch_update_fn = jax.jit(batch_update_fn_)
        self._hessian_update_fn = jax.jit(hessian_update_fn_)
        self._batch_hessian_update_fn = jax.jit(batch_hessian_update_fn_)

        if self.batch_size is None:
            self.init_fn = self._init_fn
            self.inference_fn = self._inference_fn
            self.update_fn = self._update_fn
            self.hessian_update_fn = self._hessian_update_fn
            dummy_inputs = jnp.ones([input_size, 2])
            dummy_side_info = jnp.ones([input_size])
        else:
            self.init_fn = self._batch_init_fn
            self.inference_fn = self._batch_inference_fn
            self.update_fn = self._batch_update_fn
            # NOTE not using batched - not implemented / possible
            self.hessian_update_fn = self._hessian_update_fn
            dummy_inputs = jnp.ones([self.batch_size, input_size, 2])
            dummy_side_info = jnp.ones([self.batch_size, input_size])

        # initialise the GGLN
        self.gln_params, self.gln_state = self.init_fn(
            next(self._rng), dummy_inputs, dummy_side_info)

        if init_bias_weights is not None:
            print("Setting bias weights")
            self.set_bias_weights(init_bias_weights)

    def _transform_to_positve(self, feat):
        """Transform from [-1, 1] to [0, 1], jitted"""
        if self.feat_mean == 0.:
            return jnp.asarray(0.5) + feat / jnp.asarray(2.)
        elif self.feat_mean == 0.5:
            return feat  # already in range
        else:
            raise ValueError(f"feat mean {self.feat_mean} invalid")

    def predict(
            self, inputs, target=None, return_sigma=False, nodewise=False,
            use_newtons=False,
    ):
        """Performs predictions and updates for the GGLN

        Args:
            inputs (jnp.ndarray): A (N, context_dim) array of input
                features
            target (Optional[jnp.ndarray]): A (N, outputs) array of
                targets. If provided, the GGLN parameters are updated
                toward the target. Else, predictions are returned.
            return_sigma (bool): if True, return a tuple of mu, sigma
            nodewise (bool): if True, return the predictions for all
                nodes. Else, just the final layer
            use_newtons (bool): if True, use Newton's method of
                approximation rather than a standard gradient descent.
                Note - assumes delta_weights is small.
        Returns:
            mu, product of Gaussians (or see return_sgima)
        """
        if (
            (target is not None and any((return_sigma, nodewise)))
            or (target is None and use_newtons)
        ):
            raise NotImplementedError()
        # Sanitise inputs
        input_features = jnp.asarray(inputs)
        # Input mean usually the x values, but can just be a PDF
        # standard deviation is so that sigma_squared spans whole space
        gln_input = self.transform_to_positive(input_features)
        initial_pdfs = (
            # jnp.full_like(input_features, self.feat_mean),
            gln_input,
            jnp.full_like(input_features, 1.),
        )
        target = jnp.asarray(target) if target is not None else None

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
                self.gln_params, self.gln_state, inputs_with_sig_sq, side_info,
            )
            return_nodes = slice(0, predictions.shape[-2]) if nodewise else -1
            returns = [predictions[..., return_nodes, 0]]
            if return_sigma:
                returns.append(predictions[..., return_nodes, 1])
            if len(returns) == 1:
                return returns[0]
            else:
                return tuple(returns)
        else:
            # if a target is provided, update the GLN parameters
            if use_newtons:
                update_fn = self.hessian_update_fn
            else:
                update_fn = self.update_fn
            # TODO compiling hessian update function is - extremely slow
            (_, new_gln_params), _ = update_fn(
                self.gln_params, self.gln_state, inputs_with_sig_sq, side_info,
                target, learning_rate=self.lr)

            self.gln_params = new_gln_params
            self.check_weights()

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
            converge_epochs (int): number of epochs to run to
                convergence for
            debug (bool): print more info

        Returns:
            ns (jnp.ndarray): pseudocounts for each state
            alphas (jnp.ndarray): alpha parameters of a beta
                distribution, per-state in the batch
            betas (jnp.ndarray): beta parameters of a beta distribution,
                per-state in the batch
        """
        if debug:
            print(f"\nUncert estimate for {self.name}, lr={self.lr}")
        initial_lr = self.lr
        initial_params = hk.data_structures.to_immutable_dict(self.gln_params)

        pre_convergence_means = self.predict(states)

        # Converge so that assumption that delta_w is small when minimising
        # L(w+delta_w) to find delta_w (Taylor's expansion)
        # However note that whole history (x_batch) might not include update
        # states if sampled differently.
        # Probably don't use hessian udpate here?
        # TODO - not converged after 20 epochs; still predicting sub-0.8
        #  Theoretical amount of steps, given LR? For now, try "more".
        for n_conv in range(converge_epochs):
            if debug:
                print(f"Conv step {n_conv}/{converge_epochs}")
            self.predict(x_batch, y_batch)
        self.check_weights()

        # post_convergence_means = self.predict(states)
        if debug:
            print("Pre convergence")
            print(pre_convergence_means)
            # print("Post convergence")
            # print(post_convergence_means)
        # assert post_convergence_means.shape == (states.shape[0],), (
        #     f"{post_convergence_means.shape}, {states.shape[0]}")
        fake_targets = jnp.stack(
            # TODO - investigate the pseudocount method; it's not consistently
            #  0 -> less, 1 -> more at the moment
            (pre_convergence_means,
             jnp.full(states.shape[0], 0.),
             jnp.full(states.shape[0], 1.)),
            axis=1)
        fake_means = jnp.empty((states.shape[0], fake_targets.shape[1]))
        converged_ws = hk.data_structures.to_immutable_dict(self.gln_params)
        for i, s in enumerate(states):
            for j, fake_target in enumerate(fake_targets[i]):
                # Update towards a fake data point using Newton's method
                xs = jnp.expand_dims(s, 0)
                ys = jnp.expand_dims(fake_target, 0)
                self.predict(xs, ys, use_newtons=True)
                new_est = jnp.squeeze(self.predict(jnp.expand_dims(s, 0)), 0)
                fake_means = jax.ops.index_update(fake_means, (i, j), new_est)

                # Clean up
                self.copy_values(converged_ws)
                self.update_learning_rate(initial_lr)

        # TODO 2 - try going back to a single update? But is this hessian
        #  anything like what we want?
        #  Although, currently the sum of the hessians does mean that a larger
        #  batch size may well be washing-out the fake data point that we
        #  should be being concerned about. Suggests we don't want to be
        #  increasing batch size?
        #  Or when we increase batch size, incease by a proportional amount?

        if max_est_scaling is not None:
            fake_means /= max_est_scaling
        updated_to_current_est = fake_means[:, 0]
        # Added clip, as usually falling outside is just a rounding error
        # rather than indicative of something wrong. A little risky as it
        # may mask terrible behaviour. At least checking current est (above)
        # should help
        biased_ests = jnp.clip(fake_means[:, 1:], a_min=0., a_max=1.)

        # greater = biased_ests[:, 0] > updated_to_current_est
        # lesser = biased_ests[:, 1] < updated_to_current_est
        # if jnp.any(jnp.logical_and(greater, lesser)):
        #     raise ValueError("Cannot hack")

        if debug:
            print(f"Post-scaling midpoints\n{updated_to_current_est}")
            print(f"Post-scaling fake zeros\n{biased_ests[:, 0]}")
            print(f"Post-scaling fake ones\n{biased_ests[:, 1]}")

        ns, alphas, betas = self.pseudocount(
            updated_to_current_est, biased_ests, debug=debug)

        # Definitely reset state
        self.copy_values(initial_params)
        self.lr = initial_lr

        # TEMP - save ns
        save = False
        if save:
            experiment = "single_update"
            os.makedirs(
                os.path.join("batched_hessian", experiment), exist_ok=True)
            join = lambda p: os.path.join(
                "batched_hessian", experiment, f"{self.name}_{p}")
            if os.path.exists(join("prev_n.npy")):
                prev_n = np.load(join("prev_n.npy"))
                prev_n = np.concatenate((prev_n, np.expand_dims(ns, axis=0)), axis=0)
                prev_s = np.load(join("prev_s.npy"))
                prev_s = np.concatenate(
                    (prev_s, np.expand_dims(states, axis=0)),
                    axis=0)
                n_updates = np.load(join("prev_n_update.npy"))
                n_updates = np.concatenate((n_updates, [self.update_count]))
            else:
                prev_n = np.expand_dims(ns, axis=0)
                prev_s = np.expand_dims(states, axis=0)
                n_updates = np.array([self.update_count])
            # updated in a mo
            assert prev_n.shape[0] == prev_s.shape[0] == n_updates.shape[0]
            np.save(join("prev_n.npy"), prev_n)
            np.save(join("prev_s.npy"), prev_s)
            np.save(join("prev_n_update.npy"), n_updates)
            ##############

        self.update_count += 1

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
            debug (bool): print to command line debugging info
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
            print(f"WARN - some estimates out of range {in_range}"
                  f"\nActual:\n{actual_estimates}\nFake:\n{fake_estimates}")
        if jnp.any(actual_estimates[:, None] == fake_estimates):
            print(
                f"WARN: some actual estimates and fake estimates are the same"
                f"Actual:\n{actual_estimates}\nFakes:\n{fake_estimates}")

        diff = (
            actual_estimates[:, None]
            - fake_estimates) * jnp.asarray([1., -1.])
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
            print(f"ns=\n{ns}\nalphas=\n{alphas}\nbetas=\n{betas}")

        assert jnp.all(ns >= 0)\
               and jnp.all(alphas > 0.) and jnp.all(betas > 0.), (
            f"\nns=\n{ns}\nalphas=\n{alphas}\nbetas=\n{betas}")

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
            if jnp.isnan(v["weights"]).any():
                raise ValueError(f"{self.name} has NaNs in weights")

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

    def weights_equal(self, compare_with_params, debug=False):
        equals = []
        for layer_key, self_layer_key in zip(
                compare_with_params.keys(), self.gln_params.keys()):
            subkeys = list(compare_with_params[layer_key].keys())
            assert len(subkeys) == 1 and subkeys[0] == "weights", subkeys
            equals.append(
                jnp.all(
                    self.gln_params[self_layer_key]["weights"]
                    == compare_with_params[layer_key]["weights"]))
        if debug:
            print(f"Comparing given weights with {self.name}")
            print(f"Layers equal: {equals}")
        return all(equals)
