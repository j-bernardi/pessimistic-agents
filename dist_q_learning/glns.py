import haiku as hk
import jax
import jax.numpy as jnp
import tree

from gated_linear_networks import gaussian


# TEMP
import os
import numpy as np
######


JAX_RANDOM_KEY = jax.random.PRNGKey(0)
print("KEY TYPE", JAX_RANDOM_KEY)


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
            min_sigma_sq=0.001,
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
            self._rng = hk.PRNGSequence(JAX_RANDOM_KEY)

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

        self.transform_to_positive = jax.jit(self._transform_to_positve)

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

    @staticmethod
    def _transform_to_positve(feat):
        """Transform from [-1, 1] to [0, 1], jitted"""
        return jnp.asarray(0.5) + feat / jnp.asarray(2.)

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
        input_features = jnp.asarray(inputs)
        # Input mean usually the x values, but can just be a PDF
        # standard deviation is so that sigma_squared spans whole space
        if self.feat_mean == 0.:
            gln_input = self.transform_to_positive(input_features)
        elif self.feat_mean == 0.5:
            gln_input = input_features
        else:
            raise ValueError(f"feat mean {self.feat_mean} invalid")
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

    def uncertainty_estimate(
            self, states, x_batch, y_batch, max_est_scaling=None,
            converge_epochs=20, debug=False):
        """Get parameters to Beta distribution defining uncertainty

        Caps all predictions between [0, 1]. If real and fake_0 == 0,
        the pseudocount goes very high, but will hopefully be capped by
        fake_1. The alternative is a risk of negative alphas or betas.

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

        def batch_learn(xx, yy):
            for ii in range(0, xx.shape[0], self.batch_size):
                self.predict(
                    xx[ii:ii+self.batch_size], yy[ii:ii+self.batch_size])

        # TEMP - not converging
        # for n_conv in range(converge_epochs):
        #     if debug:
        #         print(f"Conv step {n_conv}/{converge_epochs}")
        #     self.predict(x_batch, y_batch)
        # self.check_weights()
        # post_convergence_means = self.predict(states)

        if debug:
            print("Pre convergence")
            print(pre_convergence_means)
            # print("Post convergence")
            # print(post_convergence_means)
        # assert post_convergence_means.shape == (states.shape[0],), (
        #     f"{post_convergence_means.shape}, {states.shape[0]}")
        fake_targets = jnp.stack(
            (jnp.full(states.shape[0], 0.),
             jnp.full(states.shape[0], 1.)),
            axis=1)
        fake_means = [[] for _ in range(fake_targets.shape[1])]
        converged_ws = hk.data_structures.to_immutable_dict(self.gln_params)
        converged_ws2 = hk.data_structures.to_immutable_dict(self.gln_params)
        for j in range(fake_targets.shape[1]):
            # Update towards a fake data point using Newton's method
            # TODO - contention on whether linear bs or root(bs) here
            # Using linear for higher PC ; not needed as BS preserved
            # self.update_learning_rate(initial_lr / self.batch_size)
            # TODO to improve batch selection - select a batch of states that
            #  are reasonably well spaced. 2 samples next to each other may
            #  unreasonably increase the uncertainty
            self.predict(states, fake_targets[:, j])
            self.update_learning_rate(initial_lr)
            batch_learn(x_batch, y_batch)
            new_ests = self.predict(states)
            fake_means[j] = new_ests
            # Clean up
            self.copy_values(converged_ws)

        assert self.weights_equal(converged_ws2)
        fake_means = jnp.asarray(fake_means)

        # Now do 1 update before calculating current estimate
        batch_learn(x_batch, y_batch)
        current_est = self.predict(states)  # TODO removed clips here
        # current_est = jnp.clip(self.predict(states), a_min=0., a_max=1.)
        self.copy_values(initial_params)
        ###

        if max_est_scaling is not None:
            fake_means /= max_est_scaling
        # Added clip, as usually falling outside is just a rounding error
        # rather than indicative of something wrong. A little risky as it
        # may mask terrible behaviour. At least checking current est (above)
        # should help
        # TODO - removed clips here
        biased_ests = fake_means.T

        # greater = biased_ests[:, 0] > updated_to_current_est
        # lesser = biased_ests[:, 1] < updated_to_current_est
        # if jnp.any(jnp.logical_and(greater, lesser)):
        #     raise ValueError("Cannot hack")

        if debug:
            print(f"Post-scaling midpoints\n{current_est}")
            print(f"Post-scaling fake zeros\n{biased_ests[:, 0]}")
            print(f"Post-scaling fake ones\n{biased_ests[:, 1]}")

        ns, alphas, betas = self.pseudocount(
            current_est, biased_ests, debug=debug, lr=1.)

        # Definitely reset state
        self.copy_values(initial_params)
        self.lr = initial_lr

        # TEMP - save ns
        experiment = "batched_higher_rf_0_inf_zero_minsig_q3"
        os.makedirs(
            os.path.join("pseudocount_invest", experiment), exist_ok=True)
        join = lambda p: os.path.join(
            "pseudocount_invest", experiment, f"{self.name}_{p}")
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
        np.save(join("prev_hist_x.npy"), x_batch)
        np.save(join("prev_hist_y.npy"), y_batch)
        np.save(join("prev_n.npy"), prev_n)
        np.save(join("prev_s.npy"), prev_s)
        np.save(join("prev_n_update.npy"), n_updates)
        ##############

        self.update_count += 1

        return ns, alphas, betas

    @staticmethod
    def pseudocount(actual_estimates, fake_estimates, debug=False, lr=1.):
        """Return a pseudocount given biased values

        Recover count with the assumption that delta_mean = delta_sum_val / n
        I.e. reverse engineer so that n = delta_sum_val / delta_mean

        Args:
            actual_estimates: the estimate of the current mean about
                which to estimate uncertainty via pseudocount
            fake_estimates (np.ndarray): the estimates biased towards
                the GLN's min_val, max_val
            debug (bool): print to command line debugging info
            lr (float): learning rate assumed
        Returns:
            ns, alphas, betas

        TODO:
          - Decide if we really want the learning rate
        """
        assert actual_estimates.shape[0] == fake_estimates.shape[0]
        assert fake_estimates.ndim == 2 and fake_estimates.shape[1] == 2\
               and actual_estimates.ndim == 1
        in_range = [
            jnp.all(jnp.logical_and(x >= 0, x <= 1.))
            for x in (actual_estimates, fake_estimates)]
        if not all(in_range):
            print(
                f"WARN - some estimates out of range {in_range}"
                f"\nActual:\n{actual_estimates}\nFake:\n{fake_estimates}")
        equal = actual_estimates[:, None] == fake_estimates
        if jnp.any(equal):
            print(f"WARN: some values equal\n{equal}"
                  f"\n{actual_estimates}\n{fake_estimates}")
        fake_diff = (fake_estimates[:, 1] - fake_estimates[:, 0])
        if jnp.any(fake_diff < 0):
            # Probably because of high sigma sq?
            print("FAKE DIFF IS LESS THAN 0")
            fake_diff = jnp.where(fake_diff < 0, 1e-8, fake_diff)

        delta_est = (
            jnp.minimum(1, actual_estimates + lr)
            - jnp.maximum(0, actual_estimates - lr)
        )
        if jnp.any(delta_est < 0):
            print("DELTA EST IS LESS THAN 0")
            delta_est = jnp.where(delta_est < 0, 0, delta_est)

        # NOTE - ommitted the -1 from the diff !=0 term
        ns = jnp.where(fake_diff == 0., 1e8, delta_est / fake_diff)

        # TODO - this is a little hacky to say whenever diff is negative
        lessthan = (ns < 0)
        if jnp.any(lessthan):
            print("WARN - some values less than")
            ns = jnp.where(lessthan, 1, ns)

        clipped_est = jnp.clip(actual_estimates, a_min=0., a_max=1.)
        alphas = clipped_est * ns + 1.
        betas = (1. - clipped_est) * ns + 1.

        if debug:
            print(f"ns=\n{ns}\nalphas=\n{alphas}\nbetas=\n{betas}")

        assert jnp.all(ns >= 0)\
               and jnp.all(alphas > 0.) and jnp.all(betas > 0.), (
            f"\nFake ests{fake_estimates}\nactual ests{actual_estimates}"
            f"\nclipped ests\n{clipped_est}"
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
