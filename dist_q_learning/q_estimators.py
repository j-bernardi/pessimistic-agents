import abc

import scipy.stats
import numpy as np
import jax.numpy as jnp
import torch as tc

import glns
from estimators import (
    Estimator, BURN_IN_N, DEFAULT_GLN_LAYERS, get_burnin_states)
from utils import geometric_sum
import bayes_by_backprop


class QTableEstimator(Estimator, abc.ABC):
    """Base class for both the finite and infinite horizon update

    Implements both types of Q table. Attribute q_table: ndim=3,
    (states, actions, num_horizons+1)

    Infinite Horizon Q Estimator (bootstrapping). num_horizons=1 (so final
    dim represents (0-step-future-R (always == 0), inf_R==Q_val).

    And Finite Horizon Temporal Difference Q Estimator, as in:
        https://arxiv.org/pdf/1909.03906.pdf

        Recursively updates the next num_horizons estimates of the reward:
            Q_0(s, a) = mean_{t : s_t = s} r_t
            V_i(s) = max_a Q_i(s, a)
            Q_{i+1}(s, a) = mean_{t : s_t = s}[
                (1 - gamma) * r_t  + gamma * V_i(s_{t+1})
            ]
    """
    def __init__(
            self,
            num_states,
            num_actions,
            gamma,
            lr,
            q_table_init_val=0.,
            scaled=True,
            has_mentor=True,
            horizon_type="inf",
            num_horizons=1,
            **kwargs
    ):
        """Initialise a Q Estimator of infinite or finite horizon

        Additional Args:
            q_table_init_val (float): Value to initialise Q table at
            scaled: if True, all Q values are in scaled range [0, 1],
                rather than the actual discounted sum or rewards.
            has_mentor: if there is no mentor, we must set an initial
                random action probability (to explore, at first).
            horizon_type (str): one of "inf" or "finite". Defines
                estimator behaviour
            num_horizons (int): Number of steps to look ahead into the
                future. Must be 1 if horizon_type="inf" - store only 1
                Q value (inf)
            kwargs: see super()
        """
        super().__init__(lr, scaled=scaled, **kwargs)
        self.num_states = num_states
        self.num_actions = num_actions
        self.gamma = gamma
        self.q_table_init_val = q_table_init_val

        self.transition_table = np.zeros(
            (num_states, num_actions, num_states), dtype=int)

        self.random_act_prob = None if has_mentor else 1.
        self.horizon_type = horizon_type
        self.num_horizons = num_horizons

        if self.horizon_type not in ("inf", "finite"):
            raise ValueError(f"Must be in inf, finite: {horizon_type}")
        if not isinstance(self.num_horizons, int) or self.num_horizons < 0:
            raise ValueError(f"Must be int number of steps: {self.num_horizons}")

        if self.horizon_type == "finite":
            if self.scaled:
                raise NotImplementedError(
                    f"Not defined for scaled Q values. Try --unscale-q ?")
            if self.num_horizons <= 0:
                raise ValueError(f"Must be > 0 future steps: {self.num_horizons}")
        elif self.num_horizons != 1:
            raise ValueError(
                f"Num steps must be == 1 for inf horizon: {self.num_horizons}")

        # account for Q_0, add one in size to the "horizons" dimension
        self.q_table = self.make_q_estimator()

    def make_q_estimator(self):
        """Define the Q table for finite horizon
        TODO:
            - Geometirc series to scale each init q, not actually all
              the same
        """
        # Q_0 and Q_inf in finite case
        q_table = np.zeros(
            (self.num_states, self.num_actions, self.num_horizons + 1))
        q_table += self.q_table_init_val  # Initialise all to init val

        # Q_0 always init to 0 in finite case: future is 0 from now
        q_table[:, :, 0] = 0.
        return q_table

    def reset(self):
        self.transition_table = np.zeros(
            (self.num_states, self.num_actions, self.num_states), dtype=int)
        self.q_table = self.make_q_estimator()

    def estimate(self, state, action, h=None, q_table=None):
        """Estimate the future Q, using this estimator (or q_table)

        Args:
            state (int): the current state
            action (int): the action taken
            h (int): the horizon we are estimating for. If None, use the
                final horizon, (usually used for choosing actions, and
                handles infinite case).
            q_table (np.ndarray): the Q table to estimate the value
                from, if None use self.q_table as default.

        Returns:
            q_estimate (float): Estimate of the q value for horizon h
        """
        if q_table is None:
            q_table = self.q_table

        if h is None:
            h = -1

        return q_table[state, action, h]

    def update_estimator(
            self, state, action, q_target, horizon=None, update_table=None,
            lr=None
    ):
        """Update the q_table (or arg) towards q_target

        horizon (Optional[int]): the index of the Q-horizon to update.
            If None, update the final index (handles infinite case).
        """

        if update_table is None:
            update_table = self.q_table

        if lr is None:
            lr = self.get_lr(state, action)

        if horizon is None:
            horizon = -1

        update_table[state, action, horizon] += lr * (
                q_target - update_table[state, action, horizon])

        self.decay_lr()

    def get_random_act_prob(self, decay_factor=5e-5, min_random=0.05):
        if (self.random_act_prob is not None
                and self.random_act_prob > min_random):
            self.random_act_prob *= (1. - decay_factor)
        return self.random_act_prob

    def get_lr(self, state=None, action=None):
        """ Calculate the learning rate for updating the Q table.

        If self.lr is specified use that, otherwise use 1/(n+1),
        where n is the number of times the state has been visted.

        Args:
            state (int): The state the Q table is being updated for
            action (int): The action the Q table is being updated for

        """
        if self.lr is not None:
            return self.lr
        else:
            return 1. / (1. + self.transition_table[state, action, :].sum())


class BasicQTableEstimator(QTableEstimator):
    """Uses FHTDQ with a basic update to r + gamma * Q_h"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def update(self, history):
        for state, action, reward, next_state, done in history:
            self.transition_table[state, action, next_state] += 1

            # If finite, num steps is 0 and does 1 update
            for h in range(1, self.num_horizons + 1):
                # Estimate Q_h
                if not done:
                    next_q = np.max([
                        self.estimate(
                            next_state, action_i,
                            h=None if self.horizon_type == "inf" else h-1
                        ) for action_i in range(self.num_actions)]
                    )
                else:
                    next_q = 0.

                scaled_r = (1. - self.gamma) * reward \
                    if self.scaled else reward
                q_target = scaled_r + self.gamma * next_q

                super().update_estimator(
                    state, action, q_target,
                    horizon=None if self.horizon_type == "inf" else h
                )


class QuantileQEstimator(QTableEstimator):
    """A Q table estimator that makes pessimistic updates e.g. using IRE

    Updates using algorithm 3 in the QuEUE specification.
    """

    def __init__(
            self,
            quantile,
            immediate_r_estimators,
            use_pseudocount=False,
            **kwargs
    ):
        """Set up the QEstimator for the given quantile

        'Burn in' the quantiles by calling 'update' with an artificial
        historical reward - Algorithm 4. E.g. call update with r=i, so
        that it updates theta_i_a, parameters for this estimator, s.t.:
            expected(estimate(theta_i_a)) for i=0.1 -> 0.1

        Args:
            quantile (float): the pessimism-quantile that this estimator
                is estimating the future-Q value for.
            immediate_r_estimators (list[ImmediateRewardEstimator]): A
                list of IRE objects, indexed by-action
        """
        super().__init__(**kwargs)
        if quantile <= 0. or quantile > 1.:
            raise ValueError(f"Require 0. < q_val <= 1. {quantile}")

        self.quantile = quantile  # the value of the quantile
        self.use_pseudocount = use_pseudocount
        self.immediate_r_estimators = immediate_r_estimators

    def update(self, history, capture_alpha_beta=None):
        """Algorithm 3. Use history to update future-Q quantiles.

        The Q-estimator stores estimates of multiple quantiles in the
        distribution (with respect to epistemic uncertainty) of the
        Q-value.

        It updates by boot-strapping at a given state-action pair.

        Args:
            history (list): (state, action, reward, next_state, done)
                tuples
            capture_alpha_beta (tuple): A (state, action) tuple where,
                if given, return a list of all the alpha, beta
                parameters encountered for the final horizon only:
                    [((ire_alpha, ire_beta), (q_alpha, q_beta)), ...]

        Updates parameters for this estimator, theta_i_a
        """
        alpha_betas = []
        for state, action, reward, next_state, done in history:
            self.transition_table[state, action, next_state] += 1

            for h in range(1, self.num_horizons + 1):
                if not done:
                    future_q = np.max([
                        self.estimate(
                            next_state, action_i,
                            h=None if self.horizon_type == "inf" else h-1
                        ) for action_i in range(self.num_actions)]
                    )
                else:
                    future_q = 0.

                ire = self.immediate_r_estimators[action]
                ire_alpha, ire_beta = ire.expected_with_uncertainty(state)
                iv_i = scipy.stats.beta.ppf(self.quantile, ire_alpha, ire_beta)
                # Note: not scaled
                scaled_iv_i = (1. - self.gamma) * iv_i if self.scaled else iv_i

                q_target = self.gamma * future_q + scaled_iv_i

                self.update_estimator(
                    state, action, q_target,
                    horizon=None if self.horizon_type == "inf" else h
                )

                # Account for uncertainty in the state transition function
                q_ai = self.estimate(
                    state, action, h=None if self.horizon_type == "inf" else h)
                if not self.use_pseudocount:
                    n = self.transition_table[state, action, :].sum()
                else:
                    fake_q_ai = []
                    for fake_target in [0., 1.]:
                        fake_q_table = self.q_table.copy()
                        self.update_estimator(
                            state, action, fake_target,
                            update_table=fake_q_table,
                            horizon=None if self.horizon_type == "inf" else h)
                        fake_q_ai.append(
                            self.estimate(
                                state, action,
                                h=None if self.horizon_type == "inf" else h,
                                q_table=fake_q_table)
                        )

                    n_ai0 = fake_q_ai[0] / (q_ai - fake_q_ai[0])
                    n_ai1 = (1. - fake_q_ai[1]) / (fake_q_ai[1] - q_ai)
                    # in the original algorithm this min was also over the
                    # quantiles
                    # as well as the fake targets
                    n = np.min([n_ai0, n_ai1])

                if not self.scaled and self.horizon_type == "finite":
                    max_q = geometric_sum(1., self.gamma, h)
                elif not self.scaled:
                    max_q = geometric_sum(1., self.gamma, "inf")
                else:
                    max_q = 1.

                q_alpha = q_ai * n + 1.
                q_beta = (max_q - q_ai) * n + 1.
                q_target_transition = scipy.stats.beta.ppf(
                    self.quantile, q_alpha, q_beta)

                self.update_estimator(
                    state, action, q_target_transition,
                    lr=1./(1. + self.transition_table[state, action, :].sum()),
                    horizon=None if self.horizon_type == "inf" else h
                )
                self.total_updates += 1

                # Only capture last horizon alpha beta
                if capture_alpha_beta and h == self.num_horizons\
                        and (state, action) == capture_alpha_beta:
                    alpha_betas.append(
                        ((ire_alpha, ire_beta), (q_alpha, q_beta)))

        return alpha_betas


class QEstimatorIRE(QTableEstimator):
    """A basic Q table estimator which uses IRE instead of reward

    Does not do "2nd update" (i.e. skips transition uncertainty)

    Implements both pessimistic IRE and mean IRE estimates, flagged with
        `quantile is None`

    Updates (as 'usual') towards the target of:
        gamma * Q(s', a*) + (1. - gamma) * IRE(state)
    """

    def __init__(self, quantile, immediate_r_estimators, **kwargs):
        """Init the quantile and IREs needed for updates on pess IRE

        Args:
            quantile (Optional[float]): the quantile value. If None,
                use expected IRE (mean).
            immediate_r_estimators: per action
            kwargs: see super()
        """
        super().__init__(**kwargs)
        self.immediate_r_estimators = immediate_r_estimators
        self.quantile = quantile  # If true, use quantile

    def update(self, history):
        for state, action, reward, next_state, done in history:
            self.transition_table[state, action, next_state] += 1

            # If finite, num steps is 0 and does 1 update
            for h in range(1, self.num_horizons + 1):
                # Estimate Q_h
                if not done:
                    next_q = np.max([
                        self.estimate(
                            next_state, action_i,
                            h=None if self.horizon_type == "inf" else h-1
                        ) for action_i in range(self.num_actions)]
                    )
                else:
                    next_q = 0.

                ire_estimator = self.immediate_r_estimators[action]
                if self.quantile is not None:
                    ire_alpha, ire_beta = (
                        ire_estimator.expected_with_uncertainty(state))
                    ire = scipy.stats.beta.ppf(
                        self.quantile, ire_alpha, ire_beta)
                else:
                    ire = ire_estimator.estimate(state)

                scaled_ire = (1. - self.gamma) * ire if self.scaled else ire
                q_target = scaled_ire + self.gamma * next_q

                super().update_estimator(
                    state, action, q_target,
                    horizon=None if self.horizon_type == "inf" else h)


class QuantileQEstimatorGaussianGLN(Estimator):
    """A  GGLN Q estimator that makes pessimistic updates e.g. using IRE

    Updates using algorithm 3 in the QuEUE specification.

    Uses IREs (which are also GGLNs) to get pessimistic estimates of the next
    reward, to train a GGLN to estimate the Q value.
    """

    def __init__(
            self, quantile, immediate_r_estimators, dim_states, num_actions,
            gamma, layer_sizes=None, context_dim=4, feat_mean=0.5,
            lr=1e-4, scaled=True, burnin_n=BURN_IN_N, burnin_val=None,
            horizon_type="inf", num_steps=1, batch_size=1):
        """Set up the GGLN QEstimator for the given quantile

        Burns in the GGLN to the burnin_val (default is the quantile value)
        for burnin_n steps, where the state for burn in is randomly sampled.

        Args:
            quantile (float): the pessimism-quantile that this estimator
                is estimating the future-Q value for.
            immediate_r_estimators (
                    list[ImmediateRewardEstimator_GLN_gaussian]): A list
                of IRE objects, indexed by-action
            dim_states (int): the dimension of the state space,
                eg, in the 2D cliffworld dim_states = 2
            num_actions (int): the number of actions (4 in cliffworld)
            gamma (float): the discount rate
            layer_sizes (List[int]): The number of neurons in each layer
                for the GGLN
            context_dim (int): the number of hyperplanes used to make the
                halfspaces
            feat_mean (float): mean of all possible inputs to GLN (not
                side info). Typically 0.5, or 0.
            lr (float): the learning rate
            scaled (bool): NOT CURRENTLY IMPLEMENTED
            burnin_val (Optional[float]): the value we burn in the
                estimator with. Defaults to quantile value

        """
        super().__init__(lr, scaled=scaled, burnin_n=burnin_n)

        if quantile <= 0. or quantile > 1. or isinstance(quantile, int):
            raise ValueError(f"Require 0. < q <= 1. {quantile}")

        self.quantile = quantile  # the value of the quantile

        self.immediate_r_estimators = immediate_r_estimators
        self.dim_states = dim_states
        self.num_actions = num_actions
        self.context_dim = context_dim
        self.gamma = gamma

        self.layer_sizes = (
            DEFAULT_GLN_LAYERS if layer_sizes is None else layer_sizes)

        self.horizon_type = horizon_type
        self.num_steps = num_steps
        self.batch_size = batch_size

        if self.horizon_type not in ("inf", "finite"):
            raise ValueError(f"Must be in inf, finite: {horizon_type}")
        if not isinstance(self.num_steps, int) or self.num_steps < 0:
            raise ValueError(f"Must be int number of steps: {self.num_steps}")

        if self.horizon_type == "finite":
            # if self.scaled:
            #     raise NotImplementedError(f"Not defined for scaled Q values")
            if self.num_steps <= 0:
                raise ValueError(f"Must be > 0 future steps: {self.num_steps}")
        elif self.num_steps != 1:
            raise ValueError(
                f"Num steps must be == 0 for inf horizon: {self.num_steps}")

        if burnin_val is None:
            burnin_val = self.quantile

        self._model = None
        self._target_model = None
        self.make_q_estimator(self.layer_sizes, burnin_val, feat_mean)

    def make_q_estimator(self, layer_sizes, burnin_val, mean):

        def model_maker(num_acts, num_steps, weights_from=None, prefix=""):
            """Returns list of lists of model[action][horizon] = GLN"""
            models = []
            q_str = f"{self.quantile:.3f}".replace(".", "_")
            for action in range(num_acts):
                act_models = []
                for s in range(num_steps + 1):
                    if s == 0:
                        act_models.append(None)  # never needed - reduce risk
                        continue
                    act_step_gln = glns.GGLN(
                        name=f"{prefix}QuantileQ_a{action}_s{s}_q{q_str}",
                        layer_sizes=layer_sizes,
                        input_size=self.dim_states,
                        context_dim=self.context_dim,
                        feat_mean=mean,
                        lr=self.lr,
                        batch_size=self.batch_size,
                        # min_sigma_sq=0.5,
                        # bias_len=3,
                        bias_max_mu=1.
                        # init_bias_weights=[None, None, None],
                        # init_bias_weights=[0.1, 0.2, 0.1]
                    )
                    if weights_from is not None:
                        weights_to_copy = weights_from[action][s].gln_params
                        act_step_gln.copy_values(weights_to_copy)
                    act_models.append(act_step_gln)

                models.append(act_models)
            return models

        self._model = model_maker(self.num_actions, self.num_steps)
        self._target_model = model_maker(
            self.num_actions, self.num_steps, weights_from=self._model,
            prefix="Target",
        )

        # set the value to burn in the estimator
        if burnin_val is None:
            burnin_val = self.quantile

        if self.burnin_n > 0:
            print(f"Burning in Q Estimators to {burnin_val:.4f}")
        for i in range(0, self.burnin_n, self.batch_size):
            # using random inputs from  the space [-0.05, 1.05]^dim_states
            # the space is larger than the actual state space that the
            # agent will encounter, to hopefully mean that it burns in
            # correctly around the edges
            states = get_burnin_states(mean, self.batch_size, self.dim_states)
            for step in range(1, self.num_steps + 1):
                for a in range(self.num_actions):
                    self.update_estimator(
                        states=states,
                        action=a,
                        q_targets=jnp.full(self.batch_size, burnin_val),
                        horizon=step)
        self.update_target_net()  # update the burned in weights

    def model(self, action, horizon, target=False, safe=True):
        if self.horizon_type == "inf" and horizon not in (1, -1) and safe:
            raise ValueError(f"Unneeded access {horizon}")
        if horizon == 0:
            print("WARN - horizon 0 is not trained")
        if target:
            return self._target_model[action][horizon]
        else:
            return self._model[action][horizon]

    def reset(self):
        raise NotImplementedError("Not yet implemented")

    def estimate(
            self, states, action, h=None, target=False, debug=False):
        """Estimate the future Q, using this estimator

        Args:
            states (jnp.array): the current state from which the Q
                value is being estimated
            action (int): the action to estimate for
            h (int): the timestep horizon ahead to estimate for
            target (bool): whether to use the target net or not
            debug (bool): extra printing
        """
        assert isinstance(action, (int, np.integer)), type(action)
        assert states.ndim == 2, f"states={states.shape}"

        if h == 0 and not self.horizon_type == "inf":
            return 0.  # hardcode a zero value as the model should be None
        h = -1 if h is None else h
        model = self.model(action=action, horizon=h, target=target)
        return model.predict(states)

    def update(
            self, history_batch, update_action, convergence_data=None,
            ire_scale=2., q_scale=8., debug=False):
        """Algorithm 3. Use history to update future-Q quantiles.

        The Q-estimator stores estimates of multiple quantiles in the
        distribution (with respect to epistemic uncertainty) of the
        Q-value.

        It updates by boot-strapping at a given state-action pair.

        Args:
            history_batch (Tranisition): Tuple of the
                (states, actions, rewards, next_states, dones) arrays
                that form the batch for updating
            update_action (int): the action to be updating
            convergence_data (Tuple[jnp.array]): list of
                (states, actions, rewards, next_states, dones) arrays
                that form the batch to converge the network on for
                uncertainty estimates
            debug (bool): print information
            q_scale (float): amount to multiply transition uncertainty
                pseudocount for Q by. Higher is less pessimistic about
                epistemic uncertainty.
            ire_scale (float): amount to multiply ire pseudocount by.
                Higher is less pessimistic about epistemic uncertainty.

        Updates parameters for this estimator, theta_i_a
        """
        if debug:
            print("\nStarting update\n")
        if len(history_batch) == 0:
            return

        self.update_target_net(debug=debug)
        ire = self.immediate_r_estimators[update_action]

        for h in range(1, self.num_steps + 1):
            if debug:
                print(f"Current Q(state) estimates:\n"
                      f"{self.estimate(history_batch.state, update_action)}")
            future_q_value_ests = jnp.stack([
                self.estimate(
                    history_batch.next_state, a,
                    h=None if self.horizon_type == "inf" else h - 1,
                    target=True, debug=debug,
                ) for a in range(self.num_actions)])
            max_future_q_vals = jnp.max(future_q_value_ests, axis=0)
            future_qs = jnp.where(history_batch.done, 0., max_future_q_vals)

            if debug:
                # print("Q value ests", future_q_value_ests)
                if not jnp.all(future_qs == max_future_q_vals):
                    print("max vals", max_future_q_vals)
                print(f"Future Q {future_qs}")
            ire_ns, ire_alphas, ire_betas = ire.model.uncertainty_estimate(
                states=history_batch.state,
                x_batch=convergence_data.state,
                y_batch=convergence_data.reward,
                scale_n=ire_scale,
                debug=debug)
            IV_is = scipy.stats.beta.ppf(
                self.quantile, ire_alphas, ire_betas)
            if debug:
                print(f"s=\n{history_batch.state}")
                print(f"IRE alphas=\n{ire_alphas}\nbetas=\n{ire_betas}")
                print(f"IV_is at q_({self.quantile:.4f}):\n{IV_is}")

            # Q target = r + (h-1)-step future from next state (future_qs)
            # so to scale q to (0, 1) (exc gamma), scale by (~h)
            if self.horizon_type == "inf":
                scaled_IV_is = (
                    1. - self.gamma) * IV_is if self.scaled else IV_is
                q_targets = scaled_IV_is + self.gamma * future_qs
            else:
                # TODO - incorporate gamma or keep approx?
                q_targets = IV_is / h + future_qs * (h - 1) / h
            if debug:
                print("Doing update using IRE uncertainty...")
                print(f"s=\n{history_batch.state}")
                print(f"action={update_action} {type(update_action)}")
                print(f"IRE q_targets combined=\n{q_targets}")
            # TODO lr must be scalar... Also what?
            ire_lr = self.get_lr(ns=ire_ns, scale=ire_scale)
            if debug:
                print(f"IRE LR=\n{ire_lr}")
            self.update_estimator(
                states=history_batch.state,
                action=update_action,
                q_targets=q_targets,
                horizon=None if self.horizon_type == "inf" else h,
                lr=ire_lr)

            # Do the transition uncertainty estimate
            # For scaling:
            if not self.scaled and self.horizon_type == "finite":
                max_q = geometric_sum(1., self.gamma, h)
            elif not self.scaled:
                max_q = geometric_sum(1., self.gamma, "inf")
            else:
                max_q = 1.

            # TODO use target or not?
            if convergence_data.state is not None:
                if self.horizon_type == "inf":
                    scaled_conv_rs = (
                        (1. - self.gamma) * convergence_data.reward)\
                        if self.scaled else convergence_data.reward
                    max_conv_targets = jnp.max(
                        jnp.asarray([
                            self.estimate(
                                convergence_data.next_state,
                                action=a,
                                target=False,  # now larger
                                h=None if self.horizon_type == "inf" else h - 1
                            ) for a in range(self.num_actions)]),
                        axis=0)
                    conv_targets = scaled_conv_rs + jnp.where(
                        convergence_data.done,
                        0.,
                        self.gamma * max_conv_targets)
                else:
                    # q_targets = IV_is / h + future_qs * (h - 1) / h
                    raise NotImplementedError()
            else:
                conv_targets = None

            trans_ns, q_alphas, q_betas = self.model(
                action=update_action, horizon=h).uncertainty_estimate(
                    history_batch.state,
                    x_batch=convergence_data.state,
                    y_batch=conv_targets,
                    max_est_scaling=max_q,
                    scale_n=q_scale,
                    debug=debug,
                )

            q_target_transitions = scipy.stats.beta.ppf(
                self.quantile, q_alphas, q_betas)
            if debug:
                print(f"Trans Q quantile vals\n{q_target_transitions}")
                print(f"{q_target_transitions.shape}, "
                      f"{history_batch.state.shape}")
            assert q_target_transitions.shape[0]\
                   == history_batch.state.shape[0], (
                f"{q_target_transitions.shape}, "
                f"{history_batch.state.shape}")
            if max_q != 1.:
                q_target_transitions *= max_q
                if debug:
                    print(f"Scaled:\n{q_target_transitions}")
                    print("Learning scaled transition Qs")

            trans_lr = self.get_lr(ns=trans_ns, scale=q_scale)
            if debug:
                print(f"Trans LR {trans_lr:.4f}")
            if debug:
                current_vals_debug = jnp.max(jnp.asarray([
                    self.estimate(
                        history_batch.state, a,
                        h=None if self.horizon_type == "inf" else h - 1,
                        target=False, debug=False,
                    ) for a in range(self.num_actions)]), axis=0)
                print(f"Current values:\n{current_vals_debug}")
                print(f"tgt higher?\n"
                      f"{q_target_transitions - current_vals_debug > 0}")
                print(f"Averaged target\n{q_target_transitions}")
            self.update_estimator(
                states=history_batch.state,
                action=update_action,
                q_targets=q_target_transitions,
                lr=trans_lr,
                horizon=None if self.horizon_type == "inf" else h)
        self.total_updates += 1
        if self.total_updates % 150 == 0 and self.lr > 0.02:
            self.lr *= 0.95

    def update_estimator(
            self, states, action, q_targets, horizon=None, lr=None):
        """Update the underlying GLN, i.e. the 'estimator'

        Args:
            states: the x data to update with
            action: the action indexing the GLN to update
            q_targets: the y targets to update towards
            horizon: the horizon to
            lr: lr to update with (default to standard lr)
        """
        # Sanitise inputs
        assert states.ndim == 2 and q_targets.ndim == 1\
               and states.shape[0] == q_targets.shape[0], (
                f"s={states.shape}, a={action}, q={q_targets.shape}")

        if lr is None:
            lr = self.get_lr()

        if horizon is None:
            horizon = -1

        update_gln = self.model(
            action=action, horizon=int(horizon), target=False)
        current_lr = update_gln.lr
        update_gln.update_learning_rate(lr)
        print(f"Updating {update_gln.name} lr {update_gln.lr}")
        update_gln.predict(states, target=q_targets)
        update_gln.update_learning_rate(current_lr)

    def get_lr(self, ns=None, scale=None):
        """
        Returns the learning rate.

        Optionally takes the pseudocount n and calculates
        the learning rate. If no n is provided, uses the predefined
        learning rate.

        """
        assert not (ns is None and self.lr is None), (
            "Both n and self.lr cannot be None")

        # TEMP - skipping the 1/lr scaling
        if self.lr is not None:
            return self.lr
        else:
            assert False
        assert ns is not None
        lr = 1 / (ns + 1) if ns is not None else self.lr
        if scale is not None:
            assert ns is not None
            lr *= scale
        return lr

    def update_target_net(self, debug=False):
        """Update the target model to latest estimator params"""
        for a in range(self.num_actions):
            for s in range(1, self.num_steps + 1):
                target_gln = self.model(action=a, horizon=s, target=True)
                estimator_gln = self.model(action=a, horizon=s, target=False)
                if debug:
                    print(f"Updating {target_gln.name} weights "
                          f"with {estimator_gln.name} weights")
                target_gln.copy_values(estimator_gln.gln_params)


class QuantileQEstimatorBayes(Estimator):
    """A Q estimator that makes pessimistic updates e.g. using IRE

    Uses BayesByBackprop or Monte Carlo Dropout, depending on
    initialiser. Updates using algorithm 3 in the QuEUE specification.

    Uses IREs (which are also GGLNs) to get pessimistic estimates of the next
    reward, to train a GGLN to estimate the Q value.
    """

    def __init__(
            self, quantile, immediate_r_estimator, dim_states, num_actions,
            gamma, layer_sizes=None, context_dim=4, feat_mean=0.5,
            lr=1e-4, scaled=True, burnin_val=None, horizon_type="inf",
            num_steps=1, batch_size=1, net_init=bayes_by_backprop.BBBNet,
            **net_kwargs):
        """Set up the GGLN QEstimator for the given quantile

        Burns in the GGLN to the burnin_val (default is the quantile value)
        for burnin_n steps, where the state for burn in is randomly sampled.

        Args:
            quantile (float): the pessimism-quantile that this estimator
                is estimating the future-Q value for.
            immediate_r_estimator (ImmediateRewardEstimatorBBB]): An IRE
                estimator with an output for each action
            dim_states (int): the dimension of the state space,
                eg, in the 2D cliffworld dim_states = 2
            num_actions (int): the number of actions (4 in cliffworld)
            gamma (float): the discount rate
            layer_sizes (List[int]): The number of neurons in each layer
                for the GGLN
            context_dim (int): the number of hyperplanes used to make the
                halfspaces
            feat_mean (float): mean of all possible inputs to GLN (not
                side info). Typically 0.5, or 0.
            lr (float): the learning rate
            scaled (bool): NOT CURRENTLY IMPLEMENTED
            burnin_n (int): the number of steps we burn in for
            burnin_val (Optional[float]): the value we burn in the
                estimator with. Defaults to quantile value
            net_init (callable): network to use

        """
        super().__init__(lr, scaled=scaled)

        self.net_init = net_init

        if quantile <= 0. or quantile > 1.:
            raise ValueError(f"Require 0. < q <= 1. {quantile}")

        self.quantile = quantile

        self.immediate_r_estimator = immediate_r_estimator
        self.dim_states = dim_states
        self.num_actions = num_actions
        self.context_dim = context_dim
        self.gamma = gamma

        self.horizon_type = horizon_type
        self.num_steps = num_steps
        self.batch_size = batch_size

        if self.horizon_type not in ("inf", "finite"):
            raise ValueError(f"Must be in inf, finite: {horizon_type}")
        if not isinstance(self.num_steps, int) or self.num_steps < 0:
            raise ValueError(f"Must be int number of steps: {self.num_steps}")

        if self.horizon_type == "finite":
            # if self.scaled:
            #     raise NotImplementedError(f"Not defined for scaled Q values")
            if self.num_steps <= 0:
                raise ValueError(f"Must be > 0 future steps: {self.num_steps}")
        elif self.num_steps != 1:
            raise ValueError(
                f"Num steps must be == 1 for inf horizon: {self.num_steps}")

        if burnin_val is None:
            burnin_val = self.quantile

        self.model = None
        self.target_model = None
        self.make_q_estimator(burnin_val, feat_mean, **net_kwargs)

    def make_q_estimator(self, burnin_val, mean, **net_kwargs):

        def model_maker(num_acts, num_horizons, weights_from=None, prefix=""):
            """Returns a network with an output per-action per-horizon"""
            q_str = f"{self.quantile:.3f}".replace(".", "_")
            network = self.net_init(
                name=f"{prefix}QuantileQ_s{num_horizons}_q{q_str}",
                num_actions=num_acts,
                num_horizons=num_horizons,
                input_size=self.dim_states,
                feat_mean=mean,
                lr=self.lr,
                batch_size=self.batch_size,
                sigmoid_vals=self.scaled,
                **net_kwargs
            )
            if weights_from is not None:
                weights_to_copy = weights_from.net.state_dict()
                network.copy_values(weights_to_copy)
            return network

        self.model = model_maker(self.num_actions, self.num_steps)
        self.target_model = model_maker(
            self.num_actions, self.num_steps, weights_from=self.model,
            prefix="Target")

        # set the value to burn in the estimator
        if burnin_val is None:
            burnin_val = self.quantile

        if self.burnin_n > 0:
            print(f"Burning in Q Estimators to {burnin_val:.4f} "
                  f"for {self.burnin_n}")
        for step in range(1, self.num_steps + 1):
            stepped_val = geometric_sum(burnin_val, self.gamma, step)
            targets = tc.full((self.batch_size, 1), stepped_val)
            print(f"Burning in Q Estimator step {step} to {stepped_val:.4f}")
            for i in range(0, self.burnin_n, self.batch_size):
                # using random inputs from  the space [-0.05, 1.05]^dim_states
                # the space is larger than the actual state space that the
                # agent will encounter, to hopefully mean that it burns in
                # correctly around the edges
                states = get_burnin_states(
                    mean, self.batch_size, self.dim_states, library="torch")
                actions = tc.randint(
                    low=0,
                    high=self.num_actions,
                    size=(self.batch_size, 1),
                    dtype=tc.int64)
                self.update_estimator(
                    states=states,
                    actions=actions,
                    q_targets=targets,
                    horizon=step)
        self.model.make_optimizer()  # reset optim
        self.update_target_net()  # update the burned in weights

    def reset(self):
        raise NotImplementedError("Not yet implemented")

    def estimate(self, states, h=None, target=False, debug=False):
        """Estimate the future Q, using this estimator

        Args:
            states (tc.Tensor): the current state from which the Q
                value is being estimated
            h (int): the timestep horizon ahead to estimate for
            target (bool): whether to use the target net or not
            debug (bool): extra printing
        """
        assert states.ndim == 2, f"states={states.shape}"

        # hardcode a zero value as the model should be None
        if h == 0 and not self.horizon_type == "inf":
            return tc.full((states.shape[0], self.num_actions), 0.)
        h = self.num_steps if h is None else h
        ALL_PREDS = self.model.predict(states, horizon=h)
        assert ALL_PREDS.shape[-1] == self.num_actions
        return ALL_PREDS

    def update(self, history_batch, debug=False):
        """Algorithm 3. Use history to update future-Q quantiles.

        The Q-estimator stores estimates of multiple quantiles in the
        distribution (with respect to epistemic uncertainty) of the
        Q-value.

        It updates by boot-strapping at a given state-action pair.

        Args:
            history_batch (Tuple[tc.Tensor]): tuple of
                (states, actions, rewards, next_states, dones) arrays
                that form the batch for updating
            debug (bool): print information

        Updates parameters for this estimator, theta_i_a
        """
        if debug:
            print("\nStarting update\n")
        if len(history_batch) == 0:
            return

        self.update_target_net(debug=debug)
        states, actions, rewards, next_states, dones = history_batch

        ire = self.immediate_r_estimator

        for h in range(1, self.num_steps + 1):
            with tc.no_grad():
                future_q_value_ests = self.estimate(
                    next_states,
                    h=None if self.horizon_type == "inf" else h - 1,
                    target=True,
                    debug=debug)
            max_future_q_vals, _ = tc.max(
                future_q_value_ests, dim=1, keepdim=True)
            future_qs = tc.where(
                dones, tc.tensor(0., dtype=tc.float), max_future_q_vals)

            if debug:
                print(f"\nHORIZON {h}")
                # print("Q value ests", future_q_value_ests)
                if not tc.all(future_qs == max_future_q_vals):
                    print("max vals", max_future_q_vals.squeeze())
                print(f"Future Q {future_qs.squeeze()}")

            with tc.no_grad():
                IV_is = ire.model.uncertainty_estimate(
                    states=states[:, 0:2],  # x, v only
                    actions=actions,
                    debug=debug,
                    quantile=self.quantile).unsqueeze(1)  # preserve dimensionality

            if debug:
                print(f"IV_is at q_({self.quantile:.4f}):\n{IV_is.squeeze()}")

            # Q target = r + (h-1)-step future from next state (future_qs)
            # so to scale q to (0, 1) (exc gamma), scale by (~h)
            if self.horizon_type == "inf":
                scaled_IV_is = (
                    1. - self.gamma) * IV_is if self.scaled else IV_is
                q_targets = scaled_IV_is + self.gamma * future_qs
            else:
                if self.scaled:
                    # TODO - incorporate gamma or keep approx?
                    q_targets = IV_is / h + future_qs * (h - 1) / h
                else:
                    q_targets = IV_is + self.gamma * future_qs
            if debug:
                # print("Doing update using IRE uncertainty...")
                curr = self.estimate(states, h=h)
                print(f"Current Q estimates="
                      f"\n{tc.gather(curr, 1, actions).squeeze()}")
                print(f"IRE q_targets combined (g={self.gamma})="
                      f"\n{q_targets.squeeze()}")

            # self.update_estimator(
            #     states=states,
            #     actions=actions,
            #     q_targets=q_targets,
            #     horizon=None if self.horizon_type == "inf" else h,
            # )

            with tc.no_grad():
                # Do the transition uncertainty estimate
                q_target_transitions = self.target_model.uncertainty_estimate(
                    states, actions, quantile=self.quantile, debug=debug,
                    horizon=h).unsqueeze(1)  # keep dimensionality
            assert q_target_transitions.shape[0] == states.shape[0], (
                f"{q_target_transitions.shape}, {states.shape}")
            avg_target = (q_targets + q_target_transitions) / 2.
            if debug:
                print(f"Quantile Q targets\n{q_target_transitions.squeeze()}")
                print(f"Combined Q targets\n{avg_target.squeeze()}")

            self.update_estimator(
                states=states,
                actions=actions,
                q_targets=avg_target,
                horizon=None if self.horizon_type == "inf" else h,
            )

    def update_estimator(
            self, states, actions, q_targets, horizon=None, lr=None):
        """Update the underlying GLN, i.e. the 'estimator'

        Args:
            states: the x data to update with
            actions: the actions taken
            q_targets: the y targets to update towards
            horizon: the horizon to
            lr: lr to update with (default to standard lr)
        """
        # Sanitise inputs
        assert states.ndim == 2\
               and q_targets.shape[1] == 1\
               and states.shape[0] == q_targets.shape[0], (
                f"s={states.shape}, q={q_targets.shape}")

        self.model.predict(
            states, actions=actions, target=q_targets, horizon=horizon)

    def get_lr(self, ns=None):
        """
        Returns the learning rate.

        Optionally takes the pseudocount n and calculates
        the learning rate. If no n is provided, uses the predefined
        learning rate.

        """
        assert not (ns is None and self.lr is None), (
            "Both n and self.lr cannot be None")

        return 1 / (ns + 1) if self.lr is None else self.lr

    def update_target_net(self, debug=False):
        """Update the target model to latest estimator params"""
        if debug:
            print(f"Updating {self.target_model.name} weights "
                  f"with {self.model.name} weights")
        self.target_model.copy_values(self.model.net.state_dict())
