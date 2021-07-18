import abc
import copy
import scipy.stats
import numpy as np
import jax
import jax.numpy as jnp
from haiku.data_structures import to_immutable_dict

import glns
from estimators import Estimator, BURN_IN_N, DEFAULT_GLN_LAYERS
from utils import geometric_sum


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
            gamma, layer_sizes=None, context_dim=4, lr=1e-4, scaled=True,
            burnin_n=BURN_IN_N, burnin_val=None, horizon_type="inf",
            num_steps=1, batch_size=1):
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
            lr (float): the learning rate
            scaled (bool): NOT CURRENTLY IMPLEMENTED
            burnin_n (int): the number of steps we burn in for
            burnin_val (Optional[float]): the value we burn in the
                estimator with. Defaults to quantile value

        """
        super().__init__(lr, scaled=scaled)

        if quantile <= 0. or quantile > 1.:
            raise ValueError(f"Require 0. < q <= 1. {quantile}")

        self.quantile = quantile  # the 'i' index of the quantile

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

        self.model = None
        self.target_model = None
        self.make_q_estimator(self.layer_sizes, burnin_val, burnin_n)

    def make_q_estimator(self, layer_sizes, burnin_val, burnin_n):

        def model_maker(num_acts, num_steps, weights_from=None):
            """Returns list of lists of model[action][horizon] = GLN"""
            models = []
            for a in range(num_acts):
                act_models = []
                for s in range(num_steps + 1):
                    act_step_gln = glns.GGLN(
                        layer_sizes=layer_sizes,
                        input_size=self.dim_states,
                        context_dim=self.context_dim,
                        bias_len=3,
                        lr=self.lr,
                        min_sigma_sq=0.5,
                        batch_size=self.batch_size,
                        # init_bias_weights=[0.1, 0.2, 0.1]
                    )
                    if weights_from is not None:
                        params = weights_from[a][s].gln_params
                        # Copy operation, in haiku
                        act_step_gln.gln_params = to_immutable_dict(params)
                    act_models.append(act_step_gln)

                models.append(act_models)
            return models

        self.model = model_maker(self.num_actions, self.num_steps)
        self.target_model = model_maker(
            self.num_actions, self.num_steps, weights_from=self.model)

        # set the value to burn in the estimator
        if burnin_val is None:
            burnin_val = self.quantile

        if burnin_n > 0:
            print("Burning in Q Estimator")

        for i in range(0, burnin_n, self.batch_size):
            # using random inputs from  the space [-0.05, 1.05]^dim_states
            # the space is larger than the actual state space that the
            # agent will encounter, to hopefully mean that it burns in
            # correctly around the edges
            states = 1.1 * jax.random.uniform(
                glns.JAX_RANDOM_KEY, (self.batch_size, self.dim_states)) - 0.05
            for step in range(1, self.num_steps):
                for a in range(self.num_actions):
                    self.update_estimator(
                        states, a, jnp.full(self.batch_size, burnin_val),
                        horizon=step)

    def reset(self):
        raise NotImplementedError("Not yet implemented")

    def estimate(self, states, actions, h=None, model=None):
        """Estimate the future Q, using this estimator

        Args:
            states (jnp.ndarray): the current state from which the Q
                value is being estimated
            actions (jnp.ndarray): the actions taken
            model: the estimator (GGLN) used to estimate the Q value,
            if None, then use self.model as default
        """
        assert actions.ndim == 1 and states.ndim == 2\
               and states.shape[0] == actions.shape[0]
        if h == 0 and not self.horizon_type == "inf":
            return 0.

        if model is None:
            model = self.model

        if h is None:
            h = -1

        returns = jnp.empty(states.shape[0])
        for update_action in range(self.num_actions):
            if not jnp.any(actions == update_action):
                continue
            estimate_with = model[update_action][h]
            idxs = jnp.squeeze(
                jnp.argwhere(actions == update_action), -1).astype(jnp.int8)
            xs = states[idxs]
            if xs.ndim == 1 and idxs.shape[0] == 1:
                # Sometimes the squeeze and idx squeezes out the correct dim
                xs = np.expand_dims(xs, 0)
            elif xs.ndim == 1:
                raise RuntimeError()
            ys = estimate_with.predict(xs)
            returns = jax.ops.index_update(returns, idxs, ys)

        return returns

    def update(self, history_batch, convergence_data, debug=False):
        """Algorithm 3. Use history to update future-Q quantiles.

        The Q-estimator stores estimates of multiple quantiles in the
        distribution (with respect to epistemic uncertainty) of the
        Q-value.

        It updates by boot-strapping at a given state-action pair.

        Args:
            history_batch (list): list of
                (state, action, reward, next_state, done) tuples that
                will form the batch for updating
            convergence_data (list): list of
                (state, action, reward, next_state, done) tuples that
                will form the batch to converge the network on

        Updates parameters for this estimator, theta_i_a
        """
        if len(history_batch) == 0:
            return

        self.update_target_net()

        batch_tuple = tuple(map(jnp.array, [i for i in zip(*history_batch)]))
        convergence_tuple = tuple(
            map(jnp.array, [i for i in zip(*convergence_data)]))
        nones_found = [
            jnp.any(jnp.logical_or(jnp.isnan(x), x == None))
            for x in batch_tuple]
        assert not any(nones_found), f"Nones found in arrays: {nones_found}"
        states, actions, rewards, next_states, dones = batch_tuple
        conv_states, conv_actions, conv_rewards, _, _ = convergence_tuple

        for update_action in range(self.num_actions):
            if not jnp.any(actions == update_action):
                continue
            idxs = jnp.argwhere(
                actions == update_action).squeeze(-1).astype(jnp.int8)
            conv_idxs = jnp.argwhere(
                conv_actions == update_action).squeeze(-1).astype(jnp.int8)
            assert conv_idxs.shape[0], "Only supported if data to converge on"
            ire = self.immediate_r_estimators[update_action]
            for h in range(1, self.num_steps + 1):
                future_q_value_ests = jnp.array([
                    self.estimate(
                        next_states[idxs],
                        jnp.full_like(idxs, a),
                        h=None if self.horizon_type == "inf" else h - 1,
                        model=self.target_model,
                    )
                    for a in range(self.num_actions)])
                max_future_q_vals = jnp.max(future_q_value_ests, axis=0)
                future_qs = jnp.where(dones[idxs], 0., max_future_q_vals)

                if debug:
                    print("Q value ests", future_q_value_ests)
                    print("max vals", max_future_q_vals)
                    print("Future Q", future_qs)

                ire_alphas, ire_betas, ns = ire.model.uncertainty_estimate(
                    states=states[idxs],
                    x_batch=conv_states[conv_idxs],
                    y_batch=conv_rewards[conv_idxs],
                    debug=debug)
                IV_is = scipy.stats.beta.ppf(
                    self.quantile, ire_alphas, ire_betas)
                if debug:
                    print(f"s=\n{states[idxs]}")
                    print(f"IRE alphas=\n{ire_alphas}\nbetas=\n{ire_betas}")
                    print(f"BETA SAMPLES q_({self.quantile:.5f})")
                    print(f"IV_is:\n{IV_is}")

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
                    print(f"s=\n{states[idxs]}")
                    print(f"actions=\n{actions[idxs]}")
                    assert jnp.all(jnp.logical_or(actions == 0, actions == 1))
                    print(f"q_targets=\n{q_targets}")
                # TODO lr must be scalar... Also what?
                # TODO - do this update at the end, rather than use the model
                #  for the next transition estimate?
                self.update_estimator(
                    states[idxs],
                    update_action,
                    q_targets,
                    horizon=None if self.horizon_type == "inf" else h,
                    lr=self.get_lr(ns=ns))

                # Do the transition uncertainty estimate
                # For scaling:
                if not self.scaled and self.horizon_type == "finite":
                    max_q = geometric_sum(1., self.gamma, h)
                elif not self.scaled:
                    max_q = geometric_sum(1., self.gamma, "inf")
                else:
                    max_q = 1.

                trans_ns, q_alphas, q_betas =\
                    self.model[update_action][h].uncertainty_estimate(
                        states[idxs],
                        x_batch=conv_states[conv_idxs],
                        y_batch=self.estimate(
                            conv_states[conv_idxs],
                            actions=conv_actions[conv_idxs],
                            model=self.target_model),
                        max_est_scaling=max_q,
                        debug=debug,
                    )

                q_target_transitions = scipy.stats.beta.ppf(
                    self.quantile, q_alphas, q_betas)
                if debug:
                    print(f"Transition Q values\n{q_target_transitions}")
                    print(f"{q_target_transitions.shape}, {idxs.shape}")
                assert q_target_transitions.shape[0] == idxs.shape[0], (
                    f"{q_target_transitions.shape}, {idxs.shape}")
                # TODO - right operation here?
                q_target_transitions /= max_q
                if debug:
                    print(f"Scaled:\n{q_target_transitions}")
                    print("Learning scaled transition Qs")
                # TODO - batch the learning rate?
                self.update_estimator(
                    states[idxs],
                    update_action,
                    q_target_transitions,
                    lr=self.get_lr(ns=trans_ns),
                    horizon=None if self.horizon_type == "inf" else h)

    def update_estimator(
            self, states, action, q_targets, horizon=None, update_model=None,
            lr=None):
        """Update the underlying GLN, i.e. the 'estimator'

        Args:
            states: the x data to update with
            action: the action indexing the GLN to update
            q_targets: the y targets to update towards
            horizon: the horizon to
            update_model: the model to update (default to self.model)
            lr: lr to update with (default to standard lr)
        """
        # Sanitise inputs
        assert states.ndim == 2 and q_targets.ndim == 1\
               and states.shape[0] == q_targets.shape[0], (
            f"s={states.shape}, a={action}, q={q_targets.shape}")

        if update_model is None:
            update_model = self.model

        if lr is None:
            lr = self.get_lr()

        if horizon is None:
            horizon = -1

        update_gln = update_model[action][int(horizon)]
        update_gln.update_learning_rate(lr)
        update_gln.predict(states, target=q_targets)

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

    def update_target_net(self, update_model=None):
        """Update update_model (default target model) to latest params"""
        for a in range(self.num_actions):
            for s in range(self.num_steps):
                if update_model is None:
                    gln_to_update = self.target_model[a][s]
                else:
                    gln_to_update = update_model[a][s]
                # Copy in haiku:
                gln_to_update.gln_params = to_immutable_dict(
                    self.model[a][s].gln_params)


# TODO - update function not yet attempted to be batched at all!
#  Incomplete
class QuantileQEstimatorGaussianSigmaGLN(QuantileQEstimatorGaussianGLN):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    # TODO - batch, and add history to model
    def update(self, history_batch):
        """Algorithm 3. Use history to update future-Q quantiles.

        The Q-estimator stores estimates of multiple quantiles in the
        distribution (with respect to epistemic uncertainty) of the
        Q-value.

        It updates by boot-strapping at a given state-action pair.

        Args:
            history_batch (list[tuple]): list of
                (state, action, reward, next_state, done) tuples that
                will for form the batch

        Updates parameters for this estimator, theta_i_a
        """
        if len(history_batch) == 0:
            return

        self.update_target_net()

        states, actions, rewards, next_states, dones = map(
            jnp.array, [i for i in zip(*history_batch)])

        future_qs = jnp.where(
            dones, 0.,
            jnp.max(
                self.estimate(
                    jnp.full(
                        (states.shape[0], self.num_actions, states.shape[1]),
                        states),
                    jnp.arange(start=0, stop=self.num_actions),
                    model=self.target_model),
                axis=1)
        )

        for update_action in range(self.num_actions):
            if not jnp.any(actions == update_action):
                continue
            idxs = jnp.argwhere(actions == update_action)
            idxs = idxs.squeeze(-1).astype(jnp.int8)
            ire = self.immediate_r_estimators[update_action]

            IV_is = jnp.empty_like(idxs)
            for i, s in enumerate(states[idxs]):
                ire_mean, ire_sigma = ire.estimate_with_sigma(s)
                IV_is[i] = scipy.stats.norm.ppf(self.quantile, ire_mean, ire_sigma)
            # print(ire_alpha, ire_beta)
            # print(f'IV_i: {IV_is}')
            # if not ire_success:
            #     print('bad ire update')
            #     continue
            # n = ire_alpha + ire_beta
            # print(n)
            q_targets = self.gamma * future_qs[idxs] + (1. - self.gamma) * IV_is
            # q_target = IV_i + 1
            # q_target = self.gamma * future_q + IV_i

            # print(f'q_target: {q_target}')
            # print(f'future_q: {future_q}')
            # gln_params1 = copy.copy(self.model[action].gln_params)

            self.update_estimator(
                states[idxs], update_action, q_targets, lr=self.get_lr())

            q_ais = self.estimate(states, actions)
            gln_params2 = copy.copy(self.model[update_action][s].gln_params)

            # diff_keys = [k for k in gln_params1 if gln_params1[k] != gln_params2]
            # print(diff_keys)
            fake_q_ai = []

            # for fake_target in [0., 1.]:
            #     # self.fake_model = copy.copy(self.model)
            #     # self.update_estimator(
            #     #     state, action, q_target/q_target * fake_target, lr=self.get_lr(), update_model=self.fake_model)
            #     # fake_q_ai.append(copy.copy(self.estimate(state, action, model=self.fake_model)))
            #     for k in range(100):
            #         self.update_estimator(
            #             state, action, q_target/q_target * fake_target, update_model=self.model,lr=self.get_lr())
            #     fake_q_ai.append(self.estimate(state, action, model=self.model))
            #     self.model[action].gln_params = copy.copy(gln_params)

            # print(f'{action} : {fake_q_ai}, True: {q_ai}')
            # if  not fake_q_ai[0]==fake_q_ai[1]:


            #     n_ai0 = fake_q_ai[0] / (q_ai - fake_q_ai[0])
            #     n_ai1 = (1. - fake_q_ai[1]) / (fake_q_ai[1] - q_ai)
            #     # in the original algorithm this min was also over the quantiles
            #     # as well as the fake targets
            #     n = jnp.min([n_ai0, n_ai1])
            #     if n < 0.:
            #         n = 0.
            #     q_alpha = q_ai * n + 1.
            #     q_beta = (1. - q_ai) * n + 1.

            #     q_target_transition = scipy.stats.beta.ppf(
            #         self.quantile, q_alpha, q_beta)
            #     print(f'q_target_transition: {q_target_transition}')
            #     self.update_estimator(state, action, q_target_transition, lr=self.get_lr(n=n))

        # for i in range(self.num_actions):
        #     print(f'updates {i}: {self.model[i].update_count}')
        #     print(f'nans: {i}: {self.model[i].update_nan_count}')
