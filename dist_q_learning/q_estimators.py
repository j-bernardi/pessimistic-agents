import abc
import scipy.stats
import numpy as np

from estimators import Estimator


class BaseQEstimator(Estimator, abc.ABC):

    def __init__(
            self, num_states, num_actions, gamma, lr, q_table_init_val=0.,
            scaled=True, has_mentor=True, **kwargs
    ):
        """Instantiate a base Q Estimator

        Additional Args:
            q_table_init_val (float): Value to initialise Q table at
            scaled: if True, all Q values are in scaled range [0, 1],
                rather than the actual discounted sum or rewards.
            has_mentor: if there is no mentor, we must set an initial
                random action probability (to explore, at first).
        """
        super().__init__(lr, scaled=scaled, **kwargs)
        self.num_states = num_states
        self.num_actions = num_actions
        self.gamma = gamma
        self.q_table_init_val = q_table_init_val

        self.transition_table = np.zeros(
            (num_states, num_actions, num_states), dtype=int)

        self.random_act_prob = None if has_mentor else 1.

    def reset(self):
        self.transition_table = np.zeros(
            (self.num_states, self.num_actions, self.num_states), dtype=int)

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


class FiniteHorizonQEstimator(BaseQEstimator, abc.ABC):
    """Base class for a finite horizon update

    Implements Finite Horizon Temporal Difference Q Estimator. As in:
        https://arxiv.org/pdf/1909.03906.pdf

    Recursively updates the next num_steps estimates of the reward:
        Q_0(s, a) = mean_{t : s_t = s} r_t
        V_i(s) = max_a Q_i(s, a)
        Q_{i+1}(s, a) = mean_{t : s_t = s}[
            (1 - gamma) * r_t  + gamma * V_i(s_{t+1})
        ]
    """

    def __init__(self, num_steps, **kwargs):
        super().__init__(**kwargs)

        if self.scaled:
            raise NotImplementedError("Not defined for scaled Q values")

        if not isinstance(num_steps, int):
            raise ValueError(f"Must be finite steps for FHTD: {num_steps}")
        self.num_steps = num_steps

        # account for Q_0, add one in size to the "horizons" dimension
        self.q_table = self.make_q_estimator()

    @classmethod
    def get_steps_constructor(cls, num_steps):
        """Return a constructor that matches InfiniteHorizonQEstimator

        num_steps is pre-baked into an init function, which is returned
        and matches the arg signature of InfiniteHorizonQEstimator.

        Useful for the main script.
        """

        def init_with_n_steps(**kwargs):
            return cls(num_steps=num_steps, **kwargs)

        return init_with_n_steps

    def make_q_estimator(self):
        """Define the Q table for finite horizon
        TODO:
            - Geometirc series for each q, not actually all the same
        """
        q_table = np.zeros(
            (self.num_states, self.num_actions, self.num_steps + 1))
        q_table += self.q_table_init_val  # Initialise all to init val
        q_table[:, :, 0] = 0.  # Q_0 always init to 0: future is 0 from now
        return q_table

    def reset(self):
        super().reset()
        self.q_table = self.make_q_estimator()

    def estimate(self, state, action, h=None, q_table=None):
        """Estimate the future Q, using this estimator

        Args:
            state (int): the current state
            action (int): the action taken
            h (int): the horizon we are estimating for. If None, use the
                final horizon, (usually used for choosing actions).
            q_table (np.ndarray): the Q table to estimate the value
                from, if None use self.q_table as default

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
        """New update for this special case Q table, requires h

        New args:
            horizon (int): the index of the Q-horizon to update. Default
                to None to match argument signature of super (a bit of
                a hack)
        """

        if update_table is None:
            update_table = self.q_table

        if lr is None:
            lr = self.get_lr(state, action)

        update_table[state, action, horizon] += lr * (
                q_target - update_table[state, action, horizon])

        self.decay_lr()


class InfiniteHorizonQEstimator(BaseQEstimator, abc.ABC):

    num_steps = "inf"

    def __init__(self, **kwargs):
        """
        Additional args:
            num_steps (int, str): The number of steps into the future to
                estimate the Q value for. Defaults to None, for infinite
                horizon.
        """
        super().__init__(**kwargs)
        self.q_table = np.zeros((self.num_states, self.num_actions))
        self.q_table += self.q_table_init_val

    def reset(self):
        super().reset()
        self.q_table = np.zeros(
            (self.num_states, self.num_actions)) + self.q_table_init_val

    def estimate(self, state, action, q_table=None):
        """Estimate the future Q, using this estimator

        Args:
            state (int): the current state from which the Q value is
                being estimated
            action (int): the action taken
            q_table (np.ndarray): the Q table to estimate the value
                from, if None use self.q_table as default
        """
        if q_table is None:
            q_table = self.q_table

        return q_table[state, action]

    def update_estimator(
            self, state, action, q_target, update_table=None, lr=None):
        """Make a single update given an S, A, Target tuple.

        Args:
            state:
            action:
            q_target:
            update_table:
            lr: The learning rate with which to update towards the
                target val. If None, use 1/(1+n_s_a).
        """
        if update_table is None:
            update_table = self.q_table

        if lr is None:
            lr = self.get_lr(state, action)

        update_table[state, action] += lr * (
                q_target - update_table[state, action])
        self.decay_lr()


class FiniteBasicQEstimator(FiniteHorizonQEstimator):
    """Uses FHTDQ with a basic update to r + gamma * Q_h"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def update(self, history):
        for state, action, reward, next_state, done in history:
            self.transition_table[state, action, next_state] += 1

            for h in range(1, self.num_steps + 1):
                # Estimate Q_h
                if not done:
                    next_q = np.max([
                        self.estimate(next_state, action_i, h-1)
                        for action_i in range(self.num_actions)]
                    )
                else:
                    next_q = 0.

                # TODO what was this for? Is it needed now?
                # if h == 1:
                #     next_q = reward

                assert not self.scaled, "Q value must not be scaled"
                q_target = reward + self.gamma * next_q

                super().update_estimator(state, action, q_target, horizon=h)


class FiniteQuantileQEstimator(FiniteHorizonQEstimator):
    """A Q table estimator that makes pessimistic updates e.g. using IRE

    Updates using algorithm 3 in the QuEUE specification.
    """

    def __init__(
            self, quantile,
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
        TODO:
            Optional scaling, finite horizon Q estimators
        """
        super().__init__(**kwargs)
        if quantile <= 0. or quantile > 1.:
            raise ValueError(f"Require 0. < q_val <= 1. {quantile}")

        self.quantile = quantile  # the value of the quantile
        self.use_pseudocount = use_pseudocount
        self.immediate_r_estimators = immediate_r_estimators

    def update(self, history):
        """Algorithm 3. Use history to update future-Q quantiles.

        The Q-estimator stores estimates of multiple quantiles in the
        distribution (with respect to epistemic uncertainty) of the
        Q-value.

        It updates by boot-strapping at a given state-action pair.

        Args:
            history (list): (state, action, reward, next_state, done) tuples

        Updates parameters for this estimator, theta_i_a
        """
        if self.scaled:
            raise NotImplementedError(
                "Not implemented scaled version due to complex scaling rules")

        for state, action, reward, next_state, done in history:
            self.transition_table[state, action, next_state] += 1

            for h in range(1, self.num_steps + 1):
                if not done:
                    future_q = np.max([
                        self.estimate(next_state, action_i, h-1)
                        for action_i in range(self.num_actions)]
                    )
                else:
                    future_q = 0.

                ire = self.immediate_r_estimators[action]
                ire_alpha, ire_beta = ire.expected_with_uncertainty(state)
                iv_i = scipy.stats.beta.ppf(self.quantile, ire_alpha, ire_beta)
                # Note: not scaled
                scaled_iv_i = (1. - self.gamma) * iv_i if self.scaled else iv_i

                q_target = self.gamma * future_q + scaled_iv_i

                self.update_estimator(state, action, q_target, horizon=h)

                # Account for uncertainty in the state transition function
                # UPDATE - TODO - check uses current horizon
                q_ai = self.estimate(state, action, h)
                if not self.use_pseudocount:
                    n = self.transition_table[state, action, :].sum()
                else:
                    fake_q_ai = []
                    for fake_target in [0., 1.]:
                        fake_q_table = self.q_table.copy()
                        self.update_estimator(
                            state, action, fake_target,
                            update_table=fake_q_table, horizon=h)
                        fake_q_ai.append(
                            self.estimate(
                                state, action, h, q_table=fake_q_table)
                        )

                    n_ai0 = fake_q_ai[0] / (q_ai - fake_q_ai[0])
                    n_ai1 = (1. - fake_q_ai[1]) / (fake_q_ai[1] - q_ai)
                    # in the original algorithm this min was also over the
                    # quantiles
                    # as well as the fake targets
                    n = np.min([n_ai0, n_ai1])

                q_alpha = q_ai * n + 1.
                q_beta = (1. - q_ai) * n + 1.
                q_target_transition = scipy.stats.beta.ppf(
                    self.quantile, q_alpha, q_beta)

                self.update_estimator(
                    state, action, q_target_transition,
                    lr=1./(1. + self.transition_table[state, action, :].sum()),
                    horizon=h
                )
                self.total_updates += 1


class InfiniteQuantileQEstimator(InfiniteHorizonQEstimator):
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
            init_to_zero (bool): if True, init Q table to 0. instead of
                'burining-in' quantile value

        TODO:
            Optional scaling
        """
        # "Burn in" quantile with init value argument
        super().__init__(**kwargs)
        if quantile <= 0. or quantile > 1.:
            raise ValueError(f"Require 0. < q_val <= 1. {quantile}")

        self.quantile = quantile  # the value of the quantile
        self.use_pseudocount = use_pseudocount
        self.immediate_r_estimators = immediate_r_estimators

    def update(self, history):
        """Algorithm 3. Use history to update future-Q quantiles.

        The Q-estimator stores estimates of multiple quantiles in the
        distribution (with respect to epistemic uncertainty) of the
        Q-value.

        It updates by boot-strapping at a given state-action pair.

        Args:
            history (list): (state, action, reward, next_state, done) tuples

        Updates parameters for this estimator, theta_i_a
        """

        for state, action, reward, next_state, done in history:
            self.transition_table[state, action, next_state] += 1

            # The update towards the IRE if there is a reward to be gained
            # UPDATE 1
            if not done:
                future_q = np.max([
                    self.estimate(next_state, action_i)
                    for action_i in range(self.num_actions)]
                )
            else:
                future_q = 0.

            ire = self.immediate_r_estimators[action]
            ire_alpha, ire_beta = ire.expected_with_uncertainty(state)
            IV_i = scipy.stats.beta.ppf(self.quantile, ire_alpha, ire_beta)
            scaled_iv_i = (1. - self.gamma) * IV_i if self.scaled else IV_i

            q_target = self.gamma * future_q + scaled_iv_i

            self.update_estimator(state, action, q_target)

            # Account for uncertainty in the state transition function
            # UPDATE 2
            q_ai = self.estimate(state, action)
            if not self.use_pseudocount:
                n = self.transition_table[state, action, :].sum()
            else:
                fake_q_ai = []
                for fake_target in [0., 1.]:
                    fake_q_table = self.q_table.copy()
                    self.update_estimator(
                        state, action, fake_target, update_table=fake_q_table)
                    fake_q_ai.append(self.estimate(state, action, fake_q_table))

                n_ai0 = fake_q_ai[0] / (q_ai - fake_q_ai[0])
                n_ai1 = (1. - fake_q_ai[1]) / (fake_q_ai[1] - q_ai)
                # in the original algorithm this min was also over the quantiles
                # as well as the fake targets
                n = np.min([n_ai0, n_ai1])

            q_alpha = q_ai * n + 1.
            q_beta = (1. - q_ai) * n + 1.
            q_target_transition = scipy.stats.beta.ppf(
                self.quantile, q_alpha, q_beta)

            self.update_estimator(
                state, action, q_target_transition,
                lr=1./(1. + self.transition_table[state, action, :].sum())
            )
            self.total_updates += 1


class InfiniteQuantileQEstimatorSingle(InfiniteQuantileQEstimator):
    """Override the pessimitic QEstimator to do a single Q estimate

    TODO - complete theory and update logic
    """

    def __init__(self, next_state_expectation=False, **kwargs):
        self.ns_estimator = kwargs.pop("ns_estimator")
        super().__init__(**kwargs)
        self.next_state_expectation = next_state_expectation

    def update(self, history):
        """See super() update - and diff"""

        for state, action, reward, next_state, done in history:
            self.transition_table[state, action, next_state] += 1

            # The update towards the IRE if there is a reward to be gained
            ire = self.immediate_r_estimators[action]
            ire_alpha, ire_beta = ire.expected_with_uncertainty(state)
            iv_i = scipy.stats.beta.ppf(self.quantile, ire_alpha, ire_beta)
            scaled_iv_i = (1. - self.gamma) * iv_i if self.scaled else iv_i

            # TRANSITION UNCERTAINTY
            next_state_freqs = self.ns_estimator.frequencies(state, action)
            next_state_probs = next_state_freqs / np.sum(next_state_freqs)

            # No s_i -> s transform necessary as [state_i] := range(num_states)

            if self.next_state_expectation:
                next_state_vals = [  # [Q(s', a*) for s' in all next_states]
                    np.amax(
                        [self.estimate(s, a) for a in range(self.num_actions)])
                    for s, f in enumerate(next_state_freqs)]  # if f > 0
                # EXPECTED future q
                future_q = np.sum(next_state_probs * next_state_vals)
            else:
                # Assume at least 1 step everywhere to be uber paranoid
                pess_next_state_freqs = np.where(
                    next_state_freqs == 0, 1, next_state_freqs)
                # Order from most to least likely (descending):
                ordered_pess_ns_f = [
                    (i, f) for i, f in enumerate(pess_next_state_freqs)]
                ordered_pess_ns_f.sort(key=lambda tup: -tup[1])
                num_to_consider = 1
                total_prob = ordered_pess_ns_f[0]
                while ordered_pess_ns_f[num_to_consider][1] < (
                        1. - self.quantile):
                    total_prob += ordered_pess_ns_f[num_to_consider]
                    num_to_consider += 1
                considering = ordered_pess_ns_f[:num_to_consider]

                # VALUES
                # Property 1: Must select the (epistemically) pessimistic val in
                # limit uncertainty -> 1. Epistemically optimistic val in limit
                # -> 0
                considered_vals = [  # [Q(s', a*) for s' in poss next_states]
                    np.amax([
                        self.estimate(s_i, a)
                        for a in range(self.num_actions)]
                    ) for s_i, _ in considering
                ]

                # EPISTEMICALLY PESSIMISTIC FUTURE Q
                # TODO - this doesn't fairly sample. In the limit, it will
                #  tend toward the modal value, rather than the expected value.
                #  Perhaps instead of min, it should just be an expectation
                #  across the considered states? Still fuzzes expectation and

                # TODO 2 - what happens if there's 100 next states all with 1%
                #  probability?

                # TODO - it should be more like "sample from this distribution
                #  (probabilities), but account for epistemic perturbations of
                #  some size (dictated by q?) and select the worst variation.
                #  Such that in the end the perturbations add nothing.

                # NOTE - works fine when T(s', s,a) is deterministic
                future_q = np.amin(considered_vals)

            q_target = self.gamma * future_q + scaled_iv_i

            self.update_estimator(state, action, q_target)
            self.total_updates += 1


class InfiniteBasicQEstimator(InfiniteHorizonQEstimator):
    """A basic Q table estimator

    Updates (as 'usual') towards the target of:
        gamma * Q(s', a*) + (1. - gamma) * reward
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def update(self, history):
        for state, action, reward, next_state, done in history:
            self.transition_table[state, action, next_state] += 1

            if not done:
                future_q = np.max([
                    self.estimate(next_state, action_i)
                    for action_i in range(self.num_actions)]
                )
            else:
                future_q = 0.

            scaled_r = (1. - self.gamma) * reward if self.scaled else reward
            q_target = self.gamma * future_q + scaled_r

            self.update_estimator(state, action, q_target)


class InfiniteQMeanIREEstimator(InfiniteHorizonQEstimator):
    """A basic Q table estimator which uses mean IRE instead of reward

    Updates (as 'usual') towards the target of:
        gamma * Q(s', a*) + (1. - gamma) * IRE(state)

    TODO:
        Could be a special case of the QPessIREEstimator, with a flag
        (e.g. quantile=None) to use .estimate rather than with
        uncertainty.
    """

    def __init__(self, immediate_r_estimators, **kwargs):

        super().__init__(**kwargs)
        self.immediate_r_estimators = immediate_r_estimators

    def update(self, history):
        for state, action, reward, next_state, done in history:
            self.transition_table[state, action, next_state] += 1

            if not done:
                future_q = np.max([
                    self.estimate(next_state, action_i)
                    for action_i in range(self.num_actions)]
                )
            else:
                future_q = 0.

            ire = self.immediate_r_estimators[action].estimate(state)
            scaled_ire = (1. - self.gamma) * ire if self.scaled else ire

            q_target = self.gamma * future_q + scaled_ire
            self.update_estimator(state, action, q_target)


class InfiniteQPessIREEstimator(InfiniteHorizonQEstimator):
    """A basic Q table estimator which uses mean IRE instead of reward

    Does not do the "2nd" update (for transition uncertainty)

    Updates (as 'usual') towards the target of:
        gamma * Q(s', a*) + (1. - gamma) * IRE_qi(state)
    """

    def __init__(self, quantile, immediate_r_estimators, **kwargs):
        """

        quantile (float): the quantile value
        immediate_r_estimators:
        kwargs: see super()
        """
        super().__init__(**kwargs)

        self.quantile = quantile
        self.immediate_r_estimators = immediate_r_estimators

    def update(self, history):
        for state, action, reward, next_state, done in history:
            self.transition_table[state, action, next_state] += 1

            if not done:
                future_q = np.max([
                    self.estimate(next_state, action_i)
                    for action_i in range(self.num_actions)]
                )
            else:
                future_q = 0.

            ire_estimator = self.immediate_r_estimators[action]
            ire_alpha, ire_beta = ire_estimator.expected_with_uncertainty(state)
            ire_i = scipy.stats.beta.ppf(self.quantile, ire_alpha, ire_beta)

            scaled_ire = (1. - self.gamma) * ire_i if self.scaled else ire_i
            q_target = self.gamma * future_q + scaled_ire

            # No 2nd update!
            self.update_estimator(state, action, q_target)
