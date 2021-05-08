import abc
import scipy.stats
import numpy as np

from estimators import Estimator
from utils import geometric_sum


class QTableEstimator(Estimator, abc.ABC):
    """Base class for both the finite and infinite horizon update

    Implements both types of Q table. Attribute q_table: ndim=3,
    (states, actions, num_steps+1)

    Infinite Horizon Q Estimator (bootstrapping). num_steps=1 (so final
    dim represents (0-step-future-R (always == 0), inf_R==Q_val).

    And Finite Horizon Temporal Difference Q Estimator, as in:
        https://arxiv.org/pdf/1909.03906.pdf

        Recursively updates the next num_steps estimates of the reward:
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
            num_steps=1,
            **kwargs
    ):
        """Initialise a Q Estimator of infinite or finite horizon

        Additional Args:
            q_table_init_val (float): Value to initialise Q table at
            scaled: if True, all Q values are in scaled range [0, 1],
                rather than the actual discounted sum or rewards.
            has_mentor: if there is no mentor, we must set an initial
                random action probability (to explore, at first).
            horizon_type (str): one of "inf" or "finite". Defines estimator
                behaviour
            num_steps (int): Number of steps to look ahead into the future.
                Must be 1 if horizon_type="inf" - store only 1 Q value (inf)
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
        self.num_steps = num_steps

        if self.horizon_type not in ("inf", "finite"):
            raise ValueError(f"Must be in inf, finite: {horizon_type}")
        if not isinstance(self.num_steps, int) or self.num_steps < 0:
            raise ValueError(f"Must be int number of steps: {self.num_steps}")

        if self.horizon_type == "finite":
            if self.scaled:
                raise NotImplementedError(
                    f"Not defined for scaled Q values. Try --unscale-q ?")
            if self.num_steps <= 0:
                raise ValueError(f"Must be > 0 future steps: {self.num_steps}")
        elif self.num_steps != 1:
            raise ValueError(
                f"Num steps must be == 1 for inf horizon: {self.num_steps}")

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
            (self.num_states, self.num_actions, self.num_steps + 1))
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
            for h in range(1, self.num_steps + 1):
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

            for h in range(1, self.num_steps + 1):
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
            for h in range(1, self.num_steps + 1):
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
