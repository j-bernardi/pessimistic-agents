import abc
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline
import scipy.stats
NP_RANDOM_GEN = np.random.Generator(np.random.PCG64())


def sample_beta(a, b, n=1):
    """Sample Beta(alpha=a, beta=b), return 1d array size n."""
    return NP_RANDOM_GEN.beta(a, b, size=n)


def plot_beta(a, b, show=True, n_samples=10000):
    """Plot a beta distribution, given these parameters."""
    ax = plt.gca()
    sampled_vals = sample_beta(a, b, n=n_samples)
    n_bins = n_samples // 100
    ps, xs = np.histogram(sampled_vals, bins=n_bins, density=True)
    # convert bin edges to centers
    xs = xs[:-1] + (xs[1] - xs[0]) / 2
    f = UnivariateSpline(xs, ps, s=n_bins)  # smooths data

    ax.set_title(f"Beta distribution for alpha={a}, beta={b}")
    ax.set_ylabel("PDF")
    ax.set_xlabel("E(reward)")
    ax.set_xlim((0, 1))

    ax.plot(xs, f(xs))
    if show:
        plt.show()

    return ax


class Estimator(abc.ABC):
    """Abstract definition of an estimator"""
    def __init__(self, lr, min_lr=0.05, lr_decay=0., scaled=True):
        """

        Args:
            lr (Optional[float]): The initial learning rate. If None,
                override the self.get_lr() method to return a custom
                learning rate (e.g. num_visits[state, action]).
            min_lr (float): If lr gets below min_lr, stop decaying
            lr_decay (float): Reduce the learning rate by a factor of
                (1. - lr_decay)
            scaled (bool): if True, all estimates should be in [0, 1],
                else reflect the true value of the quantity being
                estimated.
        """
        self.lr = lr
        self.min_lr = min_lr
        self.lr_decay = lr_decay
        self.scaled = scaled

        self.total_updates = 0

    @abc.abstractmethod
    def estimate(self, **args):
        """Use the estimator to generate an estimate given state"""
        return

    @abc.abstractmethod
    def update(self, history):
        """Update the estimator given some experience"""
        return

    @abc.abstractmethod
    def reset(self):
        """Reset as if the estimator had just been made"""
        return

    def get_lr(self):
        """Returns the learning rate"""
        return self.lr

    def decay_lr(self):
        """Reduce lr by factor (1. - self.lr_decay), if above the min"""
        if self.lr is not None:
            if self.lr > self.min_lr and self.lr_decay is not None:
                self.lr *= (1. - self.lr_decay)


class ImmediateNextStateEstimator(Estimator):

    def __init__(self, num_states, num_actions):
        """Create an action-specific IRE.

        Args:
            action: the action this estimator is specific to.
        """
        super().__init__(lr=None)
        self.num_states = num_states
        self.num_actions = num_actions

        # Maps state to a list of next-states
        self.state_table = np.zeros(
            (self.num_states, self.num_actions, self.num_states),
            dtype=int
        )

    def reset(self):
        self.state_table = np.zeros(
            (self.num_states, self.num_actions, self.num_states),
            dtype=int
        )

    def frequencies(self, state, action, from_dict=None):
        """Provide the frequencies of next-states

        E.g. T(s'|s, a) = frequencies(s, a) / SUM(frequencies(s, a))

        Args:
            state (int): the current state to estimate next state from
            action (int): the action taken to estimate next state from
            from_dict (dict[int, list[float])): the state table to
                use. If None, default to self.state_table.

        Returns:
            The (n_states,)-shape array of frequencies for each next
            state given the current state, action.
        """

        if from_dict is None:
            from_dict = self.state_table

        if state not in from_dict or np.all(from_dict[state, action] == 0):
            # raise ValueError("Decide what to do on first encounter")
            pass

        frequencies = from_dict[state, action]
        return frequencies

    def estimate(self, state, action, from_dict=None):
        """Provide the next state with the highest likelihood

        Args:
            state (int): the current state to estimate next state from
            action (int): the action taken to estimate next state from
            from_dict (dict[int, list[float])): the state table to
                use. If None, default to self.state_table.

        Returns:
            Index of the most likely next state (int)
        """
        freqs = self.frequencies(state, action, from_dict=from_dict)
        return np.argmax(freqs)

    def expected_with_uncertainty(self, state, action):
        """Epistemic uncertainty distribution over next s
        Args:
            state (int): the current state to estimate next reward from
            action (int): the action being taken

        Returns:
            TODO ?
            list of alpha, betas: defining beta distribution over
            next reward
        """

        freqs = self.frequencies(state, action)

        # TODO - a beta distribution over every probability?

        for i, fake_r in enumerate(fake_rewards):
            fake_dict = {state: self.state_dict[state].copy()}
            self.update([(state, fake_r)], fake_dict)
            fake_means[i] = self.estimate(state, fake_dict)

        # n is the pseudo-count of number of times we've been in this
        # state. We take the n that gives the most uncertainty?
        # TODO - just use n = len(rewards[state])

        # TODO - handle nans if current_mean is already 0. or 1.

        # Assume the base estimator’s mean tracks the sample mean.
        # To recover a distribution (w.r.t epistemic uncertainty) over the true
        # mean, let n be the estimated data count in the sample.
        # Because μ0 ≈ μn /(n + 1)
        n_0 = (  # TEMP handling of / 0 error
            fake_means[0] / (current_mean - fake_means[0])
            if current_mean != fake_means[0] else None
        )

        # Because μ1 ≈ (μn + 1)/(n + 1)
        n_1 = (
            (1. - fake_means[1]) / (fake_means[1] - current_mean)
            if current_mean != fake_means[1] else None
        )

        # Take min, or ignore the None value by making it the larger number
        n = min((n_0 or n_1 + np.abs(n_1)), (n_1 or n_0 + np.abs(n_0)))

        assert n >= 0., f"Unexpected n {n}, from ({n_0}, {n_1})"
        alpha = current_mean * n + 1.  # pseudo-successes (r=1)
        beta = (1. - current_mean) * n + 1.  # pseudo-failures (r=0)
        return alpha, beta

    def update(self, history, update_dict=None):
        """Algorithm 1. Use experience to update estimate of immediate r

        Args:
            history (list[tuple]): list of (state, reward) tuples to add
            update_dict (dict, None): the state_dict to apply the update
                to. Defaults to self.state_dict.
        """

        if update_dict is None:
            update_dict = self.state_table

        for (state, action, reward, next_state, done) in history:
            update_dict[state, action, next_state] += 1
            self.total_updates += 1


class ImmediateRewardEstimator(Estimator):
    """Estimates the next reward given a current state and an action"""

    def __init__(self, action, use_pseudocount=False):
        """Create an action-specific IRE.

        Args:
            action: the action this estimator is specific to.
        """
        super().__init__(lr=None)
        self.action = action
        self.use_pseudocount = use_pseudocount

        # Maps state to a list of next-rewards
        self.state_dict = {}

    def reset(self):
        self.state_dict = {}

    def estimate(self, state, from_dict=None):
        """Provide the mean immediate reward estimate for an action

        Args:
            state (int): the current state to estimate next reward from
            from_dict (dict[int, list[float])): the state dict to
                calculate the mean from. If None, default to
                self.state_dict.

        Returns:
            Mean of the encountered rewards
        """

        if from_dict is None:
            from_dict = self.state_dict

        if state not in self.state_dict:
            raise NotImplementedError("Decide what to do on first encounter")

        reward_mean, _ = from_dict[state]

        return reward_mean

    def expected_with_uncertainty(self, state):
        """Algorithm 2. Epistemic Uncertainty distribution over next r

        Obtain a pseudo-count by updating towards r = 0 and 1 with fake
        data, observe shift in estimation of next reward, given the
        current state.

        Args:
            state: the current state to estimate next reward from

        Returns:
            alpha, beta: defining beta distribution over next reward
        """
        if self.use_pseudocount:
            current_mean = self.estimate(state)
            fake_rewards = [0., 1.]
            fake_means = np.empty((2,))
            # TODO - something more efficient than copy? Actually add, then strip?
            for i, fake_r in enumerate(fake_rewards):
                fake_dict = {state: self.state_dict[state].copy()}
                self.update([(state, fake_r)], fake_dict)
                fake_means[i] = self.estimate(state, fake_dict)

            # n is the pseudo-count of number of times we've been in this
            # state. We take the n that gives the most uncertainty?

            # TODO - handle nans if current_mean is already 0. or 1.

            # Assume the base estimator’s mean tracks the sample mean.
            # To recover a distribution (w.r.t epistemic uncertainty) over the true
            # mean, let n be the estimated data count in the sample.
            # Because μ0 ≈ μn /(n + 1)
            n_0 = (  # TEMP handling of / 0 error
                fake_means[0] / (current_mean - fake_means[0])
                if current_mean != fake_means[0] else None
            )

            # Because μ1 ≈ (μn + 1)/(n + 1)
            n_1 = (
                (1. - fake_means[1]) / (fake_means[1] - current_mean)
                if current_mean != fake_means[1] else None
            )

            # Take min, or ignore the None value by making it the larger number
            n = min((n_0 or n_1 + np.abs(n_1)), (n_1 or n_0 + np.abs(n_0)))
            assert n >= 0., f"Unexpected n {n}, from ({n_0}, {n_1})"
        else:
            current_mean, n = self.state_dict[state]

        alpha = current_mean * n + 1.  # pseudo-successes (r=1)
        beta = (1. - current_mean) * n + 1.  # pseudo-failures (r=0)

        return alpha, beta

    def update(self, history, update_dict=None):
        """Algorithm 1. Use experience to update estimate of immediate r

        Args:
            history (list[tuple]): list of (state, reward) tuples to add
            update_dict (dict, None): the state_dict to apply the update
                to. Defaults to self.state_dict.
        """

        if update_dict is None:
            update_dict = self.state_dict

        for state, reward in history:
            if state not in update_dict:
                update_dict[state] = (reward, 1)
            else:
                current_r, current_n = update_dict[state]
                new_r = (current_r * current_n + reward) / (current_n + 1)
                update_dict[state] = (new_r, current_n + 1)

            self.total_updates += 1


class MentorQEstimator(Estimator):

    def __init__(
            self, num_states, num_actions, gamma, lr=0.1, scaled=True,
            init_val=1.):
        """Set up the QEstimator for the mentor

        Rather than using a Q-table with shape num_states * num_actions
        we use a list with shape num_states, and update this "Q-list"
        for the given state regardless of what action was taken.
        This is allowed because we never will choose an action from this,
        only compare Q-values to decide when to query the mentor.

        Args:
            num_states (int): the number of states in the environment
            num_actions (int): the number of actions
            gamma (float): the discount rate for Q-learning
        """
        self._init_lr = lr
        self._init_val = init_val
        super().__init__(lr, scaled=scaled)
        self.num_actions = num_actions
        self.num_states = num_states
        self.gamma = gamma

        self.q_list = np.ones(self.num_states) * self._init_val
        self.n = np.zeros(self.num_states)

    def reset(self):
        self.lr = self._init_lr
        self.q_list = np.ones(self.num_states) * self._init_val
        self.n = np.zeros(self.num_states)

    def update(self, history):
        """Update the mentor model's Q-list with a given history.

        In practice this history will be for actions when the
        mentor decided the action.

        Args:
            history (list): (state, action, reward, next_state) tuples
        """

        for state, action, reward, next_state, done in history:
            self.n[state] += 1 

            next_q_val = self.estimate(next_state) if not done else 0.
            scaled_r = (1 - self.gamma) * reward if self.scaled else reward

            q_target = scaled_r + self.gamma * next_q_val

            self.update_estimator(state, q_target)
            self.total_updates += 1

    def update_estimator(self, state, target_q_val, update_table=None):
        if update_table is None:
            update_table = self.q_list

        update_table[state] += self.get_lr(state) * (
                target_q_val - update_table[state])
        self.decay_lr()

    def estimate(self, state, q_list=None):
        """Estimate the future Q, using this estimator

        Estimate the Q value of what the mentor would choose, regardless
        of what action is taken.

        Args:
            state (int): the current state from which the Q value is being
                estimated
            q_list(np.ndarray): the Q table to estimate the value from,
                if None use self.Q_list as default
        """

        if q_list is None:
            q_list = self.q_list

        return q_list[state]

    def get_lr(self, state=None):
        """Calculate the learning rate for updating the Q table.

        If self.lr is specified use that, otherwise use 1/(n+1),
        where n is the number of times the state has been visited.

        Args:
            state (int): The state the Q table is being updated for
        """
        if self.lr is not None:
            return self.lr
        else:
            return 1. / (1. + self.n[state])


class MentorFHTDQEstimator(Estimator):

    def __init__(
            self, num_states, num_actions, num_steps, gamma, lr=0.1, scaled=True,
            init_val=1.
    ):
        """Set up the Finite horizon QEstimator for the mentor

        Args:
            num_states (int): the number of states in the environment
            num_actions (int): the number of actions
            gamma (float): the discount rate for Q-learning
        """
        self._init_val = init_val

        super().__init__(lr, scaled=scaled)
        self.num_actions = num_actions
        self.num_states = num_states
        self.num_steps = num_steps
        self.gamma = gamma
        # Add an extra one for the Q_zero est
        self.q_list = np.ones(
            (self.num_states, self.num_steps + 1)) * self._init_val
        self.q_list[:, 0] = 0.  # Q_0 starts at 0
        self.n = np.zeros(self.num_states)

        self.total_updates = np.zeros(self.num_steps + 1, dtype=int)

    def reset(self):
        # Add an extra one for the Q_zero est
        self.q_list = np.ones(
            (self.num_states, self.num_steps + 1)) * self._init_val
        self.q_list[:, 0] = 0.  # Q_0 starts at 0
        self.n = np.zeros(self.num_states)

        self.total_updates = np.zeros(self.num_steps + 1, dtype=int)

    @classmethod
    def get_steps_constructor(cls, num_steps):
        """Return a constructor that matches the MentorQEstimator's"""

        def init_with_n_steps(
                num_states, num_actions, gamma=0.99, lr=0.1, scaled=False,
                init_val=1.
        ):
            return cls(
                num_states=num_states, num_actions=num_actions,
                num_steps=num_steps, gamma=gamma, lr=lr, scaled=scaled,
                init_val=init_val
            )

        return init_with_n_steps

    def update(self, history):
        """Update the mentor model's Q-list with a given history.

        In practice this history will be for actions when the
        mentor decided the action.

        Args:
            history (list): (state, action, reward, next_state) tuples
        """

        for state, action, reward, next_state, done in history:
            self.n[state] += 1

            for h in range(1, self.num_steps + 1):
                next_q = self.estimate(next_state, h-1) if not done else 0.
                scaled_r = (1 - self.gamma) * reward if self.scaled else reward
                assert not self.scaled, "Q value must not be scaled for FH"
                q_target = scaled_r + self.gamma * next_q

                self.update_estimator(state, q_target, horizon=h)

                self.total_updates[h] += 1

    def update_estimator(
            self, state, target_q_val, horizon=None, update_table=None):
        if update_table is None:
            update_table = self.q_list

        update_table[state, horizon] += self.get_lr(state) * (
                target_q_val - update_table[state, horizon]
        )
        self.decay_lr()

    def estimate(self, state, horizon=None, q_list=None):
        """Estimate the future Q, using this estimator

        Estimate the Q value of what the mentor would choose, regardless
        of what action is taken.

        Args:
            state (int): the current state from which the Q value is
                being estimated
            horizon (Optional[int]): which timestep horizon to estimate.
                If None, returns the farthest horizon
            q_list (np.ndarray): the Q table to estimate the value from,
                if None use self.Q_list as default
        """
        if q_list is None:
            q_list = self.q_list
        if horizon is None:
            horizon = -1
        return q_list[state, horizon]

    def get_lr(self, state=None):
        """ Calculate the learning rate for updating the Q table.
        
        If self.lr is specified use that, otherwise use 1/(n+1),
        where n is the number of times the state has been visted.

        Args:
            state (int): The state the Q table is being updated for

        """
        if self.lr is not None:
            return self.lr
        else:
            return 1. / (1. + self.n[state])


class BaseQEstimator(Estimator, abc.ABC):

    def __init__(
            self, num_states, num_actions, gamma, lr, q_table_init_val=0.,
            scaled=True, num_steps=None,
    ):
        """Instantiate

        Args (also see Estimator):
            num_states:
            num_actions:
            gamma:
            lr:
            q_table_init_val (float): Value to initialise Q table at
            scaled:
            num_steps (int, str): The number of steps into the future
                to estimate the Q value for. Defaults to None, for
                infinite horizon.
        """
        self._q_table_init_val = q_table_init_val
        super().__init__(lr, scaled=scaled)
        self.num_actions = num_actions
        self.num_states = num_states
        self.gamma = gamma

        self.num_steps = num_steps

        self.transition_table = np.zeros(
            (num_states, num_actions, num_states), dtype=int)
        self.q_table = np.zeros((num_states, num_actions)) + q_table_init_val

        self.random_act_prob = None

    def reset(self):
        self.transition_table = np.zeros(
            (self.num_states, self.num_actions, self.num_states), dtype=int)
        self.q_table = np.zeros(
            (self.num_states, self.num_actions)) + self._q_table_init_val

    def get_random_act_prob(self, decay_factor=5e-5, min_random=0.05):
        if (self.random_act_prob is not None
                and self.random_act_prob > min_random):
            self.random_act_prob *= (1. - decay_factor)
        return self.random_act_prob

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
        if update_table is None:
            update_table = self.q_table

        if lr is None:
            lr = self.get_lr(state, action)

        update_table[state, action] += lr * (
                q_target - update_table[state, action])
        self.decay_lr()

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
            return 1 / (1 + self.transition_table[state, action, :].sum())


class QuantileQEstimator(BaseQEstimator):
    """A Q table estimator that makes pessimistic updates e.g. using IRE

    Updates using algorithm 3 in the QuEUE specification.
    """
    horizon_type = "infinite"

    def __init__(
            self,
            quantile,
            immediate_r_estimators,
            gamma,
            num_states,
            num_actions,
            lr=0.1,
            use_pseudocount=False,
            init_to_zero=False,
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
            Optional scaling, finite horizon Q estimators
        """
        # "Burn in" quantile with init value argument
        super().__init__(
            num_states, num_actions, gamma, lr,
            q_table_init_val=0. if init_to_zero else quantile)
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


class QuantileQEstimatorSingleOrig(QuantileQEstimator):
    """Override the update to do a single Q estimate"""

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


class QEstimator(BaseQEstimator):
    """A basic Q table estimator

    Updates (as 'usual') towards the target of:
        gamma * Q(s', a*) + (1. - gamma) * reward
    """

    def __init__(
            self, num_states, num_actions, gamma, lr=0.1, has_mentor=False,
            scaled=True
    ):
        super().__init__(
            num_states, num_actions, gamma, lr, scaled=scaled,
            q_table_init_val=0.
        )

        self.has_mentor = has_mentor
        self.random_act_prob = None if self.has_mentor else 1.

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


class QMeanIREEstimator(BaseQEstimator):
    """A basic Q table estimator which uses mean IRE instead of reward

    Updates (as 'usual') towards the target of:
        gamma * Q(s', a*) + (1. - gamma) * IRE(state)

    TODO:
        Could be a special case of the QPessIREEstimator, with a flag
        (e.g. quantile=None) to use .estimate rather than with
        uncertainty.
    """

    def __init__(
            self, num_states, num_actions, gamma, immediate_r_estimators,
            lr=0.1, has_mentor=False, scaled=True
    ):

        super().__init__(num_states, num_actions, gamma, lr, scaled=scaled)

        self.has_mentor = has_mentor
        self.immediate_r_estimators = immediate_r_estimators

        self.random_act_prob = None if self.has_mentor else 1.

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


class QPessIREEstimator(BaseQEstimator):
    """A basic Q table estimator which uses mean IRE instead of reward

    Does not do the "2nd" update (for transition uncertainty)

    Updates (as 'usual') towards the target of:
        gamma * Q(s', a*) + (1. - gamma) * IRE_qi(state)
    """

    def __init__(
            self, quantile, num_states, num_actions, gamma,
            immediate_r_estimators, lr=0.1, has_mentor=False, scaled=True
    ):
        super().__init__(
            num_states, num_actions, gamma, lr, scaled=scaled,
            q_table_init_val=quantile
        )

        self.quantile = quantile
        self.has_mentor = has_mentor
        self.immediate_r_estimators = immediate_r_estimators

        self.random_act_prob = None if self.has_mentor else 1.

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


class FHTDQEstimator(BaseQEstimator):
    """A basic Q table estimator for the finite horizon

    Finite Horizon Temporal Difference Q Estimator
    As in: https://arxiv.org/pdf/1909.03906.pdf

    Recursively updates the next num_steps estimates of the reward:
        Q_0(s, a) = mean_{t : s_t = s} r_t
        V_i(s) = max_a Q_i(s, a)
        Q_{i+1}(s, a) = mean_{t : s_t = s}[
            (1 - gamma) * r_t  + gamma * V_i(s_{t+1})
        ]
    """

    def __init__(
            self, num_states, num_actions, num_steps, gamma=0.99, lr=0.1,
            has_mentor=False, q_table_init_val=0., scaled=False
    ):
        if not isinstance(num_steps, int):
            raise ValueError(f"Must be finite steps for FHTD: {num_steps}")
        if scaled:
            raise NotImplementedError("Not defined for scaled Q values")

        self._q_table_init_val = q_table_init_val

        super().__init__(
            num_states, num_actions, gamma, lr, scaled=scaled,
            num_steps=num_steps
        )

        # account for Q_0, add one in size to the "horizons" dimension
        self.q_table = np.zeros((num_states, num_actions, num_steps + 1))
        self.q_table += q_table_init_val
        self.q_table[:, :, 0] = 0.  # Q_0 always init to 0: future is 0 from now

        self.random_act_prob = None if has_mentor else 1.

    def reset(self):
        super().reset()
        self.q_table = np.zeros(
            (self.num_states, self.num_actions, self.num_steps + 1))
        self.q_table += self._q_table_init_val
        self.q_table[:, :, 0] = 0.  # Q_0 always init to 0: future is 0 from now

    @classmethod
    def get_steps_constructor(cls, num_steps):
        """Return a constructor that matches the QEstimator's"""
        def init_with_n_steps(
                num_states, num_actions, gamma=0.99, lr=0.1, has_mentor=False,
                q_table_init_val=0., scaled=False
        ):
            return cls(
                num_states=num_states, num_actions=num_actions,
                num_steps=num_steps, gamma=gamma, lr=lr, has_mentor=has_mentor,
                q_table_init_val=q_table_init_val, scaled=scaled
            )
        return init_with_n_steps

    def estimate(self, state, action, h=None, q_table=None):
        """Estimate the future Q, using this estimator

        Args:
            state (int): the current state from which the Q value is
                being estimated
            action (int): the action taken
            h (int): the horizon we are estimating for, if None then
                use the final horizon, which is used for choosing
                which action to take.
            q_table (np.ndarray): the Q table to estimate the value
                from, if None use self.q_table as default

        Returns:
            q_estimate (float):
        """
        if q_table is None:
            q_table = self.q_table

        if h is None:
            h = -1

        return q_table[state, action, h]

    def update(self, history):
        for state, action, reward, next_state, done in history:

            self.transition_table[state, action, next_state] += 1


            for h in range(1, self.num_steps + 1):
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

                self.update_estimator(state, action, q_target, horizon=h)

    def update_estimator(
            self, state, action, q_target, horizon=None, update_table=None):
        """New update for this special case Q table, requires h

        New args:
            horizon (int): the index of the Q-horizon to update. Default
                to None to match argument signature of super (a bit of
                a hack)
        """

        if update_table is None:
            update_table = self.q_table

        update_table[state, action, horizon] += self.get_lr(state, action) * (
                q_target - update_table[state, action, horizon])

        self.decay_lr()
