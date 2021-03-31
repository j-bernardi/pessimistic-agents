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

    ax.plot(xs, f(xs))
    if show:
        plt.show()

    return ax


class Estimator(abc.ABC):

    def __init__(self):
        pass

    @abc.abstractmethod
    def estimate(self, state):
        """Use the estimator to generate an estimate given state"""
        return

    @abc.abstractmethod
    def update(self, history):
        """Update the estimator given some experience"""
        return


class ImmediateRewardEstimator(Estimator):
    """Estimates the next reward given a current state and an action

    TODO:
        Only store number and the current value to recover mean
        (erring on saving too much for now)
    """
    def __init__(self, action):
        """Create an action-specific IRE.

        Args:
            action: the action this estimator is specific to.
        """
        super().__init__()
        self.action = action

        # Maps state to a list of next-rewards
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

        rewards = from_dict[state]

        return np.mean(rewards)

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
            update_dict = self.state_dict

        for state, reward in history:
            if state not in update_dict:
                update_dict[state] = [reward]
            else:
                update_dict[state].append(reward)


class QEstimator(Estimator):

    def __init__(
            self,
            quantile,
            immediate_r_estimators,
            gamma,
            num_states,
            num_actions,
            lr=0.1,
            use_pseudocount=False
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
        super().__init__()
        if quantile <= 0. or quantile > 1.:
            raise ValueError(f"Require 0. < q <= 1. {quantile}")
        self.quantile = quantile  # the 'i' index of the quantile
        self.num_actions = num_actions
        self.num_states = num_states
        self.gamma = gamma
        self.lr = lr
        self.use_pseudocount = use_pseudocount
        self.immediate_r_estimators = immediate_r_estimators

        # Algorithm 4 burn in Quantile for Q table
        self.Q_table = np.ones((num_states, num_actions)) * self.quantile

        self.transition_table = np.zeros(
            (num_states, num_actions, num_states), dtype=int)

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
            if not done:
                V_i = np.max([
                    self.estimate(next_state, action_i)
                    for action_i in range(self.num_actions)]
                )
            else:
                V_i = 0.

            ire = self.immediate_r_estimators[action]
            ire_alpha, ire_beta = ire.expected_with_uncertainty(state)
            IV_i = scipy.stats.beta.cdf(self.quantile, ire_alpha, ire_beta)

            Q_target = self.gamma * V_i + (1. - self.gamma) * IV_i

            self.update_estimator(state, action, Q_target)

            # Account for uncertainty in the state transition function
            q_ai = self.estimate(state, action)
            if not self.use_pseudocount:
                n = self.transition_table[state, action, :].sum()

            else:
                fake_q_ai = []
                for fake_target in [0., 1.]:
                    fake_Q_table = self.Q_table.copy()
                    self.update_estimator(
                        state, action, fake_target, update_table=fake_Q_table)
                    fake_q_ai.append(self.estimate(state, action, fake_Q_table))

                n_ai0 = fake_q_ai[0] / (q_ai - fake_q_ai[0])
                n_ai1 = (1. - fake_q_ai[1]) / (fake_q_ai[1] - q_ai)
                # in the original algorithm this min was also over the quantiles
                # as well as the fake targets
                n = np.min([n_ai0, n_ai1])

            q_alpha = q_ai * n + 1.
            q_beta = (1. - q_ai) * n + 1.
            Q_target_transition = scipy.stats.beta.cdf(
                self.quantile, q_alpha, q_beta)
            
            self.update_estimator(state, action, Q_target_transition)

    def update_estimator(self, state, action, Q_target, update_table=None):
        if update_table is None:
            update_table = self.Q_table

        update_table[state, action] += self.lr * (
                Q_target - update_table[state, action])

    def estimate(self, state, action, Q_table=None):
        """Estimate the future Q, using this estimator

        Args:
            state (int): the current state from which the Q value is being
                estimated
            action (int): the action taken
            Q_table(np.ndarray): the Q table to estimate the value from, 
                if None use self.Q_table as default
        """

        if Q_table is None:
            Q_table = self.Q_table

        return Q_table[state, action]


class MentorQEstimator(Estimator):

    def __init__(self, num_states, num_actions, gamma, lr=0.1):
        """Set up the QEstimator for the mentor

        Rather than using a Q-table with shape num_states * num_actions
        we use a list with shape num_states, and update this "Q-list"
        for the given state regardless of what action was taken.
        This is allowed because we never will choose an action from this,
        only compare Q-values to decide when to query the mentor. 

        Args:
            num_states (int): the number of states in the environment
            num_actions (int): the number of actions
            gamma (float): the discount rate for Q-learing
        """
        super().__init__()        
        self.num_actions = num_actions
        self.num_states = num_states
        self.gamma = gamma
        self.lr = lr
        self.Q_list = np.ones(num_states)

    def update(self, history):
        """Update the mentor model's Q-list with a given history.

        In practice this history will be for actions when the 
        mentor decided the action.

        Args:
            history (list): (state, action, reward, next_state) tuples
        """

        for state, action, reward, next_state, done in history:

            V_i = self.estimate(next_state) if not done else 0.
            
            Q_target = self.gamma * V_i + (1 - self.gamma) * reward
            # Q_target = reward + self.gamma * V_i 

            self.update_estimator(state, Q_target)

    def update_estimator(self, state, Q_target, update_table=None):
        if update_table is None:
            update_table = self.Q_list

        update_table[state] += self.lr * (Q_target - update_table[state]) 

    def estimate(self, state, Q_list=None):
        """Estimate the future Q, using this estimator

        Estimate the Q value of what the mentor would choose, regardless
        of what action is taken.

        Args:
            state (int): the current state from which the Q value is being
                estimated
            Q_list(np.ndarray): the Q table to estimate the value from, 
                if None use self.Q_list as default
        """

        if Q_list is None:
            Q_list = self.Q_list

        return Q_list[state]
