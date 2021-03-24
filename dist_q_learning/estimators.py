import abc
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline

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

        Update towards r = 0 and 1 with fake data, observe shift in
        estimation of next reward, given the current state.

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
            # TODO - should be expectation value of [estimate] - no diff here?
            fake_means[i] = self.estimate(state, fake_dict)

        # TODO - handle nans if current_mean is already 0. or 1.

        # Assume the base estimator’s mean tracks the sample mean.
        # To recover a distribution (w.r.t epistemic uncertainty) over the true
        # mean, let n be the estimated data count in the sample.
        # Because μ 0 ≈ μn/(n + 1)
        n_0 = (  # TEMP handling of / 0 error
            fake_means[0] / (current_mean - fake_means[0])
            if current_mean != fake_means[0] else None
        )

        # Because μ 1 ≈ (μn + 1)/(n + 1)
        n_1 = (
            (1. - fake_means[1]) / (fake_means[1] - current_mean)
            if current_mean != fake_means[1] else None
        )

        # Take min, or ignore the None value by making it the larger number
        n = min((n_0 or n_1 + np.abs(n_1)), (n_1 or n_0 + np.abs(n_0)))

        assert n >= 0., f"Unexpected n {n}, from ({n_0}, {n_1})"
        alpha = (1. - current_mean) * n + 1.
        beta = current_mean * n + 1.
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

    def __init__(self, quantile, immediate_r_estimators):
        """Set up the QEstimator for the given quantile & action

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

        # TODO - burn-in via updating

    def update(self, history):
        """Algorithm 3. Use history to update future-Q quantiles.

        The Q-estimator stores estimates of multiple quantiles in the
        distribution (with respect to epistemic uncertainty) of the
        Q-value.

        It updates by boot-strapping at a given state-action pair.

        Args:
            history (list): (state, action, reward, next_state) tuples

        Updates parameters for this estimator, theta_i_a
        """
        pass

    def estimate(self, state):
        """Estimate the future Q, using this estimator

        Args:
            state (): the current state from which the Q value is being
                estimated
        """
        pass
