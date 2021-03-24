import torch as tc
import numpy as np


class Estimator:

    def __init__(self):
        pass


class ImmediateRewardEstimator(Estimator):
    """

    """
    def __init__(self, action):
        """

        Args:
            action: the action this estimator is specific to.
                TEMP: it may not be necessary to hold different action
                IREs for every action - could instead store it all
                here and index the estimators with it.
        """
        super().__init__()

    def estimate(self, action):
        """Provide the mean(?) estimate for an action
        Args:
            action: TEMP - we may store a separate estimator for evrey
                action - not sure yet.
        """
        pass

    def expected_with_uncertainty(self, state):
        """Algorithm 2. Epistemic Uncertainty distribution over next r

        Update towards r = 0 and 1 with fake data, observe shift in
        estimation of next reward, given the current state.

        Args:
            state: the current state (is it the reward of this state?)
        """
        pass

    def update(self, history):
        """Algorithm 1. Use experience to update estimate of immediate r

        history (list): (state, reward) tuples
        """
        pass


class QEstimator(Estimator):

    def __init__(self, quantile, action):
        """Algorithm 4. Burn in the quantiles.

        Set up such that expected(estimate(theta_i_a)) for i=0.1 -> r=0.1

        Args:
            as before, unsure if QEstimator will be a diff one per
            action, or if each qunatile estimator stores all the params
            for each action (more like deep Q)
        """
        super().__init__()
        self.quantile = quantile  # the 'i' index of the quantile
        # TODO - action?

    def update(self, history):
        """Algorithm 3. Use history to update future-Q quantiles.

        The Q-estimator stores estimates of multiple quantiles in the
        distribution (with respect to epistemic uncertainty) of the
        Q-value.

        It updates by boot-strapping at a given state-action pair.

        history (list): (state, action, reward, next_state) tuples
        """
        pass

    def estimate(self, action, immediate_r_estimator):
        """Estimate the future Q for the given quantile and action indices

        TODO - action index here or in the init?
        """
