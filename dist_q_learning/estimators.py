import torch as tc
import numpy as np


class Estimator:

    def __init__(self):
        pass


class ImmediateRewardEstimator(Estimator):
    """Estimates the next reward given a current state and an action

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

    def estimate(self, state, action):
        """Provide the mean immediate reward estimate for an action

        Args:
            state: the current state to estimate next reward from
            action: the action taken from this state, leading to next
                state (and expected reward). TEMP - we may store a
                separate estimator for every action - which would render
                this unnecessary (action would be internal class state)
                - not sure yet.
        """
        pass

    def expected_with_uncertainty(self, state, action):
        """Algorithm 2. Epistemic Uncertainty distribution over next r

        Update towards r = 0 and 1 with fake data, observe shift in
        estimation of next reward, given the current state.

        Args:
            state: the current state to estimate next reward from
            action: the action taken from this state, leading to next
                state (and expected reward). TEMP - we may store a
                separate estimator for every action - which would render
                this unnecessary (action would be internal class state)
                - not sure yet.

        Returns:
            beta distribution over next reward
        """
        pass

    def update(self, history):
        """Algorithm 1. Use experience to update estimate of immediate r

        history (list): (state, reward) tuples
        """
        pass


class QEstimator(Estimator):

    def __init__(self, quantile, action):
        """Set up the QEstimator for the given quantile & action

        'Burn in' the quantiles by calling 'update' with an artificial
        historical reward - Algorithm 4. E.g. call update with r=i, so
        that it updates theta_i_a, parameters for this estimator, s.t.:
            expected(estimate(theta_i_a)) for i=0.1 -> 0.1

        Args:
            quantile (float): the pessimism-quantile that this estimator
                is estimating the future-Q value for.
            action (int): as before, unsure if QEstimator will be a diff
                one per action, or if each qunatile estimator stores all
                the params for each action (more like deep Q)
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

        Args:
            history (list): (state, action, reward, next_state) tuples

        Updates parameters for this estimator, theta_i_a
        """
        pass

    def estimate(self, state, action, immediate_r_estimator):
        """Estimate the future Q, using this estimator

        Args:
            state (): the current state from which the Q value is being
                estimated
            action: the action corresponding to the set of parameters,
                to use to make this estimation. TEMP - we may store a
                separate estimator for every action - which would render
                this unnecessary (action would be internal class state)
                - not sure yet.
            immediate_r_estimator (ImmediateRewardEstimator): the
                estimator corresponding to quartile i, used to
                estimate the immediate value for quartile i (IV_i)
        """
        pass
