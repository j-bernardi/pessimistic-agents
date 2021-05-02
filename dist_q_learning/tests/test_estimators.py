import numpy as np
import unittest

from q_estimators import QuantileQEstimator
from estimators import (
    ImmediateRewardEstimator, plot_beta, MentorQEstimator,
    ImmediateRewardEstimator_GLN_gaussian,
    QuantileQEstimator_GLN_gaussian,
)


class TestImmediateRewardEstimator(unittest.TestCase):

    def test_estimate(self):
        """Test estimate method returns expected value from history"""
        fake_ire = ImmediateRewardEstimator(action=0)
        fake_ire.state_dict = {0: (0.6, 3), 1: (0.5, 2)}

        assert fake_ire.estimate(0) == 0.6
        assert fake_ire.estimate(1) == 0.5

    def test_expected_with_uncertainty(self):
        """Test the uncertainty estimate over the IRE"""
        test_cases = [
            (0.5, 1.5, 1.5)
        ]

        ire = ImmediateRewardEstimator(action=0)
        for init_r, exp_a, exp_b in test_cases:
            ire.state_dict[0] = (init_r, 1)  # undecided what to do if empty
            a, b = ire.expected_with_uncertainty(0)
            print("ALPHA, BETA", a, b)
            assert a == exp_a and b == exp_b, f"{exp_a}: {a}, {exp_b}: {b}"

    def test_plotting(self):
        a, b = 1.5, 1.5
        plot_beta(a, b, show=False)

    def test_update(self):
        """Test data can be added to the state_dict with update.

        Also tests the fake_dict operation.
        """

        ire = ImmediateRewardEstimator(action=0)
        fake_dict = {2: (0.5, 1)}
        state_rew_history = [(0, 1.), (0, 0.), (1, 1.)]

        for t, update_dict in enumerate((None, fake_dict)):
            print("Subtest", t)
            ire.update(state_rew_history, update_dict=update_dict)
            assert ire.state_dict == {0: (0.5, 2), 1: (1., 1)}, ire.state_dict

        assert fake_dict == {0: (0.5, 2), 1: (1., 1), 2: (0.5, 1)}, fake_dict


class TestQEstimator(unittest.TestCase):

    def initialise_IREs(self):
        IREs=[]
        for i in range(2):
            IREs.append(ImmediateRewardEstimator(i))
            IREs[i].state_dict = {0: (0.6, 3), 1: (0.4, 3)}
        
        return IREs

    def test_estimate(self):
        ires = self.initialise_IREs()
        estimator = QuantileQEstimator(
            quantile=0.5, immediate_r_estimators=ires, gamma=0.99,
            num_states=4, num_actions=2, lr=1.
        )
        estimator.estimate(0, 1)

    def test_update(self):
        ires = self.initialise_IREs()
        q = 0.5
        estimator = QuantileQEstimator(
            quantile=q, immediate_r_estimators=ires, gamma=0.99, num_states=4,
            num_actions=2, lr=1., q_table_init_val=q,
        )
        print(estimator.q_table)
        assert np.all(estimator.q_table[:, :, -1] == q)
        # s, a, r, s', d
        estimator.update([(0, 1, 0.9, 2, False), (1, 0, 0.9, 3, True)])
        # assert Q.Q_table[0,1] == Q.quantile + Q.lr *
        # (self.IREs[1].estimate(0)*(1 - Q.gamma) + Q.gamma * 0.5 - Q.quantile)
        print(estimator.q_table)


class TestMentorQEstimator(unittest.TestCase):

    def test_estimate(self):
        mentor_estimator = MentorQEstimator(
            num_states=4, num_actions=2, gamma=0.99)

        mentor_estimator.estimate(0)

    def test_update(self):
        mentor_estimator = MentorQEstimator(
            num_states=4, num_actions=2, gamma=0.99)
        assert np.all(mentor_estimator.q_list == 1)
        print(mentor_estimator.q_list)
        # s, a, r, s', d
        mentor_estimator.update([(0, 1, 0.9, 2, False), (1, 0, 0.9, 3, True)])
        print(mentor_estimator.q_list)


class TestImmediateRewardEstimator_GLN_gaussian(unittest.TestCase):

    def test_estimate(self):
        """Test estimate method returns expected value from history"""
        fake_ire = ImmediateRewardEstimator_GLN_gaussian(action=0)
    #     fake_ire.state_dict = {0: [0., 0.8, 1.], 1: [0., 1.]}
        fake_ire.estimate(np.array([1, 3]))
    #     assert fake_ire.estimate(0) == 0.6
    #     assert fake_ire.estimate(1) == 0.5

    def test_expected_with_uncertainty(self):
        """Test the uncertainty estimate over the IRE"""
        test_cases = [
            (0.5, 1.5, 1.5)
        ]

        ire = ImmediateRewardEstimator_GLN_gaussian(action=0, lr=0.01)
        for init_r, exp_a, exp_b in test_cases:
            a, b = ire.expected_with_uncertainty(np.array([3, 4]))
            print("ALPHA, BETA", a, b)

    # def test_plotting(self):
    #     a, b = 1.5, 1.5
    #     plot_beta(a, b, show=False)

    def test_update(self):
        """

        """
        ire = ImmediateRewardEstimator_GLN_gaussian(action=0, burnin_n=1)
        print(f"Estimate before: {ire.estimate([0., 0.5])}")

        state_rew_history = [([0., 0.5], 0.9),
        ([0., 0.5], 0.8), ([-0.5, 0.5], 0.2)]

        for i in range(10):
            state_rew_history = [(
                [2 * np.random.rand() - 1, 2 * np.random.rand() - 1], 0.)]
            ire.update(state_rew_history)

        # print(ire.estimate([0., 0.5]))

        print(f"Estimate after: {ire.estimate([0., 0.5])}")


class TestQEstimator_GLN_gaussian(unittest.TestCase):
    @unittest.expectedFailure
    def initialise_IREs(self):
        IREs=[]
        for i in range(2):
            IREs.append(ImmediateRewardEstimator_GLN_gaussian(
                action=i, burnin_n=100, layer_sizes=[4]))

        return IREs

    @unittest.expectedFailure
    def test_estimate(self):
        IREs = self.initialise_IREs()
        Q = QuantileQEstimator_GLN_gaussian(
            0.5, IREs, 2, 4, 0.99, layer_sizes=[4], lr=0.01,
            burnin_n=10)
        Q_est = Q.estimate([0.4, 0.5], 1)
        print(f"Q estimate: {Q_est}")

    @unittest.expectedFailure
    def test_update(self):
        IREs = self.initialise_IREs()
        Q = QuantileQEstimator_GLN_gaussian(
            0.5, IREs, 2, 4, 0.99, layer_sizes=[4], lr=0.1,
            burnin_n=100)

        Q_est = Q.estimate([0.4, 0.5], 1)
        print(f"Q estimate2: {Q_est}")

        state = [0.4, 0.5]
        state2 = [0.2, -0.3]

        Q.update(
            [(state, 1, 0.9, state2, False), (state, 0, 0.9, state2, True)])

        Q_est = Q.estimate([0.4, 0.5], 1)
        print(f"Q estimate3: {Q_est}")
