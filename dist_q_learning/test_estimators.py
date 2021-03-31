import numpy as np
from unittest import TestCase

from estimators import (
    ImmediateRewardEstimator, plot_beta, QEstimator, MentorQEstimator)


class TestImmediateRewardEstimator(TestCase):

    def test_estimate(self):
        """Test estimate method returns expected value from history"""
        fake_ire = ImmediateRewardEstimator(action=0)
        fake_ire.state_dict = {0: [0., 0.8, 1.], 1: [0., 1.]}

        assert fake_ire.estimate(0) == 0.6
        assert fake_ire.estimate(1) == 0.5

    def test_expected_with_uncertainty(self):
        """Test the uncertainty estimate over the IRE"""
        test_cases = [
            (0.5, 1.5, 1.5)
        ]

        ire = ImmediateRewardEstimator(action=0)
        for init_r, exp_a, exp_b in test_cases:
            ire.state_dict[0] = [init_r]  # undecided what to do if empty
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
        fake_dict = {2: [0.5]}
        state_rew_history = [(0, 1.), (0, 0.), (1, 1.)]

        for t, update_dict in enumerate((None, fake_dict)):
            print("Subtest", t)
            ire.update(state_rew_history, update_dict=update_dict)
            assert ire.state_dict == {0: [1., 0.], 1: [1.]}, ire.state_dict

        assert fake_dict == {0: [1., 0.], 1: [1.], 2: [0.5]}, fake_dict


class TestQEstimator(TestCase):

    def initialise_IREs(self):
        IREs=[]
        for i in range(2):
            IREs.append(ImmediateRewardEstimator(i))
            IREs[i].state_dict = {0: [0., 0.8, 1.], 1: [0., 1., 0.3]}
        
        return IREs

    def test_estimate(self):
        IREs = self.initialise_IREs()
        Q = QEstimator(0.5, IREs, 0.99, 4, 2)
        Q.estimate(0, 1)
        print('hi')

    def test_update(self):
        IREs =  self.initialise_IREs()

        Q = QEstimator(0.5, IREs, 0.99, 4, 2, lr=10)
        assert np.all(Q.Q_table == 0.5)
        print(Q.Q_table)
        Q.update([(0, 1, 0.9, 2), (1, 0, 0.9, 3)])
        # assert Q.Q_table[0,1] == Q.quantile + Q.lr *
        # (self.IREs[1].estimate(0)*(1 - Q.gamma) + Q.gamma * 0.5 - Q.quantile)
        print(Q.Q_table)


class TestMentorQEstimator(TestCase):

    def test_estimate(self):
        MQ = MentorQEstimator(4, 2, 0.99)

        MQ.estimate(0)

    def test_update(self):
        MQ = MentorQEstimator(4, 2, 0.99)
        assert np.all(MQ.Q_list == 1)
        print(MQ.Q_list)
        MQ.update([(0, 1, 0.9, 2), (1, 0, 0.9, 3)])
        print(MQ.Q_list)