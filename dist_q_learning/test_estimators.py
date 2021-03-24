from unittest import TestCase

from estimators import ImmediateRewardEstimator, plot_beta


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
