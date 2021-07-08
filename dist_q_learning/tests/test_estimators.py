import numpy as np
import unittest

from haiku.data_structures import to_immutable_dict

from utils import plot_beta
from q_estimators import QuantileQEstimator, QuantileQEstimatorGaussianGLN
from estimators import (
    ImmediateRewardEstimator, MentorQEstimator,
    ImmediateRewardEstimatorGaussianGLN)


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


class TestImmediateRewardEstimatorGaussianGLN(unittest.TestCase):

    test_state = np.array([[0.4, 0.5], [0.2, 0.3]])

    def test_estimate(self):
        """Test estimate method runs"""
        fake_ire = ImmediateRewardEstimatorGaussianGLN(
            action=0, layer_sizes=[2, 1], burnin_n=10)
        est = fake_ire.estimate(np.array([[0.3, 0.4]]))
        print("EST", est)

    def test_expected_with_uncertainty(self):
        """Test the uncertainty estimate over the IRE runs"""
        ire = ImmediateRewardEstimatorGaussianGLN(
            action=0, lr=1e-4, context_dim=2, layer_sizes=[2, 1],
            burnin_n=10, burnin_val=0.5
        )
        params = to_immutable_dict(ire.model.gln_params)
        lr = ire.model.lr
        assert lr == ire.lr
        batch_x = np.repeat(self.test_state[0:1], 25, axis=0)
        values = np.full(len(batch_x), 0.5)
        ns, alphas, betas = ire.model.uncertainty_estimate(
            self.test_state[0:1],
            batch_x,
            values,
            converge_epochs=2, debug=True)
        print(ns, alphas, betas)

        lr_after = ire.model.lr
        assert lr_after == ire.lr == lr

        def flat(d):
            result = []
            if hasattr(d, "keys"):
                for k in d.keys():
                    result.extend(flat(d[k]))
            else:
                result.extend(d)
            return result

        params_after = to_immutable_dict(ire.model.gln_params)
        print(params)
        params = np.array(flat(params))
        params_after = np.array(flat(params_after))
        print(params.shape)
        print(params)

        assert np.all(params_after == params)

    def test_update(self):
        """Test that updating the GLN updates towards the target"""
        ire = ImmediateRewardEstimatorGaussianGLN(
            action=0, burnin_n=10, burnin_val=0., lr=1e-3, context_dim=2,
            layer_sizes=[2, 1])
        init_estimate = ire.estimate(self.test_state)
        print(f"Estimate before: {init_estimate}")

        for i in range(10):
            state_rew_history = (self.test_state, [0.5] * len(self.test_state))
            ire.update(state_rew_history, tup=True)
        next_est = ire.estimate(self.test_state)
        print(f"Estimate after: {next_est}")

        assert not np.any(init_estimate == next_est)
        assert np.all(np.abs(next_est - 0.5) < np.abs(init_estimate - 0.5))


class TestQEstimatorGaussianGLN(unittest.TestCase):

    test_state = np.array([[0.4, 0.5], [0.2, 0.3]])
    test_acts = np.array([0, 1])
    num_acts = 2

    def initialise_IREs(self, num_a):
        ires = []
        for i in range(num_a):
            ires.append(
                ImmediateRewardEstimatorGaussianGLN(
                    action=i,
                    burnin_n=10,
                    burnin_val=0.5,
                    layer_sizes=[4, 1],
                    lr=1e-5,
                ))
        return ires

    def test_estimate(self):
        IREs = self.initialise_IREs(self.num_acts)
        Q = QuantileQEstimatorGaussianGLN(
            quantile=0.5, immediate_r_estimators=IREs,
            dim_states=2, num_actions=self.num_acts, gamma=0.99, layer_sizes=[4, 1],
            lr=0.01, burnin_n=10, batch_size=2)

        Q_est = Q.estimate(states=self.test_state, actions=self.test_acts)
        print(f"Q estimate: {Q_est}")

    def test_update(self):
        IREs = self.initialise_IREs(self.num_acts)
        Q = QuantileQEstimatorGaussianGLN(
            quantile=0.5, immediate_r_estimators=IREs, dim_states=2,
            num_actions=self.num_acts, gamma=0.99, layer_sizes=[4, 1], lr=1e-4,
            burnin_n=10, burnin_val=0.5)

        Q_est2 = Q.estimate(self.test_state, self.test_acts)
        print(f"Q estimate2: {Q_est2}")

        n_state = [0.4, 0.5]
        n_state2 = [0.2, 0.3]

        two_data = [
            (self.test_state[0], self.test_acts[0], 0.5, n_state, False),
            (self.test_state[1], self.test_acts[1], 0.5, n_state2, True)]
        for _ in range(10):
            Q.update(two_data, two_data, debug=True)

        Q_est3 = Q.estimate(self.test_state, self.test_acts)
        print(f"Q estimate3: {Q_est3}")

        assert not np.any(Q_est3 == Q_est2)
