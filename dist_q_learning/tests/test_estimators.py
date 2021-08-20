import numpy as np

import unittest
import jax.numpy as jnp
import jax
import haiku as hk
from haiku.data_structures import to_immutable_dict, to_mutable_dict

from glns import GGLN
from gated_linear_networks.gaussian import _unpack_inputs, GatedLinearNetwork
from gated_linear_networks.base import GatedLinearNetwork as\
    base_GatedLinearNetwork
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
            action=0, lr=1e-4, context_dim=2, layer_sizes=[3, 1],
            burnin_n=10, burnin_val=0.5
        )
        init_params_d = to_immutable_dict(ire.model.gln_params)
        lr = ire.model.lr
        assert lr == ire.lr
        batch_x = np.repeat(self.test_state, 25, axis=0)
        values = np.full(len(batch_x), 0.5)
        print("BATCH", batch_x.shape, values.shape)
        ns, alphas, betas = ire.model.uncertainty_estimate(
            self.test_state,
            batch_x,
            values,
            converge_epochs=2, debug=True)
        print(ns, alphas, betas)

        lr_after = ire.model.lr
        assert lr_after == ire.lr == lr
        assert ire.model.weights_equal(init_params_d)

    def test_update(self):
        """Test that updating the GLN updates towards the target"""
        ire = ImmediateRewardEstimatorGaussianGLN(
            action=0, burnin_n=10, burnin_val=0., lr=1e-3, context_dim=2,
            layer_sizes=[2, 1])
        init_estimate = ire.estimate(self.test_state)
        print(f"Estimate before: {init_estimate}")

        for i in range(10):
            state_rew_history = (self.test_state, [0.5] * len(self.test_state))
            ire.update(state_rew_history)
        next_est = ire.estimate(self.test_state)
        print(f"Estimate after: {next_est}")

        assert not np.any(init_estimate == next_est)
        assert np.all(np.abs(next_est - 0.5) < np.abs(init_estimate - 0.5))

    def test_copy(self):

        def weights_equal(p1, p2):
            for k1, k2 in zip(p1, p2):
                return np.all(p1[k1]["weights"] == p2[k2]["weights"])

        shared_params = dict(
            layer_sizes=[6, 1],
            input_size=4,
            context_dim=3,
            batch_size=32,
            lr=0.001,
            min_sigma_sq=0.001,
            # init_bias_weights=[None, None, None],
            bias_max_mu=1,
        )
        model = GGLN(name=f"test", **shared_params)
        print(f"\nParam type init {type(model.gln_params)}")
        initial_params = to_immutable_dict(model.gln_params)
        print(f"\ntype of the hk copy {type(initial_params)}")
        assert weights_equal(model.gln_params, initial_params)

        # Do an update to change the weights of model 1
        model.predict(jnp.ones((32, 4)), jnp.full(32, 0.5))
        print(f"\ntype after update {type(model.gln_params)}")
        params_after_update = to_immutable_dict(model.gln_params)
        print(f"type after update and immut {type(model.gln_params)}")
        assert weights_equal(model.gln_params, params_after_update)
        model.copy_values(initial_params, debug=True)
        assert weights_equal(model.gln_params, initial_params)
        assert not weights_equal(model.gln_params, params_after_update)
        print(f"\ntype after custom copy {type(model.gln_params)}")
        params_after_copy = to_immutable_dict(model.gln_params)
        print(f"type after custom copy and immut {type(params_after_copy)}")

        # Make a fresh model
        model.copy_values(params_after_update)
        model2 = GGLN(name=f"test2", **shared_params)
        assert not weights_equal(model2.gln_params, model.gln_params)
        model.copy_values(model2.gln_params, debug=True)
        assert weights_equal(model2.gln_params, model.gln_params)
        assert not weights_equal(model2.gln_params, params_after_update)

    def test_haiku_mutibility(self):
        shared_params = dict(
            layer_sizes=[2, 1],
            input_size=4,
            context_dim=4,
            batch_size=32,
            lr=0.001,
            min_sigma_sq=0.001,
            init_bias_weights=[None, None, None],
            bias_max_mu=1,
        )
        model = GGLN(name=f"test", **shared_params)
        layer_0 = list(model.gln_params.keys())[0]
        immut_params = to_immutable_dict(model.gln_params)
        mut_params = to_mutable_dict(model.gln_params)
        immut_w = immut_params[layer_0]["weights"]
        mut_w = mut_params[layer_0]["weights"]

        def assert_add(w_array):
            """Can't update a jax array"""
            raised = False
            try:
                 w_array += 2.
            except ValueError as ve:
                if "output array is read-only" in str(ve):
                    raised = True
                else:
                    raise
            assert raised

        # Can't add to jax arrays
        assert_add(immut_w)
        assert_add(mut_w)

        # Can assign to mutable, not immutable
        mut_params["new_key"] = "new"
        try:
            immut_params["new_key"] = "new"
            assert False
        except TypeError as te:
            assert "does not support item assignment" in str(te), te

        # Nor to existing weights
        try:
            immut_params[layer_0] = "new"
            assert False
        except TypeError as te:
            assert "does not support item assignment" in str(te), te

    def test_gln_predict(self):
        shared_params = dict(
            layer_sizes=[2, 1],
            input_size=2,
            context_dim=2,
            batch_size=32,
            lr=0.001,
            min_sigma_sq=0.001,
            # init_bias_weights=[None, None, None],
            bias_max_mu=1,
        )
        model = GGLN(name=f"test", **shared_params)
        model2 = GGLN(name=f"test2", **shared_params)
        initial_params = to_immutable_dict(model.gln_params)
        initial_params2 = to_immutable_dict(model.gln_params)

        assert model.weights_equal(initial_params, True)
        assert model.weights_equal(initial_params2, True)  # all init to same

        model.predict(self.test_state, jnp.array([1., 1.]))
        model2.predict(self.test_state, jnp.array([0., 0.]))

        # they updated
        assert not model.weights_equal(initial_params, True)
        assert not model2.weights_equal(initial_params2, True)

        # And are different
        assert not model.weights_equal(model2.gln_params, True)


class TestGLNGeneral(unittest.TestCase):
    shared_params = dict(
        output_sizes=[6, 1],
        context_dim=3,
    )

    def test_get_context(self):
        # model = base_GatedLinearNetwork(**self.shared_params)
        # Batch, side info size
        side_info = np.random.random(size=(10, 4))
        # Context dim, side info size
        hyperplanes = np.random.random(size=(3, 4))
        hyperplane_bias = np.random.random(size=3)  # context dim

        # Batched
        batched = base_GatedLinearNetwork._compute_context(
            side_info, hyperplanes, hyperplane_bias)

        stack = []
        for s in side_info:
            stack.append(
                base_GatedLinearNetwork._compute_context(
                    s, hyperplanes, hyperplane_bias)
            )
        stacked = np.stack(stack)
        assert np.all(batched == stacked)

    def test_unpack(self):
        input_size = 4
        inputs_2d = np.ones((input_size, 2))
        inputs_2d[:, 0] = 5
        mu, sig = _unpack_inputs(inputs_2d)
        assert np.all(mu == np.ones(input_size) * 5)
        assert np.all(sig == np.ones(input_size))

        mu, sig = _unpack_inputs(inputs_2d[0])
        assert mu == 5 and sig == 1

        batch = np.stack([inputs_2d, 2 * inputs_2d, 3 * inputs_2d])
        mus, sigs = _unpack_inputs(batch)
        mu_1, sig_1 = _unpack_inputs(inputs_2d)
        mu_2, sig_2 = _unpack_inputs(inputs_2d * 2)
        assert np.all(mus[0] == mu_1) and np.all(sigs[0] == sig_1)
        assert np.all(mus[1] == mu_2) and np.all(sigs[1] == sig_2)

    def test_add_bias(self):
        # model = GatedLinearNetwork(**self.shared_params)
        # Batch, side info size
        def add_bias_factory(inps):
            return GatedLinearNetwork(
                **self.shared_params)._add_bias(inps)
        print("\nHK TRANSFORM")
        init_f, add_bias = hk.without_apply_rng(
            hk.transform_with_state(add_bias_factory))

        batch_size = 10
        input_size = 4
        mu_sig = 2
        print("\nINITTING")
        params, state = init_f(
            next(hk.PRNGSequence(jax.random.PRNGKey(0))),
            jnp.ones([batch_size, input_size, mu_sig]),  # dummy in
            # jnp.ones([batch_size, input_size])  # dummy side
        )

        batch_inputs = np.random.random((batch_size, input_size, mu_sig))

        # Batched (like in hessian updates)
        print("\nCALLING BATCHED")
        batched, _ = add_bias(params, state, batch_inputs)

        stack = []
        for inp in batch_inputs:
            bias, _ = add_bias(params, state, inp)
            stack.append(bias)
        stacked = np.stack(stack)

        print(stacked.shape)
        print(batched.shape)

        assert np.all(batched == stacked), (
            f"{batched.shape}\n{stacked.shape}")

    def test_project_weights(self):
        # model = GatedLinearNetwork(**self.shared_params)
        # Batch, side info size
        def project_weights_factory(inps, ws, min_sigs):
            return GatedLinearNetwork(
                **self.shared_params)._project_weights(inps, ws, min_sigs)

        print("\nHK TRANSFORM")
        init_f, project_weights = hk.without_apply_rng(
            hk.transform_with_state(project_weights_factory))

        batch_size = 10
        input_size = 4
        mu_sig = 2
        context_dim = 3
        bias_len = 3
        print("\nINITTING")
        params, state = init_f(
            next(hk.PRNGSequence(jax.random.PRNGKey(0))),
            jnp.ones([batch_size, input_size + bias_len, mu_sig]),  # dummy in
            jnp.ones([batch_size, input_size + bias_len]),  # dummy w; projected
            1.  # dummy min sig sq
        )

        batch_inputs = np.random.random(
            (batch_size, input_size + bias_len, mu_sig))
        batch_weights = np.random.random(
            (batch_size, input_size + bias_len))

        # Batched (like in hessian updates)
        print("\nCALLING BATCHED")
        batched, _ = project_weights(
            params, state, batch_inputs, batch_weights, 0.5)

        stack = []
        for inp, w in zip(batch_inputs, batch_weights):
            bias, _ = project_weights(params, state, inp, w, 0.5)
            stack.append(bias)
        stacked = np.stack(stack)

        assert batched.shape == stacked.shape
        for i, (b, s) in enumerate(zip(batched, stacked)):
            print("batch item", i, np.allclose(b, s, atol=1e-4))
            print(s)
            print(b)

        assert np.allclose(batched, stacked, atol=1e-4)


class TestQEstimatorGaussianGLN(unittest.TestCase):

    test_state = jnp.asarray(
        [[0.4, 0.5, 0.3], [0.2, 0.3, 0.3]] * 6)
    test_acts = jnp.asarray([0, 1])
    num_acts = 2

    def initialise_IREs(self, num_a):
        ires = []
        for i in range(num_a):
            ires.append(
                ImmediateRewardEstimatorGaussianGLN(
                    action=i,
                    burnin_n=10,
                    input_size=3,
                    burnin_val=0.5,
                    layer_sizes=[9, 1],
                    lr=1e-5,
                ))
        return ires

    def test_estimate(self):
        IREs = self.initialise_IREs(self.num_acts)
        Q = QuantileQEstimatorGaussianGLN(
            quantile=0.5, immediate_r_estimators=IREs,
            dim_states=3, num_actions=self.num_acts, gamma=0.99,
            layer_sizes=[4, 1], lr=0.01, burnin_n=10, batch_size=2,
            horizon_type="inf")

        for a in self.test_acts:
            # Repeats same states for 2 actions - dummy
            Q_est = Q.estimate(states=self.test_state, action=a)
        print(f"Q estimate: {Q_est}")

    def test_update(self):
        IREs = self.initialise_IREs(self.num_acts)
        Q = QuantileQEstimatorGaussianGLN(
            quantile=0.5, immediate_r_estimators=IREs, dim_states=3,
            num_actions=self.num_acts, gamma=0.99, layer_sizes=[17, 1],
            lr=1e-4, burnin_n=10, burnin_val=0.5, horizon_type="inf",
            batch_size=2)
        # Not needed
        assert Q.model(action=0, horizon=0, safe=False) is None\
               and Q.model(action=1, horizon=0, safe=False) is None
        # targets initialised to the same
        assert Q.model(action=0, horizon=1).weights_equal(
            Q.model(action=0, horizon=1, target=True).gln_params)
        Q_est2 = None
        for a in self.test_acts:
            Q_est2 = Q.estimate(self.test_state, a)
            print(f"Q estimate2 ({a}): {Q_est2}")

        n_state = jnp.asarray([0.4, 0.5, 0.3])
        n_state2 = jnp.asarray([0.2, 0.3, 0.3])

        tuple_data = (
            self.test_state,
            self.test_acts,
            jnp.full(self.test_state.shape[0], 0.5),
            jnp.asarray([n_state, n_state2] * (self.test_state.shape[0] // 2)),
            jnp.asarray(
                [False] * (self.test_state.shape[0] // 2)
                + [True] * (self.test_state.shape[0] // 2)),
        )
        for _ in range(10):
            Q.update(
                tuple_data,
                int(self.test_acts[0]),
                convergence_data=tuple_data,
                debug=True)
            Q.update(
                tuple_data,
                int(self.test_acts[1]),
                convergence_data=tuple_data,
                debug=True)

        # After update, target model is fixed but model updates
        updated_params = to_immutable_dict(
            Q.model(action=1, horizon=1).gln_params)
        assert not Q.model(
            action=1, horizon=1, target=True).weights_equal(
                updated_params)
        Q_est3 = None
        for a in self.test_acts:
            Q_est3 = Q.estimate(self.test_state, a)
            print(f"Q estimate3 ({a}): {Q_est3}")

        assert not jnp.any(Q_est3 == Q_est2)

        # Now check we can update the target model
        Q.update_target_net()
        assert Q.model(action=1, horizon=1, target=True).weights_equal(
            updated_params)
