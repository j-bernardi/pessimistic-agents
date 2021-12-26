import abc
import jax
import jax.numpy as jnp
import numpy as np
import torch as tc

import glns
import bayes_by_backprop
from utils import (
    vec_stack_batch, stack_batch, JaxRandom, geometric_sum, jnp_batch_apply,
    device_put_id)

DEFAULT_BURN_IN_N = 10000  # 0
DEFAULT_GLN_LAYERS = [64, 32, 1]
DEFAULT_GLN_LAYERS_IRE = [32, 16, 1]
GLN_CONTEXT_DIM = 4
JAX_RANDOM_CLS = JaxRandom()


def get_burnin_states(feat_mean, batch_size, dim_states, library="jax"):
    if library == "jax":
        unifrm = JAX_RANDOM_CLS.uniform
    elif library == "torch":
        unifrm = tc.rand
    else:
        raise NotImplementedError(library)
    size = (batch_size, dim_states)
    if feat_mean == 0.5:
        states = 1.2 * unifrm(size) - 0.1
    elif feat_mean == 0.:
        states = 2.2 * unifrm(size) - 1.1
    else:
        raise ValueError(f"Unexpected feat mean {feat_mean}")
    return states


class Estimator(abc.ABC):
    """Abstract definition of an estimator"""
    def __init__(
            self, lr, min_lr=0.02, lr_step=(None, None), scaled=True,
            burnin_n=DEFAULT_BURN_IN_N):
        """

        Args:
            lr (Optional[float]): The initial learning rate. If None,
                override the self.get_lr() method to return a custom
                learning rate (e.g. num_visits[state, action]).
            lr_step (tuple): (N update steps, decay factor)
            min_lr (float): If lr gets below min_lr, stop decaying
            scaled (bool): if True, all estimates should be in [0, 1],
                else reflect the true value of the quantity being
                estimated.
        """
        self.lr = lr
        self.lr_step_size = lr_step[0] if lr_step is not None else None
        self.lr_decay = lr_step[1] if lr_step is not None else None
        self.min_lr = min_lr
        self.scaled = scaled
        self.burnin_n = burnin_n
        self.total_updates = 0
        self.device_id = None

    def to_device(self, x):
        return device_put_id(x, self.device_id)

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

    def step_decay(self):
        """Every N steps, decay the learning rate by factor"""
        if self.lr_step_size is None:
            return
        if self.total_updates % self.lr_step_size == 0 and self.lr > self.min_lr:
            self.lr *= self.lr_decay


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
        """Set up the QEstimator as a model for the mentor in the finite case.

        Rather than using a Q-table with shape num_states * num_actions
        we use a list (flattened table) with shape num_states, and,
        whenever the mentor takes an action, update this "Q-list"
        for the given state regardless of what action the mentor has taken.
        This is allowed because we never will choose an action from this,
        only compare Q-values to decide when to query the mentor.
        The goal is to approximate how good the mentor would have been given some state.

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

        # the q_list is updated in update_estimator.
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

    # Whenever the agent sees the mentor act, this is called.
    def update_estimator(self, state, target_q_val, update_table=None):
        if update_table is None:
            update_table = self.q_list

        update_table[state] += self.get_lr(state) * (
                target_q_val - update_table[state])
        self.step_decay()

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
            self, num_states, num_actions, num_horizons, gamma, lr=0.1, scaled=True,
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
        self.num_horizons = num_horizons
        self.gamma = gamma
        # Add an extra one for the Q_zero est
        self.q_list = np.ones(
            (self.num_states, self.num_horizons + 1)) * self._init_val
        self.q_list[:, 0] = 0.  # Q_0 starts at 0
        self.n = np.zeros(self.num_states)

        self.total_updates = np.zeros(self.num_horizons + 1, dtype=int)

    def reset(self):
        # Add an extra one for the Q_zero est
        self.q_list = np.ones(
            (self.num_states, self.num_horizons + 1)) * self._init_val
        self.q_list[:, 0] = 0.  # Q_0 starts at 0
        self.n = np.zeros(self.num_states)

        self.total_updates = np.zeros(self.num_horizons + 1, dtype=int)

    @classmethod
    def get_steps_constructor(cls, num_horizons):
        """Return a constructor that matches the MentorQEstimator's"""

        def init_with_n_steps(
                num_states, num_actions, gamma=0.99, lr=0.1, scaled=False,
                init_val=1.
        ):
            return cls(
                num_states=num_states, num_actions=num_actions,
                num_horizons=num_horizons, gamma=gamma, lr=lr, scaled=scaled,
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

            for h in range(1, self.num_horizons + 1):
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


class ImmediateRewardEstimatorGaussianGLN(Estimator):
    """Estimates the next reward given the current state.

    Each action has a separate single IRE.
    """

    def __init__(
            self, action, input_size=2, layer_sizes=None,
            context_dim=GLN_CONTEXT_DIM, feat_mean=0.5, lr=1e-4, lr_step=None,
            scaled=True, burnin_n=DEFAULT_BURN_IN_N, burnin_val=0.,
            batch_size=1, min_sigma=1e-3):
        """Create an action-specific IRE.

        Args:
            action (int): the action for this IRE, this is just used to 
                keep track of things, and isn't used for calcuations.
            input_size: (int): the length of the input
            layer_sizes (List[int]): The number of neurons in each layer
                for the GGLN
            context_dim (int): the number of hyperplanes used to make the 
                halfspaces
            feat_mean (float): mean of all possible inputs to GLN (not
                side info). Typically 0.5, or 0.
            lr (float): the learning rate
            lr_step (tuple): (N steps, decay factor)
            scaled (bool): NOT CURRENTLY IMPLEMENTED
            burnin_val (float): the value we burn in the estimator with
        """
        if scaled is not True:
            raise NotImplementedError("Didn't implement scaled yet")
        super().__init__(lr=lr, lr_step=lr_step, burnin_n=burnin_n)
        self.action = action
        if layer_sizes is None:
            layer_sizes = DEFAULT_GLN_LAYERS_IRE

        self.model = glns.GGLN(
            name=f"IRE_{self.action}",
            layer_sizes=layer_sizes,
            input_size=input_size,
            context_dim=context_dim,
            feat_mean=feat_mean,
            batch_size=batch_size,
            lr=lr,
            # init_bias_weights=[None, None, None],
            bias_max_mu=1.,
            min_sigma_sq=min_sigma
        )

        self.state_dict = {}
        self.update_count = 0
        self.input_size = input_size
        # burn in the estimator for burnin_n steps, with the value burnin_val
        if self.burnin_n > 0:
            print(f"Burning in IRE {action} for {self.burnin_n} "
                  f"to {burnin_val}")
        for i in range(0, self.burnin_n, batch_size):
            # using random inputs from  the space [-2, 2]^dim_states
            # the space is larger than the actual state space that the
            # agent will encounter, to burn in correctly around the edges
            states = get_burnin_states(feat_mean, batch_size, self.input_size)
            rewards = self.to_device(jnp.full(batch_size, burnin_val))
            self.update((states, rewards))

    def reset(self):
        raise NotImplementedError("Not yet implemented")

    def estimate(self, states, estimate_model=None):
        """Estimate the next reward given the current state (for this action).

        Uses estimate_model to make this estimate, if this isn't provided,
        then just use the self.model

        Args:
            states (jnp.ndarray): (b, state.size) array of states to
                estimate on.
            estimate_model (GLN): The model to make the predictions with
        """
        if estimate_model is None:
            estimate_model = self.model

        if states.shape[0] not in (1, self.model.batch_size):
            return jnp_batch_apply(
                estimate_model.predict, states, self.model.batch_size)
        else:
            return estimate_model.predict(states)

    def update(self, history_batch, update_model=None):
        """Algorithm 1. Use experience to update estimate of immediate r

        Args:
            history_batch (tuple[jnp.ndarray]): list of the
                (states, rewards) tuple to update on.
            update_model: the model to perform the update on.
        """
        if not history_batch:
            return None  # too soon

        # TEMP
        states, rewards = history_batch

        if update_model is None:
            update_model = self.model

        success = update_model.predict(states, target=rewards)
        self.update_count += states.shape[0]
        self.total_updates += 1
        self.step_decay()
        return success

    def estimate_with_sigma(self, state, estimate_model=None):
        """Estimate the next reward given the current state (for this action).

        Uses estimate_model to make this estimate, if this isn't provided,
        then just use the self.model

        """
        if estimate_model is None:
            estimate_model = self.model
        model_est, model_sigma = estimate_model.predict_with_sigma(state)

        return model_est, model_sigma


class MentorQEstimatorGaussianGLN(Estimator):

    def __init__(
            self, dim_states, num_actions, gamma, scaled=True,
            init_val=1., layer_sizes=None, context_dim=GLN_CONTEXT_DIM,
            feat_mean=0.5, lr=1e-4, lr_step=None, burnin_n=DEFAULT_BURN_IN_N,
            batch_size=1, min_sigma=1e-3
    ):
        """Set up the QEstimator for the mentor

        Rather than using num_actions Q estimators for each of the actions,
        we one estimator, and update this single estimator
        for the given state regardless of what action was taken.
        This is allowed because we never will choose an action from this,
        only compare Q-values to decide when to query the mentor.

        Args:
            dim_states (int): the number of state dimensions in the
                environment
            num_actions (int): the number of actions
            gamma (float): the discount rate for Q-learning
            scaled: NOT IMPLEMENTED
            init_val (float): the value to burn in the GGLN with
            layer_sizes (List[int]): The number of neurons in each layer
                for the GGLN
            context_dim (int): the number of hyperplanes used to make the 
                halfspaces
            feat_mean (float): mean of all possible inputs to GLN (not
                side info). Typically 0.5, or 0.
            lr (float): the learning rate
        """
        super().__init__(lr, lr_step=lr_step, scaled=scaled, burnin_n=burnin_n)
        self.num_actions = num_actions
        self.dim_states = dim_states
        self.gamma = gamma
        self.inf_scaling_factor = self.to_device(jnp.asarray(1. - self.gamma))

        layer_sizes = DEFAULT_GLN_LAYERS if layer_sizes is None else layer_sizes
        self.model = glns.GGLN(
            name="MentorQ",
            layer_sizes=layer_sizes,
            input_size=dim_states,
            context_dim=context_dim,
            feat_mean=feat_mean,
            batch_size=batch_size,
            lr=lr,
            min_sigma_sq=min_sigma,
            # init_bias_weights=[None, None, None]
        )

        if self.burnin_n > 0:
            print(f"Burning in Mentor Q Estimator to {init_val} for {self.burnin_n}")
        for _ in range(0, self.burnin_n, batch_size):
            states = get_burnin_states(feat_mean, batch_size, self.dim_states)
            self.update_estimator(
                states,
                self.to_device(jnp.full(batch_size, init_val)))

        self.total_updates = 0

    def reset(self):
        raise NotImplementedError("Not yet implemented")

    def scale_rewards(self, rs):
        return self.inf_scaling_factor * rs

    def update(self, history_batch, debug=False):
        """Update the mentor model's Q-estimator with a given history.

        In practice this history will be for actions when the
        mentor decided the action.

        Args:
            history_batch (Transition): list of
                (state, action, reward, next_state) tuples that will
                form the batch
            debug (bool): print more if true
        """

        raw_qs = self.estimate(history_batch.next_state)
        next_q_vals = jnp.where(history_batch.done, 0., raw_qs)
        if self.scaled:
            scaled_r = self.scale_rewards(history_batch.reward)
        else:
            scaled_r = history_batch.reward
        q_targets = scaled_r + self.gamma * next_q_vals
        if debug:
            print(f"Updating mentor agent on Q Targets:\n{q_targets}")
        self.update_estimator(history_batch.state, q_targets)
        self.total_updates += 1
        self.step_decay()

    def update_estimator(self, states, q_targets, update_model=None, lr=None):
        """
        states: batch of
        q_targets: batch of
        update_model:
        lr:
        """
        if update_model is None:
            update_model = self.model

        if lr is None:
            lr = self.get_lr()

        update_model.update_learning_rate(lr)
        update_model.predict(states, target=q_targets)

    def estimate(self, states, model=None):
        """Estimate the future Q, using this estimator

        Estimate the Q value of what the mentor would choose, regardless
        of what action is taken.

        Args:
            states (jnp.ndarray): shape (b, state.size), the current
                states from which the Q value is being estimated
            model (GGLN): the estimator used to estimate the value
                from, if None use self.model as default
        """
        if model is None:
            model = self.model
        if states.shape[0] not in (1, model.batch_size):
            return jnp_batch_apply(model.predict, states, bs=model.batch_size)
        else:
            return model.predict(states)

    def get_lr(self, n=None):
        """
        Returns the learning rate. 

        Optionally takes the pseudocount n and calculates
        the learning rate. If no n is provided, uses the predefined
        learning rate. 

        """
        assert not (n is None and self.lr is None), (
            "Both n and self.lr cannot be None")

        return 1 / (n + 1) if self.lr is None else self.lr


class MentorFHTDQEstimatorGaussianGLN(Estimator):

    def __init__(
            self, dim_states, num_actions, num_steps, gamma, scaled=True,
            init_val=1., layer_sizes=None, context_dim=GLN_CONTEXT_DIM,
            feat_mean=0.5, bias=True, context_bias=True, lr=1e-4, env=None,
            burnin_n=DEFAULT_BURN_IN_N, batch_size=1):
        """Set up the QEstimator for the mentor

        Rather than using num_actions Q estimators for each of the actions,
        we one estimator, and update this single estimator
        for the given state regardless of what action was taken.
        This is allowed because we never will choose an action from this,
        only compare Q-values to decide when to query the mentor.

        Args:
            num_states (int): the number of states in the environment
            num_actions (int): the number of actions
            gamma (float): the discount rate for Q-learning
            scaled: NOT IMPLEMENTED
            init_val (float): the value to burn in the GGLN with
            layer_sizes (List[int]): The number of neurons in each layer
                for the GGLN
            context_dim (int): the number of hyperplanes used to make the
                halfspaces
            feat_mean (float): mean of all possible inputs to GLN (not
                side info). Typically 0.5, or 0.
            lr (float): the learning rate
            burnin_n (int): the number of steps we burn in for
        """
        super().__init__(lr, scaled=scaled, burnin_n=burnin_n)
        self.num_actions = num_actions
        self.dim_states = dim_states
        self.num_steps = num_steps
        self.gamma = gamma
        self.context_dim = context_dim

        layer_sizes = DEFAULT_GLN_LAYERS if layer_sizes is None else layer_sizes

        self.make_q_estimator(
            layer_sizes, init_val, batch_size, feat_mean)
        # self.model = glns.GGLN(
        #     layer_sizes=layer_sizes,
        #     input_size=dim_states,
        #     context_dim=context_dim,
        # device_id
        #     lr=lr,init_bias_weights=[None, None, 1]
        # )

        # if burnin_n > 0:
        #     print("Burning in Mentor Q Estimator")
        # for i in range(burnin_n):

        #     state = 4 * jnp.random.rand(self.dim_states) - 2
        #     self.update_estimator(state, init_val)

        self.total_updates = 0

    def make_q_estimator(
            self, layer_sizes, burnin_val, batch_size, mean):
        self.model = [
            glns.GGLN(
                name="MentorFHTDQ",
                layer_sizes=layer_sizes,
                input_size=self.dim_states,
                context_dim=self.context_dim,
                feat_mean=mean,
                # bias_len=3,
                lr=self.lr,
                batch_size=batch_size,
                # min_sigma_sq=0.5,
                # init_bias_weights=[None, None, 1]
                ) for s in range(self.num_steps + 1)
        ]

        if self.burnin_n > 0:
            print(f"Burning in Mentor Q Estimator to {burnin_val} for {self.burnin_n}")

        for i in range(0, self.burnin_n, batch_size):
            # using random inputs from  the space [-2, 2]^dim_states
            # the space is larger than the actual state space that the
            # agent will encounter, to hopefully mean that it burns in
            # correctly around the edges

            states = get_burnin_states(mean, batch_size, self.dim_states)

            for step in range(self.num_steps):
                self.update_estimator(
                    states, self.to_device(jnp.full(batch_size, burnin_val)),
                    horizon=step)

    def reset(self):
        raise NotImplementedError("Not yet implemented")

    def update(self, history_batch):
        """Update the mentor model's Q-estimator with a given history.

        In practice this history will be for actions when the
        mentor decided the action.

        Args:
            history_batch (list): list of
                (state, action, reward, next_state) tuples that will
                form the batch
        """
        states, actions, rewards, next_states, dones = vec_stack_batch(
            history_batch)
        for h in range(1, self.num_steps + 1):
            next_q_vals = jnp.where(
                dones,
                self.to_device(jnp.zeros(1)),
                self.estimate(next_states, h=h-1))
            scaled_r = (1 - self.gamma) * rewards if self.scaled else rewards
            # q_target = scaled_r + self.gamma * next_q_val
            q_targets = rewards / h + next_q_vals * (h - 1) / h
            self.update_estimator(states, q_targets, horizon=h)

        self.total_updates += states.shape[0]

    def update_estimator(
            self, states, q_targets, horizon=None, update_model=None, lr=None):
        if update_model is None:
            update_model = self.model

        if lr is None:
            lr = self.get_lr()

        if horizon is None:
            horizon = -1

        update_model[horizon].update_learning_rate(lr)
        update_model[horizon].predict(states, target=q_targets)

    def estimate(self, states, h=None, model=None):
        """Estimate the future Q, using this estimator

        Estimate the Q value of what the mentor would choose, regardless
        of what action is taken.

        Args:
            states (list[jnp.ndarray]): the current state from which the
                Q value is being estimated
            model: the (GGLN) estimator used to estimate the value from,
                if None use self.model as default
        """
        if h == 0:
            return 0. 

        if model is None:
            model = self.model

        if h is None:
            h = -1

        return model[h].predict(states)

    def get_lr(self, n=None):
        """
        Returns the learning rate.

        Optionally takes the pseudocount n and calculates
        the learning rate. If no n is provided, uses the predefined
        learning rate.

        """
        assert not (n is None and self.lr is None), (
            "Both n and self.lr cannot be None")

        return 1 / (n + 1) if self.lr is None else self.lr


class ImmediateRewardEstimatorBayes(Estimator):
    """Estimates the next reward given the current state.

    Each action has a separate single IRE.
    """

    def __init__(
            self, num_actions, input_size=2, feat_mean=0.5, lr=1e-4,
            burnin_n=DEFAULT_BURN_IN_N, burnin_val=0., batch_size=1,
            net_init_func=bayes_by_backprop.BBBNet, **net_kwargs
    ):
        """Create an action-specific IRE.

        Args:
            num_actions (int): the number of actions available
            input_size: (int): the length of the input
            feat_mean (float): mean of all possible inputs to GLN (not
                side info). Typically 0.5, or 0.
            lr (float): the learning rate
            burnin_n (int): the number of steps we burn in for
            burnin_val (float): the value we burn in the estimator with
        """
        super().__init__(lr=lr, burnin_n=burnin_n)
        self.num_actions = num_actions

        self.model = net_init_func(
            name=f"IRE_Bayes",
            input_size=input_size,
            num_horizons=1,
            num_actions=num_actions,
            feat_mean=feat_mean,
            lr=lr,
            **net_kwargs
        )

        self.state_dict = {}
        self.update_count = 0
        self.input_size = input_size
        # burn in the estimator for burnin_n steps, with the value burnin_val
        print(f"Burning in {self.model.name} for {self.burnin_n} "
              f"to {burnin_val}")
        rewards = tc.full((batch_size, 1), burnin_val)
        for i in range(0, self.burnin_n, batch_size):
            actions = tc.randint(
                low=0,
                high=self.num_actions,
                size=(batch_size, 1),
                dtype=tc.int64)
            # using random inputs from  the space [-2, 2]^dim_states
            # the space is larger than the actual state space that the
            # agent will encounter, to burn in correctly around the edges
            states = get_burnin_states(
                feat_mean, batch_size, self.input_size, library="torch")
            self.update((states, actions, rewards))
        self.model.make_optimizer()

    def reset(self):
        raise NotImplementedError("Not yet implemented")

    def estimate(self, states, estimate_model=None):
        """Estimate the next reward given the current state (for this action).

        Uses estimate_model to make this estimate, if this isn't provided,
        then just use the self.model

        Args:
            states (tc.tensor): (b, state.size) array of states to
                estimate on.
            estimate_model (BBBNet): The model to make the predictions
                with. Defaults to self.model
        """
        if estimate_model is None:
            estimate_model = self.model
        model_est = estimate_model.predict(states)

        return model_est

    def update(self, history_batch, update_model=None, debug=False):
        """Algorithm 1. Use experience to update estimate of immediate r

        Args:
            history_batch (tuple[tc.Tensor]): tuple of the
                state, actions, rewards tensors to update on.
            update_model: the model to perform the update on.
            debug (additional)
        """
        if not history_batch:
            return None  # too soon
        states, acts, rewards = history_batch

        if update_model is None:
            update_model = self.model
        if debug:
            print(f"UPDATING IRE {update_model.name}")
            print(f"States\n{states.squeeze()}")
            print(f"Acts\n{acts.squeeze()}")
            print(f"Rewards\n{rewards.squeeze()}")
        success = update_model.predict(
            states, actions=acts, target=rewards, debug=debug)
        self.update_count += states.shape[0]
        self.total_updates += 1
        if self.total_updates % 50 == 0:
            self.lr *= 0.9
        return success


class MentorQEstimatorBayes(Estimator):

    def __init__(
            self, dim_states, gamma, scaled=True,
            init_val=1., feat_mean=0.5, lr=1e-4, burnin_n=DEFAULT_BURN_IN_N,
            batch_size=1, net_type=bayes_by_backprop.BBBNet, num_horizons=1,
            horizon_type="inf", **net_kwargs
    ):
        """Set up the QEstimator for the mentor

        Rather than using num_actions Q estimators for each of the actions,
        we one estimator, and update this single estimator
        for the given state regardless of what action was taken.
        This is allowed because we never will choose an action from this,
        only compare Q-values to decide when to query the mentor.

        Args:
            gamma (float): the discount rate for Q-learning
            scaled: NOT IMPLEMENTED
            init_val (float): the value to burn in the GGLN with
            feat_mean (float): mean of all possible inputs to GLN (not
                side info). Typically 0.5, or 0.
            lr (float): the learning rate
            burnin_n (int): the number of steps we burn in for
            num_horizons (int):
            horizon_type (str): one of "inf" or "finite"
        """
        super().__init__(lr, scaled=scaled, burnin_n=burnin_n)
        self.dim_states = dim_states
        self.gamma = gamma
        self.inf_scaling_factor = 1. - self.gamma
        self.num_horizons = num_horizons
        assert horizon_type in ("inf", "finite")
        self.horizon_type = horizon_type
        if self.num_horizons > 1:
            assert self.horizon_type == "finite"

        self.model = net_type(
            name="MentorQBayes",
            input_size=dim_states,
            num_actions=1,  # We only care if the mentor acted at all
            num_horizons=self.num_horizons,
            feat_mean=feat_mean,
            batch_size=batch_size,
            lr=lr,
            sigmoid_vals=self.scaled,
            **net_kwargs
        )

        if self.burnin_n > 0:
            print(f"Burning in Mentor Q Estimator to {init_val} for {self.burnin_n}")
        for h in range(1, 1 + num_horizons):
            burnin_val = geometric_sum(init_val, self.gamma, h)
            print(f"Burning in Mentor Q Estimator step {h} to {burnin_val} for {self.burnin_n}")
            for _ in range(0, self.burnin_n, batch_size):
                states = get_burnin_states(
                    feat_mean, batch_size, self.dim_states, library="torch")
                self.update_estimator(
                    states, tc.full((batch_size, 1), burnin_val), horizon=h)
        self.model.make_optimizer()
        self.total_updates = 0

    def reset(self):
        raise NotImplementedError("Not yet implemented")

    def scale_rewards(self, rs):
        return self.inf_scaling_factor * rs

    def update(self, history_batch, debug=False):
        """Update the mentor model's Q-estimator with a given history.

        In practice this history will be for actions when the
        mentor decided the action.

        Args:
            history_batch (list[tuple]): list of
                (state, action, reward, next_state) tuples that will
                form the batch
            debug (bool): print more if true
        """
        states, _, rewards, nxt_states, dones = stack_batch(
            history_batch, lib=tc)
        for h in range(1, self.num_horizons + 1):
            with tc.no_grad():
                raw_qs = self.estimate(
                    nxt_states,
                    horizon=None if self.horizon_type == "inf" else h - 1)
            next_q_vals = tc.where(
                dones, tc.zeros_like(raw_qs, dtype=tc.float), raw_qs)
            if self.scaled:
                scaled_r = self.scale_rewards(rewards)
            else:
                scaled_r = rewards
            q_targets = scaled_r + self.gamma * next_q_vals
            if debug:
                print(f"HORIZON {h}")
                print(f"Raw Q nexts\n{raw_qs.squeeze()}")
                print(f"Scaled rewards\n{scaled_r.squeeze()}")
                print(f"q_targets ({self.gamma})\n{q_targets.squeeze()}")
            if debug:
                print(f"Updating mentor agent on Q Targets:"
                      f"\n{q_targets.squeeze()}")
            self.update_estimator(
                states,
                q_targets,
                horizon=None if self.horizon_type == "inf" else h)
        self.total_updates += states.shape[0]

    def update_estimator(
            self, states, q_targets, update_model=None, lr=None, horizon=None):
        """
        states: batch of
        q_targets: batch of
        update_model:
        lr:
        """
        if update_model is None:
            update_model = self.model

        if lr is None:
            lr = self.model.lr
        update_model.update_learning_rate(lr)

        update_model.predict(
            states,
            actions=tc.full((states.shape[0], 1), 0, dtype=tc.int64),
            target=q_targets,
            horizon=horizon
        )

    def estimate(self, states, model=None, horizon=None):
        """Estimate the future Q, using this estimator

        Estimate the Q value of what the mentor would choose, regardless
        of what action is taken.

        Args:
            states (tc.tensor): shape (b, state.size), the current
                states from which the Q value is being estimated
            model (BBB): the estimator used to estimate the value
                from, if None use self.model as default
            horizon (int): which horizon to look for
        """
        if model is None:
            model = self.model
        if horizon is None:
            horizon = self.num_horizons
        if horizon == 0:
            return tc.full((states.shape[0], 1), 0.)
        return model.predict(states, horizon=horizon)

    def get_lr(self, n=None):
        """
        Returns the learning rate.

        Optionally takes the pseudocount n and calculates
        the learning rate. If no n is provided, uses the predefined
        learning rate.

        """
        assert not (n is None and self.lr is None), (
            "Both n and self.lr cannot be None")

        return 1 / (n + 1) if self.lr is None else self.lr
