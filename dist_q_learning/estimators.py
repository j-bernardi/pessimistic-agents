import abc
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline
import scipy.stats
NP_RANDOM_GEN = np.random.Generator(np.random.PCG64())
import copy

# import pygln
import glns
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
    """Abstract definition of an estimator

    Args:
        scaled (bool): if True, all estimates should be between 0 and 1
    """
    def __init__(self, lr, min_lr=0.05, lr_decay=0., scaled=True):  # 1e-6):
        self.lr = lr
        self.min_lr = min_lr
        self.lr_decay = lr_decay
        self.scaled = scaled

        # TODO - possibly per state, per action
        self.total_updates = 0

    @abc.abstractmethod
    def estimate(self, **args):
        """Use the estimator to generate an estimate given state"""
        return

    @abc.abstractmethod
    def update(self, history):
        """Update the estimator given some experience"""
        return

    def lr_fun(self):
        """Returns the learning rate"""
        return

    def decay_lr(self):

        if self.lr is not None:
            if self.lr > self.min_lr and self.lr_decay is not None:
                self.lr *= (1. - self.lr_decay)
                # self.lr *= (1. / (1. + self.lr_decay * self.total_updates))



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
        super().__init__(lr=None)
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

            self.total_updates += 1


class MentorQEstimator(Estimator):

    def __init__(
            self, num_states, num_actions, gamma, lr=0.1, scaled=True,
            init_val=1.):
        """Set up the QEstimator for the mentor

        Rather than using a Q-table with shape num_states * num_actions
        we use a list with shape num_states, and update this "Q-list"
        for the given state regardless of what action was taken.
        This is allowed because we never will choose an action from this,
        only compare Q-values to decide when to query the mentor.

        Args:
            num_states (int): the number of states in the environment
            num_actions (int): the number of actions
            gamma (float): the discount rate for Q-learning
        """
        super().__init__(lr, scaled=scaled)
        self.num_actions = num_actions
        self.num_states = num_states
        self.gamma = gamma
        self.q_list = np.ones(num_states) * init_val
        self.n = np.zeros(num_states)

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

    def update_estimator(self, state, target_q_val, update_table=None):
        if update_table is None:
            update_table = self.q_list

        update_table[state] += self.lr_fun(state) * (target_q_val - update_table[state])
        self.decay_lr()

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

    def lr_fun(self, state=None):
        """ Calculate the learning rate for updating the Q table.
        
        If self.lr is specified use that, otherwise use 1/(n+1),
        where n is the number of times the state has been visted.

        Args:
            state (int): The state the Q table is being updated for

        """


        if self.lr is not None:
            return self.lr

        else:
            return 1 / (1 + self.n[state])

class MentorFHTDQEstimator(Estimator):

    def __init__(
            self, num_states, num_actions, num_steps, gamma, lr=0.1, scaled=True,
            init_val=1.
    ):
        """Set up the Finite horizon QEstimator for the mentor

        Args:
            num_states (int): the number of states in the environment
            num_actions (int): the number of actions
            gamma (float): the discount rate for Q-learning
        """
        super().__init__(lr, scaled=scaled)
        self.num_actions = num_actions
        self.num_states = num_states
        self.num_steps = num_steps
        self.gamma = gamma
        # Add an extra one for the Q_zero est
        self.q_list = np.ones((num_states, num_steps + 1)) * init_val
        self.q_list[:, 0] = 0.  # Q_0 starts at 0
        self.n = np.zeros(num_states)

        self.total_updates = np.zeros(num_steps + 1, dtype=int)
    @classmethod
    def get_steps_constructor(cls, num_steps):
        """Return a constructor that matches the MentorQEstimator's"""

        def init_with_n_steps(
                num_states, num_actions, gamma=0.99, lr=0.1, scaled=False,
                init_val=1.
        ):
            return cls(
                num_states=num_states, num_actions=num_actions,
                num_steps=num_steps, gamma=gamma, lr=lr, scaled=scaled,
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

            for h in range(1, self.num_steps + 1):
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


        update_table[state, horizon] += self.lr_fun(state) * (
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
            q_list (np.ndarray): the Q table to estimate the value from,
                if None use self.Q_list as default
        """
        if q_list is None:
            q_list = self.q_list
        if horizon is None:
            horizon = -1
        return q_list[state, horizon]

    def lr_fun(self, state=None):
        """ Calculate the learning rate for updating the Q table.
        
        If self.lr is specified use that, otherwise use 1/(n+1),
        where n is the number of times the state has been visted.

        Args:
            state (int): The state the Q table is being updated for

        """

        if self.lr is not None:
            return self.lr

        else:
            return 1 / (1 + self.n[state])

class BaseQEstimator(Estimator, abc.ABC):

    def __init__(
            self, num_states, num_actions, gamma, lr, q_table_init_val=0.,
            scaled=True, num_steps="inf",
    ):
        super().__init__(lr, scaled=scaled)
        self.num_actions = num_actions
        self.num_states = num_states
        self.gamma = gamma

        self.num_steps = num_steps

        self.transition_table = np.zeros(
            (num_states, num_actions, num_states), dtype=int)

        self.q_table = np.zeros((num_states, num_actions)) + q_table_init_val

        self.random_act_prob = None

    def get_random_act_prob(self, decay_factor=5e-5, min_random=0.05):
        if self.random_act_prob is not None\
                and self.random_act_prob > min_random:
            self.random_act_prob *= (1. - decay_factor)
        return self.random_act_prob

    def estimate(self, state, action, q_table=None):
        """Estimate the future Q, using this estimator

        Args:
            state (int): the current state from which the Q value is
                being estimated
            action (int): the action taken
            q_table (np.ndarray): the Q table to estimate the value
                from, if None use self.q_table as default
        """
        if q_table is None:
            q_table = self.q_table

        return q_table[state, action]

    def update_estimator(self, state, action, q_target, update_table=None):
        if update_table is None:
            update_table = self.q_table


        update_table[state, action] += self.lr_fun(state, action) * (
                q_target - update_table[state, action])
        self.decay_lr()

    def lr_fun(self, state=None, action=None):
        """ Calculate the learning rate for updating the Q table.
        
        If self.lr is specified use that, otherwise use 1/(n+1),
        where n is the number of times the state has been visted.

        Args:
            state (int): The state the Q table is being updated for
            action (int): The action the Q table is being updated for

        """

        if self.lr is not None:
            return self.lr

        else:
            return 1 / (1 + self.transition_table[state, action, :].sum())

class QuantileQEstimator(BaseQEstimator):
    """A Q table estimator that makes pessimistic updates e.g. using IRE

    Updates using algorithm 3 in the QuEUE specification.
    """
    horizon_type = "infinite"

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

        TODO:
            Optional scaling, finite horizon Q estimators
        """
        # "Burn in" quantile with init value argument
        super().__init__(
            num_states, num_actions, gamma, lr, q_table_init_val=quantile)
        if quantile <= 0. or quantile > 1.:
            raise ValueError(f"Require 0. < q <= 1. {quantile}")

        self.quantile = quantile  # the 'i' index of the quantile
        self.use_pseudocount = use_pseudocount
        self.immediate_r_estimators = immediate_r_estimators

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
                future_q = np.max([
                    self.estimate(next_state, action_i)
                    for action_i in range(self.num_actions)]
                )
            else:
                future_q = 0.

            ire = self.immediate_r_estimators[action]
            ire_alpha, ire_beta = ire.expected_with_uncertainty(state)
            IV_i = scipy.stats.beta.ppf(self.quantile, ire_alpha, ire_beta)

            q_target = self.gamma * future_q + (1. - self.gamma) * IV_i

            self.update_estimator(state, action, q_target)

            # Account for uncertainty in the state transition function
            q_ai = self.estimate(state, action)
            if not self.use_pseudocount:
                n = self.transition_table[state, action, :].sum()

            else:
                fake_q_ai = []
                for fake_target in [0., 1.]:
                    fake_q_table = self.q_table.copy()
                    self.update_estimator(
                        state, action, fake_target, update_table=fake_q_table)
                    fake_q_ai.append(self.estimate(state, action, fake_q_table))

                n_ai0 = fake_q_ai[0] / (q_ai - fake_q_ai[0])
                n_ai1 = (1. - fake_q_ai[1]) / (fake_q_ai[1] - q_ai)
                # in the original algorithm this min was also over the quantiles
                # as well as the fake targets
                n = np.min([n_ai0, n_ai1])

            q_alpha = q_ai * n + 1.
            q_beta = (1. - q_ai) * n + 1.
            q_target_transition = scipy.stats.beta.ppf(
                self.quantile, q_alpha, q_beta)

            self.update_estimator(state, action, q_target_transition)
            self.total_updates += 1


class QEstimator(BaseQEstimator):
    """A basic Q table estimator

    Updates (as 'usual') towards the target of:
        gamma * Q(s', a*) + (1. - gamma) * reward
    """

    def __init__(
            self, num_states, num_actions, gamma, lr=0.1, has_mentor=False,
            scaled=True
    ):
        super().__init__(num_states, num_actions, gamma, lr, scaled=scaled)
        self.random_act_prob = None if has_mentor else 1.

    def update(self, history):
        for state, action, reward, next_state, done in history:
            self.transition_table[state, action, next_state] += 1

            if not done:
                future_q = np.max([
                    self.estimate(next_state, action_i)
                    for action_i in range(self.num_actions)]
                )
            else:
                future_q = 0.

            scaled_r = (1. - self.gamma) * reward if self.scaled else reward
            q_target = self.gamma * future_q + scaled_r

            self.update_estimator(state, action, q_target)


class QMeanIREEstimator(BaseQEstimator):
    """A basic Q table estimator which uses mean IRE instead of reward

    Updates (as 'usual') towards the target of:
        gamma * Q(s', a*) + (1. - gamma) * IRE(state)
    """

    def __init__(
            self, num_states, num_actions, gamma, immediate_r_estimators,
            lr=0.1, has_mentor=False, scaled=True
    ):

        super().__init__(num_states, num_actions, gamma, lr, scaled=scaled)
        self.random_act_prob = None if has_mentor else 1.
        self.immediate_r_estimators = immediate_r_estimators

    def update(self, history):
        for state, action, reward, next_state, done in history:
            self.transition_table[state, action, next_state] += 1

            if not done:
                future_q = np.max([
                    self.estimate(next_state, action_i)
                    for action_i in range(self.num_actions)]
                )
            else:
                future_q = 0.

            ire = self.immediate_r_estimators[action].estimate(state)
            scaled_ire = (1. - self.gamma) * ire if self.scaled else ire

            q_target = self.gamma * future_q + (1. - self.gamma) * scaled_ire
            self.update_estimator(state, action, q_target)


class FHTDQEstimator(BaseQEstimator):
    """A basic Q table estimator for the finite horizon

    Finite Horizon Temporal Difference Q Estimator
    As in: https://arxiv.org/pdf/1909.03906.pdf

    Recursively updates the next num_steps estimates of the reward:
        Q_0(s, a) = mean_{t : s_t = s} r_t
        V_i(s) = max_a Q_i(s, a)
        Q_{i+1}(s, a) = mean_{t : s_t = s}[
            (1 - gamma) * r_t  + gamma * V_i(s_{t+1})
        ]
    """

    def __init__(
            self, num_states, num_actions, num_steps, gamma=0.99, lr=0.1,
            has_mentor=False, q_table_init_val=0., scaled=False
    ):
        if scaled:
            raise NotImplementedError("Not defined for scaled Q values")
        super().__init__(
            num_states, num_actions, gamma, lr, scaled=scaled,
            num_steps=num_steps
        )

        # account for Q_0, add one in size to the "horizons" dimension
        self.q_table = np.zeros((num_states, num_actions, num_steps + 1))
        self.q_table += q_table_init_val
        self.q_table[:, :, 0] = 0.  # Q_0 always init to 0 TODO why ?

        self.random_act_prob = None if has_mentor else 1.

    @classmethod
    def get_steps_constructor(cls, num_steps):
        """Return a constructor that matches the QEstimator's"""
        def init_with_n_steps(
                num_states, num_actions, gamma=0.99, lr=0.1, has_mentor=False,
                q_table_init_val=0., scaled=False
        ):
            return cls(
                num_states=num_states, num_actions=num_actions,
                num_steps=num_steps, gamma=gamma, lr=lr, has_mentor=has_mentor,
                q_table_init_val=q_table_init_val, scaled=scaled
            )
        return init_with_n_steps

    def estimate(self, state, action, h=None, q_table=None):
        """Estimate the future Q, using this estimator

        Args:
            state (int): the current state from which the Q value is
                being estimated
            action (int): the action taken
            h (int): the horizon we are estimating for, if None then
                use the final horizon, which is used for choosing
                which action to take.
            q_table (np.ndarray): the Q table to estimate the value
                from, if None use self.q_table as default

        Returns:
            q_estimate (float):
        """
        if q_table is None:
            q_table = self.q_table

        if h is None:
            h = -1

        return q_table[state, action, h]

    def update(self, history):
        for state, action, reward, next_state, done in history:

            self.transition_table[state, action, next_state] += 1


            for h in range(1, self.num_steps + 1):
                if not done:
                    next_q = np.max([
                        self.estimate(next_state, action_i, h-1)
                        for action_i in range(self.num_actions)]
                    )
                else:
                    next_q = 0.

                # TODO what was this for? Is it needed now?
                # if h == 1:
                #     next_q = reward

                assert not self.scaled, "Q value must not be scaled"
                q_target = reward + self.gamma * next_q

                self.update_estimator(state, action, q_target, horizon=h)

    def update_estimator(
            self, state, action, q_target, horizon=None, update_table=None):
        """New update for this special case Q table, requires h

        New args:
            horizon (int): the index of the Q-horizon to update. Default
                to None to match argument signature of super (a bit of
                a hack)
        """

        if update_table is None:
            update_table = self.q_table

        update_table[state, action, horizon] += self.lr_fun(state, action) * (
                q_target - update_table[state, action, horizon])

        self.decay_lr()


class ImmediateRewardEstimator_GLN_bernoulli(Estimator):

    def __init__(self, action, input_size=2, layer_sizes=[4,4],
                 context_map_size=4, bias=True,
                 context_bias=True, lr=1e-4, scaled=True, env=None, burnin_n=0):

        super().__init__(lr=lr)
        self.action = action

        self.model = pygln.GLN(backend='numpy', layer_sizes=layer_sizes,
                          input_size=input_size, 
                          context_map_size=context_map_size,
                          num_classes=2,
                          learning_rate=lr,
                          bias=bias, context_bias=context_bias
                          )

        self.state_dict = {}
        self.update_count = 0

        if burnin_n > 0:
            print(f'Burning in IRE {action}')
        for i in range(burnin_n):
            state_rew_history = [([2*np.random.rand() - 1, 2*np.random.rand() - 1], 0.)]
            self.update(state_rew_history)

    def estimate(self, state, estimate_model=None):

        if estimate_model is None:
            estimate_model = self.model
        model_est = estimate_model.predict(state, return_probs=True)

        return model_est 

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

        for i, fake_r in enumerate(fake_rewards):
            fake_model = copy.copy(self.model)

            self.update([(state, fake_r)], fake_model)
            fake_means[i] = self.estimate(state, fake_model)


        n_0 = (  # TEMP handling of / 0 error
            fake_means[0] / (current_mean - fake_means[0])
            if current_mean != fake_means[0] else None
        )

        # Because μ1 ≈ (μn + 1)/(n + 1)
        n_1 = (
            (1. - fake_means[1]) / (fake_means[1] - current_mean)
            if current_mean != fake_means[1] else None
        )
        # print(current_mean)
        # print(n_0, n_1)
        # Take min, or ignore the None value by making it the larger number
        if n_1 is None:
            n_1 = 0.

        if n_0 is None:
            n_0 = 0.

        n = min((n_0 or n_1 + np.abs(n_1)), (n_1 or n_0 + np.abs(n_0)))
        n = np.nan_to_num(n)
        # print(f"n: {n}")
        if n < 0.:
            n = 0.
        assert n >= 0., f"Unexpected n {n}, from ({n_0}, {n_1})"
        alpha = current_mean * n + 1.  # pseudo-successes (r=1)
        beta = (1. - current_mean) * n + 1.  # pseudo-failures (r=0)
        # alpha = np.max([current_mean * n + 1., 0])  # pseudo-successes (r=1)
        # beta = np.max([(1. - current_mean) * n + 1., 0])  # pseudo-failures (r=0)
        
        
        return alpha, beta



    def update(self, history, update_model=None):
        """Algorithm 1. Use experience to update estimate of immediate r

        Args:
            history (list[tuple]): list of (state, reward) tuples to add
        """
        self.update_count += 1
        if update_model is None:
            update_model = self.model

        for state, reward in history:

            pred = update_model.predict(state, target=np.array([reward]))



class QuantileQEstimator_GLN(Estimator):

    def __init__(self, quantile, immediate_r_estimators,
                 dim_states, num_actions, gamma, layer_sizes=[4,4,4],
                 context_map_size=4, bias=True,
                 context_bias=True, lr=1e-4, scaled=True, env=None, burnin_n=0):

        super().__init__(lr, scaled=scaled)
        

        if quantile <= 0. or quantile > 1.:
            raise ValueError(f"Require 0. < q <= 1. {quantile}")

        self.quantile = quantile  # the 'i' index of the quantile
        
        self.immediate_r_estimators = immediate_r_estimators
        self.dim_states = dim_states
        self.num_actions = num_actions
        self.gamma = gamma
        self.model = [pygln.GLN(backend='numpy', layer_sizes=layer_sizes,
                          input_size=dim_states, 
                          context_map_size=context_map_size,
                          num_classes=2,
                          learning_rate=1e-4,
                          bias=bias, context_bias=context_bias
                          ) for a in range(num_actions)]

        if burnin_n > 0:
            print("Burning in Q Estimator")                          
        for i in range(burnin_n):
            state = [2*np.random.rand() - 1, 2*np.random.rand() - 1]

            for action in range(num_actions):

                self.update_estimator(state, action, self.quantile)                 
                # self.update_estimator(state, action, 0.)                 


    def estimate(self, state, action, model=None):

        if model is None:
            model =  self.model

        return model[action].predict(state, return_probs=True)

    def update(self, history):
        

        for state, action, reward, next_state, done in history:

            if not done:
                future_q = np.max([
                    self.estimate(next_state, action_i)
                    for action_i in range(self.num_actions)])
            else:
                future_q = 0.

            ire = self.immediate_r_estimators[action]
            ire_alpha, ire_beta = ire.expected_with_uncertainty(state)

            IV_i = scipy.stats.beta.ppf(self.quantile, ire_alpha, ire_beta)
            q_target = self.gamma * future_q + (1. - self.gamma) * IV_i

            self.update_estimator(state, action, q_target, lr=self.lr_fun())

            q_ai = self.estimate(state, action)

            fake_q_ai = []

            for fake_target in [0., 1.]:
                fake_model = copy.copy(self.model)
                self.update_estimator(
                    state, action, fake_target, update_model=fake_model)
                fake_q_ai.append(self.estimate(state, action, model=fake_model))
            
            n_ai0 = fake_q_ai[0] / (q_ai - fake_q_ai[0])
            n_ai1 = (1. - fake_q_ai[1]) / (fake_q_ai[1] - q_ai)
            # in the original algorithm this min was also over the quantiles
            # as well as the fake targets
            n = np.min([n_ai0, n_ai1])
            if n < 0.:
                n = 0.
            q_alpha = q_ai * n + 1.
            q_beta = (1. - q_ai) * n + 1.
            q_target_transition = scipy.stats.beta.ppf(
                self.quantile, q_alpha, q_beta)

            self.update_estimator(state, action, q_target_transition, lr=self.lr_fun(n=n))


    def update_estimator(self, state, action, q_target, update_model=None, lr=None):
        
        if update_model is None:
            update_model = self.model

        if lr is None:
            lr = self.lr_fun()
        action = int(action)
        update_gln = update_model[action]
        update_gln.update_learning_rate(lr)
        update_gln.predict(state, target=[q_target], return_probs=True)
        # print(update_model[action])
        # update_model[action].update_learning_rate(lr)

        # update_model[action].predict(state, target=[q_target])
        

    def lr_fun(self, n=None):
        """
        Returns the learning rate. 

        Optionally takes the pseudocount n and calculates
        the learning rate. If no n is provided, uses the predefined
        learning rate. 

        """
        assert not(n is None and self.lr is None), "Both n and self.lr cannot be None"


        if self.lr is None:
            return 1 / (n + 1)
        else:
            return self.lr







class MentorQEstimator_GLN(Estimator):

    def __init__(
            self, dim_states, num_actions, gamma, scaled=True,
            init_val=1.,
            layer_sizes=[4,4,4], context_map_size=4, bias=True,
            context_bias=True, lr=1e-4, env=None, burnin_n=0):
        """Set up the QEstimator for the mentor

        Rather than using a Q-table with shape num_states * num_actions
        we use a list with shape num_states, and update this "Q-list"
        for the given state regardless of what action was taken.
        This is allowed because we never will choose an action from this,
        only compare Q-values to decide when to query the mentor.

        Args:
            num_states (int): the number of states in the environment
            num_actions (int): the number of actions
            gamma (float): the discount rate for Q-learning
        """
        super().__init__(lr, scaled=scaled)
        self.num_actions = num_actions
        self.dim_states = dim_states
        self.gamma = gamma
        # self.model = pygln.GLN(backend='numpy', layer_sizes=layer_sizes,
        #                 input_size=dim_states, 
        #                 context_map_size=context_map_size,
        #                 num_classes=2,
        #                 learning_rate=1e-4,
        #                 bias=bias, context_bias=context_bias
        #                 ) 
        self.model = pygln.GLN(backend='numpy', layer_sizes=layer_sizes,
                        input_size=dim_states, 
                        context_map_size=context_map_size,
                        num_classes=2,
                        learning_rate=1e-4,
                        bias=bias, context_bias=context_bias
                        ) 
        if burnin_n > 0:
            print("Burning in Mentor Q Estimator")
        for i in range(burnin_n):

            state = [2*np.random.rand() - 1, 2*np.random.rand() - 1]
            self.update_estimator(state, init_val)   

        self.total_updates = 0


    def update(self, history):
        """Update the mentor model's Q-list with a given history.

        In practice this history will be for actions when the
        mentor decided the action.

        Args:
            history (list): (state, action, reward, next_state) tuples
        """

        for state, action, reward, next_state, done in history:
            

            next_q_val = self.estimate(next_state) if not done else 0.
            scaled_r = (1 - self.gamma) * reward if self.scaled else reward

            q_target = scaled_r + self.gamma * next_q_val

            self.update_estimator(state, q_target)
            self.total_updates += 1

    def update_estimator(self, state, q_target, update_model=None, lr=None):
        
        if update_model is None:
            update_model = self.model

        if lr is None:
            lr = self.lr_fun()

        update_model.update_learning_rate(lr)
        update_model.predict(state, target=[q_target], return_probs=True)


    def estimate(self, state, model=None):
        """Estimate the future Q, using this estimator

        Estimate the Q value of what the mentor would choose, regardless
        of what action is taken.

        Args:
            state (int): the current state from which the Q value is being
                estimated
            q_list(np.ndarray): the Q table to estimate the value from,
                if None use self.Q_list as default
        """

        if model is None:
            model =  self.model

        return model.predict(state, return_probs=True)


    def lr_fun(self, n=None):
        """
        Returns the learning rate. 

        Optionally takes the pseudocount n and calculates
        the learning rate. If no n is provided, uses the predefined
        learning rate. 

        """
        assert not(n is None and self.lr is None), "Both n and self.lr cannot be None"


        if self.lr is None:
            return 1 / (n + 1)
        else:
            return self.lr


class ImmediateRewardEstimator_GLN_gaussian(Estimator):
    """Estimates the next reward given the current state.

    Each action has a separate single IRE.
    """

    def __init__(self, action, 
                input_size=2, layer_sizes=[4,4,1], context_dim=4, 
                 lr=1e-4, scaled=True, burnin_n=0, burnin_val=0.):
        """Create an action-specific IRE.

        Args:
            action (int): the action for this IRE, this is just used to 
                keep track of things, and isn't used for calcuations.
            input_size: (int): the length of the input
            layer_sizes (List[int]): The number of neurons in each layer
                for the GGLN
            context_dim (int): the number of hyperplanes used to make the 
                halfspaces
            lr (float): the learning rate
            scaled (bool): NOT CURRENTLY IMPLEMENTED
            burnin_n (int): the number of steps we burn in for
            burnin_val (float): the value we burn in the estimator with
        """
        super().__init__(lr=lr)
        self.action = action

        self.model = glns.GGLN(layer_sizes=layer_sizes,
                          input_size=input_size, 
                          context_dim=context_dim,
                          lr=lr
                          )                          

        self.state_dict = {}
        self.update_count = 0

        # burn in the estimator for burnin_n steps, with the value burnin_val
        if burnin_n > 0:
            print(f'Burning in IRE {action}')
        for i in range(burnin_n):
            # using random inputs from  the space [-2, 2]^2
            # the space is larger than the actual state space that the
            # agent will encounter, to hopefully mean that it burns in
            # correctly around the edges
            state_rew_history = [([4*np.random.rand() - 2, 4*np.random.rand() - 2], burnin_val)]
            self.update(state_rew_history)

    def estimate(self, state, estimate_model=None):
        """Estimate the next reward given the current state (for this action).

        Uses estimate_model to make this estimate, if this isn't provided,
        then just use the self.model
        
        """
        if estimate_model is None:
            estimate_model = self.model
        model_est = estimate_model.predict(state)

        return model_est 

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

        for i, fake_r in enumerate(fake_rewards):
            fake_model = copy.copy(self.model)

            self.update([(state, fake_r)], fake_model)
            fake_means[i] = self.estimate(state, fake_model)


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
        if n_1 is None:
            n_1 = 0.

        if n_0 is None:
            n_0 = 0.

        n = min((n_0 or n_1 + np.abs(n_1)), (n_1 or n_0 + np.abs(n_0)))
        n = np.nan_to_num(n)

        if n < 0.:
            n = 0.
        assert n >= 0., f"Unexpected n {n}, from ({n_0}, {n_1})"
        alpha = current_mean * n + 1.  # pseudo-successes (r=1)
        beta = (1. - current_mean) * n + 1.  # pseudo-failures (r=0)
        # alpha = np.max([current_mean * n + 1., 0])  # pseudo-successes (r=1)
        # beta = np.max([(1. - current_mean) * n + 1., 0])  # pseudo-failures (r=0)
        
        
        return alpha, beta



    def update(self, history, update_model=None):
        """Algorithm 1. Use experience to update estimate of immediate r

        Args:
            history (list[tuple]): list of (state, reward) tuples to add
        """
        self.update_count += 1
        if update_model is None:
            update_model = self.model

        for state, reward in history:

            pred = update_model.predict(state, target=np.array([reward]))



class QuantileQEstimator_GLN_gaussian(Estimator):
    """A  GGLN Q estimator that makes pessimistic updates e.g. using IRE

    Updates using algorithm 3 in the QuEUE specification.

    Uses IREs (which are also GGLNs) to get pessimistic estimates of the next
    reward, to train a GGLN to estimate the Q value.
    """
    def __init__(self, quantile, immediate_r_estimators,
                 dim_states, num_actions, gamma, layer_sizes=[4,4,4,1],
                 context_dim=4, lr=1e-4, scaled=True, env=None, burnin_n=0,
                 burnin_val=None):
        """Set up the GGLN QEstimator for the given quantile

        Burns in the GGLN to the burnin_val (default is the quantile value)
        for burnin_n steps, where the state for burn in is randomly sampled.

        Args:
            quantile (float): the pessimism-quantile that this estimator
                is estimating the future-Q value for.
            immediate_r_estimators (list[ImmediateRewardEstimator]): A
                list of IRE objects, indexed by-action
            dim_states (int): the dimension of the state space,
                eg, in the 2D cliffworld dim_states = 2
            num_actions (int): the number of actions (4 in cliffworld)
            gamma (float): the discount rate
            layer_sizes (List[int]): The number of neurons in each layer
                for the GGLN
            context_dim (int): the number of hyperplanes used to make the 
                halfspaces
            lr (float): the learning rate
            scaled (bool): NOT CURRENTLY IMPLEMENTED
            burnin_n (int): the number of steps we burn in for
            burnin_val (float): the value we burn in the estimator with
        
        """
        super().__init__(lr, scaled=scaled)
        

        if quantile <= 0. or quantile > 1.:
            raise ValueError(f"Require 0. < q <= 1. {quantile}")

        self.quantile = quantile  # the 'i' index of the quantile
        
        self.immediate_r_estimators = immediate_r_estimators
        self.dim_states = dim_states
        self.num_actions = num_actions
        self.gamma = gamma

        self.model = [glns.GGLN(layer_sizes=layer_sizes,
                          input_size=dim_states, 
                          context_dim=context_dim,
                          lr=lr,
                          )  for a in range(num_actions)]
        
        # set the value to burn in the estimator
        if burnin_val is None:
            burnin_val = self.quantile

        if burnin_n > 0:
            print("Burning in Q Estimator")                          
        for i in range(burnin_n):
            # using random inputs from  the space [-2, 2]^2
            # the space is larger than the actual state space that the
            # agent will encounter, to hopefully mean that it burns in
            # correctly around the edges

            state = [4*np.random.rand() - 2, 4*np.random.rand() - 2]

            for action in range(num_actions):

                self.update_estimator(state, action, burnin_val)                 
                # self.update_estimator(state, action, 0.)                 


    def estimate(self, state, action, model=None):
        """Estimate the future Q, using this estimator

        Args:
            state (int): the current state from which the Q value is
                being estimated
            action (int): the action taken
            model: the estimator (GGLN) used to estimate the Q value,
            if None, then use self.model as default

        """
        if model is None:
            model =  self.model

        return model[action].predict(state)

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

            if not done:
                future_q = np.max([
                    self.estimate(next_state, action_i)
                    for action_i in range(self.num_actions)])
            else:
                future_q = 0.

            ire = self.immediate_r_estimators[action]
            ire_alpha, ire_beta = ire.expected_with_uncertainty(state)

            IV_i = scipy.stats.beta.ppf(self.quantile, ire_alpha, ire_beta)
            q_target = self.gamma * future_q + (1. - self.gamma) * IV_i

            self.update_estimator(state, action, q_target, lr=self.lr_fun())

            q_ai = self.estimate(state, action)

            fake_q_ai = []

            for fake_target in [0., 1.]:
                fake_model = copy.copy(self.model)
                self.update_estimator(
                    state, action, fake_target, update_model=fake_model)
                fake_q_ai.append(self.estimate(state, action, model=fake_model))
            
            n_ai0 = fake_q_ai[0] / (q_ai - fake_q_ai[0])
            n_ai1 = (1. - fake_q_ai[1]) / (fake_q_ai[1] - q_ai)
            # in the original algorithm this min was also over the quantiles
            # as well as the fake targets
            n = np.min([n_ai0, n_ai1])
            if n < 0.:
                n = 0.
            q_alpha = q_ai * n + 1.
            q_beta = (1. - q_ai) * n + 1.
            q_target_transition = scipy.stats.beta.ppf(
                self.quantile, q_alpha, q_beta)

            self.update_estimator(state, action, q_target_transition, lr=self.lr_fun(n=n))


    def update_estimator(self, state, action, q_target, update_model=None, lr=None):
        
        if update_model is None:
            update_model = self.model

        if lr is None:
            lr = self.lr_fun()
        action = int(action)
        update_gln = update_model[action]
        update_gln.update_learning_rate(lr)
        update_gln.predict(state, target=[q_target])

        

    def lr_fun(self, n=None):
        """
        Returns the learning rate. 

        Optionally takes the pseudocount n and calculates
        the learning rate. If no n is provided, uses the predefined
        learning rate. 

        """
        assert not(n is None and self.lr is None), "Both n and self.lr cannot be None"

        if self.lr is None:
            return 1 / (n + 1)
        else:
            return self.lr







class MentorQEstimator_GLN_gaussian(Estimator):

    def __init__(
            self, dim_states, num_actions, gamma, scaled=True,
            init_val=1.,
            layer_sizes=[4,4,4], context_dim=4, bias=True,
            context_bias=True, lr=1e-4, env=None, burnin_n=0):
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
            lr (float): the learning rate
            burnin_n (int): the number of steps we burn in for
        """
        super().__init__(lr, scaled=scaled)
        self.num_actions = num_actions
        self.dim_states = dim_states
        self.gamma = gamma

        self.model = glns.GGLN(layer_sizes=layer_sizes,
                          input_size=dim_states, 
                          context_dim=context_dim,
                          lr=lr,
                          )

        if burnin_n > 0:
            print("Burning in Mentor Q Estimator")
        for i in range(burnin_n):

            state = [4*np.random.rand() - 2, 4*np.random.rand() - 2]
            self.update_estimator(state, init_val)   

        self.total_updates = 0


    def update(self, history):
        """Update the mentor model's Q-estimator with a given history.

        In practice this history will be for actions when the
        mentor decided the action.

        Args:
            history (list): (state, action, reward, next_state) tuples
        """

        for state, action, reward, next_state, done in history:
            

            next_q_val = self.estimate(next_state) if not done else 0.
            scaled_r = (1 - self.gamma) * reward if self.scaled else reward

            q_target = scaled_r + self.gamma * next_q_val

            self.update_estimator(state, q_target)
            self.total_updates += 1

    def update_estimator(self, state, q_target, update_model=None, lr=None):
        
        if update_model is None:
            update_model = self.model

        if lr is None:
            lr = self.lr_fun()

        update_model.update_learning_rate(lr)
        update_model.predict(state, target=[q_target])


    def estimate(self, state, model=None):
        """Estimate the future Q, using this estimator

        Estimate the Q value of what the mentor would choose, regardless
        of what action is taken.

        Args:
            state (int): the current state from which the Q value is being
                estimated
            model: the (GGLN) estimator used to estimate the value from,
                if None use self.model as default
        """

        if model is None:
            model =  self.model

        return model.predict(state)


    def lr_fun(self, n=None):
        """
        Returns the learning rate. 

        Optionally takes the pseudocount n and calculates
        the learning rate. If no n is provided, uses the predefined
        learning rate. 

        """
        assert not(n is None and self.lr is None), "Both n and self.lr cannot be None"


        if self.lr is None:
            return 1 / (n + 1)
        else:
            return self.lr
