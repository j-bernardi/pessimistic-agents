import abc
import sys
import time

import jax
import torch as tc
import numpy as np
import jax.numpy as jnp
from collections import deque, namedtuple

import bayes_by_backprop
from estimators import (
    ImmediateRewardEstimator,
    MentorQEstimator,
    MentorFHTDQEstimator,
    ImmediateRewardEstimatorGaussianGLN,
    MentorQEstimatorGaussianGLN,
    BURN_IN_N, GLN_CONTEXT_DIM,
    ImmediateRewardEstimatorBayes,
    MentorQEstimatorBayes,
)

from q_estimators import (
    QuantileQEstimator,
    BasicQTableEstimator,
    QEstimatorIRE,
    QTableEstimator,
    QuantileQEstimatorGaussianGLN,
    QuantileQEstimatorBayes,
)
from utils import geometric_sum, stack_batch, JaxRandom, jnp_batch_apply

# QUANTILES = [2**k / (1 + 2**k) for k in range(-5, 5)]
QUANTILES = [0.01, 0.03, 0.06, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35]


Transition = namedtuple(
    "Transition", ("state", "action", "reward", "next_state", "done"))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = deque([], maxlen=capacity)
        self.count = 0

        self.stack = jnp.stack

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*(jax.device_put(x) for x in args)))
        self.count += 1

    def sample(self, batch_size):
        """Obtain a random sample of the history (with replacement)

        Args:
            batch_size: number of samples to pull
        """
        if not self.count:
            print("WARN - hist len is 0")
            return None
        indices = np.random.randint(
            low=0, high=min(self.count, self.capacity), size=batch_size)
        transition_list = [self.memory[i] for i in indices]
        return transition_list

    def sample_arrays(self, batch_size):
        """Return a sample as a Transition of arrays, len batch_size"""
        trans_list = self.sample(batch_size)
        return self.list_to_arrays(trans_list)

    def all_as_arrays(self, bs=None, retain_all=False):
        return self.list_to_arrays(
            self.memory, batch_apply=bs, retain_all=retain_all)

    def list_to_arrays(self, transition_list, batch_apply=None, retain_all=False):
        """Converts lists of single transitions to a Transition of
        stacked arrays

        Assumes state, reward and next state are to be stacked as device
        arrays, and actions and dones are left raw.
        """
        # A named tuple with tuples as values
        transes = Transition(*zip(*transition_list))
        if batch_apply is not None and self.count > batch_apply:
            list_of_stacked = [
                jnp_batch_apply(
                    jnp.stack, x, batch_apply, retain_all=retain_all)
                for x in transes]
        else:
            # jax_stacked = jax.vmap(jnp.stack)([x for x in transes])
            # list_of_stacked = [
            #     jax_stacked[i] for i in range(jax_stacked.shape[0])]
            list_of_stacked = list(map(jnp.stack, transes))
        return Transition(*list_of_stacked)

    def __len__(self):
        return len(self.memory)


class BaseAgent(abc.ABC):
    """Abstract implementation of an agent

    Useful as it saves state and variables that are shared by all
    agents. Also specifies the abstract methods that all agents need
    to be valid with learn() - which is the same for all agents.
    """
    def __init__(
            self,
            num_actions,
            env,
            gamma,
            sampling_strategy="last_n_steps",
            lr=0.1,
            update_n_steps=1,
            batch_size=1,
            eps_max=0.1,
            eps_min=0.01,
            mentor=None,
            scale_q_value=True,
            min_reward=0.1,
            horizon_type="inf",
            num_horizons=1,
            max_steps=np.inf,
            debug_mode=False,
            **kwargs
    ):
        """Initialise the base agent with shared params

            num_actions:
            env:
            gamma (float): the future Q discount rate
            sampling_strategy (str): one of 'random', 'last_n_steps' or
                'whole', dictating how the history should be sampled.
            lr (float): Agent learning rate (defaults to all estimators,
                unless inheriting classes defines another lr).
            update_n_steps: how often to call the update_estimators
                function
            batch_size (int): size of the history to update on
            eps_max (float): initial max value of the random
                query-factor
            eps_min (float): the minimum value of the random
                query-factor. Once self.epsilon < self.eps_min, it stops
                reducing.
            mentor (callable): A function taking args=(state, kwargs),
                returning a tuple of (act_rows, act_cols) symbolizing
                a 2d grid transform. See mentors.py.
            scale_q_value (bool): if True, Q value estimates are always
                between 0 and 1 for the agent and mentor estimates.
                Otherwise, they are an estimate of the true sum of
                rewards
            min_reward (float): if the agent value is below this
                threshold, it will not act (regardless of exceeding
                mentor value).
            horizon_type (str): one of "finite" or "inf"
            num_horizons (int): number of time steps to look into the
                future for, when calculating horizons
            max_steps (int): max number of steps to take.
            debug_mode (bool): a flag for whether to be maximally
                verbose. Only partially implemented.
        """
        assert not kwargs, f"Arguments missed: {kwargs}"
        self.num_actions = num_actions
        self.env = env
        self.gamma = gamma
        if "whole" in sampling_strategy and batch_size != 1:
            print("WARN: Default not used for BS, but sampling whole history")
        self.sampling_strategy = sampling_strategy
        self.lr = lr
        self.batch_size = batch_size
        self.update_n_steps = update_n_steps
        self.max_steps = max_steps
        self.eps_max = eps_max
        self.eps_min = eps_min
        self.scale_q_value = scale_q_value
        self.mentor = mentor
        self.min_reward = min_reward

        self.horizon_type = horizon_type
        self.num_horizons = num_horizons

        self.q_estimator = None
        self.mentor_q_estimator = None

        self.history = deque(maxlen=10000)
        self.mentor_history = deque(maxlen=10000)

        self.mentor_queries = 0
        self.total_steps = 0
        self.failures = 0

        self.mentor_queries_periodic = []
        self.rewards_periodic = []
        self.failures_periodic = []

        self.debug_mode = debug_mode

    def store_history(
            self, state, action, reward, next_state, done, mentor_acted=False):

        if mentor_acted:
            self.mentor_history.append(
                (state, action, reward, next_state, done))

        self.history.append((state, action, reward, next_state, done))

    @abc.abstractmethod
    def act(self, state):
        raise NotImplementedError("Must be implemented per agent")

    @abc.abstractmethod
    def update_estimators(self, mentor_acted=False, **kwargs):
        raise NotImplementedError()

    def epsilon(self, reduce=True):
        """Reduce the max value of, and return, the random var epsilon."""

        if self.eps_max > self.eps_min and reduce:
            self.eps_max *= 0.999

        return self.eps_max * np.random.rand()
        # return np.random.rand()*(self.eps_max - self.eps_min) + self.eps_min

    def use_epsilon(self, value_compare, reduce_if=True):
        """Return whether to use epsilon, the random var epsilon.

        Args:
            value_compare (float): the value to compare to epsilon
                (typically mentor_val - agent_val). If epsilon is
                larger, randomness won-out.
            reduce_if (bool): reduce epsilon if it wins
        """
        eps = self.epsilon(reduce=False)
        if self.horizon_type == "finite":
            eps = geometric_sum(eps, self.gamma, self.num_horizons)
        epsilon_random_wins = eps > value_compare

        if reduce_if and epsilon_random_wins and self.eps_max > self.eps_min:
            self.eps_max *= 0.999  # decay

        return epsilon_random_wins, eps

    def sample_history(self, history, strategy=None, batch_size=None):
        """Return a sample of the history

        Args:
            history: the list from which to sample indices
            strategy (Choice(["random", "last_n_steps", "whole"])):
                The sampling strategy to use to sample this history.
                If None, use self.sampling_strategy.
            batch_size (Optional[int]): if None, use self.batch_size

        Returns:
            samples (list): the sampled entries from self.history -
                i.e. (s, a, r, ns, d) tuples in the history
        """
        hist_len = len(history)
        if not hist_len:
            print("WARN - hist len is 0")
            return []
        sampling_strategy = (
            self.sampling_strategy if strategy is None else strategy)
        batch_size = self.batch_size if batch_size is None else batch_size

        if sampling_strategy == "random":
            idxs = np.random.randint(
                low=0, high=hist_len, size=batch_size)
        elif sampling_strategy == "last_n_steps":
            if hist_len < batch_size:
                print("WARN - last N doesn't exist")
                return []  # not ready yet
            idxs = range(-batch_size, 0)
        elif "whole" in sampling_strategy:
            return history
        else:
            raise ValueError(f"Sampling strategy {sampling_strategy} invalid")
        return [history[i] for i in idxs]

    def report(
            self, tot_steps, rewards_last, render_mode, queries_last=None,
            duration=None,
    ):
        """Reports on period of steps and calls any additional printing

        Args:
            tot_steps (int): total number of steps expected
            rewards_last (list): list of rewards from last reporting
                period
            render_mode (int): defines verbosity of rendering
            queries_last (int): number of mentor queries in the last
                reporting period (i.e. the one being reported).
            duration (float): time of last update
        """
        if render_mode < 0 or self.total_steps <= 0:
            return
        if render_mode > 0:
            self.env.print_spacer()
        if rewards_last:
            if len(rewards_last) > 1:
                rew = sum(rewards_last)
            else:
                rew = rewards_last[0]
        else:
            rew = "-"
        report = (
            f"Step {self.total_steps} / {tot_steps} : "
            f"{100 * self.total_steps / tot_steps:.0f}% : "
            f"F {self.failures} - R (last N) {rew:.2f}")
        if queries_last is not None:
            report += (
                f" - M (last N) {queries_last} (total={self.mentor_queries})")
        if duration is not None:
            report += f" - T {duration:.1f}s"

        print(report)
        self.additional_printing(render_mode)

        if render_mode > 0:
            self.env.print_newlines()

        sys.stdout.flush()

    def additional_printing(self, render_mode):
        """Defines class-specific printing"""
        return None


class FiniteAgent(BaseAgent, abc.ABC):

    def __init__(self, num_states, track_transitions=None, **kwargs):
        """Set finite-specific state

        Args:
            num_states: number of (integer) states
            track_transitions (list): List of transition definitions
                (state, action, next_state), where any can be None to
                indicate "any". Tracks number of times the transition is
                observed. Keeps dict of transitions[s][a][s'] = N_(s,a,s')
            kwargs: pass to BaseAgent
        """
        super().__init__(**kwargs)

        self.num_states = num_states
        if track_transitions is not None:
            self.transitions = {}
            for s, a, ns in track_transitions:
                assert all(
                    x is None or int(x) or x == 0 for x in (s, a, ns)), (
                    f"Misuse: {s}: {type(s)}, {a}, {ns} should all be ints")
                if s not in self.transitions:
                    self.transitions[s] = {}
                if a not in self.transitions[s]:
                    self.transitions[s][a] = {}
                if ns not in self.transitions[s][a]:
                    self.transitions[s][a][ns] = [0, 0]  # initial count
        else:
            self.transitions = None

    def learn(
            self, num_steps, report_every_n=500, render=1,
            reset_every_ep=False,
            early_stopping=0
    ):
        """Let the agent loose in the environment, and learn!

        Args:
            num_steps (int): Number of steps to learn for.
            report_every_n (int): Number of steps per reporting period
            render (int): Render mode 0, 1, 2
            reset_every_ep (bool): If true, resets state every
                reporting period, else continues e.g. infinite env.
            early_stopping (int): If sum(mentor_queries[-es:]) == 0,
                stop (and return True)

        Returns:
            True if stopped querying mentor for `early_stopping`
                reporting periods
            False if never stopped querying for all num_eps
            None if early_stopping = 0 (e.g. not-applicable)
        """

        if self.total_steps != 0:
            print("WARN: Agent already trained", self.total_steps)
        period_rewards = []  # initialise

        state = int(self.env.reset())
        while self.total_steps <= num_steps:
            action, mentor_acted = self.act(state)
            next_state, reward, done, _ = self.env.step(action)
            next_state = int(next_state)

            self.store_history(
                state, action, reward, next_state, done, mentor_acted)
            period_rewards.append(reward)
            _ = self.track_transition(
                state, action, next_state, mentor_acted=mentor_acted)

            assert reward is not None, (
                f"Reward None at ({state}, {action})->{next_state}, "
                f"Mentor acted {mentor_acted}. Done {done}")
            if render > 0:
                # First rendering should not return N lines
                self.env.render(in_loop=self.total_steps > 0)
            if self.total_steps and self.total_steps % self.update_n_steps == 0:
                self.update_estimators(mentor_acted=mentor_acted)
            state = next_state
            if done:
                self.failures += 1
                # print('failed')
                state = int(self.env.reset())

            if self.total_steps % report_every_n == 0:
                if self.total_steps > 0 and reset_every_ep:
                    state = int(self.env.reset())
                prev_queries = np.sum(self.mentor_queries_periodic)
                self.mentor_queries_periodic.append(
                    self.mentor_queries - prev_queries)
                self.report(
                    num_steps, period_rewards, render_mode=render,
                    queries_last=self.mentor_queries_periodic[-1])
                prev_failures = np.sum(self.failures_periodic)
                self.failures_periodic.append(self.failures - prev_failures)
                self.rewards_periodic.append(sum(period_rewards))
                period_rewards = []  # reset

                if (
                        early_stopping
                        and len(self.mentor_queries_periodic) > early_stopping
                        and sum(self.mentor_queries_periodic[
                                -early_stopping:]) == 0):
                    return True

            self.total_steps += 1

        return False if early_stopping else None

    def track_transition(self, s, a, ns, mentor_acted=False):
        """If flagged for tracking, track a transition.

        Returns:
            None if not tracking any
        """
        assert all((isinstance(x, int) or x is None) for x in (s, a, ns)), (
            f"Misuse: {s}, {a}, {ns} should all be ints")

        # First, replace anything not found with None, the more general option.
        # None means "any". So specified transitions override.
        if self.transitions is None:
            return None
        if s not in self.transitions:
            s = None
        if s in self.transitions and a not in self.transitions[s]:
            a = None
        if (s in self.transitions and a in self.transitions[s]) \
                and ns not in self.transitions[s][a]:
            ns = None

        # Then increment the transition, if found
        if s in self.transitions:
            if a in self.transitions[s]:
                if ns in self.transitions[s][a]:
                    if mentor_acted:
                        self.transitions[s][a][ns][1] += 1  # mentor count
                    else:
                        self.transitions[s][a][ns][0] += 1  # agent count
                    return True
        return False


class MentorAgent(FiniteAgent):
    """An agent that provides a way to call the mentor at every timestep

    Simply does what the mentor does - so only implements act()
    (and dummy methods for the abstract methods)
    """
    def __init__(self, **kwargs):
        """Implement agent that just does what mentor would do

        No Q estimator required - always acts via `mentor` callable.
        """
        super().__init__(**kwargs)
        if self.mentor is None:
            raise ValueError("MentorAgent must have a mentor")

    def act(self, state):
        """Act, given the current state

        Returns:
             mentor_action (int): the action the mentor takes
             mentor_acted (bool): always True
        """
        # TODO: change this to account for continuous states as well.
        mentor_action = self.env.map_grid_act_to_int(
            self.mentor(
                self.env.map_int_to_grid(state),
                kwargs={'state_shape': self.env.state_shape})
        )
        return mentor_action, True

    def update_estimators(self, mentor_acted=False, **kwargs):
        """Nothing to do"""
        pass

    def store_history(
            self, state, action, reward, next_state, done, mentor_acted=True):
        """Nothing to do"""
        pass


class BaseFiniteQAgent(FiniteAgent, abc.ABC):
    """A base agent that acts upon a Q-value estimate of finite actions

    Assumes self.q_estimator has a method estimate(state, action), which
    is used to select the next action with the highest value.

    Inheriting classes must implement the self.q_estimator to be
    whatever version the algorithm requires, and the other methods are
    implemented to succesfully update that q_estimator (and supporting
    estimators), given the agent's experience.

    Optionally uses a mentor (depending on whether parent self.mentor is
    None).
    """

    def __init__(
            self, eps_a_max=None, eps_a_min=0.05, eps_a_decay=(1.-1e-6),
            **kwargs):
        super().__init__(**kwargs)
        self.eps_a_max = eps_a_max  # unused by default (unless set)
        self.eps_a_min = eps_a_min  # minimum noise-range, over 0.
        self.eps_a_decay = eps_a_decay  # factor to decay each step
        if self.eps_a_max is not None:
            assert self.eps_a_max > self.eps_a_min

    def act(self, state):
        """Act, given a state, depending on future Q of each action

        Args:
            state (int): current state

        Returns:
            action (int): The action to take
            mentor_acted (bool): Whether the mentor selected the action
        """
        eps_rand_act = self.q_estimator.get_random_act_prob()
        if eps_rand_act is not None and np.random.rand() < eps_rand_act:
            assert self.mentor is None
            return int(np.random.randint(self.num_actions)), False

        if self.scale_q_value:
            scaled_min_r = self.min_reward  # Defined as between [0, 1]
            scaled_eps = self.epsilon()
            scaled_max_r = 1.
        else:
            horizon_steps = self.q_estimator.num_horizons\
                if self.horizon_type == "finite" else "inf"
            scaled_min_r = geometric_sum(
                self.min_reward, self.gamma, horizon_steps)
            scaled_eps = geometric_sum(
                self.epsilon(), self.gamma, horizon_steps)
            scaled_max_r = geometric_sum(
                1., self.gamma, horizon_steps)

        values = np.array(
            [self.q_estimator.estimate(state, action_i)
             for action_i in range(self.num_actions)]
        )
        # Add some uniform, random noise to the action values, if flagged
        if self.eps_a_max is not None:
            values += self.action_noise()
            values = np.minimum(values, scaled_max_r)

        # Choose randomly from any jointly maximum values
        max_vals = values == np.amax(values)
        max_nonzero_val_idxs = np.flatnonzero(max_vals)
        tie_broken_proposed_action = np.random.choice(max_nonzero_val_idxs)
        agent_max_action = int(tie_broken_proposed_action)

        if self.mentor is not None:
            mentor_value = self.mentor_q_estimator.estimate(state)
            # Defer if
            # 1) min_reward > agent value, based on r > eps
            agent_value_too_low = values[agent_max_action] <= scaled_min_r
            # 2) mentor value > agent value + eps
            prefer_mentor = mentor_value > (
                values[agent_max_action] + scaled_eps)

            if agent_value_too_low or prefer_mentor:
                mentor_action = self.env.map_grid_act_to_int(
                    self.mentor(
                        self.env.map_int_to_grid(state),
                        kwargs={'state_shape': self.env.state_shape})
                )
                # print("called mentor")
                self.mentor_queries += 1
                return mentor_action, True

        return agent_max_action, False

    def action_noise(self):
        """Produce random noise to

        Noise is centred on 0., and is in range [-eps_max, +eps_max].

        eps_max is decayed to a minimum of eps_min, every time it is
        called.
        """
        if self.eps_a_max is None:
            return 0.

        rand_vals = (
                (np.random.rand(self.num_actions) - 0.5)  # centre on 0.
                * self.eps_a_max  # scale
        )
        assert np.all(np.abs(rand_vals) <= 0.5 * self.eps_a_max), (
            f"{rand_vals}, {self.eps_a_max}")
        # Decay
        if self.eps_a_max > self.eps_a_min:
            self.eps_a_max *= self.eps_a_decay

        return rand_vals


class PessimisticAgent(BaseFiniteQAgent):
    """The faithful, original Pessimistic Distributed Q value agent

    The q_estimator used is QuantileQEstimator, by default. This has
    state that defines the quantile (inferred in the init function of
    this agent.

    The estimator used is an argument to the init function,
    because in the future we may want to try a single-step update
    version of the QEstimator.
    """

    def __init__(
            self,
            num_actions,
            num_states,
            env,
            gamma,
            mentor,
            quantile_i,
            quantile_estimator_init=QuantileQEstimator,
            train_all_q=False,
            init_to_zero=False,
            **kwargs
    ):
        """Initialise the faithful agent

        Additional Kwargs:
            mentor (callable): a function taking tuple (state, kwargs),
                returning an integer action. Not optional for the
                pessimistic agent.
            quantile_i (int): the index of the quantile from QUANTILES to use
                for taking actions.
            quantile_estimator_init (callable): the init function for
                the type of Q Estimator to use for the agent. Choices:
                QuantileQEstimatorSingleOrig or default.
            train_all_q (bool): if False, trains only the Q estimator
                corresponding to quantile_i (self.q_estimator)
            init_to_zero (bool): if True, initialises the Q table to 0.
                rather than 'burning-in' quantile value

        Kwargs:
            capture_alphas (tuple): The (state, action) tuple to capture
                the alpha beta information for, from the Q Estimator.
                Used for visualisation. They are generated like:
                [((ire_alpha, ire_beta), (q_alpha, q_beta)), ...]
        """
        self.capture_alphas = kwargs.pop("capture_alphas", False)
        super().__init__(
            num_actions=num_actions, num_states=num_states, env=env,
            gamma=gamma, mentor=mentor, **kwargs
        )

        if self.mentor is None:
            raise NotImplementedError("Pessimistic agent requires a mentor")

        self.quantile_i = quantile_i

        # Create the estimators
        self.IREs = [ImmediateRewardEstimator(a) for a in range(num_actions)]

        self.QEstimators = [
            quantile_estimator_init(
                quantile=q,
                immediate_r_estimators=self.IREs,
                gamma=gamma,
                num_states=num_states,
                num_actions=num_actions,
                lr=self.lr,
                q_table_init_val=0. if init_to_zero else QUANTILES[i],
                horizon_type=self.horizon_type,
                num_horizons=self.num_horizons,
                scaled=self.scale_q_value,
            ) for i, q in enumerate(QUANTILES) if (
                i == self.quantile_i or train_all_q)
        ]
        self.q_estimator = self.QEstimators[
            self.quantile_i if train_all_q else 0]

        if self.horizon_type == "inf":
            self.mentor_q_estimator = MentorQEstimator(
                num_states=num_states, num_actions=num_actions, gamma=gamma,
                lr=self.lr, scaled=self.scale_q_value)
        elif self.horizon_type == "finite":
            self.mentor_q_estimator = MentorFHTDQEstimator(
                num_states=num_states, num_actions=num_actions, gamma=gamma,
                lr=self.lr, num_horizons=self.num_horizons,
                scaled=self.scale_q_value)

        self.alpha_betas = [] if self.capture_alphas else None

    def reset_estimators(self):
        for ire in self.IREs:
            ire.reset()
        for q_est in self.QEstimators:
            q_est.reset()
        self.mentor_q_estimator.reset()

    def update_estimators(self, mentor_acted=False, **kwargs):
        """Update all estimators with a random batch of the histories.

        Mentor-Q Estimator
        ImmediateRewardEstimators (currently only for the actions in the
            sampled batch that corresponds with the IRE).
        Q-estimator (for every quantile)
        """
        if self.sampling_strategy == "whole_reset":
            self.reset_estimators()

        if mentor_acted:
            mentor_history_samples = self.sample_history(self.mentor_history)

            self.mentor_q_estimator.update(mentor_history_samples)

        history_samples = self.sample_history(self.history)

        # This does < batch_size updates on the IREs. For history-handling
        # purposes. Possibly sample batch_size per-action in the future.
        for IRE_index, IRE in enumerate(self.IREs):
            IRE.update(
                [(s, r) for s, a, r, _, _ in history_samples if IRE_index == a])

        for i, q_estimator in enumerate(self.QEstimators):
            alpha_betas = q_estimator.update(
                history_samples, capture_alpha_beta=self.capture_alphas)

            if self.capture_alphas\
                    and (i == self.quantile_i or len(self.QEstimators) == 1):
                self.alpha_betas.extend(alpha_betas)

    def additional_printing(self, render):
        super().additional_printing(render_mode=render)
        if render > 0:
            print(
                f"M {self.mentor_queries_periodic[-1]} ({self.mentor_queries})")
        if render > 1:
            print("Additional for finite pessimistic")
            # Q table has dim (States, actions, n_horiz)=(s, a, 2) in inf case
            print(f"Q table\n{self.q_estimator.q_table[:, :, -1]}")
            if self.horizon_type == "finite":
                print(f"Mentor Q table\n"
                      f"{self.mentor_q_estimator.q_list[..., -1]}")
            else:
                print(f"Mentor Q table\n{self.mentor_q_estimator.q_list}")
            if self.q_estimator.lr is not None:
                print(
                    f"Learning rates: "
                    f"QEst {self.q_estimator.lr:.4f}, "
                    f"Mentor V {self.mentor_q_estimator.lr:.4f}"
                )
                if self.eps_a_max is not None:
                    print(f"\nEpsilon max: {self.eps_max:.4f}")


class BaseQTableAgent(BaseFiniteQAgent, abc.ABC):
    """The base implementation of an agent calculative Q with a table

    The self.q_estimator is a basic Q table. It is updated in the
    simplest manner, and implements no notion of pessimism.

    It can optionally use a mentor, as defined in BaseQAgent.act().
    """
    def __init__(
            self,
            num_actions,
            num_states,
            env,
            gamma,
            q_estimator_init=BasicQTableEstimator,
            **kwargs
    ):
        """Initialise the basic Q table agent

        Inheriting classes should pass the correct q_estimator_init to
        this init function, so that the correct estimation is
        implemented. All other methods defining updates should be
        re-implented too (see examples of agents inheriting this class)

        Additional Args:
            q_estimator_init (callable): the init function for the type of
                Q estimator to use for the class (e.g. finite, infinite
                horizon).
            mentor_q_estimator_init (callable): the init function for
                the type of Q estimator to use for the class (e.g.
                finite, infinite horizon)
        """
        super().__init__(
            num_actions=num_actions, num_states=num_states, env=env,
            gamma=gamma, **kwargs
        )

        self.q_estimator = q_estimator_init(
            num_states=num_states,
            num_actions=num_actions,
            gamma=gamma,
            lr=self.lr,
            has_mentor=self.mentor is not None,
            scaled=self.scale_q_value,
            horizon_type=self.horizon_type,
            num_horizons=self.num_horizons
        )

        if not self.scale_q_value:
            # geometric sum of max rewards per step (1.) for N steps (up to inf)
            init_mentor_val = geometric_sum(
                1., self.gamma, self.q_estimator.num_horizons)
        else:
            init_mentor_val = 1.

        if self.horizon_type == "inf" and self.mentor is not None:
            self.mentor_q_estimator = MentorQEstimator(
                num_states=num_states, num_actions=num_actions, gamma=gamma,
                lr=self.lr, scaled=self.scale_q_value, init_val=init_mentor_val)
        elif self.horizon_type == "finite" and self.mentor is not None:
            self.mentor_q_estimator = MentorFHTDQEstimator(
                num_states=num_states, num_actions=num_actions, gamma=gamma,
                lr=self.lr, scaled=self.scale_q_value, init_val=init_mentor_val,
                num_horizons=self.num_horizons)

        self.history = deque(maxlen=10000)
        if self.mentor is None:
            self.mentor_history = None

    def reset_estimators(self):
        self.q_estimator.reset()
        if self.mentor_q_estimator is not None:
            self.mentor_q_estimator.reset()

    def update_estimators(self, mentor_acted=False, **kwargs):

        if self.sampling_strategy == "whole_reset":
            self.reset_estimators()

        history_samples = self.sample_history(self.history)
        self.q_estimator.update(history_samples)

        if mentor_acted:
            assert self.mentor is not None
            mentor_history_samples = self.sample_history(self.mentor_history)
            self.mentor_q_estimator.update(mentor_history_samples)

    def additional_printing(self, render_mode):
        """Called by the episodic reporting super method"""
        if render_mode > 0:
            if self.mentor is not None:
                print(
                    f"Mentor queries {self.mentor_queries_periodic[-1]} "
                    f"({self.mentor_queries})")
            else:
                print(f"Epsilon {self.q_estimator.random_act_prob}")

        if render_mode > 1:
            print("Additional for QTableAgent")
            print(f"M {self.mentor_queries} ")
            if len(self.q_estimator.q_table.shape) == 3:
                print(f"Agent Q\n{self.q_estimator.q_table[:, :, -1]}")
                if self.mentor_q_estimator.q_list.ndim > 1:
                    print(
                        f"Mentor Q table\n"
                        f"{self.mentor_q_estimator.q_list[:, -1]}")
                else:
                    print(
                        f"Mentor Q table\n{self.mentor_q_estimator.q_list[:]}")
            else:
                print(f"Agent Q\n{self.q_estimator.q_table}")
                print(f"Mentor Q table\n{self.mentor_q_estimator.q_list}")
            if self.mentor_q_estimator.lr is not None:
                print(
                    f"Learning rates: "
                    f"QEst {self.q_estimator.lr:.4f}, "
                    f"Mentor V {self.mentor_q_estimator.lr:.4f}"
                )


class QTableAgent(BaseQTableAgent):
    """A basic Q table agent - no pessimism or IREs, just a Q table

    No further state is required - the base defaults to the QTableAgent
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        assert isinstance(self.q_estimator, QTableEstimator)

    def additional_printing(self, render_mode):
        super().additional_printing(render_mode)
        if render_mode:
            print("Action noise max eps", self.eps_a_max)


class BaseQTableIREAgent(BaseQTableAgent, abc.ABC):
    """The base implementation of a Q-table agent updating with IREs

    This acts in exactly the same way as a QTable Agent, except instead
    of updating towards the *actual* reward received, it keeps an
    ImmediateRewardEstimator for each action, and updates towards the
    *estimate* of the immediate next reward.

    This can be done pessimistically, or with the mean estimate,
    as-implemented by inheriting agents.
    """
    def __init__(self, num_actions, num_states, env, gamma, **kwargs):
        super().__init__(
            num_actions=num_actions, num_states=num_states, env=env,
            gamma=gamma, **kwargs)

        self.IREs = [
            ImmediateRewardEstimator(a) for a in range(self.num_actions)]

    def reset_estimators(self):
        """Reset all estimators for a base Q table agent with IREs"""
        super().reset_estimators()
        for ire in self.IREs:
            ire.reset()

    def update_estimators(self, mentor_acted=False, **kwargs):
        """Update Q Estimator and IREs"""

        if self.sampling_strategy == "whole_reset":
            self.reset_estimators()

        history_samples = self.sample_history(self.history)

        if mentor_acted:
            assert self.mentor is not None
            mentor_history_samples = self.sample_history(self.mentor_history)
            self.mentor_q_estimator.update(mentor_history_samples)

        # This does < batch_size updates on the IREs. For history-handling
        # purposes. Possibly sample batch_size per-action in the future.
        for IRE_index, IRE in enumerate(self.IREs):
            IRE.update(
                [(s, r) for s, a, r, _, _ in history_samples if IRE_index == a])

        self.q_estimator.update(history_samples)


class QTableMeanIREAgent(BaseQTableIREAgent):
    """Q Table agent that uses the expectation of IRE to update

    Instead of actual reward, use the mean estimate of the next reward.

    Used as a sanity-check for (mean) IRE functionality, without
    the transition-uncertainty update.
    """

    def __init__(self, num_actions, num_states, env, gamma, **kwargs):
        """Init the base agent and override the Q estimator"""
        super().__init__(
            num_actions=num_actions, num_states=num_states, env=env,
            gamma=gamma, **kwargs)

        if not hasattr(self, "q_estimator"):
            raise ValueError(
                "Hacky way to assert that q_estimator is overridden")
        self.q_estimator = QEstimatorIRE(
            quantile=None,  # indicates to use expectation
            immediate_r_estimators=self.IREs,
            num_states=self.num_states,
            num_actions=self.num_actions,
            gamma=self.gamma,
            lr=self.lr,
            has_mentor=self.mentor is not None,
            scaled=self.scale_q_value,
            horizon_type=self.horizon_type,
            num_horizons=self.num_horizons,
        )


class QTablePessIREAgent(BaseQTableIREAgent):
    """Q Table agent that uses *pessimistic* IRE to update

    Instead of actual reward, use the epistemically pessimistic estimate
    of the next reward.

    Used as a sanity-check for pessimistic IRE functionality, without
    the transition-uncertainty update.
    """

    def __init__(
            self, num_actions, num_states, env, gamma, quantile_i,
            init_to_zero=False, **kwargs
    ):
        """Init the base agent and override the Q estimator"""
        super().__init__(
            num_actions=num_actions, num_states=num_states, env=env,
            gamma=gamma, **kwargs)

        if not hasattr(self, "q_estimator"):
            raise ValueError(
                "Hacky way to assert that q_estimator is overridden")
        self.quantile_i = quantile_i
        self.q_estimator = QEstimatorIRE(
            quantile=QUANTILES[self.quantile_i],
            immediate_r_estimators=self.IREs,
            num_states=self.num_states,
            num_actions=self.num_actions,
            gamma=gamma,
            lr=self.lr,
            has_mentor=self.mentor is not None,
            scaled=self.scale_q_value,
            q_table_init_val=0. if init_to_zero else QUANTILES[self.quantile_i],
            horizon_type=self.horizon_type,
            num_horizons=self.num_horizons,
        )


class ContinuousAgent(BaseAgent, abc.ABC):
    """Tuned to the CartPole problem, at the moment"""

    def __init__(self, dim_states, burnin_n=BURN_IN_N, **kwargs):
        self.dim_states = dim_states
        self.burnin_n = burnin_n
        super().__init__(**kwargs)

    def learn(
            self, num_steps, report_every_n=500, render=1,
            reset_every_ep=False, early_stopping=0):
        """Let the agent loose in the environment, and learn!

        For continuous agents.

        Args:
            num_steps (int): Number of steps to learn for.
            report_every_n (int): Number of steps per reporting period
            render (int): Render mode 0, 1, 2
            reset_every_ep (bool): If true, resets state every
                reporting period, else continues e.g. infinite env.
            early_stopping (int): If sum(mentor_queries[-es:]) == 0,
                stop (and return True)

        Returns:
            True if stopped querying mentor for `early_stopping`
                reporting periods
            False if never stopped querying for all num_eps
            None if early_stopping = 0 (e.g. not-applicable)
        """

        if reset_every_ep:
            raise NotImplementedError("Not implemented reset_every_step")
        if early_stopping:
            raise NotImplementedError("Not implemented early stopping")
        if self.total_steps != 0:
            print("WARN: Agent already trained", self.total_steps)

        period_rewards = []  # initialise
        steps_last_fails = 0
        time_last = time.time()

        state = self.env.reset()
        while self.total_steps <= num_steps:
            action, mentor_acted = self.act(state)
            next_state, reward, done, _ = self.env.step(action)

            if self.debug_mode:
                print("Received reward", reward)

            self.store_history(
                state, action, reward, next_state, done, mentor_acted)
            period_rewards.append(reward)

            assert reward is not None, (
                f"Reward None at ({state}, {action})->{next_state}, "
                f"Mentor acted {mentor_acted}. Done {done}")
            if render > 0:
                self.env.render()

            if done:
                print(f"FAILED at {self.total_steps - steps_last_fails} - "
                      f"state transition:\n{state} ->\n{next_state}")
                # TODO or steps == max steps for env!
                #  Need some failure condition
                steps_last_fails = self.total_steps

                # Check if angular, then x displacement out of bounds
                if self.env.min_val is None:
                    min_x_val = -self.env.x_threshold
                    max_x_val = self.env.x_threshold
                    min_t_val = -self.env.theta_threshold_radians
                    max_t_val = self.env.theta_threshold_radians
                else:
                    min_x_val = min_t_val = self.env.min_val
                    max_x_val = max_t_val = 1.
                if not (min_t_val < next_state[2] < max_t_val):
                    self.failures += 1
                    print("Fell over - counting failure")
                elif not (min_x_val < next_state[0] < max_x_val):
                    print("Failed by outside x range - doesn't count to total")
                    invert = getattr(self, "invert_mentor", None)
                    if invert is not None:
                        self.invert_mentor = not invert
                        print(f"Inverting mentor to: {self.invert_mentor}")
                else:
                    raise RuntimeError(
                        f"Unexpected failure for cartpole!\n{state}")
                state = self.env.reset()
            else:
                state = next_state

            self.total_steps += 1

            if self.total_steps % self.update_n_steps == 0:
                self.update_estimators(debug=self.debug_mode)

            if self.total_steps > 1 and self.total_steps % report_every_n == 0:
                prev_queries = sum(self.mentor_queries_periodic)
                self.mentor_queries_periodic.append(
                    self.mentor_queries - prev_queries)
                period_rewards_sum = np.sum(np.stack(period_rewards))
                self.report(
                    num_steps, [period_rewards_sum], render_mode=render,
                    queries_last=self.mentor_queries_periodic[-1],
                    duration=time.time() - time_last)
                prev_failures = sum(self.failures_periodic)
                self.failures_periodic.append(
                    self.failures - prev_failures)
                self.rewards_periodic.append(period_rewards_sum)
                period_rewards = []  # reset
                time_last = time.time()

    def reset_estimators(self):
        raise NotImplementedError("Not yet implemented")


class MentorAgentGLN(ContinuousAgent):
    """An agent that provides a way to call the mentor at every timestep

    Simply does what the mentor does - so only implements act()
    (and dummy methods for the abstract methods)
    """
    def __init__(self, **kwargs):
        """Implement agent that just does what mentor would do

        No Q estimator required - always acts via `mentor` callable.
        """
        super().__init__(**kwargs)
        if self.mentor is None:
            raise ValueError("MentorAgent must have a mentor")

    def act(self, state):
        """Act, given the current state

        Returns:
             mentor_action (int): the action the mentor takes
             mentor_acted (bool): always True
        """
        return self.mentor(state), True

    def update_estimators(self, mentor_acted=False, **kwargs):
        """Nothing to do"""
        pass

    def store_history(
            self, state, action, reward, next_state, done, mentor_acted=True):
        """Nothing to do"""
        pass


class ContinuousPessimisticAgentGLN(ContinuousAgent):
    """Agent that can act in a continuous, multidimensional state space.

    Uses GGLNs as function approximators for the IRE estimators,
    the Q estimators and the mentor Q estimators.
    """

    def __init__(
            self,
            dim_states,
            quantile_i,
            train_all_q=False,
            init_to_zero=False,
            q_init_func=QuantileQEstimatorGaussianGLN,
            invert_mentor=None,
            **kwargs
    ):
        """Initialise function for a base agent

        Args (additional to base):
            mentor: a function taking (state, kwargs), returning an
                integer action.
            quantile_i: the index of the quantile from QUANTILES to use
                for taking actions.

            eps_max: initial max value of the random query-factor
            eps_min: the minimum value of the random query-factor.
                Once self.epsilon < self.eps_min, it stops reducing.
        """
        if init_to_zero:
            raise NotImplementedError("Only implemented for quantile burn in")
        if kwargs.get("update_n_steps", 2) == 1:
            print("Warning: not using batches for GLN learning")

        super().__init__(dim_states=dim_states, **kwargs)

        self.quantile_i = quantile_i

        self.Q_val_temp = 0.
        self.mentor_Q_val_temp = 0.

        self._train_all_q = train_all_q
        self._q_init_func = q_init_func
        self.make_estimators()
        self.update_calls = 0

        self.jax_random = JaxRandom()

        self.invert_mentor = invert_mentor
        self.mentor_history = ReplayMemory(10000)
        self.history = [ReplayMemory(10000) for _ in range(self.num_actions)]

    def store_history(
            self, state, action, reward, next_state, done, mentor_acted=False):
        """Bin sars tuples into action-specific history, and/or mentor"""
        if mentor_acted:
            self.mentor_history.push(state, action, reward, next_state, done)
        self.history[action].push(state, action, reward, next_state, done)

    def sample_history(
            self, history, strategy=None, batch_size=None, actions=None):
        """Not used in GLNs"""
        NotImplementedError()

    def make_estimators(self):
        # Create the estimators
        self.IREs = [
            ImmediateRewardEstimatorGaussianGLN(
                action=a,
                input_size=self.dim_states,
                lr=self.lr,
                feat_mean=self.env.mean_val,
                burnin_n=self.burnin_n,
                layer_sizes=[64, 64, 1],
                context_dim=GLN_CONTEXT_DIM,
                batch_size=self.batch_size,
                burnin_val=0.
            ) for a in range(self.num_actions)
        ]

        self.QEstimators = [
            self._q_init_func(
                quantile=q,
                immediate_r_estimators=self.IREs,
                dim_states=self.dim_states,
                num_actions=self.num_actions,
                gamma=self.gamma,
                layer_sizes=[64, 64, 1],
                context_dim=GLN_CONTEXT_DIM,
                feat_mean=self.env.mean_val,
                lr=self.lr,
                burnin_n=self.burnin_n,
                burnin_val=None,
                batch_size=self.batch_size,
            ) for i, q in enumerate(QUANTILES) if (
                i == self.quantile_i or self._train_all_q)
        ]

        self.q_estimator = self.QEstimators[
            self.quantile_i if self._train_all_q else 0]

        self.mentor_q_estimator = MentorQEstimatorGaussianGLN(
            self.dim_states,
            self.num_actions,
            self.gamma,
            lr=self.lr,
            feat_mean=self.env.mean_val,
            layer_sizes=[64, 64, 1],
            context_dim=GLN_CONTEXT_DIM,
            burnin_n=self.burnin_n,
            init_val=1.,
            batch_size=self.batch_size)

    def act(self, state):
        state = jax.lax.stop_gradient(state)
        values = jnp.asarray([
            jnp.squeeze(
                self.q_estimator.estimate(
                    jnp.expand_dims(state, 0), action=a))
            for a in range(self.num_actions)])

        if self.debug_mode:
            print("Q Est values", values)
        assert not jnp.any(jnp.isnan(values)), (values, state)

        # Choose randomly from any jointly-maximum values
        max_vals = (values == values.max())
        proposed_action = self.jax_random.choice(jnp.flatnonzero(max_vals))

        if self.mentor is None:
            mentor_acted = False
            if self.jax_random.uniform() < self.epsilon():
                action = self.jax_random.randint(
                    (1,), minval=0, maxval=self.num_actions)
            else:
                action = proposed_action
        else:
            # Defer if predicted value < min, based on r > eps
            mentor_value = self.mentor_q_estimator.estimate(
                jnp.expand_dims(state, 0))
            mentor_pref_magnitude = (mentor_value - values[proposed_action])
            scaled_min_r = self.min_reward
            if not self.scale_q_value:
                scaled_min_r /= (1. - self.gamma)
                mentor_pref_magnitude *= (1. - self.gamma)  # scale down
            self.mentor_Q_val_temp = mentor_value
            eps_wins, eps_val = self.use_epsilon(mentor_pref_magnitude)
            prefer_mentor = not eps_wins
            agent_value_too_low = values[proposed_action] <= scaled_min_r
            if self.debug_mode:
                print(f"Agent value={values[proposed_action]:.4f}")
                print(f"Mentor value={mentor_value[0]:.4f} - "
                      f"eps={eps_val:.4f} "
                      + ("not scaled " if not self.scale_q_value else " ") +
                      f"query={agent_value_too_low or prefer_mentor}")
            if agent_value_too_low or prefer_mentor:
                action = self.mentor(state, invert=self.invert_mentor)
                mentor_acted = True
                self.mentor_queries += 1
            else:
                action = proposed_action
                mentor_acted = False

        # return int(action), mentor_acted
        return action, mentor_acted

    def update_estimators(
            self, debug=False, sample_converge=True, perturb=False):
        """Update all estimators with a random batch of the histories.

        1) Mentor-Q Estimator
        2) ImmediateRewardEstimators (currently only for the actions in
           the sampled batch that corresponds with the IRE).
        3) Q-estimator (for every quantile)

        Args:
            debug (bool): print more output
            sample_converge (bool): whether to pass this history as
                convergence data for the update
            perturb (bool): after 10, 100 steps, discontinuously move
                the agent (usually for pseudocount experimentation)
        """
        if debug:
            print(f"\nUPDATE CALL {self.update_calls}\n")
            print("Updating Mentor Q Estimator...")
        stacked_mentor_batch = self.mentor_history.sample_arrays(
            self.batch_size)
        self.mentor_q_estimator.update(stacked_mentor_batch, debug=debug)

        for a in range(self.num_actions):
            if debug:
                print(f"Sampling for updates to action {a} estimators")
            stacked_batch = self.history[a].sample_arrays(self.batch_size)

            if debug:
                print(f"Updating IRE {a}...")
            self.IREs[a].update((stacked_batch.state, stacked_batch.reward))

            # Update each quantile estimator being kept...
            for n, q_estimator in enumerate(self.QEstimators):
                if debug:
                    print(f"Updating Q estimator {n} action {a}...")
                # use stack_batch as deque is not valid jax type for jitting
                # TODO - this can return history lengths < batch size
                q_estimator.update(
                    stacked_batch,
                    update_action=a,
                    convergence_data=(
                        self.history[a].all_as_arrays(
                            self.batch_size,
                            retain_all=self.history[a].count < self.batch_size)
                        if sample_converge else Transition(*([None] * 5))),
                    debug=self.debug_mode)

        self.update_calls += 1

    def additional_printing(self, render_mode):
        if render_mode > 1:
            ires = []
            preds = []
            samples = self.history[0].sample_arrays(10)
            for a in range(self.num_actions):
                ires.append(self.IREs[a].estimate(samples.state))
                preds.append(
                    self.q_estimator.estimate(samples.state, action=a))
            ires = jnp.stack(ires).T
            preds = jnp.stack(preds).T

            mentor_q = self.mentor_q_estimator.estimate(samples.state)
            print("State\t\t\t\t\t\t  -> IRE,\t\t\tQ estimates,\t\t"
                  "Mentor Q value\tReward")
            for i in range(samples.state.shape[0]):
                print(f"{samples.state[i]} -> {ires[i]}, {preds[i]}, "
                      f"{mentor_q[i]}, {samples.reward[i]}")

            q_lrs = [
                self.q_estimator.lr for a in range(self.num_actions)]
            print(f"Q lr {q_lrs}")
            print(f"M lr {self.mentor_q_estimator.lr}")
            print(f"IRE lr "
                  f"{[self.IREs[a].lr for a in range(self.num_actions)]}")

            if self.horizon_type == "finite":
                print(f"Finite horizons for s={samples.state[0].numpy()}")
                lst = []
                for h in range(self.num_horizons + 1):
                    h_pred = [
                        self.q_estimator.estimate(
                            samples.state[:1], action=a, h=h)
                        for a in range(self.num_actions)]
                    lst.append(f"horizon {h}: {', '.join(h_pred)}")
                print("\n".join(lst))


class ContinuousPessimisticAgentBayes(ContinuousAgent):
    """Agent that uses BayesByBackprop for uncertainty estimates

    Acts in continuous state spaces.
    """

    def __init__(
            self,
            dim_states,
            quantile_i,
            train_all_q=False,
            init_to_zero=False,
            net_type=bayes_by_backprop.BBBNet,
            invert_mentor=None,
            **kwargs
    ):
        """Initialise function for a Bayesian agent"""
        if init_to_zero:
            raise NotImplementedError("Only implemented for quantile burn in")
        if kwargs.get("update_n_steps", 2) == 1:
            print("Warning: not using batches for GLN learning")
        dropout_rate = kwargs.pop("dropout_rate", 0.5)

        super().__init__(dim_states=dim_states, **kwargs)
        self.net_type = net_type
        self.quantile_i = quantile_i

        self._train_all_q = train_all_q

        self.update_calls = 0
        self.invert_mentor = invert_mentor
        self.ire = None
        self.QEstimators = None
        self.q_estimator = None
        self.mentor_q_estimator = None
        self.make_estimators(dropout_rate=dropout_rate)

        if self.scale_q_value and self.horizon_type == "finite":
            raise ValueError()

    def store_history(
            self, state, action, reward, next_state, done, mentor_acted=False):
        """Bin sars tuples into action-specific history, and/or mentor"""
        sars = (
            tc.as_tensor(state, dtype=tc.float),
            tc.as_tensor([action], dtype=tc.int64),
            tc.as_tensor([reward], dtype=tc.float),
            tc.as_tensor(next_state, dtype=tc.float),
            tc.as_tensor([done], dtype=tc.bool))
        if mentor_acted:
            self.mentor_history.append(sars)
        self.history.append(sars)

    def make_estimators(self, **net_kwargs):
        if self.num_horizons > 1:
            assert (self.horizon_type == "finite")
        # Create the estimators
        self.ire = ImmediateRewardEstimatorBayes(
            num_actions=self.num_actions,
            input_size=2,  # x, v only
            lr=self.lr,
            lr_steps=500,
            lr_gamma=0.9,
            feat_mean=self.env.mean_val,
            burnin_n=self.burnin_n,
            batch_size=self.batch_size,
            burnin_val=QUANTILES[self.quantile_i],
            net_init_func=self.net_type,
            scaled=self.scale_q_value,
            **{**net_kwargs, **{"hidden_sizes": [128, 32] * 4}},
        )

        self.QEstimators = [
            QuantileQEstimatorBayes(
                quantile=q,
                immediate_r_estimator=self.ire,
                dim_states=self.dim_states,
                num_actions=self.num_actions,
                num_steps=self.num_horizons,
                horizon_type=self.horizon_type,
                gamma=self.gamma,
                feat_mean=self.env.mean_val,
                lr=self.lr,
                lr_steps=500,
                lr_gamma=0.9,
                burnin_n=self.burnin_n,
                burnin_val=QUANTILES[self.quantile_i],
                batch_size=self.batch_size,
                net_init=self.net_type,
                scaled=self.scale_q_value,
                **{**net_kwargs, **{"hidden_sizes": [128, 32] * 4}},
            ) for i, q in enumerate(QUANTILES) if (
                i == self.quantile_i or self._train_all_q)
        ]

        self.q_estimator = self.QEstimators[
            self.quantile_i if self._train_all_q else 0]

        mentor_kwargs = {
            k: v for k, v in net_kwargs.items() if k != "dropout_rate"}
        mentor_kwargs.update({"hidden_sizes": (128, 128, 128)})
        self.mentor_q_estimator = MentorQEstimatorBayes(
            self.dim_states,
            self.gamma,
            num_horizons=self.num_horizons,
            horizon_type=self.horizon_type,
            lr=self.lr * 1.1,
            lr_steps=500,
            lr_gamma=0.9,
            feat_mean=self.env.mean_val,
            burnin_n=self.burnin_n,
            init_val=1.,
            batch_size=self.batch_size,
            net_type=self.net_type,
            scaled=self.scale_q_value,
            dropout_rate=0.05,
            **mentor_kwargs
        )

    def reset_estimators(self):
        self.make_estimators()

    @tc.no_grad()
    def act(self, state):
        state_tensor = tc.unsqueeze(tc.as_tensor(state, dtype=tc.float), 0)
        values = tc.squeeze(self.q_estimator.estimate(state_tensor), 0).numpy()

        if self.debug_mode:
            print("Q Est values", values)
        assert not np.any(np.isnan(values)), (values, state)

        # Choose randomly from any jointly-maximum values
        max_vals = (values == values.max())
        proposed_action = np.random.choice(np.flatnonzero(max_vals))

        if self.mentor is None:
            mentor_acted = False
            if np.random.rand() < self.epsilon():
                action = np.random.randint(
                    size=(1,), low=0,
                    high=self.num_actions)
            else:
                action = proposed_action
        else:
            # Defer if predicted value < min, based on r > eps
            mentor_value = tc.squeeze(
                self.mentor_q_estimator.estimate(state_tensor), 0)
            mentor_pref_magnitude = (mentor_value - values[proposed_action])
            scaled_min_r = self.min_reward
            if not self.scale_q_value:
                scaled_min_r = geometric_sum(
                    self.min_reward, self.gamma, self.num_horizons)
            eps_wins, eps_val = self.use_epsilon(mentor_pref_magnitude)
            prefer_mentor = not eps_wins
            agent_value_too_low = values[proposed_action] <= scaled_min_r
            if self.debug_mode:
                print(f"Agent value={values[proposed_action]:.4f}")
                print(f"Mentor value={mentor_value[0]:.4f} - "
                      f"eps={eps_val:.4f} "
                      + ("not scaled " if not self.scale_q_value else " ") +
                      f"query={agent_value_too_low or prefer_mentor}")
            if agent_value_too_low or prefer_mentor:
                action = self.mentor(state, invert=self.invert_mentor)
                mentor_acted = True
                self.mentor_queries += 1
            else:
                action = proposed_action
                mentor_acted = False

        return int(action), mentor_acted

    def update_estimators(self, debug=False, sample_converge=True):
        """Update all estimators with a random batch of the histories.

        Mentor-Q Estimator
        ImmediateRewardEstimators (currently only for the actions in the
            sampled batch that corresponds with the IRE).
        Q-estimator (for every quantile)
        """
        if debug:
            print(f"\nUPDATE CALL {self.update_calls}\n")
            print("Updating Mentor Q Estimator...")
        mentor_history_samples = self.sample_history(self.mentor_history)
        self.mentor_q_estimator.update(mentor_history_samples, debug=debug)

        history_samples = self.sample_history(self.history)
        stacked_batch = stack_batch(history_samples, lib=tc)
        sts, acts, rs, _, _ = stacked_batch
        self.ire.update((sts[:, 0:2], acts, rs), debug=debug)

        # Update each quantile estimator being kept...
        for n, q_estimator in enumerate(self.QEstimators):
            if debug:
                print(f"Updating Q estimator {n}...")
            q_estimator.update(stacked_batch, debug=self.debug_mode)
        self.update_calls += 1

    def additional_printing(self, render_mode):
        if render_mode > 1:
            samples = self.sample_history(self.history, batch_size=10)
            sample_states, _, sample_rs, _, _ = stack_batch(samples, lib=tc)
            ires = self.ire.estimate(sample_states[:, 0:2])  # x, v only
            preds = self.q_estimator.estimate(sample_states).squeeze()
            mentor_q = self.mentor_q_estimator.estimate(sample_states)
            print("State\t\t\t\t\t\t  -> IRE,\t\t\tQ estimates,\t\t"
                  "Mentor Q value\tReward")
            for i in range(sample_states.shape[0]):
                print(f"{sample_states[i].numpy()} -> {ires[i].numpy()}, "
                      f"{preds[i].numpy()}, {mentor_q[i].numpy()}, "
                      f"{sample_rs[i].numpy()}")

            def get_lr(estimator):
                if estimator.model.lr_schedule is not None:
                    return estimator.model.lr_schedule.get_last_lr()
                else:
                    return estimator.model.lr

            print(f"Q lr {get_lr(self.q_estimator)}")
            print(f"M lr {get_lr(self.mentor_q_estimator)}")
            print(f"IRE lr {get_lr(self.ire)}")
            if self.horizon_type == "finite":
                print(f"Finite horizons for s={sample_states[0].numpy()}")
                lst = []
                for h in range(self.num_horizons + 1):
                    h_pred = self.q_estimator.estimate(
                        sample_states[:1], h=h)
                    lst.append(f"horizon {h}: {h_pred.squeeze().numpy()}")
                print("\n".join(lst))
