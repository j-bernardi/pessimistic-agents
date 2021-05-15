import abc
import numpy as np
from collections import deque

from estimators import (
    ImmediateRewardEstimator, MentorQEstimator,
    MentorFHTDQEstimator,
    ImmediateRewardEstimator_GLN_gaussian,
    QuantileQEstimator_GLN_gaussian,
    MentorQEstimator_GLN_gaussian
)

from q_estimators import (
    QuantileQEstimator, BasicQTableEstimator, QEstimatorIRE, QTableEstimator,
)
from utils import geometric_sum

QUANTILES = [2**k / (1 + 2**k) for k in range(-5, 5)]


class BaseAgent(abc.ABC):
    """Abstract implementation of an agent

    Useful as it saves state and variables that are shared by all
    agents. Also specifies the abstract methods that all agents need
    to be valid with learn() - which is the same for all agents.
    """
    def __init__(
            self,
            num_actions,
            num_states,
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
            min_reward=1e-6,
            horizon_type="inf",
            num_steps=1,
            track_transitions=None,
    ):
        """Initialise the base agent with shared params

            num_actions:
            num_states:
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
            min_reward (float):
            horizon_type (str): one of "finite" or "inf"
            num_steps (int): number of time steps to look into the
                future for, when calculating horizons
            track_transitions (list): List of transition definitions
                (state, action, next_state), where any can be None to
                indicate "any". Tracks number of times the transition is
                observed. Keeps dict of transitions[s][a][s'] = N_(s,a,s')
        """
        self.num_actions = num_actions
        self.num_states = num_states
        self.env = env
        self.gamma = gamma
        if "whole" in sampling_strategy and batch_size != 1:
            print("WARN: Default not used for BS, but sampling whole history")
        self.sampling_strategy = sampling_strategy
        self.lr = lr
        self.batch_size = batch_size
        self.update_n_steps = update_n_steps
        self.eps_max = eps_max
        self.eps_min = eps_min
        self.scale_q_value = scale_q_value
        self.mentor = mentor
        self.min_reward = min_reward

        self.horizon_type = horizon_type
        self.num_steps = num_steps

        self.q_estimator = None
        self.mentor_q_estimator = None

        self.mentor_queries = 0
        self.total_steps = 0
        self.failures = 0

        self.mentor_queries_periodic = []
        self.rewards_periodic = []
        self.failures_periodic = []

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
            self, num_steps, report_every_n=500, render=1, reset_every_ep=False,
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
            if self.total_steps % report_every_n == 0:
                queries = self.mentor_queries_periodic[-1]\
                    if self.mentor_queries_periodic else -1
                self.report(
                    num_steps, period_rewards, render_mode=render,
                    queries_last=queries)
                if reset_every_ep:
                    state = int(self.env.reset())
                period_rewards = []  # reset

            action, mentor_acted = self.act(state)
            next_state, reward, done, _ = self.env.step(action)

            _ = self.track_transition(
                state, action, next_state, mentor_acted=mentor_acted)
            assert reward is not None, (
                f"Reward None at ({state}, {action})->{next_state}, "
                f"Mentor acted {mentor_acted}. Done {done}")

            period_rewards.append(reward)
            next_state = int(next_state)

            if render > 0:
                # First rendering should not return N lines
                self.env.render(in_loop=self.total_steps > 0)

            self.store_history(
                state, action, reward, next_state, done, mentor_acted)

            if self.total_steps % self.update_n_steps == 0:
                self.update_estimators(mentor_acted=mentor_acted)

            state = next_state
            if done:
                self.failures += 1
                # print('failed')
                state = int(self.env.reset())

            if self.total_steps % report_every_n == 0:
                if self.total_steps == 0:
                    self.mentor_queries_periodic.append(self.mentor_queries)
                    self.failures_periodic.append(self.failures)
                else:
                    self.mentor_queries_periodic.append(
                        self.mentor_queries
                        - np.sum(self.mentor_queries_periodic))
                    self.failures_periodic.append(
                        self.failures - np.sum(self.failures_periodic))
                self.rewards_periodic.append(sum(period_rewards))

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

    @abc.abstractmethod
    def act(self, state):
        raise NotImplementedError("Must be implemented per agent")

    @abc.abstractmethod
    def store_history(
            self, state, action, reward, next_state, done, mentor_acted):
        raise NotImplementedError("Must be implemented per agent")

    @abc.abstractmethod
    def update_estimators(self, mentor_acted=False):
        raise NotImplementedError()

    def epsilon(self, reduce=True):
        """Reduce the max value of, and return, the random var epsilon."""

        if self.eps_max > self.eps_min and reduce:
            self.eps_max *= 0.999

        return self.eps_max * np.random.rand()

    def sample_history(self, history):
        """Return a sample of the history

        Uses:
            self.sampling_strategy (
                Choice(["random", "last_n_steps", "whole"])): See defs
        """

        if self.sampling_strategy == "random":
            idxs = np.random.randint(
                low=0, high=len(history), size=self.batch_size)

        elif self.sampling_strategy == "last_n_steps":
            assert self.batch_size == self.update_n_steps
            idxs = range(-self.batch_size, 0)

        elif "whole" in self.sampling_strategy:
            return history

        else:
            raise ValueError(
                f"Sampling strategy {self.sampling_strategy} invalid")

        return [history[i] for i in idxs]

    def report(
            self, tot_steps, rewards_last, render_mode, queries_last=None
    ):
        """Reports on period of steps and calls any additional printing

        Args:
            tot_steps (int): total number of steps expected
            rewards_last (list): list of rewards from last reporting
                period
            render_mode (int): defines verbosity of rendering
            queries_last (int): number of mentor queries in the last
                reporting period (i.e. the one being reported).
        """
        if render_mode < 0 or self.total_steps <= 0:
            return
        if render_mode > 0:
            print(self.env.get_spacer())
        report = (
            f"Step {self.total_steps} / {tot_steps} : "
            f"{100 * self.total_steps / tot_steps:.0f}%) - F {self.failures} - "
            f"R (last ep) {(sum(rewards_last) if rewards_last else '-'):.0f}")
        if queries_last is not None:
            report += f" - M (last ) {queries_last}"

        print(report)
        self.additional_printing(render_mode)

        if render_mode > 0:
            print("\n" * (self.env.state_shape[0] - 1))

    def additional_printing(self, render_mode):
        """Defines class-specific printing"""
        return None


class MentorAgent(BaseAgent):
    """An agent that provides a way to call the mentor at every timestep

    Simply does what the mentor does - so only implements act()
    (and dummy methods for the abstract methods)
    """
    def __init__(self, num_actions, num_states, env, gamma, mentor, **kwargs):
        """Implement agent that just does what mentor would do

        No Q estimator required - always acts via `mentor` callable.
        """
        if mentor is None:
            raise ValueError("MentorAgent must have a mentor")
        super().__init__(
            num_actions=num_actions, num_states=num_states, env=env,
            gamma=gamma, mentor=mentor, **kwargs
        )

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

    def update_estimators(self, mentor_acted=False):
        """Nothing to do"""
        pass

    def store_history(
            self, state, action, reward, next_state, done, mentor_acted):
        """Nothing to do"""
        pass


class BaseQAgent(BaseAgent, abc.ABC):
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
            horizon_steps = self.q_estimator.num_steps\
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
            prefer_mentor = mentor_value > values[agent_max_action] + scaled_eps

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


class PessimisticAgent(BaseQAgent):
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
        """
        super().__init__(
            num_actions=num_actions, num_states=num_states, env=env,
            gamma=gamma, mentor=mentor, **kwargs
        )

        if self.mentor is None:
            raise NotImplementedError("Pessimistic agent requires a mentor")

        self.quantile_i = quantile_i

        self.history = deque(maxlen=10000)
        self.mentor_history = deque(maxlen=10000)

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
                q_table_init_val=0. if init_to_zero else QUANTILES[q],
                horizon_type=self.horizon_type,
                num_steps=self.num_steps,
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
                lr=self.lr, num_steps=self.num_steps, scaled=self.scale_q_value)

    def reset_estimators(self):
        for ire in self.IREs:
            ire.reset()
        for q_est in self.QEstimators:
            q_est.reset()
        self.mentor_q_estimator.reset()

    def store_history(
            self, state, action, reward, next_state, done, mentor_acted):

        if mentor_acted:
            self.mentor_history.append(
                (state, action, reward, next_state, done))

        self.history.append((state, action, reward, next_state, done))

    def update_estimators(self, mentor_acted=False):
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

        for q_estimator in self.QEstimators:
            q_estimator.update(history_samples)

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


class BaseQTableAgent(BaseQAgent, abc.ABC):
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
            num_steps=self.num_steps
        )

        if not self.scale_q_value:
            # geometric sum of max rewards per step (1.) for N steps (up to inf)
            init_mentor_val = geometric_sum(
                1., self.gamma, self.q_estimator.num_steps)
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
                num_steps=self.num_steps)

        self.history = deque(maxlen=10000)
        if self.mentor is not None:
            self.mentor_history = deque(maxlen=10000)
        else:
            self.mentor_history = None

    def reset_estimators(self):
        self.q_estimator.reset()
        if self.mentor_q_estimator is not None:
            self.mentor_q_estimator.reset()

    def update_estimators(self, mentor_acted=False):

        if self.sampling_strategy == "whole_reset":
            self.reset_estimators()

        history_samples = self.sample_history(self.history)
        self.q_estimator.update(history_samples)

        if mentor_acted:
            assert self.mentor is not None
            mentor_history_samples = self.sample_history(self.mentor_history)
            self.mentor_q_estimator.update(mentor_history_samples)

    def store_history(
            self, state, action, reward, next_state, done, mentor_acted=False):
        if mentor_acted:
            assert self.mentor is not None
            self.mentor_history.append((state, action, reward, next_state, done))

        self.history.append((state, action, reward, next_state, done))

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

    def update_estimators(self, mentor_acted=False):
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
            num_steps=self.num_steps,
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
            num_steps=self.num_steps,
        )


class FinitePessimisticAgent_GLNIRE(BaseAgent):
    """Agent that solves the finite case with GLN function approximator

    Implements the pessimistic algorithm, but with a function
    approximator.
    """

    def __init__(
            self,
            num_actions,
            dim_states,
            env,
            gamma,
            mentor,
            quantile_i,
            burnin_n=2,
            train_all_q=False,
            init_to_zero=False,
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
        super().__init__(
            num_actions=num_actions, num_states=None, env=env,
            gamma=gamma, mentor=mentor, **kwargs
        )
        if init_to_zero:
            raise NotImplementedError("Only implemented for quantile burn in")

        self.quantile_i = quantile_i
        self.dim_states = dim_states

        self.history = deque(maxlen=10000)
        self.mentor_history = deque(maxlen=10000)

        self.Q_val_temp = 0.
        self.mentor_Q_val_temp = 0.

        # Create the estimators
        default_layer_sizes = [4] * 16 + [1]
        self.IREs = [
            ImmediateRewardEstimator_GLN_gaussian(
                a, lr=self.lr, burnin_n=burnin_n,
                layer_sizes=default_layer_sizes, context_dim=4
            ) for a in range(num_actions)
        ]

        self.QEstimators = [
            QuantileQEstimator_GLN_gaussian(
                q, self.IREs, dim_states, num_actions, gamma,
                layer_sizes=default_layer_sizes,
                context_dim=4,
                lr=self.lr,
                burnin_n=burnin_n
            ) for i, q in enumerate(QUANTILES) if (
                i == self.quantile_i or train_all_q)
        ]

        self.q_estimator = self.QEstimators[
            self.quantile_i if train_all_q else 0]

        self.mentor_q_estimator = MentorQEstimator_GLN_gaussian(
            dim_states, num_actions, gamma, lr=self.lr,
            layer_sizes=default_layer_sizes, context_dim=4, burnin_n=burnin_n,
            init_val=1.)

    def reset_estimators(self):
        raise NotImplementedError("Not yet implemented")

    def act(self, state):
        values = np.array([
            self.q_estimator.estimate(state, action_i)
            for action_i in range(self.num_actions)
        ])

        values = np.nan_to_num(values)

        # Choose randomly from any jointly maximum values
        max_vals = values == values.max()
        proposed_action = int(np.random.choice(np.flatnonzero((max_vals))))
        self.Q_val_temp = values[proposed_action]
        action = proposed_action

        if self.mentor is None:
            if np.random.rand() < self.epsilon():
                action = np.random.randint(self.num_actions)
            mentor_acted = False
        else:
            # Defer if predicted value < min, based on r > eps
            scaled_min_r = self.min_reward
            eps = self.epsilon()
            if not self.scale_q_value:
                scaled_min_r /= (1. - self.gamma)
                eps /= (1. - self.gamma)
            mentor_value = self.mentor_q_estimator.estimate(state)
            self.mentor_Q_val_temp = mentor_value
            prefer_mentor = mentor_value > (values[proposed_action] + eps)
            agent_value_too_low = values[proposed_action] <= scaled_min_r
            if agent_value_too_low or prefer_mentor:
                state_for_mentor = state * 3.5 + 3.5
                action = self.env.map_grid_act_to_int(
                    self.mentor(state_for_mentor,
                        kwargs={'state_shape': self.env.state_shape})
                )
                mentor_acted = True
                # print('called mentor')
                self.mentor_queries += 1
            else:
                action = proposed_action
                mentor_acted = False

        return action, mentor_acted

    def store_history(
            self, state, action, reward, next_state, done, mentor_acted=False):

        if mentor_acted:
            self.mentor_history.append(
                (state, action, reward, next_state, done))

        self.history.append((state, action, reward, next_state, done))

    def update_estimators(self, mentor_acted=False):
        """Update all estimators with a random batch of the histories.

        Mentor-Q Estimator
        ImmediateRewardEstimators (currently only for the actions in the
            sampled batch that corresponds with the IRE).
        Q-estimator (for every quantile)
        """
        if mentor_acted and self.batch_size <= len(self.mentor_history):
            mentor_history_samples = self.sample_history(
                self.mentor_history)
            self.mentor_q_estimator.update(mentor_history_samples)

        history_samples = self.sample_history(self.history)

        # This does < batch_size updates on the IREs. For history-handling
        # purposes. Possibly sample batch_size per-action in the future.
        for IRE_index, IRE in enumerate(self.IREs):
            IRE.update(
                [(s, r) for s, a, r, _, _ in history_samples if IRE_index == a])

        for q_estimator in self.QEstimators:
            q_estimator.update(history_samples)

    def additional_printing(self, render):
        super().additional_printing(render_mode=render)
        if render and self.mentor is not None:
            print(f"M {self.mentor_queries} ")
        if render > 1:

            if self.mentor is None:

                if self.q_estimator.lr is not None:
                    print(
                        f"Learning rates: "
                        f"QEst {self.q_estimator.lr:.4f}")
            else:
                print("Additional for finite pessimistic")
                if np.isnan(self.Q_val_temp):
                    print('Q VAL IS NAN')
                else:
                    print(f"Q val\n{self.Q_val_temp}")
                print(f"mentor Q val\n{self.mentor_Q_val_temp}")
                if self.q_estimator.lr is not None:
                    print(
                        f"Learning rates: "
                        f"QEst {self.q_estimator.lr:.4f}"
                        f"Mentor V {self.mentor_q_estimator.lr:.4f}")

    def learn(self, num_steps, report_every_n=500, render=1):

        if self.total_steps != 0:
            print("WARN: Agent already trained", self.total_steps)
        period_reward = []  # initialise
        state = self.env.map_int_to_grid(int(self.env.reset()))/3.5-1
        for step in range(num_steps):
            if step % report_every_n == 0:
                self.report(num_steps, period_reward, render_mode=render)
                # state = self.env.map_int_to_grid(int(self.env.reset()))/3.5-1
                period_reward = []  # reset

            action, mentor_acted = self.act(state)
            next_state_int, reward, done, _ = self.env.step(action)
            next_state = self.env.map_int_to_grid(next_state_int)/3.5-1
            period_reward.append(reward)

            if render:
                # First rendering should not return N lines
                self.env.render(in_loop=self.total_steps > 0)

            self.store_history(
                state, action, reward, next_state, done, mentor_acted)

            self.total_steps += 1

            if self.total_steps % self.update_n_steps == 0:
                self.update_estimators(mentor_acted=mentor_acted)

            state = next_state[:]
            if done:
                self.failures += 1
                # print('failed')
                state = self.env.map_int_to_grid(int(self.env.reset()))/3.5-1

            if step % report_every_n == 0:
                if step == 0:
                    self.mentor_queries_periodic.append(self.mentor_queries)
                else:
                    self.mentor_queries_periodic.append(
                        self.mentor_queries
                        - np.sum(self.mentor_queries_periodic))


class ContinuousPessimisticAgent_GLN(BaseAgent):
    """Agent that can act in a continuous, multidimensional state space.

    Uses GGLNs as function approximators for the IRE estimators,
    the Q estimators and the mentor Q estimators.
    """

    def __init__(
            self,
            num_actions,
            dim_states,
            env,
            gamma,
            mentor,
            quantile_i,
            burnin_n=1000,
            train_all_q=False,
            init_to_zero=False,
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
        super().__init__(
            num_actions=num_actions, num_states=None, env=env,
            gamma=gamma, mentor=mentor, update_n_steps=100, batch_size=100, **kwargs
        )
        if init_to_zero:
            raise NotImplementedError("Only implemented for quantile burn in")

        self.quantile_i = quantile_i
        self.dim_states = dim_states

        self.history = deque(maxlen=10000)
        self.mentor_history = deque(maxlen=10000)

        self.Q_val_temp = 0.
        self.mentor_Q_val_temp = 0.

        print('USING CONTINUOUS AGENT')
        # Create the estimators
        default_layer_sizes = [4] * 4 + [1]
        self.IREs = [
            ImmediateRewardEstimator_GLN_gaussian(
                a, input_size=self.dim_states, lr=self.lr, burnin_n=burnin_n,
                layer_sizes=default_layer_sizes, context_dim=4
            ) for a in range(num_actions)
        ]

        self.QEstimators = [
            QuantileQEstimator_GLN_gaussian(
                q, self.IREs, dim_states, num_actions, gamma,
                layer_sizes=default_layer_sizes, context_dim=4,
                lr=self.lr, burnin_n=burnin_n, burnin_val=None
            ) for i, q in enumerate(QUANTILES) if (
                i == self.quantile_i or train_all_q)
        ]

        self.q_estimator = self.QEstimators[
            self.quantile_i if train_all_q else 0]

        self.mentor_q_estimator = MentorQEstimator_GLN_gaussian(
            dim_states, num_actions, gamma, lr=self.lr,
            layer_sizes=default_layer_sizes, context_dim=4, burnin_n=burnin_n,
            init_val=1.)

    def reset_estimators(self):
        raise NotImplementedError("Not yet implemented")

    def act(self, state):
        values = np.array([
            self.q_estimator.estimate(state, action_i)
            for action_i in range(self.num_actions)
        ])

        values = np.nan_to_num(values)

        # Choose randomly from any jointly maximum values
        max_vals = values == values.max()
        proposed_action = int(np.random.choice(np.flatnonzero((max_vals))))
        self.Q_val_temp = values[proposed_action]
        action = proposed_action

        if self.mentor is None:
            if np.random.rand() < self.epsilon():
                action = np.random.randint(self.num_actions)
            mentor_acted = False
        else:
            # Defer if predicted value < min, based on r > eps
            scaled_min_r = self.min_reward
            eps = self.epsilon()
            if not self.scale_q_value:
                scaled_min_r /= (1. - self.gamma)
                eps /= (1. - self.gamma)
            mentor_value = self.mentor_q_estimator.estimate(state)
            self.mentor_Q_val_temp = mentor_value
            prefer_mentor = mentor_value > (values[proposed_action] + eps)
            agent_value_too_low = values[proposed_action] <= scaled_min_r
            if agent_value_too_low or prefer_mentor:

                action = self.mentor(state)

                mentor_acted = True
                # print('called mentor')
                self.mentor_queries += 1
            else:
                action = proposed_action
                mentor_acted = False

        return action, mentor_acted

    def store_history(
            self, state, action, reward, next_state, done, mentor_acted=False):

        if mentor_acted:
            self.mentor_history.append(
                (state, action, reward, next_state, done))

        self.history.append((state, action, reward, next_state, done))

    def update_estimators(self, mentor_acted=False):
        """Update all estimators with a random batch of the histories.

        Mentor-Q Estimator
        ImmediateRewardEstimators (currently only for the actions in the
            sampled batch that corresponds with the IRE).
        Q-estimator (for every quantile)
        """
        if mentor_acted and self.batch_size <= len(self.mentor_history):
            mentor_history_samples = self.sample_history(
                self.mentor_history)
            self.mentor_q_estimator.update(mentor_history_samples)

        history_samples = self.sample_history(self.history)

        # This does < batch_size updates on the IREs. For history-handling
        # purposes. Possibly sample batch_size per-action in the future.
        for IRE_index, IRE in enumerate(self.IREs):
            IRE.update(
                [(s, r) for s, a, r, _, _ in history_samples if IRE_index == a])

        for q_estimator in self.QEstimators:
            q_estimator.update(history_samples)

    def learn(
            self, num_steps, report_every_n=500, render=1, reset_every_ep=False,
            early_stopping=0):
        if reset_every_ep:
            raise NotImplementedError("Not implemented reset_every_ep")
        if early_stopping:
            raise NotImplementedError("Not implemented early stopping")
        if self.total_steps != 0:
            print("WARN: Agent already trained", self.total_steps)

        period_reward = []  # initialise
        state = self.env.reset()
        for step in range(num_steps):
            self.report(num_steps, period_reward, render_mode=render)

            # state = self.env.map_int_to_grid(int(self.env.reset()))/3.5-1
            period_reward = []  # reset

            action, mentor_acted = self.act(state)
            next_state, reward, done, _ = self.env.step(action)
            period_reward.append(reward)

            if render:
                # First rendering should not return N lines
                # self.env.render(in_loop=self.total_steps > 0)
                self.env.render()


            self.store_history(
                state, action, reward, next_state, done, mentor_acted)

            self.total_steps += 1

            if self.total_steps % self.update_n_steps == 0:
                self.update_estimators(mentor_acted=mentor_acted)

            state = next_state[:]
            if done:
                self.failures += 1
                # print('failed')
                state = self.env.reset()
            if step % report_every_n == 0:
                if step == 0:
                    self.mentor_queries_periodic.append(self.mentor_queries)
                else:
                    self.mentor_queries_periodic.append(
                        self.mentor_queries
                        - np.sum(self.mentor_queries_periodic))

    def report(
            self, tot_steps, reward_last, render_mode, queries_last=None):
        """Reports on period of steps and calls any additional printing

        Args:
            tot_steps (int): total number of steps
            reward_last (list): list of rewards from last report period
            render_mode (int): defines verbosity of rendering
            queries_last (int): number of mentor queries in the last
                reporting period (i.e. the one being reported).
        """
        if render_mode < 0:
            return
        if self.total_steps > 0:
            title = (
                f"Steps {self.total_steps}/{tot_steps} ("
                f"{100 * self.total_steps / tot_steps} %)")
            print(title)
            self.additional_printing(render_mode)
            report = (
                f"F {self.failures} - R (last ep) "
                f"{(sum(reward_last) if reward_last else '-'):.0f}"
            )
            if queries_last is not None:
                report += f" - M (last ep) {queries_last}"

            print(report)

    def additional_printing(self, render):
        if render and self.mentor is not None:
            print(f"M {self.mentor_queries} ")
        if render > 1:

            if self.mentor is None:

                if self.q_estimator.lr is not None:
                    print(
                        f"Learning rates: "
                        f"QEst {self.q_estimator.lr:.4f}")
            else:
                print("Additional for continuous pessimistic")
                if np.isnan(self.Q_val_temp):
                    print('Q VAL IS NAN')
                else:
                    print(f"Q val\n{self.Q_val_temp}")
                print(f"mentor Q val\n{self.mentor_Q_val_temp}")
                if self.q_estimator.lr is not None:
                    print(
                        f"Learning rates: "
                        f"QEst {self.q_estimator.lr:.4f}"
                        f"Mentor V {self.mentor_q_estimator.lr:.4f}")
