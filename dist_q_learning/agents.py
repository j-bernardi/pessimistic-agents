import abc
import numpy as np
from collections import deque

from estimators import (
    ImmediateRewardEstimator, QuantileQEstimator, MentorQEstimator, QEstimator,
    QMeanIREEstimator, QPessIREEstimator, QuantileQEstimatorSingleOrig,
    ImmediateNextStateEstimator
)

QUANTILES = [2**k / (1 + 2**k) for k in range(-5, 5)]


def geometric_sum(r_val, gamm, steps):
    # Two valid ways to specify infinite steps
    if steps is None or steps == "inf":
        return r_val / (1. - gamm)
    else:
        return r_val * (1. - gamm ** steps) / (1. - gamm)


class BaseAgent(abc.ABC):

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
            min_reward=1e-6
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
        """
        self.num_actions = num_actions
        self.num_states = num_states
        self.env = env
        self.gamma = gamma
        if sampling_strategy == "whole" and batch_size != 1:
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

        self.q_estimator = None
        self.mentor_q_estimator = None  # TODO - put in a

        self.mentor_queries = 0
        self.total_steps = 0
        self.failures = 0

        self.mentor_queries_per_ep = []

    def learn(self, num_eps, steps_per_ep=500, render=1, reset_every_ep=False):

        if self.total_steps != 0:
            print("WARN: Agent already trained", self.total_steps)
        ep_reward = []  # initialise
        step = 0

        state = int(self.env.reset())

        for ep in range(num_eps):
            queries = self.mentor_queries_per_ep[-1]\
                if self.mentor_queries_per_ep else -1
            self.report_episode(
                step, ep, num_eps, ep_reward, render_mode=render,
                queries_last=queries
            )
            if reset_every_ep:
                state = int(self.env.reset())
            ep_reward = []  # reset
            for step in range(steps_per_ep):
                action, mentor_acted = self.act(state)

                next_state, reward, done, _ = self.env.step(action)
                ep_reward.append(reward)
                next_state = int(next_state)

                if render:
                    # First rendering should not return N lines
                    self.env.render(in_loop=self.total_steps > 0)

                self.store_history(
                    state, action, reward, next_state, done, mentor_acted)

                self.total_steps += 1

                if self.total_steps % self.update_n_steps == 0:
                    self.update_estimators(mentor_acted=mentor_acted)

                state = next_state
                if done:
                    self.failures += 1
                    # print('failed')
                    state = int(self.env.reset())

            if ep == 0:
                self.mentor_queries_per_ep.append(self.mentor_queries)
            else:
                self.mentor_queries_per_ep.append(
                    self.mentor_queries - np.sum(self.mentor_queries_per_ep)
                )

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

        elif self.sampling_strategy == "whole":
            return history

        else:
            raise ValueError(
                f"Sampling strategy {self.sampling_strategy} invalid")

        return [history[i] for i in idxs]

    def report_episode(
            self, s, ep, num_eps, last_ep_reward, render_mode,
            queries_last=None
    ):
        """Reports standard episode and calls any additional printing

        Args:
            s (int): num steps in last episode
            ep (int): current episode
            num_eps (int): total number of episodes
            last_ep_reward (list): list of rewards from last episode
            render_mode (int): defines verbosity of rendering
            queries_last (int): number of mentor queries in the last
                episode (i.e. the one being reported).
        """
        if ep % 1 == 0 and ep > 0:
            if render_mode:
                print(self.env.get_spacer())
            episode_title = f"Episode {ep}/{num_eps} ({self.total_steps})"
            print(episode_title)
            self.additional_printing(render_mode)
            report = (
                f"{episode_title} S {s} - F {self.failures} - R (last ep) "
                f"{(sum(last_ep_reward) if last_ep_reward else '-'):.0f}"
            )
            if queries_last is not None:
                report += f" - M (last ep) {queries_last}"

            print(report)

            if render_mode:
                print("\n" * (self.env.state_shape[0] - 1))

    def additional_printing(self, render_mode):
        """Defines class-specific printing"""
        return None


class BaseQAgent(BaseAgent, abc.ABC):
    """Augment the BaseAgent with an acting method based on Q tables

    Optionally uses a mentor (depending on whether parent self.mentor is
    None)
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

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

        values = np.array(
            [self.q_estimator.estimate(state, action_i)
             for action_i in range(self.num_actions)]
        )

        # Choose randomly from any jointly maximum values
        max_vals = values == np.amax(values)
        max_nonzero_val_idxs = np.flatnonzero(max_vals)
        tie_broken_proposed_action = np.random.choice(max_nonzero_val_idxs)
        agent_max_action = int(tie_broken_proposed_action)

        if self.scale_q_value:
            scaled_min_r = self.min_reward  # Defined as between [0, 1]
            scaled_eps = self.epsilon()
        else:
            scaled_min_r = geometric_sum(
                self.min_reward, self.gamma, self.q_estimator.num_steps)
            scaled_eps = geometric_sum(
                self.epsilon(), self.gamma, self.q_estimator.num_steps)

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


class FinitePessimisticAgent(BaseQAgent):
    """The faithful, original Pessimistic Distributed Q value agent"""

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
                the type of Q Estimator to use for the agent.
            train_all_q (bool): if False, trains only the Q estimator
                corresponding to quantile_i (self.q_estimator)
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

        if quantile_estimator_init is QuantileQEstimatorSingleOrig:
            self.next_state_estimator = ImmediateNextStateEstimator(
                self.num_states, self.num_actions)
            est_kwargs = {"ns_estimator": self.next_state_estimator}
        else:
            self.next_state_estimator = None
            est_kwargs = {}

        self.QEstimators = [
            quantile_estimator_init(
                quantile=q, immediate_r_estimators=self.IREs, gamma=gamma,
                num_states=num_states, num_actions=num_actions, lr=self.lr,
                **est_kwargs
            ) for i, q in enumerate(QUANTILES) if (
                i == self.quantile_i or train_all_q)
        ]
        self.q_estimator = self.QEstimators[
            self.quantile_i if train_all_q else 0]

        self.mentor_q_estimator = MentorQEstimator(
            num_states, num_actions, gamma, lr=self.lr)

    def reset_estimators(self):
        for ire in self.IREs:
            ire.reset()
        for q_est in self.QEstimators:
            q_est.reset()
        self.mentor_q_estimator.reset()

        if self.next_state_estimator is not None:
            self.next_state_estimator.reset()

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
        if self.sampling_strategy == "whole":
            self.reset_estimators()

        if mentor_acted:
            mentor_history_samples = self.sample_history(self.mentor_history)

            self.mentor_q_estimator.update(mentor_history_samples)

        history_samples = self.sample_history(self.history)

        # This does < batch_size updates on the IREs. For history-handling
        # purposes. Possibly sample batch_size per-action in the future.
        for IRE_index, IRE in enumerate(self.IREs):
            IRE.update(
                [(s, r) for s, a, r, _, _ in history_samples if IRE_index == a]
            )

        if self.next_state_estimator is not None:
            self.next_state_estimator.update(history_samples)

        for q_estimator in self.QEstimators:
            q_estimator.update(history_samples)

    def additional_printing(self, render):
        if render:
            print(f"M {self.mentor_queries_per_ep[-1]} ({self.mentor_queries})")
        if render > 1:
            print("Additional for finite pessimistic")
            print(f"Q table\n{self.q_estimator.q_table}")
            print(f"Mentor Q table\n{self.mentor_q_estimator.q_list}")
            if self.q_estimator.lr is not None:
                print(
                    f"Learning rates: "
                    f"QEst {self.q_estimator.lr:.4f}, "
                    f"Mentor V {self.mentor_q_estimator.lr:.4f}"
                    f"\nEpsilon max: {self.eps_max:.4f}"
                )


class BaseQTableAgent(BaseQAgent):
    """The base implementation of a Q-table agent"""
    def __init__(
            self,
            num_actions,
            num_states,
            env,
            gamma,
            q_estimator_init=QEstimator,
            mentor_q_estimator_init=MentorQEstimator,
            **kwargs
    ):
        """Initialise the basic Q table agent

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
            num_states, num_actions, gamma=gamma, lr=self.lr,
            has_mentor=self.mentor is not None, scaled=self.scale_q_value
        )

        if not self.scale_q_value:
            # geometric sum of max rewards per step (1.) for N steps (up to inf)
            init_val = geometric_sum(1., self.gamma, self.q_estimator.num_steps)
        else:
            init_val = 1.

        self.mentor_q_estimator = mentor_q_estimator_init(
            num_states, num_actions, gamma, self.lr, scaled=self.scale_q_value,
            init_val=init_val)
        self.history = deque(maxlen=10000)

        if self.mentor is not None:
            self.mentor_history = deque(maxlen=10000)
        else:
            self.mentor_history = None

    def reset_estimators(self):
        self.q_estimator.reset()
        self.mentor_q_estimator.reset()

    def update_estimators(self, mentor_acted=False):

        if self.sampling_strategy == "whole":
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
        if render_mode:
            if self.mentor is not None:
                print(
                    f"Mentor queries {self.mentor_queries_per_ep[-1]} "
                    f"({self.mentor_queries})")
            else:
                print(f"Epsilon {self.q_estimator.random_act_prob}")

        if render_mode > 1:
            print("Additional for QTableAgent")
            print(f"M {self.mentor_queries} ")
            if len(self.q_estimator.q_table.shape) == 3:
                print(f"Agent Q\n{self.q_estimator.q_table[:, :, -1]}")
                print(f"Mentor Q table\n{self.mentor_q_estimator.q_list[:, -1]}")
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
    """A basic Q table agent - no pessimism or IREs, just a Q table"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class BaseQTableIREAgent(BaseQTableAgent):

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

        if self.sampling_strategy == "whole":
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
    """Q Table agent that uses IRE to update, instead of actual reward

    Does so by using the QMeanIREEstimator as its Q Estimator
    """

    def __init__(self, num_actions, num_states, env, gamma, **kwargs):
        """Init the base agent and override the Q estimator"""
        super().__init__(
            num_actions=num_actions, num_states=num_states, env=env,
            gamma=gamma, **kwargs)

        if not hasattr(self, "q_estimator"):
            raise ValueError(
                "Hacky way to assert that q_estimator is overridden")
        self.q_estimator = QMeanIREEstimator(
            self.num_states, self.num_actions, self.gamma, self.IREs,
            lr=self.lr, has_mentor=self.mentor is not None,
            scaled=self.scale_q_value
        )


class QTablePessIREAgent(BaseQTableIREAgent):
    """Q Table agent that uses *pessimistic* IRE to update

    Does so by using the QPessIREEstimator as its q_estimator
    """

    def __init__(
            self, num_actions, num_states, env, gamma, quantile_i, **kwargs):
        """Init the base agent and override the Q estimator"""
        super().__init__(
            num_actions=num_actions, num_states=num_states, env=env,
            gamma=gamma, **kwargs)

        if not hasattr(self, "q_estimator"):
            raise ValueError(
                "Hacky way to assert that q_estimator is overridden")
        self.quantile_i = quantile_i
        self.q_estimator = QPessIREEstimator(
            QUANTILES[self.quantile_i], self.num_states, self.num_actions,
            gamma, self.IREs, lr=self.lr, has_mentor=self.mentor is not None
        )
