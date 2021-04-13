import abc
import numpy as np
from collections import deque

from estimators import (
    ImmediateRewardEstimator, QuantileQEstimator, MentorQEstimator, QEstimator,
    QMeanIREEstimator, QIREEstimator
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
            gamma:
            sampling_strategy (str): one of 'random', 'last_n_steps' or
                'whole', dictating how the history should be sampled.
            update_n_steps: how often to call the update_estimators
                function
            batch_size (int): size of the history to update on
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
            print("WARN: Defaults not used for BS, but sampling whole history")
        self.sampling_strategy = sampling_strategy
        self.batch_size = batch_size
        self.update_n_steps = update_n_steps
        self.eps_max = eps_max
        self.eps_min = eps_min
        self.scale_q_value = scale_q_value
        self.mentor = mentor
        self.min_reward = min_reward

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

    def epsilon(self):
        """Reduce the max value of, and return, the random var epsilon."""

        if self.eps_max > self.eps_min:
            self.eps_max *= 0.999

        return self.eps_max * np.random.rand()

    def sample_history(self, history):
        """Return a sample of the history"""
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
            self, s, ep, num_eps, last_ep_reward, render_mode, queries_last=None
    ):
        """Reports standard episode and calls any additional printing

        Args:
            s (int): num steps in last episode
            ep (int): current episode
            num_eps (int): total number of episodes
            last_ep_reward (list): list of rewards from last episode
            render_mode (int): defines verbosity of rendering
        """
        if ep % 1 == 0 and ep > 0:
            if render_mode:
                print(self.env.get_spacer())
            report = (
                f"Episode {ep}/{num_eps} ({self.total_steps}) - "
                f"S {s} - "
                f"F {self.failures} - R (last ep) "
                f"{(sum(last_ep_reward) if last_ep_reward else '-'):.0f}"
            )
            if queries_last is not None:
                report += f" - M (last ep) {queries_last}"

            print(report)
            self.additional_printing(render_mode)

            if render_mode:
                print("\n" * (self.env.state_shape[0] - 1))

    def additional_printing(self, render_mode):
        """Defines class-specific printing"""
        return None


class FinitePessimisticAgent(BaseAgent):
    """The faithful, original Pessimistic Distributed Q value agent"""

    def __init__(
            self,
            num_actions,
            num_states,
            env,
            gamma,
            mentor,
            quantile_i,
            sampling_strategy="last_n_steps",
            update_n_steps=1,
            batch_size=1,
            lr=0.1,
            eps_max=0.1,
            eps_min=0.01,
            min_reward=1e-6,
            scale_q_value=True
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
            num_actions=num_actions, num_states=num_states, env=env,
            gamma=gamma, sampling_strategy=sampling_strategy,
            update_n_steps=update_n_steps, batch_size=batch_size,
            eps_max=eps_max, eps_min=eps_min, mentor=mentor,
            min_reward=min_reward, scale_q_value=scale_q_value,
        )

        if self.mentor is None:
            raise NotImplementedError("Pessimistic agent requires a mentor")

        self.quantile_i = quantile_i

        self.history = deque(maxlen=10000)
        self.mentor_history = deque(maxlen=10000)

        # Create the estimators
        self.IREs = [ImmediateRewardEstimator(a) for a in range(num_actions)]

        self.QEstimators = [
            QuantileQEstimator(
                q, self.IREs, gamma, num_states, num_actions, lr=lr)
            for q in QUANTILES
        ]

        self.mentor_q_estimator = MentorQEstimator(
            num_states, num_actions, gamma, lr=lr)

    def reset_estimators(self):
        for ire in self.IREs:
            ire.reset()
        for q_est in self.QEstimators:
            q_est.reset()
        self.mentor_q_estimator.reset()

    def act(self, state):
        values = np.array([
            self.QEstimators[self.quantile_i].estimate(state, action_i)
            for action_i in range(self.num_actions)
        ])

        # Choose randomly from any jointly maximum values
        max_vals = values == np.amax(values)
        proposed_action = int(np.random.choice(np.flatnonzero(max_vals)))

        # Defer if predicted value < min, based on r > eps
        scaled_min_r = self.min_reward
        eps = self.epsilon()
        if not self.scale_q_value:
            scaled_min_r /= (1. - self.gamma)
            eps /= (1. - self.gamma)
        mentor_value = self.mentor_q_estimator.estimate(state)

        prefer_mentor = mentor_value > (values[proposed_action] + eps)
        agent_value_too_low = values[proposed_action] <= scaled_min_r
        if agent_value_too_low or prefer_mentor:
            action = self.env.map_grid_act_to_int(
                self.mentor(
                    self.env.map_int_to_grid(state),
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
                [(s, r) for s, a, r, _, _ in history_samples if IRE_index == a])

        for q_estimator in self.QEstimators:
            q_estimator.update(history_samples)

    def additional_printing(self, render):
        if render:
            print(f"M {self.mentor_queries_per_ep[-1]} ({self.mentor_queries})")
        if render > 1:
            print("Additional for finite pessimistic")
            print(f"Q table\n{self.QEstimators[self.quantile_i].q_table}")
            print(f"Mentor Q table\n{self.mentor_q_estimator.q_list}")
            if self.QEstimators[self.quantile_i].lr is not None:
                print(
                    f"Learning rates: "
                    f"QEst {self.QEstimators[self.quantile_i].lr:.4f}, "
                    f"Mentor V {self.mentor_q_estimator.lr:.4f}")


class QTableAgent(BaseAgent):
    """A basic Q table agent"""
    def __init__(
            self, num_actions, num_states, env, gamma,
            sampling_strategy="last_n_steps", lr=0.1,
            update_n_steps=1, batch_size=1, mentor=None,
            eps_max=0.1, eps_min=0.01, q_estimator_init=QEstimator,
            mentor_q_estimator_init=MentorQEstimator,
            scale_q_value=True, min_reward=1e-6
    ):
        """Initialise the basic Q table agent

        Additional Args:
            q_table_init (callable): the init function for the type of
                Q estimator to use for the class (e.g. finite, infinite
                horizon)
        """
        super().__init__(
            num_actions=num_actions, num_states=num_states, env=env,
            gamma=gamma, sampling_strategy=sampling_strategy,
            update_n_steps=update_n_steps, batch_size=batch_size,
            eps_max=eps_max, eps_min=eps_min, mentor=mentor,
            scale_q_value=scale_q_value, min_reward=min_reward
        )

        self.q_estimator = q_estimator_init(
            num_states, num_actions, gamma=gamma, lr=lr,
            has_mentor=self.mentor is not None,
            scaled=self.scale_q_value
        )

        if not self.scale_q_value:
            # geometric sum of max rewards per step (1.) for N steps (up to inf)
            init_val = geometric_sum(1., self.gamma, self.q_estimator.num_steps)
        else:
            init_val = 1.

        self.mentor_q_estimator = mentor_q_estimator_init(
            num_states, num_actions, gamma, lr, scaled=self.scale_q_value,
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

    def act(self, state):
        eps_rand_act = self.q_estimator.get_random_act_prob()
        if eps_rand_act is not None and np.random.rand() < eps_rand_act:
            assert self.mentor is None
            return int(np.random.randint(self.num_actions)), False

        values = np.array([
            self.q_estimator.estimate(state, action_i)
            for action_i in range(self.num_actions)
        ])

        # Randomize if there is 2 vals at the max
        max_vals = values == np.amax(values)
        assert max_vals.shape == (4,)
        proposed_action = int(np.random.choice(np.flatnonzero(max_vals)))

        # Defer if predicted value < min, based on r > eps
        scaled_min_r = self.min_reward
        scaled_eps = self.epsilon()
        if not self.scale_q_value:
            scaled_min_r = geometric_sum(
                scaled_min_r, self.gamma, self.q_estimator.num_steps)
            scaled_eps = geometric_sum(
                scaled_eps, self.gamma, self.q_estimator.num_steps)

        if self.mentor is not None:
            mentor_value = self.mentor_q_estimator.estimate(state)

            prefer_mentor = mentor_value > values[proposed_action] + scaled_eps
            agent_value_too_low = values[proposed_action] <= scaled_min_r

            if agent_value_too_low or prefer_mentor:
                # print(
                #     f"Ag low {agent_value_too_low}: "
                #     f"agent_val {values[proposed_action]:3f} < "
                #     f"scaled_r_eps: {scaled_min_r:3f}"
                #     f"\nmentor {prefer_mentor}: mentor_val {mentor_value:3f} "
                #     f"- eps: {scaled_eps:3f}"
                # )
                action = self.env.map_grid_act_to_int(
                    self.mentor(
                        self.env.map_int_to_grid(state),
                        kwargs={'state_shape': self.env.state_shape})
                )
                # print("called mentor")
                mentor_acted = True
                self.mentor_queries += 1
            else:
                action = proposed_action
                mentor_acted = False
        else:
            action = proposed_action
            mentor_acted = False

        return action, mentor_acted

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


class QTableMeanIREAgent(QTableAgent):
    """Q Table agent that uses IRE to update, instead of actual reward

    Otherwise exactly the same
    """
    def __init__(
            self, num_actions, num_states, env, gamma, lr=0.1,
            sampling_strategy="last_n_steps", update_n_steps=1, batch_size=1,
            mentor=None, eps_max=0.1, eps_min=0.01, min_reward=1e-6,
            scale_q_value=True
    ):
        super().__init__(
            num_actions=num_actions, num_states=num_states, env=env,
            gamma=gamma, update_n_steps=update_n_steps, batch_size=batch_size,
            eps_max=eps_max, eps_min=eps_min, mentor=mentor, lr=lr,
            min_reward=min_reward, sampling_strategy=sampling_strategy,
            scale_q_value=scale_q_value
        )

        self.IREs = [
            ImmediateRewardEstimator(a) for a in range(self.num_actions)]
        self.q_estimator = QMeanIREEstimator(
            self.num_states, self.num_actions, gamma, self.IREs,
            lr=lr, has_mentor=self.mentor is not None, scaled=self.scale_q_value
        )

    def reset_estimators(self):
        for ire in self.IREs:
            ire.reset()
        self.q_estimator.reset()

    def update_estimators(self, mentor_acted=False):

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


class QTablePessIREAgent(QTableAgent):
    """Q Table agent that uses *pessimistic* IRE to update

    Otherwise exactly the same as the Q table agent
    """
    def __init__(
            self, num_actions, num_states, env, gamma, quantile_i, lr=0.1,
            sampling_strategy="last_n_steps", update_n_steps=1, batch_size=1,
            mentor=None, eps_max=0.1, eps_min=0.01, min_reward=1e-6,
            scale_q_value=True
    ):
        super().__init__(
            num_actions=num_actions, num_states=num_states, env=env,
            gamma=gamma, update_n_steps=update_n_steps, batch_size=batch_size,
            eps_max=eps_max, eps_min=eps_min, mentor=mentor, lr=lr,
            min_reward=min_reward, sampling_strategy=sampling_strategy,
            scale_q_value=scale_q_value
        )
        self.quantile_i = quantile_i

        self.IREs = [
            ImmediateRewardEstimator(a) for a in range(self.num_actions)
        ]

        self.q_estimator = QIREEstimator(
            self.quantile_i, self.num_states, self.num_actions, gamma,
            self.IREs, lr=lr, has_mentor=self.mentor is not None
        )

    def reset_estimators(self):
        for ire in self.IREs:
            ire.reset()
        self.q_estimator.reset()

    def update_estimators(self, mentor_acted=False):

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
