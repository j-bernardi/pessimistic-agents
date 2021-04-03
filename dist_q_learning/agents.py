import abc
import numpy as np
from collections import deque

from estimators import (
    ImmediateRewardEstimator, QuantileQEstimator, MentorQEstimator, QEstimator, QMeanIREEstimator)

QUANTILES = [2**k / (1 + 2**k) for k in range(-5, 5)]


class BaseAgent(abc.ABC):

    def __init__(
            self,
            num_actions,
            num_states,
            env,
            gamma,
            update_n_steps=1,
            batch_size=1,
            eps_max=0.1,
            eps_min=0.01,
            mentor=None
    ):
        """Initialise the base agent with shared params

            num_actions:
            num_states:
            env:
            gamma:
            update_n_steps: how often to call the update_estimators
                function
            batch_size: size of the history to update on
        """
        self.num_actions = num_actions
        self.num_states = num_states
        self.env = env
        self.gamma = gamma
        self.batch_size = batch_size
        self.update_n_steps = update_n_steps
        self.eps_max = eps_max
        self.eps_min = eps_min

        self.mentor = mentor
        self.mentor_queries = 0
        self.total_steps = 0
        self.failures = 0

    def learn(self, num_eps, steps_per_ep=500, render=1):

        if self.total_steps != 0:
            print("WARN: Agent already trained", self.total_steps)
        ep_reward = []  # initialise
        step = 0

        for ep in range(num_eps):
            self.report_episode(
                step, ep, num_eps, ep_reward,
                render_mode=render
            )

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
                    self.update_estimators(
                        random_sample=False, mentor_acted=mentor_acted)

                state = next_state
                if done:
                    self.failures += 1
                    # print('failed')
                    break

    @abc.abstractmethod
    def act(self, state):
        raise NotImplementedError("Must be implemented per agent")

    @abc.abstractmethod
    def store_history(
            self, state, action, reward, next_state, done, mentor_acted):
        raise NotImplementedError("Must be implemented per agent")

    @abc.abstractmethod
    def update_estimators(self, random_sample=None, mentor_acted=False):
        raise NotImplementedError()

    def epsilon(self):
        """Reduce the max value of, and return, the random var epsilon."""

        if self.eps_max > self.eps_min:
            self.eps_max *= 0.999

        return self.eps_max * np.random.rand()

    def sample_history(self, history, random_sample=False):
        """Return a sample of the history"""
        if random_sample:
            idxs = np.random.randint(
                low=0, high=len(history), size=self.batch_size)
        else:
            assert self.batch_size == self.update_n_steps
            idxs = range(-self.batch_size, 0)

        return [history[i] for i in idxs]

    def report_episode(
            self, s, ep, num_eps, last_ep_reward, render_mode
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
            print(
                f"Episode {ep}/{num_eps} ({self.total_steps}) - "
                f"S {s} - "
                f"F {self.failures} - R (last ep) "
                f"{(sum(last_ep_reward) if last_ep_reward else '-'):.0f}"
            )

            self.additional_printing(render_mode)

            if render_mode:
                print("\n" * (self.env.state_shape[0] - 1))

    def additional_printing(self, render_mode):
        """Defines class-specific printing"""
        return None


class FinitePessimisticAgent(BaseAgent):

    def __init__(
            self,
            num_actions,
            num_states,
            env,
            gamma,
            mentor,
            quantile_i,
            update_n_steps=1,
            batch_size=1,
            lr=0.1,
            eps_max=0.1,
            eps_min=0.01,
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
            gamma=gamma, update_n_steps=update_n_steps, batch_size=batch_size,
            eps_max=eps_max, eps_min=eps_min, mentor=mentor
        )

        assert self.mentor is not None

        self.quantile_i = quantile_i

        self.history = deque(maxlen=10000)
        self.mentor_history = deque(maxlen=10000)

        # Create the estimators
        self.IREs = [ImmediateRewardEstimator(a) for a in range(num_actions)]

        self.QEstimators = [
            QuantileQEstimator(
                q, self.IREs, gamma, num_states, num_actions, lr=lr)
            for q in QUANTILES]

        self.MentorQEstimator = MentorQEstimator(
            num_states, num_actions, gamma, lr=lr)

    def act(self, state):
        values = np.array([
            self.QEstimators[self.quantile_i].estimate(state, action_i)
            for action_i in range(self.num_actions)
        ])

        max_vals = values == values.max()
        proposed_action = int(np.random.choice(np.flatnonzero(max_vals)))

        mentor_value = self.MentorQEstimator.estimate(state)
        if (
                mentor_value > values[proposed_action] + self.epsilon()
                or values[proposed_action] <= 0.
        ):
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

    def update_estimators(self, random_sample=False, mentor_acted=False):
        """Update all estimators with a random batch of the histories.

        Mentor-Q Estimator
        ImmediateRewardEstimators (currently only for the actions in the
            sampled batch that corresponds with the IRE).
        Q-estimator (for every quantile)
        """
        if mentor_acted:
            mentor_history_samples = self.sample_history(
                self.mentor_history, random_sample=random_sample)

            self.MentorQEstimator.update(mentor_history_samples)

        history_samples = self.sample_history(
            self.history, random_sample=random_sample)

        # This does < batch_size updates on the IREs. For history-handling
        # purposes. Possibly sample batch_size per-action in the future.
        for IRE_index, IRE in enumerate(self.IREs):
            IRE.update(
                [(s, r) for s, a, r, _, _ in history_samples if IRE_index == a])

        for q_estimator in self.QEstimators:
            q_estimator.update(history_samples)

    def additional_printing(self, render):
        def return_callable():
            if render > 1:
                print("M {self.mentor_queries} ")
                print(
                    f"Q table\n{self.QEstimators[self.quantile_i].q_table}")
                print(f"Mentor Q table\n{self.MentorQEstimator.Q_list}")
                print(
                    f"Learning rates: "
                    f"QEst {self.QEstimators[self.quantile_i].lr:.4f}, "
                    f"Mentor V {self.MentorQEstimator.lr:.4f}")
        return return_callable


class QTableAgent(BaseAgent):

    def __init__(
            self, num_actions, num_states, env, gamma, lr=0.1,
            update_n_steps=1, batch_size=1, mentor=None,
            eps_max=0.1, eps_min=0.01,
    ):
        """

        """
        super().__init__(
            num_actions=num_actions, num_states=num_states, env=env,
            gamma=gamma, update_n_steps=update_n_steps, batch_size=batch_size,
            eps_max=eps_max, eps_min=eps_min, mentor=mentor
        )

        has_mentor = self.mentor is not None
        self.q_estimator = QEstimator(num_states, num_actions, gamma, lr=lr, has_mentor=has_mentor)
        self.mentor_q_estimator = MentorQEstimator(
            num_states, num_actions, gamma, lr)
        self.history = deque(maxlen=10000)

        if has_mentor:
            self.mentor_history = deque(maxlen=10000)
        else:
            self.mentor_history = None

    def update_estimators(self, random_sample=False, mentor_acted=False):
        history_samples = self.sample_history(
            self.history, random_sample=random_sample)
        self.q_estimator.update(history_samples)

        if mentor_acted:
            assert self.mentor is not None
            mentor_history_samples = self.sample_history(self.mentor_history)
            self.mentor_q_estimator.update(mentor_history_samples)

    def act(self, state):
        self.q_estimator.reduce_random_act_prob()
        if (
                self.q_estimator.random_act_prob is not None
                and np.random.rand() < self.q_estimator.random_act_prob
        ):
            return int(np.random.randint(self.num_actions)), False

        values = np.array([
            self.q_estimator.estimate(state, action_i)
            for action_i in range(self.num_actions)
        ])

        max_vals = values == values.max()
        assert max_vals.shape == (4,)
        proposed_action = int(np.random.choice(np.flatnonzero(max_vals)))

        if self.mentor is not None:
            mentor_value = self.mentor_q_estimator.estimate(state)
            if (mentor_value > values[proposed_action] + self.epsilon()
                    or values[proposed_action] <= 0.):
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

        if render_mode:
            if self.mentor is not None:
                print(f"Mentor Queries {self.mentor_queries}")
                print(f"Mentor Q vals {self.mentor_q_estimator.Q_list}")
            else:
                print(f"Epsilon {self.q_estimator.random_act_prob}")

        if render_mode > 1:
            print(f"Agent Q\n{self.q_estimator.q_table}")

class QTableIREAgent(QTableAgent):

    def __init__(
                self, num_actions, num_states, env, gamma, lr=0.1,
                update_n_steps=1, batch_size=1, mentor=None,
                eps_max=0.1, eps_min=0.01,
    ):
        """

        """
        super().__init__(
            num_actions=num_actions, num_states=num_states, env=env,
            gamma=gamma, update_n_steps=update_n_steps, batch_size=batch_size,
            eps_max=eps_max, eps_min=eps_min, mentor=mentor, lr=lr
        )
    
        self.IREs = [ImmediateRewardEstimator(a) for a in range(self.num_actions)]
        self.q_estimator = QMeanIREEstimator(
            self.num_states, self.num_actions, gamma, self.IREs,
            lr=lr, has_mentor=self.mentor is not None)


    def update_estimators(self, random_sample=False, mentor_acted=False):
        history_samples = self.sample_history(
            self.history, random_sample=random_sample)

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

    def additional_printing(self, render_mode):

        if render_mode:
            if self.mentor is not None:
                print(f"Mentor Queries {self.mentor_queries}")
                print(f"Mentor Q vals {self.mentor_q_estimator.Q_list}")
            else:
                print(f"Epsilon {self.q_estimator.random_act_prob}")

        if render_mode > 1:
            print(f"Agent Q\n{self.q_estimator.q_table}")

