import abc
import numpy as np
from collections import deque

from estimators import (
    ImmediateRewardEstimator, QuantileQEstimator, MentorQEstimator, QEstimator,
    QMeanIREEstimator
)

QUANTILES = [2**k / (1 + 2**k) for k in range(-5, 5)]


def geometric_sum(r_val, gamm, steps):

    if steps == "inf":
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
            min_reward=1e-6
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
            eps_max=eps_max, eps_min=eps_min, mentor=mentor,
            min_reward=min_reward
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

        self.mentor_q_estimator = MentorQEstimator(
            num_states, num_actions, gamma, lr=lr)

    def act(self, state):
        values = np.array([
            self.QEstimators[self.quantile_i].estimate(state, action_i)
            for action_i in range(self.num_actions)
        ])

        # Choose randomly from any jointly maximum values
        max_vals = values == values.max()
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

            self.mentor_q_estimator.update(mentor_history_samples)

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
        if render:
            print(f"M {self.mentor_queries} ")
        if render > 1:
            print("Additional for finite pessimistic")
            print(f"Q table\n{self.QEstimators[self.quantile_i].q_table}")
            print(f"Mentor Q table\n{self.mentor_q_estimator.q_list}")
            print(
                f"Learning rates: "
                f"QEst {self.QEstimators[self.quantile_i].lr:.4f}, "
                f"Mentor V {self.mentor_q_estimator.lr:.4f}")


class QTableAgent(BaseAgent):

    def __init__(
            self, num_actions, num_states, env, gamma, lr=0.1,
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
            gamma=gamma, update_n_steps=update_n_steps, batch_size=batch_size,
            eps_max=eps_max, eps_min=eps_min, mentor=mentor,
            scale_q_value=scale_q_value, min_reward=min_reward
        )

        self.q_estimator = q_estimator_init(
            num_states, num_actions, gamma=gamma, lr=lr,
            has_mentor=self.mentor is not None,
            scaled=self.scale_q_value
        )

        if not self.scale_q_value:
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

    def update_estimators(self, random_sample=False, mentor_acted=False):
        history_samples = self.sample_history(
            self.history, random_sample=random_sample)
        self.q_estimator.update(history_samples)

        if mentor_acted:
            assert self.mentor is not None
            mentor_history_samples = self.sample_history(self.mentor_history)
            assert history_samples == mentor_history_samples
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
        max_vals = values == values.max()
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
                print(f"Mentor Queries {self.mentor_queries}   ***")
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
            print(
                f"Learning rates: "
                f"QEst {self.q_estimator.lr:.4f}, "
                f"Mentor V {self.mentor_q_estimator.lr:.4f}"
            )


class QTableIREAgent(QTableAgent):

    def __init__(
                self, num_actions, num_states, env, gamma, lr=0.1,
                update_n_steps=1, batch_size=1, mentor=None,
                eps_max=0.1, eps_min=0.01, min_reward=1e-6
    ):
        """

        """

        super().__init__(
            num_actions=num_actions, num_states=num_states, env=env,
            gamma=gamma, update_n_steps=update_n_steps, batch_size=batch_size,
            eps_max=eps_max, eps_min=eps_min, mentor=mentor, lr=lr,
            min_reward=min_reward
        )
    
        self.IREs = [
            ImmediateRewardEstimator(a) for a in range(self.num_actions)]
        self.q_estimator = QMeanIREEstimator(
            self.num_states, self.num_actions, gamma, self.IREs,
            lr=lr, has_mentor=self.mentor is not None
        )

    def update_estimators(self, random_sample=False, mentor_acted=False):
        history_samples = self.sample_history(
            self.history, random_sample=random_sample)

        if mentor_acted:
            assert self.mentor is not None
            mentor_history_samples = self.sample_history(
                self.mentor_history, random_sample=random_sample)
            self.mentor_q_estimator.update(mentor_history_samples)

        # This does < batch_size updates on the IREs. For history-handling
        # purposes. Possibly sample batch_size per-action in the future.
        for IRE_index, IRE in enumerate(self.IREs):
            IRE.update(
                [(s, r) for s, a, r, _, _ in history_samples if IRE_index == a])

        self.q_estimator.update(history_samples)
