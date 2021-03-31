import numpy as np
from collections import deque

from estimators import ImmediateRewardEstimator, QEstimator, MentorQEstimator

QUANTILES = [2**k / (1 + 2**k) for k in range(-5, 5)]


class FinitePessimisticAgent:

    def __init__(
            self,
            num_actions,
            num_states,
            env,
            mentor,
            quantile_i,
            gamma,
            lr=1.,
            eps_max=0.1,
            eps_min=0.01,
            update_n_steps=1,
            batch_size=1,
    ):
        """Initialise

            num_actions:
            num_states:
            env:
            mentor: a function taking (state, kwargs), returning an
                integer action.
            quantile_i: the index of the quantile from QUANTILES to use
                for taking actions.
            gamma:
            eps_max: initial max value of the random query-factor
            eps_min: the minimum value of the random query-factor.
                Once self.epsilon < self.eps_min, it stops reducing.
            batch_size: size of the history to update on
        """
        self.quantile_i = quantile_i
        self.num_actions = num_actions
        self.num_states = num_states
        self.env = env
        self.gamma = gamma
        self.mentor = mentor
        self.eps_max = eps_max
        self.eps_min = eps_min
        self.batch_size = batch_size
        self.update_n_steps = update_n_steps

        self.IREs = [ImmediateRewardEstimator(a) for a in range(num_actions)]

        self.history = deque(maxlen=10000)
        self.mentor_history = deque(maxlen=10000)

        self.total_steps = 0
        self.mentor_queries = 0
        self.failures = 0

        self.QEstimators = [
            QEstimator(q, self.IREs, gamma, num_states, num_actions, lr=lr)
            for q in QUANTILES
        ]
        self.MentorQEstimator = MentorQEstimator(
            num_states, num_actions, gamma, lr=lr)

    def learn(self, num_eps, steps_per_ep=500, render=True):

        if self.total_steps != 0:
            print("WARN: Agent already trained", self.total_steps)

        for ep in range(num_eps):
            if ep % 1 == 0 and ep > 0:
                if render:
                    print(self.env.get_spacer())
                print(
                    f"Episode {ep}/{num_eps} ({self.total_steps}) "
                    f"- F {self.failures} - M {self.mentor_queries}"
                )
                if render:
                    print("\n" * (self.env.state_shape[0] - 1))
                # print("Mentor Q\n", self.MentorQEstimator.Q_list)
                # print("Q Est\n", self.QEstimators[self.quantile_i].Q_table)
            state = int(self.env.reset())

            for step in range(steps_per_ep):
                values = [
                    self.QEstimators[self.quantile_i].estimate(state, action_i)
                    for action_i in range(self.num_actions)
                ]
                proposed_action = int(np.argmax(values))

                mentor_value = self.MentorQEstimator.estimate(state)
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

                next_state, reward, done, _ = self.env.step(action)
                next_state = int(next_state)
                if render:
                    self.env.render(in_loop=self.total_steps > 0)

                if mentor_acted:
                    self.mentor_history.append(
                        (state, action, reward, next_state))
                
                self.history.append((state, action, reward, next_state))

                self.total_steps += 1

                if self.total_steps % self.update_n_steps == 0:
                    self.update_estimators(random_sample=False)
                
                state = next_state
                if done:
                    self.failures += 1
                    # print('failed')
                    break

    def update_estimators(self, random_sample=False):
        """Update all estimators with a random batch of the histories.

        Mentor-Q Estimator
        ImmediateRewardEstimators (currently only for the actions in the
            sampled batch that corresponds with the IRE).
        Q-estimator (for every quantile)
        """
        if random_sample:
            mentor_idxs = np.random.randint(
                low=0, high=len(self.mentor_history), size=self.batch_size)
            agent_idxs = np.random.randint(
                low=0, high=len(self.history), size=self.batch_size)

        else:
            assert self.batch_size == self.update_n_steps
            mentor_idxs = range(-self.batch_size, 0)
            agent_idxs = range(-self.batch_size, 0)

        mentor_history_samples = [
            self.mentor_history[i] for i in mentor_idxs]
        self.MentorQEstimator.update(mentor_history_samples)

        history_samples = [self.history[i] for i in agent_idxs]
        # This does < batch_size updates on the IREs. For history-handling
        # purposes. Possibly sample batch_size per-action in the future.
        for IRE_index, IRE in enumerate(self.IREs):
            IRE.update(
                [(s, r) for s, a, r, _ in history_samples if IRE_index == a])
        
        for q_estimator in self.QEstimators:
            q_estimator.update(history_samples)

    def epsilon(self):
        """Reduce the max value of, and return, the random var epsilon."""

        if self.eps_max > self.eps_min:
            self.eps_max *= 0.999

        return self.eps_max * np.random.rand()
