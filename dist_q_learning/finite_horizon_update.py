import time
import numpy as np

from env import FiniteStateCliffworld


class FiniteHorizonAgent:

    def __init__(self, env, num_states, num_actions, num_horizons):
        self.env = env
        self.num_states = num_states
        self.num_actions = num_actions
        self.num_horizons = num_horizons

        # Set Q_h for all h to 1. (arbitrary initialisation)
        self.q_tables = np.zeros((num_states, num_actions, num_horizons + 1))
        # Q_0 is always init to 0
        #  future is 0 steps from now, so 0 reward to accrue
        self.q_tables[:, :, 0] = 0.

        self.lr = 1.
        self.gamma = 1.  # not required < 1 for finite case!
        self.random_eps = 0.5

    def learn(self, n_steps, render=False):
        """Run n steps in the environment and learn"""
        state = self.env.reset()
        fails_last_n = 0
        report_freq = n_steps // 10
        decay = 10 ** (np.log10(1e-4) / n_steps)  # min p 1e-4
        for s in range(n_steps):

            if render:
                time.sleep(0.01)
                self.env.render(in_loop=True)

            # Use the future-most horizon, the -1 dimension
            if np.random.random() < self.random_eps:
                action = np.random.choice(range(self.num_actions))
            else:
                values = [
                    self.q_tables[state, a, -1]
                    for a in range(self.num_actions)]
                action = np.argmax(values)

            if self.random_eps > 0.:
                self.random_eps *= decay

            next_state, reward, done, _ = self.env.step(action)

            self.update([(state, action, reward, next_state, done)])
            state = next_state

            if done:
                fails_last_n += 1
                state = self.env.reset()

            if s % report_freq == 0 and s > 0:
                print(f"Step {s}/{n_steps}\nFailures in last {report_freq} "
                      f"steps: {fails_last_n}")
                fails_last_n = 0

    def update(self, history):
        """Update a finite horizon Q Table"""
        for state, action, reward, next_state, done in history:
            for h in range(1, self.num_horizons + 1):

                # Estimate V_(h-1)(s_(t+1)): the reward accrued in h-1 steps
                # from the next state
                if not done:
                    future_q = np.max([
                        self.q_tables[next_state, action_i, h - 1]
                        for action_i in range(self.num_actions)])
                else:
                    future_q = 0.

                # Bootstrap the V_h value off the future value estimate for the
                # next state, added to the reward received in getting to that
                # next state
                q_target = reward + self.gamma * future_q

                # Update the V_h estimate towards the future estimate
                self.q_tables[state, action, h] += self.lr * (
                        q_target - self.q_tables[state, action, h])


if __name__ == "__main__":
    # A gridworld with a few modifications:
    #   Max reward on the far right
    #   Cliffs (0 reward, done) on border state)
    finite_env = FiniteStateCliffworld(state_shape=(5, 5))
    agent = FiniteHorizonAgent(
        finite_env, finite_env.num_states, finite_env.num_actions,
        num_horizons=5)

    agent.learn(n_steps=10000)
