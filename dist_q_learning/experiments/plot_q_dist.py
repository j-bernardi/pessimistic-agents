import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import scipy.stats

from utils import get_beta_plot
from agents import PessimisticAgent, QUANTILES
from env import FiniteStateCliffworld
from mentors import random_safe_mentor
from transition_defs import edge_cliff_reward_slope


def make_beta_animation(success_prob=0.6, q=0.1, n_frames=1000):
    n_sample = 10000
    font = {"size": 16}
    matplotlib.rc('font', **font)
    fig = plt.figure(figsize=(16, 9))
    ax = fig.add_subplot(111)

    ax.set_xlim(0, 1)
    max_y_pointer = [5.]
    ax.set_ylim(-1., 1.1 * max_y_pointer[0])
    ax.axhline()
    ax.axvline(success_prob, alpha=0.5)
    ax.set_xlabel("Scaled Q value")
    ax.set_ylabel("Probability density")

    title = ax.text(
        0.5, max_y_pointer[0], f"Alpha=1, Beta=1 | Q={success_prob}",
        bbox={'facecolor': 'w', 'alpha': 0.5, 'pad': 5}, ha="center")

    # Set the first lines
    xs, f = get_beta_plot(1, 1, n_samples=n_sample)
    ys = f(xs)
    x_quantile = scipy.stats.beta.ppf(q, 1, 1)
    line1, = ax.plot(xs, ys, 'r-')
    line2, = ax.plot([x_quantile, x_quantile], [0., 1.], 'b--')
    quantile_label = ax.text(success_prob, -0.4, f"q_{q}={x_quantile:.3f}")

    def data_gen():
        alpha = 1
        beta = 1
        for _ in range(n_frames):
            success = np.random.rand() < success_prob
            alpha += int(success)
            beta += int(not success)
            yield alpha, beta

    def update(data, max_y_pointer):
        alpha, beta = data

        # Get data
        xs, f = get_beta_plot(alpha, beta, n_samples=n_sample)
        ys = f(xs)
        x_quantile = scipy.stats.beta.ppf(q, alpha, beta)

        # Reset annotations, limits
        new_max = max(max_y_pointer[0], 1.1 * np.max(ys))
        if new_max > max_y_pointer[0]:
            max_y_pointer[0] = new_max
            ax.set_ylim(-1., 1.1 * max_y_pointer[0])
        title.set_text(f"Alpha={alpha}, Beta={beta} | Q={success_prob}")
        title.set_y(max_y_pointer[0])
        quantile_label.set_text(f"Q_{q} = {x_quantile:.3f}")

        # Update plots
        line1.set_xdata(xs)
        line1.set_ydata(ys)
        line2.set_xdata([x_quantile, x_quantile])
        line2.set_ydata([0., 0.9 * max_y_pointer[0]])

        return line1, line2, title, quantile_label

    ani = FuncAnimation(
        fig, lambda d: update(d, max_y_pointer), data_gen, repeat=False,
        interval=10)
    ani.save('/tmp/fakedata.gif', fps=30)
    plt.show()


def make_real_data(npy_cache="dists.npy", x_quantile=1, rollout_steps=10000):
    x, y = (3, 3)
    act_i = 0
    success_prob = x / 7  # x-coord mean reward

    font = {"size": 16}
    matplotlib.rc('font', **font)
    fig = plt.figure(figsize=(16, 9))
    ax = fig.add_subplot(111)

    ax.set_xlim(0, 1.)
    ax.set_ylim(0., 1.)
    ax.axhline(success_prob, alpha=0.5, linestyle='--')
    ax.axvline()
    ax.axvline(x_quantile, alpha=0.5)
    ax.set_xlabel("Q Quantile")
    ax.set_ylabel("Q Value")

    title = ax.text(
        0.5, 1., f"Step=0 | Q={success_prob}",
        bbox={'facecolor': 'w', 'alpha': 0.5, 'pad': 5}, ha="center")

    # Set the first lines
    ys = [0.] * len(QUANTILES)

    line1, = ax.plot(QUANTILES, ys, 'r-')
    # line2, = ax.plot([x_quantile, x_quantile], [0., 1.], 'b--')
    quantile_label = ax.text(success_prob, -0.4, f"q_{x_quantile}={0.:.3f}")

    def get_dists(fname, n_steps):
        if os.path.exists(fname):
            return np.load(fname)
        env = FiniteStateCliffworld(transition_function=edge_cliff_reward_slope)
        state_i = env.map_grid_to_int((x, y))
        agent = PessimisticAgent(
            env.num_actions, env.num_states, env, 0.99,
            mentor=random_safe_mentor, quantile_i=x_quantile, train_all_q=True,
            sampling_strategy="random", update_n_steps=10, batch_size=10,
            horizon_type="inf",
        )
        n_estimators = len(agent.QEstimators)
        distributions = np.empty((n_steps, n_estimators))
        state = int(env.reset())
        for n in range(n_steps):
            if n % 500 == 0:
                print(n, "/", n_steps)
            action, mentor_acted = agent.act(state)
            next_state, r, d, info = env.step(action)
            next_state = int(next_state)
            agent.store_history(
                state, action, r, next_state, d, mentor_acted)
            distributions[n] = np.array([
                agent.QEstimators[i].q_table[state_i, act_i, -1]
                for i in range(n_estimators)])
            agent.update_estimators(mentor_acted=mentor_acted)
            state = next_state
        np.save(fname, distributions)
        return distributions

    def data_gen(distributions):
        for i, row in enumerate(distributions):
            if i % 100 == 0:
                yield i, row

    def update(data):
        step, dist = data
        # Reset annotations, limits
        title.set_text(f"Step={step} | Q={success_prob}")
        quantile_label.set_text(f"q_{x_quantile}={dist[x_quantile]:.3f}")

        # Update plots
        line1.set_ydata(dist)
        # line2.set_xdata([x_quantile, x_quantile])
        # line2.set_ydata([0., 0.9 * max_y_pointer[0]])

        return line1, title, quantile_label

    ani = FuncAnimation(
        fig,
        update,
        lambda: data_gen(get_dists(fname=npy_cache, n_steps=rollout_steps)),
        repeat=False,
        interval=1)
    ani.save('/tmp/realdata.gif', fps=30)
    plt.show()


if __name__ == "__main__":
    # make_beta_animation()
    make_real_data(npy_cache="dists_100k.npy", rollout_steps=100000)
