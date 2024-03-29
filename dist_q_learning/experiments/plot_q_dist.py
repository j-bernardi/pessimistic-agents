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


def make_beta_animation(data_gen, success_prob=0.6, q=0.1):
    """Plots and saves a fake beta distribution over given value"""
    n_sample = 1000
    font = {"size": 16}
    matplotlib.rc('font', **font)
    fig = plt.figure(figsize=(16/2, 9/2))
    ax = fig.add_subplot(111)

    ax.set_xlim(0, 1)
    max_y_pointer = [5.]
    ax.set_ylim(-1., 1.1 * max_y_pointer[0])
    ax.axhline()
    ax.axvline(success_prob, alpha=0.5)
    ax.set_xlabel("Scaled Q value")
    ax.set_ylabel("Probability density")

    title = ax.text(
        0.5, max_y_pointer[0],
        f"Step=0 | Alpha=1.00, Beta=1.00 | Q={success_prob:.3f}",
        bbox={'facecolor': 'w', 'alpha': 0.5, 'pad': 5}, ha="center")

    # Set the first lines
    xs, ys = get_beta_plot(1, 1, n_samples=n_sample)
    x_quantile = scipy.stats.beta.ppf(q, 1, 1)
    line1, = ax.plot(xs, ys, 'r-')
    line2, = ax.plot([x_quantile, x_quantile], [0., 1.], 'b--')
    quantile_label = ax.text(success_prob, -0.8, f"q_{q:.3f}={x_quantile:.3f}")

    def update(data, max_y_point):
        step, alpha, beta = data

        # Get data
        xs, ys = get_beta_plot(alpha, beta, n_samples=n_sample)
        x_quantile = scipy.stats.beta.ppf(q, alpha, beta)

        # Reset annotations, limits
        new_max = max(max_y_point[0], 1.1 * np.max(ys))
        if new_max > max_y_point[0]:
            max_y_point[0] = new_max
            ax.set_ylim(-1., 1.1 * max_y_point[0])
        title.set_text(
            f"Step={step} | Alpha={alpha:.2f}, Beta={beta:.2f} "
            f"| Q={success_prob:.3f}")
        title.set_y(max_y_point[0])
        quantile_label.set_text(f"q_{q:.3f}={x_quantile:.3f}")

        # Update plots
        line1.set_xdata(xs)
        line1.set_ydata(ys)
        line2.set_xdata([x_quantile, x_quantile])
        line2.set_ydata([0., 0.9 * max_y_point[0]])

        return line1, line2, title, quantile_label

    ani = FuncAnimation(
        fig,
        lambda d: update(d, max_y_pointer),
        data_gen,
        repeat=False,
        interval=10)
    return ani


def make_fake_beta_animation(success_prob=0.6, q=0.1, n_frames=1000):

    def data_gen():
        alpha = 1
        beta = 1
        for i in range(n_frames):
            success = np.random.rand() < success_prob
            alpha += int(success)
            beta += int(not success)
            yield i, alpha, beta

    ani = make_beta_animation(data_gen, success_prob=success_prob, q=q)
    ani.save('/tmp/fakedata.gif', fps=10)
    plt.show()


def set_up_env(gamma, act_i=1):
    """Get env and expected targets, rewards"""
    state_shape = (5, 5)
    x, y = (2, 2)
    env = FiniteStateCliffworld(
        state_shape=state_shape,
        transition_function=edge_cliff_reward_slope)
    state_i = env.map_grid_to_int((x, y))
    new_position = np.array((x, y)) + env.map_int_act_to_grid(act_i)
    # x-coord is the mean of the gaussian reward, according to edge cliff slope
    # diff is distance from the optimal col, e.g. col 6 of 7 (index 5)
    diff = int((state_shape[1] - 1 - 1) - x)
    mean_reward = new_position[1] / state_shape[1]
    # Start with the whole future in state 5 (max positive), add the journey
    real_target = (gamma ** diff) * (state_shape[1] - 1 - 1) / state_shape[1]
    for steps_away in reversed(range(diff + 1)):
        r = (mean_reward + steps_away / state_shape[0])
        discounted_r = (1 - gamma) * (gamma ** steps_away) * r
        real_target += discounted_r

    return env, state_i, act_i, real_target, mean_reward


def rollout_agent(
        n_steps, fname, env, state_i, act_i, q_i, gamma, train_all=True):
    alphas_fname = fname.replace(".npy", "_abs.npy")
    if os.path.exists(fname):
        print("Loading", fname)
        return np.load(alphas_fname), np.load(fname)

    agent = PessimisticAgent(
        env.num_actions, env.num_states, env, gamma,
        mentor=random_safe_mentor, quantile_i=q_i, train_all_q=train_all,
        sampling_strategy="random", update_n_steps=10, batch_size=10,
        horizon_type="inf",
        capture_alphas=(state_i, act_i),  # a must!
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

    # Returns
    alpha_betas = np.array(agent.alpha_betas)
    np.save(fname, distributions)
    np.save(alphas_fname, alpha_betas)
    return alpha_betas, distributions


def make_real_beta_animation(
        plot="q", rollout_steps=1000, npy_cache="dists.npy", x_quantile=1):
    gamma = 0.99
    # Some fudge is needed because the gaussian is truncated at 0, so the tail
    # extends further to the right (since we are looking at state 2/5).
    q_value_fudge, mean_r_fudge = 0.04, 0.042
    env, state_i, act_i, real_target, mean_reward = set_up_env(gamma)
    real_target += q_value_fudge
    mean_reward += mean_r_fudge
    npy_cache = npy_cache.replace(".npy", f"_s{state_i}_a{act_i}.npy")
    alphas, _ = rollout_agent(
        rollout_steps, npy_cache, env, state_i, act_i, x_quantile, gamma,
        train_all=False)

    def data_gen():
        for step, row in enumerate(alphas):
            if step % 10 == 0:
                ires, qs = row
                if plot == "q":
                    q_alpha, q_beta = qs
                    yield step, q_alpha, q_beta
                elif plot == "ire":
                    ire_alpha, ire_beta = ires
                    yield step, ire_alpha, ire_beta
                else:
                    raise ValueError(plot)

    ani = make_beta_animation(
        data_gen,
        real_target if plot == "q" else mean_reward,
        QUANTILES[x_quantile])

    ani.save(f"/tmp/realdata_beta_{plot}.gif", fps=10)
    print("Showing", plot, "beta dist")
    plt.show()


def make_real_q_dist(npy_cache="dists.npy", x_quantile=1, rollout_steps=1000):
    gamma = 0.99
    env, state_i, act_i, real_target, _ = set_up_env(gamma)
    font = {"size": 16}
    matplotlib.rc('font', **font)
    fig = plt.figure(figsize=(16/2, 9/2))
    ax = fig.add_subplot(111)

    ax.set_xlim(0, 1.)
    ax.set_ylim(0., 1.)
    ax.axhline(real_target, alpha=0.5, linestyle='--')
    ax.axvline()
    ax.axvline(x_quantile, alpha=0.5)
    ax.set_xlabel("Q quantile 'i'")
    ax.set_ylabel("Qi(s_central, a_0)")

    title = ax.text(
        0.5, 1., f"Step=0 | Q={real_target:.3f}",
        bbox={'facecolor': 'w', 'alpha': 0.5, 'pad': 5}, ha="center")

    # Set the first lines
    ys = [0.] * len(QUANTILES)

    line1, = ax.plot(QUANTILES, ys, 'r-')
    # line2, = ax.plot([x_quantile, x_quantile], [0., 1.], 'b--')
    quantile_label = ax.text(real_target, -0.4, f"q_{x_quantile}={0.:.3f}")

    def data_gen(distributions):
        for i, row in enumerate(distributions):
            if i % 100 == 0:
                yield i, row

    def update(data):
        step, dist = data
        # Reset annotations, limits
        title.set_text(f"Step={step} | Q={real_target:.3f}")
        quantile_label.set_text(f"q_{x_quantile}={dist[x_quantile]:.3f}")

        # Update plots
        line1.set_ydata(dist)
        # line2.set_xdata([x_quantile, x_quantile])
        # line2.set_ydata([0., 0.9 * max_y_pointer[0]])

        return line1, title, quantile_label
    _, dists = rollout_agent(
        rollout_steps, npy_cache, env, state_i, act_i, x_quantile, gamma)
    ani = FuncAnimation(
        fig,
        update,
        lambda: data_gen(dists),
        repeat=False,
        interval=1)
    ani.save('/tmp/realdata_dist.gif', fps=15)
    plt.show()


if __name__ == "__main__":
    # make_fake_beta_animation(n_frames=1000)
    make_real_beta_animation(
        plot="q", npy_cache="dists_50k.npy", rollout_steps=50000)
    make_real_beta_animation(
        plot="ire", npy_cache="dists_50k.npy", rollout_steps=50000)
    # make_real_q_dist(npy_cache="dists_100k.npy", rollout_steps=100000)
