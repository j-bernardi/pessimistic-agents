import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import scipy.stats

from utils import get_beta_plot


def make_beta_animation(success_prob=0.6, q=0.1, n_frames=1000):
    n_sample = 10000
    font = {'family': 'normal', 'size': 16}
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
    plt.show()


if __name__ == "__main__":
    make_beta_animation()
