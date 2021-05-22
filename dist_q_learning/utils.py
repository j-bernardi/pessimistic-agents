import numpy as np
import matplotlib.pyplot as plt


def geometric_sum(r_val, gamm, steps):
    # Two valid ways to specify infinite steps
    if steps is None or steps == "inf":
        return r_val / (1. - gamm)
    else:
        return r_val * (1. - gamm ** steps) / (1. - gamm)


def get_beta_plot(alpha, beta, n_samples):
    """Returns xs and f (as in f(xs)) needed to plot the beta curve"""
    sampled_vals = np.random.beta(alpha, beta, size=n_samples)
    n_bins = n_samples // 100
    ps, xs = np.histogram(sampled_vals, bins=n_bins, density=True)
    # convert bin edges to centers
    xs = xs[:-1] + (xs[1] - xs[0]) / 2
    # Consider smoothing
    return xs, ps


def plot_beta(a, b, show=True, n_samples=10000):
    """Plot a beta distribution, given these parameters."""
    ax = plt.gca()
    xs, ys = get_beta_plot(a, b, n_samples)

    ax.set_title(f"Beta distribution for alpha={a}, beta={b}")
    ax.set_ylabel("PDF")
    ax.set_xlabel("E(reward)")
    ax.set_xlim((0, 1))

    ax.plot(xs, ys)
    if show:
        plt.show()

    return ax
