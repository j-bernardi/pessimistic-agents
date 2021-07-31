import scipy.stats
import numpy as np
import matplotlib.pyplot as plt

import jax
import jax.numpy as jnp


def geometric_sum(r_val, gamm, steps):
    # Two valid ways to specify infinite steps
    if steps is None or steps == "inf":
        return r_val / (1. - gamm)
    else:
        return r_val * (1. - gamm ** steps) / (1. - gamm)


def get_beta_plot(alpha, beta, n_samples):
    """Returns xs and f (as in f(xs)) needed to plot the beta curve"""
    xs = np.linspace(0., 1., num=n_samples)
    ps = scipy.stats.beta(alpha, beta).pdf(xs)
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


def stack_batch(batch, vec=False):
    """Return a stack"""
    mod = jnp if vec else np
    # Default axis is 0
    return tuple(mod.stack(x) for x in zip(*batch))


vec_stack_batch = jax.jit(lambda x: stack_batch(x, vec=True))
