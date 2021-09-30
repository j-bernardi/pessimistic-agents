import scipy.stats
import numpy as np
import matplotlib.pyplot as plt

import jax
import jax.numpy as jnp
import torch as tc
from tests.check_gpu import check_gpu


def set_gpu():
    torch_gpu_available = check_gpu()
    dev_i = None
    if torch_gpu_available and tc.cuda.device_count() > 1:
        dev_i = int(input("Input device number (or 'cpu'): "))
        tc.cuda.device(dev_i)
    return dev_i


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


def stack_batch(batch, lib=np):
    """Return a stack"""
    # Default axis is 0
    return tuple(lib.stack(x) for x in zip(*batch))


vec_stack_batch = jax.jit(lambda x: stack_batch(x, lib=jnp))


class JaxRandom:
    """Singleton for jax random numbers

    Ensures that a
    """

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            print("Creating new instance of jax random key")
            cls.key = jax.random.PRNGKey(0)
            cls._instance = super(JaxRandom, cls).__new__(cls)
        return cls._instance

    def update_key(self):
        """Split and update the internal key"""
        key, subkey = jax.random.split(self.key)
        self.key = subkey

    def uniform(self, *args, **kwargs):
        rand_nums = jax.random.uniform(self.key, *args, **kwargs)
        self.update_key()
        return rand_nums

    def choice(self, *args, **kwargs):
        choices = jax.random.choice(self.key, *args, **kwargs)
        self.update_key()
        return choices

    def randint(self, *args, **kwargs):
        rand = jax.random.randint(self.key, *args, **kwargs)
        self.update_key()
        return rand
