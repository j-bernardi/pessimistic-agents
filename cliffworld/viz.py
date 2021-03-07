"""
Animation of lines (boundaries) in cliffworld.

Each line corresponds to a positive integer, shown via adding them in order.

"""


import numpy as np
from matplotlib import pyplot as plt
import time

import pessimism_boundary
import ints_to_str

dim = 2

def boundary(i):

    ints = ints_to_str.int_to_k_ints(i, dim+1)

    coeffs, c = pessimism_boundary.AgentStates.n_ints_to_boundary(ints)

    return (coeffs, c)

def display_line_coeffs(ax, label, coeffs, c, color=(0, 0, 0.5, .1)):

    x = np.linspace(-10, 10, 101)
    # assume dim=2
    y = (-coeffs[0]*x + c)/coeffs[1]
    ax.plot(x, y, label=label, c=color)

def display_line(ax, label, m, c, color=(0, 0, 0.5, .1)):

    x = np.linspace(0, 10, 101)
    y = m*x + c
    ax.plot(x, y, label=label, c=color)

def display_boundaries(args):

    lines = [boundary(i) for i in range(args.count)]

    # lines = [(np.array([-1, -1]), -1), (np.array([1, 1]), 1)]

    for coeffs, c in lines:
        print(f"{coeffs[0]}x + {coeffs[1]}y + {c} = 0")

    ax = plt.gca()
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    plt.ion()
    plt.show()
    for coeffs, c in lines:
        display_line_coeffs(ax, f"{coeffs[0]}x + {coeffs[1]}y + {c} = 0", coeffs, c)
        plt.pause(0.1)
        ax.legend()
    plt.ioff()
    plt.show()

def display_policies(args):

    vecs = [pessimism_boundary.sample_line_policy(dim) for i in range(args.count)]
    grads = [v[1]/v[0] for v in vecs]

    for grad in grads:
        print(f"y = {grad}*x + 0")

    ax = plt.gca()
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    plt.ion()
    plt.show()
    for grad in grads:
        display_line(ax, f"y = {grad}*x + 0", grad, 0)
        plt.pause(0.1)
        ax.legend()
    plt.ioff()
    plt.show()

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Visualizations of pessimistic cliffworld agent.")

    subparsers = parser.add_subparsers(required=True)
    
    boundaries = subparsers.add_parser("boundaries")
    boundaries.add_argument("--count", type=int, default=30)
    boundaries.set_defaults(func=display_boundaries)

    policies = subparsers.add_parser("policies")
    policies.add_argument("--count", type=int, default=30)
    policies.set_defaults(func=display_policies)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
