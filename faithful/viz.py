"""
Animation of lines (boundaries) in cliffworld.

Each line corresponds to a positive integer, shown via adding them in order.

"""

import time

import numpy as np
from matplotlib import pyplot as plt
import click

import agent
import ints_to_str

dim = 2

@click.group()
def main():
	pass

@main.command(help="Visualize the boundaries that comprise the world models")
@click.option("--count", default=20)
def boundaries(count):

    lines = [boundary(i) for i in range(count)]

    # lines = [(np.array([-1, -1]), -1), (np.array([1, 1]), 1)]

    for coeffs, c in lines:
        print(f"{coeffs[0]}x + {coeffs[1]}y + {c} = 0")

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xlim(-2, 6)
    ax.set_ylim(-4, 4)
    plt.ion()
    plt.show()
    for i, (coeffs, c) in enumerate(lines):
        if i >= 2:
            sol = agent.solve_optimal_point('x_coord', lines[:i+1])
            if sol is not None:
                ax.scatter(x=[sol[0]], y=[sol[1]], s=4, c='red')
        display_line_coeffs(ax, f"{coeffs[0]}x + {coeffs[1]}y + {c} = 0", coeffs, c)
        plt.pause(0.4)
        # ax.legend()
    plt.ioff()
    plt.show()

def boundary(i):

    ints = ints_to_str.int_to_k_ints(i, dim+1)

    coeffs, c = agent.AgentStates.n_ints_to_boundary(ints)

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


if __name__ == "__main__":
    main()
