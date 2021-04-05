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

@main.command(help="Visualize the process which generates the inner convex hull describing the safe zone of a given world model")
def safezone():

    lines = [boundary(i) for i in range(20)]

    # lines = [(np.array([-1, -1]), -1), (np.array([1, 1]), 1)]

    from matplotlib import animation
    import scipy

    sols = agent.intersects(lines)
    norms = np.linalg.norm(sols, axis=-1)
    units = sols/norms[:, np.newaxis]
    inverted = units * (1/norms)[:, np.newaxis]

    hull2 = scipy.spatial.ConvexHull(inverted)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xlim(-8, 8)
    ax.set_ylim(-8, 8)
    reds = ax.scatter(x=sols[:, 0], y=sols[:, 1], s=4, c="red")
    blues = ax.scatter(x=units[:, 0], y=units[:, 1], s=2, c="blue")

    for idxs in hull2.simplices:
        ax.plot(inverted[idxs, 0], inverted[idxs, 1])

    for idxs in hull2.simplices:
        ax.plot(sols[idxs, 0], sols[idxs, 1])
    
    scatter_plot = ax.scatter(x=sols[:, 0], y=sols[:, 1], s=4, c="green")
    num_frames = 181
    def update_animation(frame):
        if frame < 30:
            pass
        elif frame < 90:
            lerp = (frame - 30) / 60
            data = (1-lerp)*sols + lerp*inverted
            scatter_plot.set_offsets(data)
        elif frame < 120:
            pass
        elif frame <= 180:
            lerp = (frame - 120) / 60
            data = lerp*sols + (1-lerp)*inverted
            scatter_plot.set_offsets(data)
        return (scatter_plot,)
    frames = list(range(num_frames))
    a = animation.FuncAnimation(fig, update_animation, frames, interval=6, repeat=True)

    plt.show()

@main.command(help="Visualize the boundaries that comprise the world models")
@click.option("--count", default=20)
def boundaries(count):


    lines = list(ints_to_str.coefficients(100))

    for coeffs, c in lines:
        print(f"{coeffs[0]}x + {coeffs[1]}y + {c} = 0")

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xlim(-6, 6)
    ax.set_ylim(-6, 6)
    plt.ion()
    plt.show()
    for i, (coeffs, c) in enumerate(lines):
        if i >= 2:
            sol = agent.solve_optimal_point('x_coord', lines[:i+1])
            if sol is not None:
                ax.scatter(x=[sol[0]], y=[sol[1]], s=4, c='red')
        display_line_coeffs(ax, f"{coeffs[0]}x + {coeffs[1]}y + {c} = 0", coeffs, c)
        plt.pause(0.3)
        # ax.legend()
    plt.ioff()
    plt.show()



def boundary(i):

    ints = ints_to_str.int_to_k_ints(i, dim+1)

    coeffs, c = agent.AgentStates.n_ints_to_boundary(ints)

    return (coeffs, c)

def display_line_coeffs(ax, label, coeffs, c, color=(0, 0, 0.5, .1)):
    if coeffs[0] == 0 and coeffs[1] == 0:
        raise NotImplementedError("Can't plot an indeterminate line. Both x and y coefficients are 0.")
    # if coeff for y is 0, it's vertical 
    elif coeffs[1] == 0:
        y = np.linspace(-10, 10, 2)
        # assume dim=2
        x = (-coeffs[1]*y + c)/coeffs[0]
    else:
        x = np.linspace(-10, 10, 2)
        # assume dim=2
        y = (-coeffs[0]*x + c)/coeffs[1]
    ax.plot(x, y, label=label, c=color)

def display_line(ax, label, m, c, color=(0, 0, 0.5, .1)):

    x = np.linspace(0, 10, 101)
    y = m*x + c
    ax.plot(x, y, label=label, c=color)


if __name__ == "__main__":
    main()
