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

def display_line(ax, label, coeffs, c):

    x = np.linspace(-10, 10, 101)
    # assume dim=2
    y = (-coeffs[0]*x + c)/coeffs[1]
    ax.plot(x, y, label=label, c=(0, 0, 0.5, .1))

lines = [boundary(i) for i in range(10)]

# lines = [(np.array([-1, -1]), -1), (np.array([1, 1]), 1)]

for coeffs, c in lines:
    print(f"{coeffs[0]}x + {coeffs[1]}y + {c} = 0")

ax = plt.gca()
ax.set_xlim(-5, 5)
ax.set_ylim(-5, 5)
plt.ion()
plt.show()
for coeffs, c in lines:
    display_line(ax, f"{coeffs[0]}x + {coeffs[1]}y + {c} = 0", coeffs, c)
    plt.pause(0.01)
    ax.legend()
plt.ioff()
plt.show()
