import timeit
import numpy as np
from scipy.spatial import ConvexHull
from collections import deque
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

import ints_to_str

# For code optimization
class CodeTimer:
    def __init__(self, name=None):
        self.name = " '"  + name + "'" if name else ''
        self.total_time = 0

    def __enter__(self):
        self.total_time -= timeit.default_timer()

    def __exit__(self, exc_type, exc_value, traceback):
        self.total_time += timeit.default_timer()

    def print(self):
        print('Code block' + self.name + ' took: ' + str(self.total_time) + ' s')


class PriorWeightOnBoundary():
    """Implements a prior over world models
    
    Given an enumeration of boundaries, each model is represented
    by a finite binary string, with a 1 at the ith bit indicating
    that the model includes boundary i.

    The prior weight of a given model is proportional to 
    [1/(n+1+X) - 1/(n+2+X)]*2^{-n}, where n is the position of
    the leading bit, and X is some positive constant.
    """
    def __init__(self):
        self.first_unchecked = 0

        # self.weights[i] is an upper bound on the sum of the
        # prior weights of models including boundary i
        self.weights = []

        # self.weights_lb[i] is a lower bound on the sum of the
        # prior weights of models including boundary i
        self.weights_lb = []

        # self.weights_leading is the sum of the prior
        # weights of models that lead with boundary i
        # (no boundaries of larger index)
        self.weights_leading = []

        # Number of boundary hypotheses that have been culled
        self.cull_count = 0
        self.total_lb = 0  # = sum(self.weights_leading)

    def add_boundary(self):
        n = self.first_unchecked + 1

        # = 1/n - 1/(n+1) + 1/2*1/(n+1)
        # this is the prior weight on all models for which the
        # (n - 1)th bit is 1 (i.e. include boundary #n-1)
        # the first two terms are the prior for this being the leading bit,
        # in which case the bit will definitiely be 1
        # the last term is because half of the prior weight when the
        # leading bit is later will also be placed on this bit being 1
        orig_weight = (1. / n) - 1./(2 * (n + 1))

        # cull_count is how many previous bits have been falsified (set to 0)
        # each one reduces by half total prior weight on models which include
        # later boundaries

        new_weight = orig_weight * 2**(-self.cull_count)

        self.weights.append(new_weight)

        orig_weight_lb = 1/n - 1/(n+1)

        new_weight_lb = orig_weight_lb * 2**(-self.cull_count)

        # Lower bound assumes all unchecked boundaries are unviable:
        self.weights_lb.append(new_weight_lb)
        self.weights_leading.append(new_weight_lb)
        self.total_lb += new_weight_lb
        
        for i in range(self.first_unchecked):
            if self.weights_lb[i] != 0:
                # increases because the lower bound was assuming this
                # boundary was inviable
                self.weights_lb[i] += self.weights_lb[-1] / 2

        self.first_unchecked += 1

    def remove_boundary(self, index):

        for i in range(self.first_unchecked):

            if i < index:
                if self.weights[i] != 0:
                    # of all the models which include the boundary to remove,
                    # half of them include boundary i.
                    self.weights[i] -= self.weights[index] / 2
                    self.weights_lb[i] -= self.weights_lb[index] / 2
            
            elif i == index:
                self.weights[i] = 0
                self.weights_lb[i] = 0
                self.weights_leading[i] = 0
            
            elif i > index:
                # half of all models which include boundary i also include
                # the boundary to remove
                self.weights[i] *= 0.5
                self.weights_lb[i] *= 0.5
                self.total_lb -= 0.5 * self.weights_leading[i]
                self.weights_leading[i] *= 0.5

        self.cull_count += 1

    def boundary_weight_up_bound(self, index):
        if index < self.first_unchecked:
            return self.weights[index]

    def boundary_weight_low_bound(self, index):
        if index < self.first_unchecked:
            return self.weights_lb[index]

    def total_ub(self):
        return (
            self.total_lb 
            + 1./(self.first_unchecked + 1) * 2 ** (-self.cull_count)
        )

    def posterior_covered_lb(self):

        return self.total_lb / self.total_ub()


class AgentStates():
    """Tracks the states visited by the agent

    """

    PRIOR_SHIFT = 30

    def __init__(self, dim=2):

        self.dim = dim

        self.weights_tracker = PriorWeightOnBoundary()

        self.boundaries = dict()
        self.last_added_boundary = -1
        self.checked_boundaries = []

        self.states = [] # list of np.arrays of shape (dim,)
        self.actions = [] # list of np.arrays of shape (dim,)

        self.explored_states_hull = None
        self.boundary_states = None

        self.timers = {
            key: CodeTimer(key)
            for key in ['add_boundary', 'add_model', 'add_state']
        }

    def display_status(self):
        print("****************************")
        print("Boundaries: ", self.boundaries)
        print("Last added boundary: ", self.last_added_boundary)
        checked_boundaries_str = ""
        for b in self.checked_boundaries:
            if b:
                checked_boundaries_str += "1"
            else:
                checked_boundaries_str += "0"
        print("Checked boundaries: ", checked_boundaries_str)
        # print("Sum model weights: ", sum_models[0])
        # print("Sum checked model weights: ", sum_checked_models[0])
        print("States: ", self.states)
        print("Boundary states: ", self.boundary_states)

    def add_state(self, state):
        """Consider the next state, and define a hull of safe states
        
        """
        with self.timers['add_state']:
    
            if self.explored_states_hull is None:
                # Create a new convex hull
                if len(self.states) > self.dim + 1:
                    self.explored_states_hull = ConvexHull(
                        np.vstack(self.states))

                    self.boundary_states = self.explored_states_hull.points[
                        self.explored_states_hull.vertices]

                    self.explored_states_hull = ConvexHull(
                        self.boundary_states)
            
            else:
                # Check if new state is in hull
                eqns = self.explored_states_hull.equations
                A = eqns[:,:-1]
                b = eqns[:,-1]

                # If new state in the existing hull, return
                interior = np.all(A.dot(state) + b <= 0)
                if interior:
                    return
                
                # Add state to the points defining the hull of explored states
                self.explored_states_hull = ConvexHull(
                    np.vstack((self.explored_states_hull.points, state))
                )
    
                self.boundary_states = self.explored_states_hull.points[
                    self.explored_states_hull.vertices]
    
                # Drop some additional states for time efficiency
                if (
                        len(self.explored_states_hull.points)
                        > 8 * len(self.boundary_states)
                ):
                    self.explored_states_hull = ConvexHull(
                        self.boundary_states)
            
            # Q: only append if not already in the interior?
            self.states.append(state)

            if self.boundary_states is None:
                self.boundary_states = self.states

            # check the boundaries against the new state
            boundary_keys = tuple(self.boundaries.keys())
            for index_b in boundary_keys:
                boundary = self.boundaries[index_b]
                if not self.state_outside_boundary(boundary, state):
                    self.remove_boundary(index_b)

    def add_next_boundary(self):

        with self.timers['add_boundary']:

            self.last_added_boundary += 1
            n = self.last_added_boundary
            # boundary = n_ints_to_boundary(range(n - (dim+1), n))
            boundary = self.n_ints_to_boundary(
                ints_to_str.int_to_k_ints(n, self.dim+1)
            )

            self.weights_tracker.add_boundary()

            if self.explored_states_hull is None:
                boundary_states = self.states
            else:
                boundary_states = self.explored_states_hull.points
    
            for state in boundary_states:
                if not self.state_outside_boundary(boundary, state):
                    self.checked_boundaries.append(False)
                    # original Q: n instead of n-1?
                    self.weights_tracker.remove_boundary(n - 1)
                    return
                    # break
            else:  # Executed if break is not called
                self.boundaries[n] = boundary
                self.checked_boundaries.append(True)
            return boundary

    def remove_boundary(self, index_b):

        self.checked_boundaries[index_b] = False
        self.weights_tracker.remove_boundary(index_b)
        
        del self.boundaries[index_b]

        if self.checked_boundaries[-1] == False:
            while self.add_next_boundary() is None:
                pass

    def prior_on_ints(self, n):
        return 1. / (
            (n + 1 + self.PRIOR_SHIFT) * (n + 2 + self.PRIOR_SHIFT)
        ) * (self.PRIOR_SHIFT + 1)

    def sum_priors_up_to_n(self, n):
        # return sum([prior_on_ints(i) for i in range(n, 0, -1)])
        return (
            1. / (self.PRIOR_SHIFT + 1) - 1. / (n + 2 + self.PRIOR_SHIFT)
        ) * (self.PRIOR_SHIFT + 1)
    
    @staticmethod
    def state_outside_boundary(boundary, state):
        line_coeffs = np.array(boundary[0])
        b = boundary[1]
        return state.dot(line_coeffs) + b >= 0
    
    @staticmethod
    def n_ints_to_boundary(ints):
        return (
            tuple(
                pos_int_to_int(ints[j] + 1) for j in range(len(ints) - 1)),
            ints[-1] + 1)

    def plot_trajectory(self, boundary_models=False):
        """Plot the animated trajectory and the final convex hull
        
        Perhaps in future the hull could update with state steps
        """
        if self.dim != 2:
            raise NotImplementedError(
                f"Animated trajectory only implemented for dim 2: {self.dim}")

        fig, ax = plt.subplots()
        xdata, ydata = [], []
        ln, = plt.plot([], [], 'rx')

        state_arr = np.vstack(self.states)

        def init():
            ax.set_xlim(-5., 5.)
            ax.set_ylim(-5, 5.)
            return ln,

        def update(frame):
            xdata.append(state_arr[frame, 0])
            ydata.append(state_arr[frame, 1])
            ln.set_data(xdata, ydata)
            return ln,

        x_simplices = np.concatenate(
            (self.boundary_states[:, 0], self.boundary_states[0:1, 0]), axis=0)
        y_simplices = np.concatenate(
            (self.boundary_states[:, 1], self.boundary_states[0:1, 1]), axis=0)

        ax.plot(x_simplices, y_simplices, 'c')
        ax.plot(
            self.boundary_states[:, 0], self.boundary_states[:, 1],
            'o', mec='r', color='none', lw=1, markersize=10
        )

        if boundary_models:
            raise NotImplementedError("TODO - plot boundary line eqns")

        ani = FuncAnimation(
            fig, update, frames=range(state_arr.shape[0]), init_func=init,
            blit=True
        )

        plt.show()


def increment_string(deque_bin_string):
    """Returns whether we had to add a bit"""
    deque_bin_string.append(False)
    increment_string_helper(deque_bin_string)
    if deque_bin_string[-1] == False:
        deque_bin_string.pop()
        return False
    else:
        return True


def increment_string_helper(deque_bin_string):
    if deque_bin_string[0] == False:
        deque_bin_string[0] = True
    else:
        deque_bin_string[0] = False
        deque_bin_string.rotate(-1)
        increment_string_helper(deque_bin_string)
        deque_bin_string.rotate(1)


def n_ints_to_model(ints, dim=2):
    n_boundaries = len(ints) // (dim + 1)
    return tuple(
        (tuple(
            pos_int_to_int(ints[i * (dim+1) + j] + 1) for j in range(dim)),
            ints[i * (dim + 1) + dim] + 1
        ) for i in range(n_boundaries)
    )


def pos_int_to_int(n):
    # modified by max to remove duplicate boundaries (due to previous code mapping both 0 and 1 to 0), but also removes horizontal boundaries.
    if n % 2 == 0:
        return n // 2
    return -((n + 1) // 2)


# TODO some assertion that boundary weights and agent states
#  have the same last n? E.g. add_boundary is loose
def main(state_steps=10, beta=0.9, dim=2):

    agent_states = AgentStates()

    for i in range(state_steps):  # 1000
        agent_states.display_status()

        print("****************************")
        print("Adding world models (boundary); iteration", i)

        while agent_states.weights_tracker.posterior_covered_lb() < beta:
            agent_states.add_next_boundary()

        agent_states.display_status()

        print("****************************")
        print("Adding state; iteration", i)
        agent_states.add_state(np.random.normal(size=2))

        agent_states.display_status()

    # for i in range(20):
    #     print("****************************")
    #     print("Adding models; iteration", i)
    #     # while sum_models[0] / (sum_models[0] + (1 -  sum_checked_models[0])) < beta: # techincally, this means models will cover *at least* beta of the posterior; might be larger than necessary
    #         # add_model_()
    #         # print_all_()
    #     print("****************************")
    #     print("Adding state; iteration", i)
    #     add_state_(np.random.normal(size=2))
    #     print_all_()

    for key in agent_states.timers:
        agent_states.timers[key].print()

    agent_states.plot_trajectory(boundary_models=True)


if __name__ == "__main__":
    main()
