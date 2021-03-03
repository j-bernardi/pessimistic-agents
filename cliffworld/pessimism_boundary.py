import numpy as np
from scipy.spatial import ConvexHull
import ints_to_str
from collections import deque

import timeit

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

timers = {}
for key in ['add_boundary', 'add_model', 'add_state']:
    timers[key] = CodeTimer(key)

def main(beta=0.9999, dim=2):
    states = [] # list of np.arrays of shape (dim,)
    actions = [] # list of np.arrays of shape (dim,)

    states_hull_pointer = [None]
    def get_boundary_states():
        if states_hull_pointer[0] is None:
            return states
        else:
            return states_hull_pointer[0].points

    boundaries = dict()
    last_added_boundary = [-1]
    checked_boundaries = [[]]
    boundary_weights = PriorWeightOnBoundary()

    def print_all_():
        print_all(boundaries, last_added_boundary, checked_boundaries, boundary_weights, states, get_boundary_states())
    def add_state_(state):
        add_state(states, states_hull_pointer, state, boundaries, checked_boundaries, last_added_boundary, boundary_weights, dim=dim)
    def add_boundary_():
        add_boundary(get_boundary_states(), checked_boundaries, boundaries, last_added_boundary, boundary_weights, dim=dim)

    for i in range(1000):
        print_all_()
        print("****************************")
        print("Adding models; iteration", i)
        while boundary_weights.posterior_covered_lb() < beta:
            add_boundary_()
        print_all_()
        print("****************************")
        print("Adding state; iteration", i)
        add_state_(np.random.normal(size=2))
        print_all_()

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

        for key in timers:
            timers[key].print()

class PriorWeightOnBoundary():
    # This implements the following prior. Given an enumeration of boundaries,
    # each model is represented by a finite binary string, with a 1 at the ith
    # bit indicating that the model includes boundary i. The prior weight of
    # a given model is proportional to [1/(n+1+X) - 1/(n+2+X)]*2^{-n}, where n
    # is the position of the leading bit, and X is some positive constant.
    def __init__(self):
        self.first_unchecked = 0
        self.weights = [] # self.weights[i] is an upper bound on the sum of the
            # prior weights of models including boundary i
        self.weights_lb = [] # self.weights[i] is a lower bound on the sum of the
            # prior weights of models including boundary i
        self.weights_leading = [] # self.weights_leading is the sum of the prior
            # weights of models that lead with boundary i (no boundaries of larger index)
        self.cull_count = 0
        self.total_lb = 0 # = sum(self.weights_leading)

    def add_boundary(self):
        n = self.first_unchecked + 1
        orig_weight = (1/n - 1/(2 * (n+1))) # = 1/n - 1/(n+1) + 1/2*1/(n+1)
            # this is the prior weight on all models for which the (n-1)th bit is 1 (i.e. include boundary #n-1)
            # the first two terms are the prior for this being the leading bit,
            # in which case the bit will definitiely be 1
            # the last term is because half of the prior weight when the leading bit is later
            # will also be placed on this bit being 1
        new_weight = orig_weight * 2**(-self.cull_count) # cull_count is how many previous bits have been falsified (set to 0)
            # each one reduces by half total prior weight on models which include later boundaries
        self.weights.append(new_weight)
        orig_weight_lb = 1/n - 1/(n+1)
        new_weight_lb = orig_weight_lb * 2**(-self.cull_count)
        self.weights_lb.append(new_weight_lb) # assuming all unchecked boundaries are unviable
        self.weights_leading.append(new_weight_lb)
        self.total_lb += new_weight_lb
        for i in range(self.first_unchecked):
            if self.weights_lb[i] != 0:
                self.weights_lb[i] += self.weights_lb[-1] / 2
                # increases because the lower bound was assuming this boundary was inviable
        self.first_unchecked += 1

    def remove_boundary(self, index):
        for i in range(self.first_unchecked):
            if i < index:
                if self.weights[i] != 0:
                    self.weights[i] -= self.weights[index] / 2 # of all the models which
                    # include the boundary to remove, half of them include boundary i.
                    self.weights_lb[i] -= self.weights_lb[index] / 2
            if i == index:
                self.weights[i] = 0
                self.weights_lb[i] = 0
                self.weights_leading[i] = 0
            if i > index:
                self.weights[i] *= 0.5 # half of all models which include boundary
                # i also include the boundary to remove
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
        return self.total_lb + 1/(self.first_unchecked + 1) * 2**(-self.cull_count)

    def posterior_covered_lb(self):
        return self.total_lb / self.total_ub()


def print_all(boundaries, last_added_boundary, checked_boundaries, boundary_weights, states, boundary_states):
    print("****************************")
    print("Boundaries: ", boundaries)
    print("Last added boundary: ", last_added_boundary[0])
    checked_boundaries_str = ""
    for bool in checked_boundaries[0]:
        if bool:
            checked_boundaries_str += "1"
        else:
            checked_boundaries_str += "0"
    print("Checked boundaries: ", checked_boundaries_str)
    # print("Sum model weights: ", sum_models[0])
    # print("Sum checked model weights: ", sum_checked_models[0])
    print("States: ", states)
    print("Boundary states: ", boundary_states)

def add_boundary(boundary_states, checked_boundaries, boundaries, last_added_boundary, boundary_weights, dim=2):
    with timers['add_boundary']:
        last_added_boundary[0] += 1
        n = last_added_boundary[0]
        boundary = n_ints_to_boundary(ints_to_str.int_to_k_ints(n, dim+1))
        boundary_weights.add_boundary()

        for state in boundary_states:
            if not check_boundary_state(boundary, state):
                checked_boundaries[0].append(False)
                boundary_weights.remove_boundary(n-1) # n instead of n-1?
                return
                # break
        else:
            boundaries[n] = boundary
            checked_boundaries[0].append(True)
        return boundary

# def im_just_curious(N=10000, n=20):
#     samples = []
#     for _ in range(n):
#         states = np.random.normal(size=(N,2))
#         hull = ConvexHull(states)
#         samples.append(hull.vertices.shape[0])
#     return sum(samples)/n

def add_state(states, states_hull_pointer, state, boundaries, checked_boundaries, last_added_boundary, boundary_weights, dim=2):
    with timers['add_state']:
        boundary_states = None
        if states_hull_pointer[0] is None:
            if len(states) > dim + 1:
                states_hull_pointer[0] = ConvexHull(np.vstack(states))
                boundary_states = states_hull_pointer[0].points[states_hull_pointer[0].vertices]
                states_hull_pointer[0] = ConvexHull(boundary_states)
        else: # if states_hull_pointer[0] is not None:
            # check if new state is in hull
            eqns = states_hull_pointer[0].equations
            A = eqns[:,:-1]
            b = eqns[:,-1]
            interior = np.all(A.dot(state) + b <= 0)
            if interior:
                return
            # add state to hull
            states_hull_pointer[0] = ConvexHull(np.vstack((states_hull_pointer[0].points, state)))
            boundary_states = states_hull_pointer[0].points[states_hull_pointer[0].vertices]
            if len(states_hull_pointer[0].points) > 8 * len(boundary_states): # just for time efficiency
                states_hull_pointer[0] = ConvexHull(boundary_states) # no need to store all those other states
        states.append(state)
        if boundary_states is None:
            boundary_states = states
        # check the boundaries against the new state
        boundary_keys = tuple(boundaries.keys())
        for index_b in boundary_keys:
            boundary = boundaries[index_b]
            if not check_boundary_state(boundary, state):
                remove_boundary(index_b, boundary_states, checked_boundaries, boundaries, last_added_boundary, boundary_weights, dim=dim)

def remove_boundary(index_b, boundary_states, checked_boundaries, boundaries, last_added_boundary, boundary_weights, dim=2):
    checked_boundaries[0][index_b] = False
    boundary_weights.remove_boundary(index_b)
    del boundaries[index_b]
    if checked_boundaries[0][-1] == False:
        while add_boundary(boundary_states, checked_boundaries, boundaries, last_added_boundary, boundary_weights, dim=dim) is None:
            pass

def increment_string(deque_bin_string): # returns whether we had to add a bit
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

PRIOR_SHIFT = 30
def prior_on_ints(n):
    return 1/((n+1+PRIOR_SHIFT)*(n+2+PRIOR_SHIFT)) * (PRIOR_SHIFT+1)

def sum_priors_up_to_n(n):
    # return sum([prior_on_ints(i) for i in range(n, 0, -1)])
    return (1/(PRIOR_SHIFT+1) - 1/(n+2+PRIOR_SHIFT)) * (PRIOR_SHIFT+1)

def check_boundary_state(boundary, state):
    return state.dot(np.array(boundary[0])) + boundary[1] >= 0

def n_ints_to_boundary(ints):
    return (tuple(pos_int_to_int(ints[j]+1) for j in range(len(ints)-1)), ints[-1]+1)

def n_ints_to_model(ints, dim=2):
    n_boundaries = len(ints) // (dim + 1)
    return tuple((tuple(pos_int_to_int(ints[i*(dim+1)+j]+1) for j in range(dim)), ints[i*(dim+1)+dim]+1) for i in range(n_boundaries))

def pos_int_to_int(n):
    if n == 1:
        return 0
    if n % 2 == 0:
        return n // 2
    return -(n - 1) // 2
