"""The definitions of discrete transition probabilities for the env

A transition function is defined like:
    transitions: {
        state_0: {
            action_0: [trans_0, ...],
            ...
        },
        ...
    }

It maps states to a dict of lists of probabilistic outcomes, indexed
by the action taken from that state. E.g:
    trans_0 = (trans_probability, next_state, reward, done)

So one can make both the reward and next state stochastic with
probability trans_probability

"""
from collections import namedtuple

import scipy.stats
import numpy as np

Transition = namedtuple("Transition", ["prob", "state_next", "reward", "done"])


def linear_slope(x):
    return x


def exponential_slope(x, steepness=2.):
    """Exponential function where x in [0, 1] maps to r in [0, 1]"""
    return np.exp(steepness * (x - 1.))


def is_boundary_state(state, env):
    """Determine whether state is in safe limit of the cliff
    Also handles states outside the limits of the grid.

    Returns:
        True if agent is on a cliff square, or outside the grid.
    """
    zero_boundary = np.any(state < env.cliff_perimeter)
    end_boundary = np.any(state >= env.state_shape - env.cliff_perimeter)
    return zero_boundary or end_boundary


def deterministic_uniform_transitions(env, r=0.7):
    """T is deterministic, r is fixed everywhere

    Arguments:
        env (env.FiniteStateCliffworld): the environment
        r (float): the reward for every non-cliff state.

    Returns:
        transitions (dict): the dict defining gym discrete transitions
            per state, per action.
        reward_range (tuple[float]): the min-nonzero and max reward
            (useful for deferral decisions).
    """
    # Repeated code ###############
    transitions = {
        i: {a: [] for a in range(env.num_actions)}
        for i in range(env.num_states)
    }
    for state_i in range(env.num_states):
        for poss_action in range(env.num_actions):
            new_state_int = env.take_int_step(
                state_i, poss_action, validate=False)
            new_grid = env.map_int_to_grid(new_state_int, validate=False)

            if is_boundary_state(new_grid, env):
                trans = [Transition(1.0, new_state_int, 0., True)]  # r=0, done
                # Repeated code ###############
            else:
                trans = [Transition(1.0, new_state_int, r, False)]  # continue
            transitions[state_i][poss_action] = trans
    return transitions, (r, r)


def edge_cliff_reward_slope(
        env, standard_dev=0.1, slope_func=linear_slope, render=0):
    """A (stochastically) increasing reward from left to right

    Deterministic states

    Cliffs are at every boundary state - e.g. the [0] and [-1] index of
    each dimension returns 0 reward and ends the episode.

    Arguments:
        env (env.FiniteStateCliffworld): the environment
        standard_dev (Optional[float]): The SD of the normal
            distribution. If None, return the mean (do not distribute)
        slope_func (callable): A function mapping x in [0, 1] to a
            reward in [0, 1]
        render (int): if >1, print the trans dict

    Returns:
        transitions (dict): the dict defining gym discrete transitions
            per state, per action.
        reward_range (tuple[float]): the min-nonzero and max reward
            (useful for deferral decisions).
    """

    # TODO - we doing uncertainty in r and next states or just  r?
    #  maybe one at a time, or a function for each...

    def get_truncated_normal(mean, sd=standard_dev, low=0., upp=1.):
        """Truncate normal between 0, 1. Shift mean accordingly."""
        def scale(x): return (x - mean) / sd
        return scipy.stats.truncnorm(scale(low), scale(upp), loc=mean, scale=sd)
    # The mean reward at each x-coord
    mean_rewards = [
        slope_func(col_i / env.state_shape[1])
        for col_i in range(env.state_shape[1])
    ]
    if standard_dev is not None:
        # Normal distributions across the x-axis, with mean shifting upwards
        reward_dists = [get_truncated_normal(mean=mr) for mr in mean_rewards]
        # Which prob quantiles to take over each reward distribution, defining
        # the transition probabilities
        reward_quantiles = np.linspace(0.05, 0.95, num=10)
    else:
        reward_dists = None
        reward_quantiles = None

    # Repeated code ###############
    transitions = {
        i: {a: [] for a in range(env.num_actions)}
        for i in range(env.num_states)
    }
    min_nonzero_r = None
    max_r = None
    for state_i in range(env.num_states):
        for poss_action in range(env.num_actions):
            new_state_int = env.take_int_step(
                state_i, poss_action, validate=False)
            new_grid = env.map_int_to_grid(new_state_int, validate=False)

            if is_boundary_state(new_grid, env):
                trans_list = [Transition(1.0, new_state_int, 0., True)]
                # Repeated code ###############
            else:  # Not a boundary state
                if standard_dev is not None:
                    trans_list = []
                    prev_cd = 0.
                    # Use normal distributions
                    for r in reward_quantiles:
                        if min_nonzero_r is None or r < min_nonzero_r:
                            min_nonzero_r = r
                        if max_r is None or r > max_r:
                            max_r = r
                        cd = reward_dists[new_grid[1]].cdf(r)
                        transition_q = Transition(
                            prob=cd-prev_cd,
                            state_next=new_state_int,
                            reward=r,  # TODO - verify correct r?
                            done=False)
                        trans_list.append(transition_q)
                        prev_cd = cd
                else:
                    # Use the exact mean
                    r = mean_rewards[new_grid[1]]
                    if min_nonzero_r is None or r < min_nonzero_r:
                        min_nonzero_r = r
                    if max_r is None or r > max_r:
                        max_r = r
                    trans_list = [
                        Transition(
                            prob=1.,
                            state_next=new_state_int,
                            reward=mean_rewards[new_grid[1]],  # x-coord
                            done=False)
                    ]

            transitions[state_i][poss_action] = trans_list
    if render:
        for s in transitions:
            print(
                "s", s, "\n\t",
                "\n\t".join(str(transitions[s][a]) for a in transitions[s]))

    assert min_nonzero_r > 0.
    return transitions, (min_nonzero_r, max_r)
