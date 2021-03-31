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

TODO:
    Should it end the episode or be 0 forever (e.g. all actions
    map to itself, with p=1, r=0)?
"""
from collections import namedtuple

import scipy.stats
import numpy as np

Transition = namedtuple("Transition", ["prob", "state_next", "reward", "done"])


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
    return transitions


def edge_cliff_reward_slope(env):
    """A stochastically increasing reward from left to right

    Deterministic states

    Cliffs are at every boundary state - e.g. the [0] and [-1] index of
    each dimension returns 0 reward and ends the episode.

    Arguments:
        env (env.FiniteStateCliffworld): the environment

    Returns:
        transitions (dict): the dict defining gym discrete transitions
            per state, per action.
    """

    # TODO - we doing uncertainty in r and next states or just  r?
    #  maybe one at a time, or a function for each...

    def get_truncated_normal(mean, sd=1., low=0., upp=1.):
        """Truncate normal between 0, 1. Shift mean accordingly."""
        def scale(x): return (x - mean) / sd
        return scipy.stats.truncnorm(scale(low), scale(upp), loc=mean, scale=sd)
    # A range of distributions across the x-axis, with mean shifting upwards
    reward_dists = [
        get_truncated_normal(mean=(col_i / env.state_shape[1]))
        for col_i in range(env.state_shape[1])
    ]
    # Which prob quantiles to take over each reward distribution, defining the
    # transition probabilities
    # TODO should probably start from > 0. as 0. is terrible, etc
    #  OR boundaries = 0 forever...
    reward_quantiles = np.linspace(0., 1., num=10)

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
                trans = []
                prev_cd = 0.
                for r in reward_quantiles:
                    cd = reward_dists[new_grid[1]].cdf(r)
                    transition_q = Transition(
                        prob=cd-prev_cd,
                        state_next=new_state_int,
                        reward=r,
                        done=False)
                    trans.append(transition_q)
                    prev_cd = cd
            transitions[state_i][poss_action] = trans
    return transitions
