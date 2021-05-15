import random

import numpy as np


def random_mentor(state, kwargs=None):
    """Take a random action, with caution thrown to the wind

    Args:
        state (Optional): dummy argument for this mentor

    Optional kwargs:
        num_actions (int): number of valid actions. Default to 4.

    Returns:
        action (tuple): random 2d grid action, moving 1 place
    """
    if kwargs is None:
        kwargs = {}
    num_actions = kwargs.get("num_actions", 4)
    all_acts = ((+1, 0), (0, +1), (-1, 0), (0, -1))
    return random.choice(all_acts)


def prudent_mentor(state, kwargs=None):
    """Step away from the closest edge (break tie randomly)

    Args:
        state (np.ndarray): the 2d array of the [row, col] coordinate
            of the agent.

    Required kwargs:
        state_shape (np.ndarray): row, col shape

    Optional kwargs:
        num_actions (int): number of valid actions. Default to 4.

    Returns:
        action (tuple): the action to take to move us away from the edge
    """
    if not isinstance(state, np.ndarray):
        raise ValueError("State represented in [row, col] np.ndarray")
    if kwargs is None:
        kwargs = {}
    state_shape = kwargs["state_shape"]

    closest_dim_from_0 = int(np.argmin(state))
    closest_val_from_0 = state[closest_dim_from_0]

    distance_from_end = state_shape - state - 1
    assert distance_from_end.size == 2  # (checking 2d is correct)
    closest_dim_from_end = int(np.argmin(distance_from_end))
    closest_val_from_end = distance_from_end[closest_dim_from_end]

    coord_increasing_acts = ((+1, 0), (0, +1))
    coord_decreasing_acts = ((-1, 0), (0, -1))
    # step in direction opposite to min distance to edge
    if np.all(state_shape % 2 == 1) and np.all(state == state_shape // 2):
        # Break centre (if a centre exists) randomly
        return random.choice(coord_decreasing_acts + coord_increasing_acts)
    elif closest_val_from_0 < closest_val_from_end:
        # Move perpendicularly to closest_dim_from_0
        # Need an INCREASING action: 1 or 3, from action mapping (in env.py)
        # '1' is in the 0 / row dim)
        return coord_increasing_acts[closest_dim_from_0]

    elif closest_val_from_end < closest_val_from_0:
        # Move perpendicularly to closest_dim_from_end
        # Need a DECREASING action: 0 or 2, from action mapping (in env.py)
        # 0 is in the 0 / row dim
        return coord_decreasing_acts[closest_dim_from_end]

    else:
        # Break distance to 2-edges tie randomly (centre already checked)
        return random.choice(
            [coord_increasing_acts[closest_dim_from_0],
             coord_decreasing_acts[closest_dim_from_end]])


def random_safe_mentor(state, kwargs=None, avoider=False):
    """Take a random, safe action (except in special state, if flagged)

    Args:
        state (np.ndarray): the 2d array of the [row, col] coordinate
            of the agent.
        kwargs (dict): see below
        avoider (bool): If false, regular random safe action. Else,
            Avoids a state-action with a given probability, all
            specified in kwargs

    Optional kwargs:
        state_from (tuple): default to env.shape - 1 - border_depth
        action_from: default to (0, -1)
        action_from_prob: relative weight of mentor taking action vs
            other allowed actions. Default 0.01
        border_depth (int): depth of the cliff for each dimension.
            Default to 1.

    Required kwargs:
        state_shape (np.ndarray): row, col shape

    Returns:
        action (tuple): the random action (of all safe actions)

    TODO:
        At the moment, border_depth assumes a uniform border depth.
    """
    if not isinstance(state, np.ndarray):
        raise ValueError("State represented in [row, col] np.ndarray")
    if kwargs is None:
        kwargs = {}
    state_shape = kwargs.pop("state_shape")

    # OPT KWARGS
    border_depth = kwargs.pop("border_depth", 1)
    # OPT AVOIDER KWARGS - default None
    state_from_tups = kwargs.pop("states_from", [])
    avoid_action_froms = kwargs.pop("actions_from", [])
    avoid_action_from_weights = kwargs.pop("avoid_act_probs", [])
    assert not kwargs  # gotta catch em all

    can_subtract = state > border_depth  # e.g. NOT index 1
    can_add = state < (state_shape - 1 - border_depth)  # e.g. NOT index -2

    adding_moves = ((+1, 0), (0, +1))  # 1, 3
    subtracting_moves = ((-1, 0), (0, -1))  # 0, 2
    to_choose_from = (
        [m for i, m in enumerate(adding_moves) if can_add[i]]
        + [m for i, m in enumerate(subtracting_moves) if can_subtract[i]])

    num_valid_acts = len(to_choose_from)
    weights = np.ones((num_valid_acts,))
    avoid_state = [
        np.all(np.array(state) == np.array(tup)) for tup in state_from_tups]
    if avoider and any(avoid_state):
        avoid_state_idx = np.flatnonzero(avoid_state)
        if len(avoid_state_idx) != 1:
            raise NotImplementedError(
                "Only one avoid act per state, at the moment")
        avoid_state_idx = avoid_state_idx[0]
        avoid_action_from = avoid_action_froms[avoid_state_idx]
        action_weight = avoid_action_from_weights[avoid_state_idx]
        if avoid_action_from not in to_choose_from:
            raise ValueError("Not intended!", to_choose_from, avoid_action_from)
        idx = to_choose_from.index(avoid_action_from)
        weights[idx] = action_weight
    else:
        idx = -1
    indices = np.arange(start=0, stop=num_valid_acts, step=1)
    chosen_act_i = np.random.choice(indices, p=weights/np.sum(weights))

    # if avoider and chosen_act_i == idx:
    #     print("UNLIKELY ACTION TAKEN BY MENTOR")

    return to_choose_from[chosen_act_i]


def cartpole_safe_mentor(state, kwargs=None):
    # cartpole policy from here: https://towardsdatascience.com/how-to-beat-the-cartpole-game-in-5-lines-5ab4e738c93f
    state
    theta, w = state[2:4]
    if abs(theta) < 0.03:
        return 0 if w < 0 else 1
    else:
        return 0 if theta < 0 else 1
