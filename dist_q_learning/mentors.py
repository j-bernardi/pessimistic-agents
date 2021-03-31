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


def random_safe_mentor(state, kwargs=None):
    """Take a random action, but mask any cliff-stepping actions

    Args:
        state (np.ndarray): the 2d array of the [row, col] coordinate
            of the agent.

    Required kwargs:
        state_shape (np.ndarray): row, col shape
        border_depth (int): depth of the cliff for each dimension.
            Default to 1.

    Returns:
        action (tuple): the random action (of all safe actions)

    TODO:
        At the moment, border_depth assumes a uniform border depth.
    """

    if kwargs is None:
        kwargs = {}
    state_shape = kwargs["state_shape"]
    border_depth = kwargs.get("border_depth", 1)

    can_subtract = state > border_depth  # e.g. NOT index 1
    can_add = state < (state_shape - 1 - border_depth)  # e.g. NOT index -2

    adding_moves = ((+1, 0), (0, +1))
    subtracting_moves = ((-1, 0), (0, -1))
    to_choose_from = (
        [m for i, m in enumerate(adding_moves) if can_add[i]]
        + [m for i, m in enumerate(subtracting_moves) if can_subtract[i]])
    return random.choice(to_choose_from)
