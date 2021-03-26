import numpy as np


def random_mentor(state, kwargs=None):
    """Take a random action, with caution thrown to the wind

    Args:
        state (Optional): dummy argument for this mentor

    Optional kwargs:
        num_actions (int): number of valid actions. Default to 4.

    Returns:
        action (int): action in the range of num_actions (default 4)
    """
    if kwargs is None:
        kwargs = {}
    num_actions = kwargs.get("num_actions", 4)

    return np.random.randint(num_actions)


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
        action (int): the action to take to move us to safety.
    """
    if kwargs is None:
        kwargs = {}
    state_shape = kwargs["state_shape"]
    num_actions = kwargs.get("num_actions", 4)

    closest_dim_from_0 = int(np.argmin(state))
    closest_val_from_0 = state[closest_dim_from_0]

    distance_from_end = state_shape - state
    assert distance_from_end.size == 2  # (checking 2d is correct)
    closest_dim_from_end = int(np.argmin(distance_from_end))
    closest_val_from_end = distance_from_end[closest_dim_from_end]

    # step in direction opposite to min
    if closest_dim_from_0 < closest_val_from_end:
        # Move perpendicularly to closest_dim_from_0
        # Need an INCREASING action: 1 or 3, from action mapping (in env.py)
        # '1' is in the 0 / row dim)
        acts = (1, 3)
        return acts[closest_dim_from_end]

    elif closest_val_from_end < closest_val_from_0:
        # Move perpendicularly to closest_dim_from_end
        # Need a DECREASING action: 0 or 2, from action mapping (in env.py)
        # 0 is in the 0 / row dim
        acts = (0, 2)
        return acts[closest_dim_from_end]

    else:
        # break tie randomly
        return np.random.randint(num_actions)


def random_safe_mentor(state, kwargs=None):
    """Take a random action, but mask any cliff-stepping actions

    Args:
        state (np.ndarray): the 2d array of the [row, col] coordinate
            of the agent.

    Required kwargs:
        state_shape (np.ndarray): row, col shape

    Optional kwargs:
        num_actions (int): number of valid actions. Default to 4.
        border_depth (np.ndarray): depth of the cliff for each dimension.
            Default to np.ones(4)

    Returns:
        action (int): the action to take to move us to safety.
    """
    raise NotImplementedError(
        "Not done yet - will require more careful info about the env")
