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


def edge_cliff_reward_slope(rows, cols, num_actions):
    """A stochastically increasing reward from left to right

    Cliffs are at every boundary state - e.g. the [0] and [-1] index of
    each dimension returns 0 reward and ends the episode.

    TODO:
        Should it end the episode or be 0 forever (e.g. all actions
        map to itself, with p=1, r=0)?
    """

    # TODO - we doing uncertainty in r and next states or just  r?
    #  maybe one at a time, or a function for each...
    num_states = rows * cols
    transitions = {
        i: {a: [] for a in range(num_actions)}
        for i in range(num_states)
    }

    for row in rows:
        for col in cols:
            if row in (0, len(rows) - 1) or col in (0, len(cols) - 1):
                # Cliff state
                pass
            else:
                # Define some slope function for r
                pass
