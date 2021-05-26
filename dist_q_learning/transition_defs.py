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
import random
import scipy.stats
import numpy as np

from collections import namedtuple

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
                            reward=r,
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


def reward_slope_stochastic_trans(
        env, standard_dev=0.1, leading_trans_prob=0.6, slope_func=linear_slope,
        render=0):
    """A fully stochastic environment

    Reward increases left to right. Distributed with gaussian (with
    shifting mean), unless std is None (see args). Transition function
    defaults to deterministic outcome with given probability, sharing
    the remaining probability among the other *safe* actions.

    Cliffs are at every boundary state - e.g. the [0] and [-1] index of
    each dimension returns 0 reward and ends the episode.

    Arguments:
        env (env.FiniteStateCliffworld): the environment
        standard_dev (Optional[float]): The SD of the normal
            distribution. If None, return the mean (do not distribute)
        leading_trans_prob (float): The probability that the
            deterministic action takes effect. 1-val is shared among the
            remaining safe actions.
        slope_func (callable): A function mapping x in [0, 1] to a
            reward in [0, 1]
        render (int): if >1, print the trans dict

    Returns:
        transitions (dict): the dict defining gym discrete transitions
            per state, per action.
        reward_range (tuple[float]): the min-nonzero and max reward
            (useful for deferral decisions).

    TODO: leading states probabilities could be random in a range e.g
        0.5-0.8
    """

    trans_p_cache = {}

    def get_truncated_normal(mean, sd=standard_dev, low=0., upp=1.):
        """Truncate normal between 0, 1. Shift mean accordingly."""
        def scale(x): return (x - mean) / sd
        return scipy.stats.truncnorm(scale(low), scale(upp), loc=mean, scale=sd)

    def split_to_all_trans(p, ns, rw, dn, other_states, other_rws=None):
        """Take the arguments for a transition, and split it into
           stochastic transitions

        We assume that all the next-states are safe, therefore they are
        NOT 'done'. This is how the next states are generated - i.e. we
        can't stochastically end up failing. Agent has to take positive
        action.

        Args:
            p: the probability reserved for this transition. Split it
                into the 4 transitions, so that together they sum to p.
            ns: the 'deterministic' next state
            rw: the 'deterministic' reward
            dn: the 'deterministic' done (or not)
            other_states: the other states that we could transition to.
                Their probabilities are random, summing to 1-p.
            other_rws: reward to give to other transitions (e.g. if
                leading next is for done). Defaults to rw, if not passed
        """
        if not len(other_states):
            return [Transition(p, ns, rw, dn)]
        p_lead = p * leading_trans_prob
        split_trans = [Transition(p_lead, ns, rw, dn)]
        # 3 numbers - must sum to p * (1. - leading_trans_prob)
        # Cache them so same numbers are used across rewards always
        if ns in trans_p_cache and tuple(other_states) in trans_p_cache[ns]:
            seed_ps = trans_p_cache[ns][tuple(other_states)]
        else:
            if ns not in trans_p_cache:
                trans_p_cache[ns] = {}
            seed_ps = np.random.random(len(other_states))
            seed_ps = seed_ps / np.sum(seed_ps)
            # cache for next time
            trans_p_cache[ns][tuple(other_states)] = seed_ps
        other_ps = p * (1. - leading_trans_prob) * seed_ps

        sum_ps = p_lead + np.sum(other_ps)
        assert np.isclose(sum_ps, p, atol=0.001), f"Not close {p}: {sum_ps}"

        if other_rws is None:
            other_rws = [rw for _ in other_states]

        for other_r, other_s, other_p in zip(other_rws, other_states, other_ps):
            split_trans.append(Transition(other_p, other_s, other_r, False))

        assert np.isclose(np.sum([t[0] for t in split_trans]), p, atol=0.001), (
            f"NOT {p}: {np.sum([t[0] for t in split_trans])}")
        return split_trans

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
        all_safe_adjacent = []
        state_i_grid = env.map_int_to_grid(state_i)

        # Collect all the possible outcomes, for stochastic transitions
        for poss_action in range(env.num_actions):
            poss_state, is_safe = env.take_int_step(
                state_i, poss_action, validate=False, return_is_safe=True)
            if is_safe:
                all_safe_adjacent.append(poss_state)

        action_dict = {}
        for action in range(env.num_actions):
            new_state_int = env.take_int_step(state_i, action, validate=False)
            new_grid = env.map_int_to_grid(new_state_int, validate=False)
            if is_boundary_state(new_grid, env):
                action_dict[action] = [Transition(1.0, new_state_int, 0., True)]
                continue
            # Remove the leading (i.e. deterministic) transition resulting from
            # 'action' from the possibilities, if it's present (e.g. was safe)
            excluded_safe_adj = [
                s for s in all_safe_adjacent
                if s != new_state_int
                and np.all(env.map_int_to_grid(s)) >= 0
                and np.all(env.map_int_to_grid(s) < env.state_shape)
            ]
            if standard_dev is not None:
                trans_list = []
                prev_cd = 0.
                # Use normal distributions
                for i, r in enumerate(reward_quantiles):
                    if min_nonzero_r is None or r < min_nonzero_r:
                        min_nonzero_r = r
                    if max_r is None or r > max_r:
                        max_r = r
                    cd = reward_dists[new_grid[1]].cdf(r)
                    other_rs = []
                    # Transform the gaussian left or right as required
                    for s in excluded_safe_adj:
                        grid = env.map_int_to_grid(s)
                        diff = grid[1] - new_grid[1]
                        new_quant_i = i + diff
                        if new_quant_i < 0\
                                or new_quant_i >= len(reward_quantiles):
                            new_quant_i = i  # can't go outside range
                        assert not np.abs(diff) > 2, (
                            f"from: {state_i_grid} lead: {new_grid}, "
                            f"stoch: {grid} (diff {diff}, i = {i})")
                        other_rs.append(reward_quantiles[new_quant_i])
                    assert 0. <= cd <= 1
                    transition_qs = split_to_all_trans(
                        p=cd-prev_cd,  # split this p amongst transitions
                        ns=new_state_int,
                        rw=r,
                        dn=False,
                        other_states=excluded_safe_adj,
                        other_rws=other_rs,
                    )
                    trans_list.extend(transition_qs)
                    prev_cd = cd
            else:
                # Use the exact mean
                r = mean_rewards[new_grid[1]]
                # states to transition to - use their exact mean
                other_rs = [
                    mean_rewards[env.map_int_to_grid(s)]
                    for s in excluded_safe_adj]
                if min_nonzero_r is None or r < min_nonzero_r:
                    min_nonzero_r = r
                if max_r is None or r > max_r:
                    max_r = r
                trans_list = split_to_all_trans(
                    p=1.,  # deterministic reward, split to ns
                    ns=new_state_int,
                    rw=mean_rewards[new_grid[1]],
                    dn=False,
                    other_states=excluded_safe_adj,
                    other_rws=other_rs,
                )
            action_dict[action] = trans_list

        transitions[state_i] = action_dict
    if render:
        for s in transitions:
            print(
                "s", s, "\n\t",
                "\n\t".join(str(transitions[s][a]) for a in transitions[s]))

    assert min_nonzero_r > 0.
    return transitions, (min_nonzero_r, max_r)


def adjustment_wrapper(
        transitions, states_from, actions_from, states_to, event_probs,
        event_rewards, original_act_rewards,
):
    """Add an event square to an already-made transitions dict

    Args:
        transitions (dict): The transition dict for every
            (state, action) in the current env. E.g:
                transitions[state][action] = (prob, next_s, r, done)
        states_from (list[int]):
        actions_from (list[int]):
        states_to (list[int]): state mapped to
        event_probs (list[float]): probability of the event happening
        event_rewards (list[float]): reward received in the event
        original_act_rewards (list[float]): change the original rewards
            associated with (s, a) to this value (for all actions).
            None does nothing.

    Returns:
        transitions (dict): updated dict
    """
    event_rewards = [r or 0. for r in event_rewards]
    for adjust_tuple in zip(
            states_from, actions_from, states_to, event_probs, event_rewards,
            original_act_rewards):

        state_from, action_from, state_to, p_event, reward, adjust_orig_r =\
            adjust_tuple
        current_list = transitions[state_from][action_from]
        current_num = len(current_list)

        # Adjust  of the original transitions at transitions[s][a]
        new_list = [
            Transition(
                prob=x[0] - (p_event / current_num),
                state_next=x[1],
                # adjust if requested (not None), but if transition leads to
                # done, keep original reward (probably 0.)
                reward=(adjust_orig_r or x[2]) if not x[3] else x[2],
                done=x[3],
            ) for x in current_list]
        # Add the boosted one!
        new_list.append(
            Transition(
                prob=p_event,
                state_next=state_to,
                reward=reward,
                done=False))
        if len(states_from) < 5:
            print("WRAPPING TRANSITIONS at s:", state_from, "a:", action_from)
            print("FROM", transitions[state_from][action_from])
            transitions[state_from][action_from] = new_list
            print("TO  ", transitions[state_from][action_from])
    return transitions


def generate_single_state_config_dict(width, boost_reward=False):
    env_config_dict = {
        "avoid_act_probs": [0.01],
        "states_from": [(width - 2, width - 2)],  # bottom-right (visually)
        "actions_from": [(-1, 0)],  # Step up (visually)
        "states_to": [(0, 0)],  # disaster
        "probs_env_event": [0.01],
        "event_rewards": [0.],
        "original_act_rewards": [1. if boost_reward else None]
    }
    return env_config_dict


def generate_every_state_config_dict(
        width, boost_rewards=False, mentor_prob=0.01, env_event_prob=0.01):
    """Generate a randomly-disastrous action for every state

    Intended that the mentor avoids this action, and the pessimistic
    agent more often avoids it than a q_table.
    """
    env_config_dict = {
        k: [] for k in (
            "avoid_act_probs", "states_from", "actions_from", "states_to",
            "probs_env_event", "event_rewards", "original_act_rewards")
    }
    for s in range(width ** 2):
        all_actions = [(-1, 0), (+1, 0), (0, -1), (0, +1)]
        y_coord = int(s % width)
        x_coord = int((s - y_coord) // width)
        if x_coord in (0, width - 1) or y_coord in (0, width - 1):
            continue  # don't change probability of disaster (edge states)
        state_tuple = (x_coord, y_coord)
        safe_actions = []
        for act in all_actions:
            new_pos = np.array(state_tuple) + np.array(act)
            if np.all(1 <= new_pos) and np.all(new_pos <= width - 2):
                safe_actions.append(act)
        # Mentor only
        env_config_dict["avoid_act_probs"].append(mentor_prob)
        # Mentor and env
        env_config_dict["states_from"].append(state_tuple)
        env_config_dict["actions_from"].append(random.choice(safe_actions))
        # Env variables only
        env_config_dict["states_to"].append((0, 0))
        env_config_dict["probs_env_event"].append(env_event_prob)
        env_config_dict["event_rewards"].append(0.)
        env_config_dict["original_act_rewards"].append(
            1. if boost_rewards else None)

    return env_config_dict
