import abc
import gym
import copy
import numpy as np
import jax
import jax.numpy as jnp
from gym.envs.toy_text import discrete

from transition_defs import (
    deterministic_uniform_transitions, adjustment_wrapper)

BACK = -1
FORWARD = 1

GRID_ACTION_MAP = {
    0: (-1, 0),  # Looks up on the render
    1: (+1, 0),  # Looks down on the render
    2: (0, -1),
    3: (0, +1),
}
ENV_ADJUST_KWARGS_KEYS = {
    "avoid_act_probs", "states_from", "actions_from", "states_to",
    "event_rewards", "probs_env_event", "original_act_rewards"}


class BaseEnv(abc.ABC):

    def __init__(self):
        pass

    @abc.abstractmethod
    def render(self, **kwargs):
        raise NotImplementedError()

    @abc.abstractmethod
    def step(self, action):
        raise NotImplementedError()

    @abc.abstractmethod
    def reset(self):
        raise NotImplementedError()

    def print_spacer(self, **kwargs):
        """Helpful for command line printing"""
        return

    def print_newlines(self, **kwargs):
        return


class FiniteStateCliffworld(discrete.DiscreteEnv, BaseEnv):
    """A finite state gridworld, as detailed by QuEUE document

    This class wraps a "grid" representation of the gym discrete
    spaces. That is, it maps each of (0, ..., n_states) to a
    position on a grid.

    Grid is a NUM_ROWS * NUM_COLS grid, indexed (row, col)

    State map:
        (row, col) -> row * NUM_ROWS + col

    Action map:
        0 -> (-1, 0) - down (looks up in the rendering)
        1 -> (+1, 0) - up (looks down in the rendering)
        2 -> (0, -1) - left
        3 -> (0, +1) - right

    TODO: currently only valid for 2d
    """

    metadata = {"render.modes": ["human"]}

    def __init__(
            self,
            state_shape=(7, 7),
            cliff_perimeter=1,
            init_agent_pos=(3, 3),  # centre
            transition_function=deterministic_uniform_transitions,
            make_env_adjusts=False,
            **adjust_kwargs,

    ):
        """Create the empty grid with an initial agent position

        If agent_pos is None, random starting position

        Args:
            state_shape (tuple): The (X, Y) shape of the grid world
            cliff_perimeter (int): The number of squares from the edge
                that are give 0-reward to be in
            init_agent_pos (tuple): The initial position of the agent on
                the grid
            make_env_adjusts (bool): if true, add events as-per kwargs
                for experimenting

        Keyword args:
            states_from (List[tuple]):
            actions_from (List[tuple]):
            states_to (List[tuple]):
            probs_event (List[float]):
            event_rewards (List[float]):
            original_act_rewards (List[float]): adjust the reward of the
                original (states_from, actions_from), e.g. to
                incentivise taking a risky action to naive agents.
        """
        self.state_shape = np.array(state_shape)
        self.cliff_perimeter = cliff_perimeter
        self.num_states = self.state_shape[0] * self.state_shape[1]  # 2d
        self.num_actions = 2 * self.state_shape.size  # +1, -1 for each dim

        # Make the initial position
        if init_agent_pos is None:
            # Random, within the safe perimeter
            init_agent_pos = np.random.randint(
                low=cliff_perimeter,
                high=self.state_shape - cliff_perimeter,
                size=2
            )
        if not isinstance(init_agent_pos, tuple):
            raise TypeError(f"Init position must be tup {type(init_agent_pos)}")
        init_agent_pos_int = self.map_grid_to_int(init_agent_pos)
        # 100% chance of indicated state
        init_agent_dist = np.eye(self.num_states)[init_agent_pos_int]

        transitions, (min_nonzero_r, max_r) = transition_function(self)
        if make_env_adjusts:
            print("Adjusting env args with wrapper kwargs")
            adjust_kwargs = copy.deepcopy(adjust_kwargs)
            adjust_kwargs.pop("avoid_act_probs")  # Not needed for env
            self.states_from = [
                self.map_grid_to_int(s) for s in
                adjust_kwargs.pop("states_from", [self.state_shape - 2])]
            self.actions_from = [
                self.map_grid_act_to_int(a) for a in
                adjust_kwargs.pop("actions_from", [(-1, 0)])]
            self.states_to = [
                self.map_grid_to_int(s) for s in
                adjust_kwargs.pop("states_to", [(1, 1)])]
            self.event_probs = adjust_kwargs.pop("probs_env_event", [0.01])
            self.event_rewards = adjust_kwargs.pop("event_rewards", [None])
            self.original_act_rewards = adjust_kwargs.pop(
                "original_act_rewards", [None])
            assert not adjust_kwargs, (
                f"Unexpected keys remain {adjust_kwargs.keys()}")
            transitions = adjustment_wrapper(
                transitions,
                states_from=self.states_from,
                actions_from=self.actions_from,
                states_to=self.states_to,
                event_probs=self.event_probs,
                event_rewards=self.event_rewards,
                original_act_rewards=self.original_act_rewards,
            )
        else:
            self.states_from = None
            self.actions_from = None
            self.states_to = None
            self.event_probs = None
            self.event_rewards = None
            self.original_act_rewards = None

        self.min_nonzero_reward = min_nonzero_r
        self.max_r = max_r

        super(FiniteStateCliffworld, self).__init__(
            nS=self.num_states,
            nA=self.num_actions,
            P=transitions,  # Transition probabilities and reward f
            isd=init_agent_dist  # initial state distribution
        )
        self.summary()

    def take_int_step(
            self, state_int, action_int, validate=True, return_is_safe=False):
        """Return the next position of the agent, given an action

        Args:
            state_int: The current state of the agent as an integer
            action_int:  The action to take as an integer
            validate: Whether to check if the new state is within the
                grid
            return_is_safe

        Returns:
            new_state_int (int): the new state in the integer reps
            is_safe (bool): if return_is_safe, returns a bool describing
                whether this state is a safe one
        """
        grid_state = self.map_int_to_grid(state_int)
        grid_act = self.map_int_act_to_grid(action_int)
        new_state = grid_state + grid_act
        is_safe = (
            np.all(new_state >= self.cliff_perimeter)
            and np.all(new_state < (self.state_shape - self.cliff_perimeter)))

        new_state_int = self.map_grid_to_int(new_state, validate=validate)

        if return_is_safe:
            return new_state_int, is_safe
        else:
            return new_state_int

    def map_grid_to_int(self, pos_tuple, validate=True):
        """Map a coordinate tuple to the integer state reps

        Operation: (row, col) -> row * NUM_ROWS + col
        """
        assert (np.all(np.array(pos_tuple) < self.state_shape)
                and np.all(np.array(pos_tuple) >= 0)) or not validate, (
            f"Invalid input coord {pos_tuple} outside {self.state_shape}")
        return pos_tuple[0] * self.state_shape[0] + pos_tuple[1]

    def map_int_to_grid(self, pos, validate=True):
        """Map an integer state reps to the coordinate tuple

        Operation: row * NUM_ROWS + col -> (row, col)

        TODO assumes square
        """
        y_coord = int(pos % self.state_shape[0])
        x_coord = int((pos - y_coord) // self.state_shape[0])
        new_coord = np.array([x_coord, y_coord])
        assert np.all(new_coord < self.state_shape) or not validate, (
            f"Invalid coord. {pos} -> {new_coord} (/ {self.state_shape})")
        return new_coord

    def map_grid_act_to_int(self, act_tuple):
        """Maps a move on the grid to an integer action

        See class docstring `action_map` for intended mapping.
        """
        assert len(act_tuple) == self.state_shape.size
        # Doesn't check every combo, but better than nothing
        assert sum(act_tuple) == 1 or sum(act_tuple) == -1

        for dim, a in enumerate(act_tuple):
            if a == BACK:
                return 2 * dim
            elif a == FORWARD:
                return 2 * dim + 1

    def map_int_act_to_grid(self, act):
        """Maps an integer action to a move on the grid

        See class docstring for mapping.
        """
        assert act < self.state_shape.size * 2
        act_arr = np.zeros((self.state_shape.size,))
        dim = act // 2
        act_arr[dim] = (BACK, FORWARD)[act % 2]

        return act_arr

    def print_spacer(self, adjust=0):
        """Defines an amount to jump the command line by for rendering"""
        print("\033[F" * (self.state_shape[0] + 1 + adjust))

    def print_newlines(self, **kwargs):
        """Prints enough lines to get beyond the output of render"""
        print("\n" * (self.state_shape[0] - 1))

    def render(self, mode="human", in_loop=True):
        """Display the cliffs and current state of the agent"""
        if in_loop:
            self.print_spacer()  # move up n-rows
        grid = np.zeros(self.state_shape, dtype=np.int8)
        grid[:self.cliff_perimeter, :] = -1.
        grid[-self.cliff_perimeter:, :] = -1.
        grid[:, :self.cliff_perimeter] = -1.
        grid[:, -self.cliff_perimeter:] = -1.

        # Visualise the adjustment square, if there is only 1 (else messy)
        if self.states_from is not None and len(self.states_from) == 1:
            sf_t = self.map_int_to_grid(self.states_from[0])
            grid[sf_t[0], sf_t[1]] = 9
            st_t = self.map_int_to_grid(self.states_to[0])
            grid[st_t[0], st_t[1]] = 8
            act = tuple(
                int(x) for x in
                self.map_int_to_grid(self.states_from[0])
                + self.map_int_act_to_grid(self.actions_from[0]))
            grid[act[0], act[1]] = 7

        a_x, a_y = self.map_int_to_grid(self.s)
        grid[a_x, a_y] = 2.
        print(grid)

    def summary(self):
        print(
            f"State shape: {self.state_shape}\n"
            f"Num states: {self.num_states}\n",
            f"cliff perim (depth) {self.cliff_perimeter}\n"
            f"Num actions: {self.num_actions}\n"
            f"Current position: {self.s}\n"
        )


class CartpoleEnv(BaseEnv):
    """A variant of CartPole-v1 that doesn't end eps unless pole falls

    Wraps the gym env and resurfaces the API.
    """

    def __init__(
            self, max_episode_steps=jnp.inf, min_nonzero=0.2, min_val=None,
            target="stand_up", random_x=False,
    ):
        """

        max_episodes: reset cartpole after this many steps.
            Default: never
        min_nonzero: the minimum non-zero reward provided by the
            environment. Not accurate, but is used by pessimistic agent
            to decide when to query if value < min_nonzero.
        min_val: the minimum value of the normalised state vector. Only
            -1, 0, None are allowed. If not None, state is returned in
            range [min_val, 1], else state is not normalised at all
        target (str): defines the reward function of this env.
            "stand_up": classic cartpole, constant reward
            "move_out": max rewards centred over 0.5, see _reward_f()
        random_x (bool): if True, start at a random x-coordinate rather
            than near 0

        """
        super().__init__()
        self.gym_env = gym.make("CartPole-v1")

        # make the env not return done unless it dies
        self.gym_env._max_episode_steps = max_episode_steps

        self.normalise = jax.jit(self._normalise)
        # Can use static argnums as gym reward should only be 0 or 1
        self.reward_f = jax.jit(self._reward_f, static_argnums=1)
        self.move_out_reward = jax.jit(self._move_out_reward)

        self.random_x = random_x
        self.num_actions = self.gym_env.action_space.n
        self.min_nonzero_reward = min_nonzero
        self.min_val = min_val
        if self.min_val is not None:
            self.mean_val = (1. + self.min_val) / 2.
        else:
            self.mean_val = 0.  # cart pole default
        self.target = target
        if self.min_val is not None:
            print(f"Normalising state to [{self.min_val}, 1]")
        else:
            print("Not normalising state")

        print(f"Target: {self.target}")
        if self.target == "move_out":
            # Max turning point at 0.5 =>
            max_r = self.move_out_reward(0.5)
            assert max_r > max(
                self.move_out_reward(0.49),
                self.move_out_reward(0.51))
            assert 0. < max_r <= 1., f"Min reward too high {max_r}"

    def _normalise(self, state):
        """Optionally transform state vector to a range bounded by 1

        Args:
            state (jnp.ndarray): 4-vector of (x_pos, v, theta, w)
        """
        if self.min_val is None:
            return state

        # Positions [-max, max] converted to [-1, 1]
        x_pos = state[0] / self.gym_env.x_threshold
        theta = state[2] / self.gym_env.theta_threshold_radians
        # Velocities [-inf, inf] converted to [0, 1]
        v = 1. / (1. + jnp.exp(-state[1]))
        w = 1. / (1. + jnp.exp(-state[3]))

        if self.min_val == 0:
            # Convert to [0, 1]
            x_pos = 0.5 + x_pos / 2.
            theta = 0.5 + theta / 2.
        elif self.min_val == -1:
            # Convert to [-1, 1]
            v = (v - 0.5) * 2.
            w = (w - 0.5) * 2.
        else:
            raise ValueError(f"Invalid: {self.min_val}")

        normed_state = jnp.asarray([x_pos, v, theta, w])

        return jnp.clip(normed_state, self.min_val, 1.)

    def reset(self):
        _ = self.gym_env.reset()  # use internal update
        if self.random_x:
            # The same way it's done in the super class, with different
            # hard-coded low/high and only for x
            # TODO - is jnp faster? But base class is np
            self.gym_env.state[0] = self.gym_env.np_random.uniform(
                low=-self.gym_env.x_threshold * 0.95,
                high=self.gym_env.x_threshold * 0.95,
                size=1)
        init_state = np.array(self.gym_env.state)
        normed_state = self.normalise(init_state)
        return jnp.asarray(normed_state)

    def _reward_f(self, normed_x, gym_reward):
        """Custom reward function

        Args:
            gym_reward (float): reward returned by gym step()
            normed_x (float): normalised x-coordinate in range (-1, 1)
        """
        if gym_reward == 0. or gym_reward is None:
            reward = gym_reward  # throw it back
        elif gym_reward != 1.:
            raise ValueError(f"Encountered reward != 1 {gym_reward}")

        elif self.target == "stand_up":
            reward = 0.8

        elif self.target == "move_out":
            if self.mean_val != 0.:
                raise ValueError("X coord must be about 0 for this RF")
            reward = self.move_out_reward(normed_x)

        else:
            raise ValueError(
                f"Unexpected reward {gym_reward}, target {self.target}")

        return reward

    def _move_out_reward(self, x):
        """Define an exponential which has nice properties

        Requires x normalised in range [-1, 1]
        Max tp at +/- 0.5
        """
        a = jnp.asarray(2.)
        rw = jnp.abs(
            (a * x) / jnp.exp(jnp.asarray(2.) * jnp.abs(x ** 2))
        ) + self.min_nonzero_reward
        return rw

    def step(self, action):
        next_state, orig_reward, done, info = self.gym_env.step(action)
        next_state = jnp.asarray(next_state)
        norm_state = self.normalise(next_state)
        try:
            x_pos = norm_state[0]
            custom_reward = self.reward_f(x_pos, orig_reward)
        except Exception as any_exception:
            raise ValueError(f"{any_exception}, state {next_state}")
        return norm_state, custom_reward, done, info

    def render(self, **kwargs):
        """Kwargs to fit pattern of other envs, but are ignored"""
        self.gym_env.render()
