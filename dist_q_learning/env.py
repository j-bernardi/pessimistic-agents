import numpy as np

from gym.envs.toy_text import discrete

BACK = -1
FORWARD = 1


class FiniteStateCliffworld(discrete.DiscreteEnv):
    """A finite state gridworld, as detailed by QuEUE document

    This class wraps a "grid" representation of the gym discrete
    spaces. That is, it maps each of (0, ..., n_states) to a
    position on a grid.

    Grid is a NUM_ROWS * NUM_COLS grid, indexed (row, col)

    State map:
        (row, col) -> row * NUM_ROWS + col

    Action map:
        0 -> (-1, 0) - down
        1 -> (+1, 0) - up
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
    ):
        """Create the empty grid with an initial agent position

        If agent_pos is None, random starting position

        Args:
            state_shape (tuple): The (X, Y) shape of the grid world
            cliff_perimeter (int): The number of squares from the edge
                that are give 0-reward to be in
            init_agent_pos (tuple): The initial position of the agent on the
                grid

        TODO:
            Parameterise the transition probabilities
            Allow an input array of probabilities to make isd stochastic
            dtypes: consider using smaller dtypes for r, etc? Is it useful?
            cliff_perimeter: could be (l, r, u, d) or (x, y)
        """
        self.state_shape = np.array(state_shape)
        self.cliff_perimeter = cliff_perimeter

        print(
            "STATE SHAPE", self.state_shape, "GRID SHAPE", self.state_shape,
            "cliff perim (depth)", self.cliff_perimeter
        )

        self.num_states = self.state_shape[0] * self.state_shape[1]  # 2d
        # +1, -1 for each state dimension
        self.num_actions = 2 * self.state_shape.size
        print("NUM ACTIONS", self.num_actions)

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
        print("INIT POS", init_agent_pos, "->", init_agent_pos_int)
        # 100% chance of indicated state
        init_agent_dist = np.eye(self.num_states)[init_agent_pos_int]

        # Define the transition probabilities
        # transitions: {state_0: {action_0: [trans_0, ...], ...}, ...}
        # Maps states to a dict of lists of (stochastic, but P=1.0) outcomes,
        # indexed by action. E.g:
        #   trans_0 = (probability, next_state, reward, done)
        # So can make both reward and next state stochastic with prob.
        transitions = {
            i: {a: [] for a in range(self.num_actions)}
            for i in range(self.num_states)
        }
        for state_i in range(self.num_states):
            for poss_action in range(self.num_actions):
                new_state_int = self.take_int_step(
                    state_i, poss_action, validate=False)
                new_grid = self.map_int_to_grid(new_state_int, validate=False)
                # This handles both being outside the limits of the end and
                # defining the cliff which we define to be a valid state within
                # the limits of the env (but ends the episode)
                if (
                        np.any(new_grid < self.cliff_perimeter)
                        or np.any(
                            new_grid >= self.state_shape - self.cliff_perimeter)
                ):
                    # Fallen over the perimeter: r=0, done
                    trans = (1.0, new_state_int, 0., True)
                else:
                    trans = (1.0, new_state_int, 0.7, False)

                transitions[state_i][poss_action] = [trans]

        super(FiniteStateCliffworld, self).__init__(
            nS=self.num_states,
            nA=self.num_actions,
            P=transitions,  # Transition probabilities and reward f
            isd=init_agent_dist  # initial state distribution
        )

    def take_int_step(self, state_int, action_int, validate=True):
        """Return the next position of the agent, given an action

        Args:
            state_int: The current state of the agent as an integer
            action_int:  The action to take as an integer
            validate: Whether to check if the new state is within the
                grid

        Returns:
            new_state_int: the new state in the integer reps
        """
        grid_state = self.map_int_to_grid(state_int)
        grid_act = self.map_int_act_to_grid(action_int)
        new_state = grid_state + grid_act
        new_state_int = self.map_grid_to_int(new_state, validate=validate)
        return new_state_int

    def map_grid_to_int(self, pos_tuple, validate=True):
        """Map a coordinate tuple to the integer state reps

        Operation: (row, col) -> row * NUM_ROWS + col
        """
        assert np.all(np.array(pos_tuple) < self.state_shape) or not validate, (
            f"Invalid input coord {pos_tuple}")
        return pos_tuple[0] * self.state_shape[0] + pos_tuple[1]

    def map_int_to_grid(self, pos, validate=True):
        """Map an integer state reps to the coordinate tuple

        Operation: row * NUM_ROWS + col -> (row, col)
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

    def render(self, mode="human"):
        """Display the cliffs and current state of the agent"""
        grid = np.zeros(self.state_shape, dtype=np.int8)
        grid[:self.cliff_perimeter, :] = -1.
        grid[-self.cliff_perimeter:, :] = -1.
        grid[:, :self.cliff_perimeter] = -1.
        grid[:, -self.cliff_perimeter:] = -1.
        a_x, a_y = self.map_int_to_grid(self.s)
        grid[a_x, a_y] = 2.
        print(grid)
