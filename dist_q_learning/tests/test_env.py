import numpy as np

from env import FiniteStateCliffworld
from unittest import TestCase


class TestFiniteStateCliffworld(TestCase):

    env = FiniteStateCliffworld()
    action_mappings = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}

    def test_take_int_step(self):
        """Test the internal step function works the same as gym

        By doing the transform 'on the grid', we reassure ourselves
        that the integer-representation is making the move we are
        expecting, for every viable action from the centre state.

        Verify that the outcome is the same as a gym state-update.

        This test has internal dependencies on most of the others.
        If others are broken, investigate those first.
        """

        # start in position 0
        self.env.reset()
        assert self.env.s == (
                self.env.state_shape[0] * self.env.state_shape[1]) // 2

        for action_i, grid_action in self.action_mappings.items():
            # Note dependency on the map_int_to_grid test:
            init_grid_position = self.env.map_int_to_grid(self.env.s)

            # Transform in grid reps. Dependency on map_grid_to_int test
            expected_new_grid_position = (
                    init_grid_position + np.array(grid_action, dtype=int))
            expected_new_int_position = self.env.map_grid_to_int(
                expected_new_grid_position)

            # Take the step using the internal step function
            new_int_pos = self.env.take_int_step(self.env.s, action_i)

            # Take the step using the gym base class step function
            self.env.step(action_i)
            gym_new_int_pos = self.env.s

            assert np.all(new_int_pos == expected_new_int_position), (
                f"\ntake_int_step: {new_int_pos}"
                f"\nTransform:     {expected_new_int_position}"
            )
            assert np.all(new_int_pos == expected_new_int_position), (
                f"Gym step result: {new_int_pos}.\n"
                f"Transform:       {expected_new_int_position}"
            )

    def test_map_grid_to_int(self):
        """Test tuples mapped to expected state integer - 2d only"""
        self.env.reset()
        next_state = 0
        for row in range(self.env.state_shape[0]):
            for col in range(self.env.state_shape[1]):
                state_int = self.env.map_grid_to_int((row, col))
                assert state_int == next_state
                next_state += 1

    def test_map_int_to_grid(self):
        """Test state ints mapped to expected tuples"""
        self.env.reset()
        next_state = 0
        for row in range(self.env.state_shape[0]):
            for col in range(self.env.state_shape[1]):
                state_tuple = self.env.map_int_to_grid(next_state)
                assert np.all(state_tuple == np.array([row, col], dtype=int))
                next_state += 1

    def test_map_grid_act_to_int(self):
        """Check tuple actions are mapped to the expected ints"""
        for act_int, act_tuple in self.action_mappings.items():
            returned = self.env.map_grid_act_to_int(act_tuple)
            assert np.all(returned == act_int), (
                f"{act_tuple} -> {returned} != {act_int}"
            )

    def test_map_int_act_to_grid(self):
        """Check int actions are mapped to the expected tuples"""
        for act_int, act_tuple in self.action_mappings.items():
            returned = self.env.map_int_act_to_grid(act_int)
            assert np.all(returned == act_tuple), (
                f"{act_int} -> {returned} != {act_tuple}"
            )
