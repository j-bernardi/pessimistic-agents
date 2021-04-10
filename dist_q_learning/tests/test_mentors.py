import unittest

import numpy as np
from unittest import TestCase

from mentors import prudent_mentor, random_mentor, random_safe_mentor

from env import FiniteStateCliffworld


class Test(TestCase):

    def test_random_mentor(self):

        states = [(1, 1), (1, 2), (2, 2), (4, 2)]
        for s in states:
            act = random_mentor(s)
            assert all(a in (-1, 0, 1) for a in act), act
            assert sum(act) == 1 or sum(act) == -1 and act[0] != act[1]

    @unittest.expectedFailure
    def test_random_safe_mentor(self):

        state_shape = (5, 5)

        #      0  1  2  3  4
        # 0 | -1 -1 -1 -1 -1
        # 1 | -1  0  0  0 -1
        # 2 | -1  0  2  0 -1
        # 3 | -1  0  0  0 -1
        # 4 | -1 -1 -1 -1 -1
        state_expected_acts = [
            [(1, 2), -1],  # e.g. (+1, 0)
            [(2, 1), -1],  # TODO - need to figure out the actions here
            [(3, 2), -1],
            [(2, 3), -1],
        ]

        for state, act in state_expected_acts:
            env = FiniteStateCliffworld(
                state_shape=state_shape, init_agent_pos=state)
            env.render()
            print('taking action, mentor act:')
            mentor_act = random_safe_mentor(
                np.array(state), kwargs={
                    "state_shape": np.array(state_shape, dtype=int)}
            )
            mentor_int_act = env.map_grid_act_to_int(mentor_act)
            print(mentor_act, "->", mentor_int_act)
            env.step(mentor_int_act)
            env.render()
            assert mentor_int_act == act, f"Expected: {act}. Got: {mentor_int_act}"

    def test_prudent_mentor(self):
        """Test we actually step away from closest edge

        Reminder:
            0 -> (-1, 0) - down
            1 -> (+1, 0) - up
            2 -> (0, -1) - left
            3 -> (0, +1) - right
        """

        state_shape = (5, 5)

        #      0  1  2  3  4
        # 0 | -1 -1 -1 -1 -1
        # 1 | -1  0  0  0 -1
        # 2 | -1  0  2  0 -1
        # 3 | -1  0  0  0 -1
        # 4 | -1 -1 -1 -1 -1
        state_expected_acts = [
            [(1, 2), 1],  # e.g. (+1, 0)
            [(2, 1), 3],
            [(3, 2), 0],
            [(2, 3), 2],
        ]

        for state, act in state_expected_acts:
            env = FiniteStateCliffworld(
                state_shape=state_shape, init_agent_pos=state)
            env.render()
            print('taking action, mentor act:')
            mentor_act = prudent_mentor(
                np.array(state), kwargs={
                    "state_shape": np.array(state_shape, dtype=int)}
            )
            mentor_int_act = env.map_grid_act_to_int(mentor_act)
            print(mentor_act, "->", mentor_int_act)
            env.step(mentor_int_act)
            env.render()
            assert mentor_int_act == act, f"Expected: {act}. Got: {mentor_int_act}"
