import numpy as np
from unittest import TestCase

from mentors import prudent_mentor, random_mentor


class Test(TestCase):

    def test_random_mentor(self):

        states = [(1, 1), (1, 2), (2, 2), (4, 2)]
        for s in states:
            act = random_mentor(s)
            assert 0 <= act <= 3

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
        # 2 | -1  0  0  0 -1
        # 3 | -1  0  0  0 -1
        # 4 | -1 -1 -1 -1 -1
        state_expected_acts = [
            [(1, 2), 1],  # e.g. (+1, 0)
            [(2, 1), 3],
            [(3, 2), 0],
            [(2, 3), 2],
        ]

        for state, act in state_expected_acts:
            mentor_act = prudent_mentor(
                np.array(state), kwargs={"state_shape": state_shape})

            assert mentor_act == act, f"Expected: {act}. Got: {mentor_act}"
