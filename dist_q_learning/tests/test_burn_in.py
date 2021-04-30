from unittest import TestCase
from estimators import ImmediateRewardEstimator_GLN_gaussian


class TestBurnIn(TestCase):

    def test_burn_in_end_value(self):
        # precision - how much is the estimation value allowed to deviate from the value we burnt it in with.
        self.epsilon = 0.1

        # TODO: add a reasonable state space here. (is there some assumption about what type the states should be?)
        states = [[0.,1.],[2.,3.,4], [0.],1.,2,(0,1),(2.,1.)]
        for num_actions in range(1,4):
            print(f"Testing with {num_actions} actions")
            ires = {}
            for a in range(num_actions):
                ires[a] = ImmediateRewardEstimator_GLN_gaussian(
                    a, burnin_n=2)
                for st in states:
                    estimation = ires[a].estimate(st)
                    print(f"IRE_{a}({st})={estimation}")
                    # TODO: add assertion for the actual output.
                    # assert estimation < epsilon
                    # TODO: we do in general burn in to zero, right?
                    # TODO: do we actually need a burn in? Can't we just initialize the weights accordingly?

