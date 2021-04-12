import unittest
from unittest import TestCase

from main import run_main, AGENTS, MENTORS, TRANSITIONS

HORIZONS = ["inf", "finite"]

combinations = []
for a in AGENTS:
    for m in MENTORS:
        for t in TRANSITIONS:
            for h in HORIZONS:
                combinations.append([a, m, t, h])


def generate_combo_test(ag, ment, trans, horiz, not_impl):
    """Generate a test given the arguments, running from command line"""

    def test_to_assign(self):
        arg_string = (
            f"--agent {ag} --mentor {ment} --trans {trans} --horizon {horiz} ")

        if agent == "pessimistic":
            arg_string += f"--quantile 2 "
        arg_string += "--steps-per-ep 2 -n 1"
        split_args = arg_string.strip(" ").split(" ")
        if not_impl:
            with self.assertRaises(NotImplementedError):
                run_main(split_args)
        else:
            run_main(split_args)
    return test_to_assign


class TestMain(TestCase):

    def test_env_display(self):
        run_main(["--env-test"])

    def test_render(self):
        run_main("-n 1 --steps-per-ep 2 --render 1 --agent q_table".split(" "))


for combo in combinations:
    agent, mentor, tran, hor = combo
    not_implemented = False
    if agent == "pessimistic":
        not_implemented = hor != "inf" or mentor == "none"
    elif agent == "q_table_ire":
        not_implemented = hor != "inf"

    test_name = f"test_{'_'.join(combo)}"
    test = generate_combo_test(
        agent, mentor, tran, hor, not_impl=not_implemented)
    setattr(TestMain, test_name, test)
