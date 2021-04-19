import unittest

from main import run_main, AGENTS, MENTORS, TRANSITIONS, SAMPLING_STRATS

HORIZONS = ["inf", "finite"]

combinations = []
for a in AGENTS:
    for m in MENTORS:
        for t in TRANSITIONS:
            for h in HORIZONS:
                for s in SAMPLING_STRATS:
                    combinations.append([a, m, t, h, s])


def generate_combo_test(
        ag, ment, trans, horiz, sam, not_impl=False, val_err=False):
    """Generate a test given the arguments, running from command line"""

    def test_to_assign(self):
        arg_string = (
            f"--agent {ag} --mentor {ment} --trans {trans} --horizon {horiz} "
            f"--sampling-strategy {sam} ")

        if horiz == "finite":
            # Can't scale Q value to [0, 1] for finite horizon (yet)
            print("UNSCALING")
            arg_string += "--unscale-q "
        if "pess" in ag:
            arg_string += "--quantile 2 "

        # Defaults for testing
        arg_string += "--steps-per-ep 2 -n 1 --state-len 4"

        split_args = arg_string.strip(" ").split(" ")
        print(f"Running args\n{arg_string}\nExpecting fail: {not_impl}")
        if not_impl:
            with self.assertRaises(NotImplementedError):
                run_main(split_args)
        elif val_err:
            with self.assertRaises(ValueError):
                run_main(split_args)
        else:
            run_main(split_args)

    return test_to_assign


class TestMain(unittest.TestCase):

    def test_env_display(self):
        run_main(["--env-test"])

    def test_render(self):
        run_main("-n 1 --steps-per-ep 2 --render 1 --agent q_table".split(" "))


for combo in combinations:
    agent, mentor, tran, hor, samp = combo
    not_implemented = False
    value_err = False
    if "pess" in agent:
        not_implemented = (
            hor == "finite" and agent in ("pess", "pess_single")
        ) or (
            mentor == "none" and agent != "q_table_pess_ire")
    elif agent == "mentor":
        value_err = mentor == "none"

    test_name = f"test_{'_'.join(combo)}"
    test = generate_combo_test(
        agent, mentor, tran, hor, samp,
        not_impl=not_implemented, val_err=value_err
    )
    setattr(TestMain, test_name, test)
