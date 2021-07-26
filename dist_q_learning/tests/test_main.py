import unittest

from main import (
    run_main, AGENTS, MENTORS, TRANSITIONS, SAMPLING_STRATS, HORIZONS,
    EVENT_WRAPPERS)

combinations = []
for a in AGENTS:
    for m in MENTORS:
        for t in TRANSITIONS:
            for w in EVENT_WRAPPERS:
                for h in HORIZONS:
                    for s in SAMPLING_STRATS:
                        combinations.append([a, m, t, w, h, s])


def generate_combo_test(
        ag, ment, trans, wrapper, horiz, sam, not_impl=False, val_err=False):
    """Generate a test given the arguments, running from command line"""
    if wrapper == "every_state_custom":
        wrapper = "every_state_custom 0.01 0.0"

    def test_to_assign(self):
        arg_string = (
            f"--env grid --agent {ag} --mentor {ment} --trans {trans} "
            f"--wrapper {wrapper} --horizon {horiz} "
            f"--sampling-strategy {sam} ")

        if horiz == "finite":
            # Can't scale Q value to [0, 1] for finite horizon (yet)
            print("UNSCALING")
            arg_string += "--unscale-q "
            arg_string += "--n-horizons 3 "  # test arg and make it faster
        if "pess" in ag:
            arg_string += "--quantile 2 "
        if "gln" in ag:
            arg_string += "--init quantile "

        # Defaults for testing
        arg_string += "--n-steps 2 -n 1 --state-len 4"

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
        run_main(["--env", "grid", "--env-test"])

    def test_env_display_cart(self):
        run_main(["--env", "cart", "--env-test"])

    def test_render(self):
        run_main([
            "--env", "grid", "-n", "1", "--n-steps", "2", "--render", "1",
            "--agent", "q_table"])


for combo in combinations:
    agent, mentor, tran, wrap, hor, samp = combo
    not_implemented = False
    value_err = False
    if "gln" in agent:
        print("Skipping gln due to long tests")
        continue
    if "cartpole" in mentor:
        print("Skipping cartpole test, not implemented fully yet")
        continue
    if "pess" in agent:
        not_implemented = (
            mentor == "none" and agent != "q_table_pess_ire")
    elif agent == "mentor":
        value_err = mentor == "none"

    test_name = f"test_{'_'.join(combo)}"
    test = generate_combo_test(
        agent, mentor, tran, wrap, hor, samp,
        not_impl=not_implemented, val_err=value_err
    )
    setattr(TestMain, test_name, test)
