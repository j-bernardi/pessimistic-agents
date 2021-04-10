import argparse
import sys

import matplotlib.pyplot as plt

from env import FiniteStateCliffworld
from agents import FinitePessimisticAgent, QTableAgent, QTableIREAgent
from mentors import random_mentor, prudent_mentor, random_safe_mentor
from estimators import QEstimator, FHTDQEstimator, MentorFHTDQEstimator
from transition_defs import (
    deterministic_uniform_transitions, edge_cliff_reward_slope)


MENTORS = {
    "prudent": prudent_mentor,
    "random": random_mentor,
    "random_safe": random_safe_mentor,
    "none": None
}
TRANSITIONS = {
    "0": deterministic_uniform_transitions,
    "1": edge_cliff_reward_slope
}

AGENTS = {
    "q_table": QTableAgent,
    "pessimistic": FinitePessimisticAgent,
    "q_table_ire": QTableIREAgent
}

NUM_STEPS = 10
HORIZONS = {
    "inf": QEstimator,
    "finite": FHTDQEstimator.get_steps_constructor(num_steps=NUM_STEPS)
}


def env_visualisation(_env):
    print("RESET STATE")
    _env.reset()
    _env.render(in_loop=False)

    print("\n\nStep every action")
    for action in range(0, 4):
        print("TAKE ACTION", action)
        returned = _env.step(action)
        print("Return tuple: ", returned)
        _env.render(in_loop=False)
    print("\n\nStep off the edge")
    rew, done = None, False
    while not done:
        obs, rew, done, _ = _env.step(0)
        _env.render(in_loop=False)
        print("Reward, done:", rew, done)
    assert rew == -0.


def get_args(arg_list):

    def choices_help(_dict):
        options = [f"{k}: {v.__name__}" for k, v in TRANSITIONS.items()]
        return "\n".join(options)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--env-test", action="store_true",
        help="Run a short visualisation of the environment")

    parser.add_argument(
        "--quantile", "-q", default=1, type=int, choices=[i for i in range(11)],
        help="The value quantile to use for taking actions")

    parser.add_argument(
        "--mentor", "-m", default="none", choices=list(MENTORS.keys()),
        help=f"The mentor providing queried actions.\n{choices_help(MENTORS)}"
    )
    parser.add_argument(
        "--trans", "-t", default="0", choices=list(TRANSITIONS.keys()),
        help=f"The mentor providing queried actions.\n"
             f"{choices_help(TRANSITIONS)}")
    parser.add_argument(
        "--agent", "-a", default="pessimistic", choices=list(AGENTS.keys()),
        help=f"The agent to use.\n{choices_help(AGENTS)}"
    )
    parser.add_argument(
        "--horizon", "-o", default="inf", choices=list(HORIZONS.keys()),
        help=f"The Q estimator to use.\n{choices_help(HORIZONS)}"
    )

    parser.add_argument(
        "--steps-per-ep", default=None, type=int,
        help=f"The number of steps before reporting an episode"
    )

    parser.add_argument("--num-episodes", "-n", default=0, type=int)
    parser.add_argument(
        "--render", "-r", type=int, default=0, help="render mode 0, 1, 2"
    )

    parser.add_argument("--plot", action="store_true", help="display the plot")

    _args = parser.parse_args(arg_list)

    if _args.horizon != "inf" and _args.agent != "q_table":
        raise NotImplementedError(
            f"Only inf horizon is implemented for {_args.agent}")

    return _args


def run_main(cmd_args):

    args = get_args(cmd_args)
    env = FiniteStateCliffworld(transition_function=TRANSITIONS[args.trans])

    if args.env_test:
        env_visualisation(env)

    agent_init = AGENTS[args.agent]
    if args.agent == "pessimistic":
        agent_kwargs = {"quantile_i": args.quantile,
                        "scale_q_value": True}

    elif args.agent == "q_table":
        agent_kwargs = {
            "q_estimator_init": HORIZONS[args.horizon],
            "scale_q_value": not args.horizon == "finite"  # don't scale if fin
        }
        if args.horizon == "finite":
            agent_kwargs["mentor_q_estimator_init"] = (
                MentorFHTDQEstimator.get_steps_constructor(num_steps=NUM_STEPS))

    else:
        agent_kwargs = {}

    if args.num_episodes > 0:
        a = agent_init(
            num_actions=env.num_actions,
            num_states=env.num_states,
            env=env,
            mentor=MENTORS[args.mentor],
            gamma=0.99,
            lr=1.,
            min_reward=env.min_nonzero_reward,
            eps_max=1.,
            eps_min=0.5,
            **agent_kwargs
        )
        learn_kwargs = {}
        if args.steps_per_ep is not None:
            learn_kwargs["steps_per_ep"] = args.steps_per_ep
        a.learn(
            args.num_episodes,
            render=args.render,
            **learn_kwargs
        )
        print("Finished! Queries per ep:")
        print(a.mentor_queries_per_ep)
        if args.plot:
            plt.plot(a.mentor_queries_per_ep)
            # plt.title(a.QEstimators[1].lr)
            plt.title(a.q_estimator.lr)
            plt.show()


if __name__ == "__main__":
    run_main(sys.argv[1:])
