import argparse

from env import FiniteStateCliffworld
from agents import FinitePessimisticAgent, QTableAgent
from mentors import random_mentor, prudent_mentor, random_safe_mentor
from transition_defs import (
    deterministic_uniform_transitions, edge_cliff_reward_slope)

MENTORS = {
    "prudent": prudent_mentor,
    "random": random_mentor,
    "random_safe": random_safe_mentor
}
TRANSITIONS = {
    "0": deterministic_uniform_transitions,
    "1": edge_cliff_reward_slope
}

AGENTS = {
    "q_table": QTableAgent,
    "pessimistic": FinitePessimisticAgent,
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


def get_args():

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
        "--mentor", "-m", default="prudent", choices=list(MENTORS.keys()),
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

    parser.add_argument("--num-episodes", "-n", default=0, type=int)
    parser.add_argument("--render", "-r", type=int, default=0)

    return parser.parse_args()


if __name__ == "__main__":

    args = get_args()
    env = FiniteStateCliffworld(transition_function=TRANSITIONS[args.trans])

    if args.env_test:
        env_visualisation(env)

    agent_init = AGENTS[args.agent]
    if args.agent == "pessimistic":
        agent_kwargs = {
            "quantile_i": args.quantile,
        }
    else:
        agent_kwargs = {}

    a = agent_init(
        num_actions=env.num_actions,
        num_states=env.num_states,
        env=env,
        mentor=MENTORS[args.mentor],
        gamma=0.99,
        lr=0.5
    )
    a.learn(args.num_episodes, render=args.render)
