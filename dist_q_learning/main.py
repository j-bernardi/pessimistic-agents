import argparse

from env import FiniteStateCliffworld
from agents import FinitePessimisticAgent
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


def env_visualisation(env):
    print("RESET STATE")
    env.reset()
    env.render(in_loop=False)

    print("\n\nStep every action")
    for action in range(0, 4):
        print("TAKE ACTION", action)
        returned = env.step(action)
        print("Return tuple: ", returned)
        env.render(in_loop=False)
    print("\n\nStep off the edge")
    rew, done = None, False
    while not done:
        obs, rew, done, _ = env.step(0)
        env.render(in_loop=False)
        print("Reward, done:", rew, done)
    assert rew == -0.


def get_args():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--env-test", action="store_true",
        help="Run a short visualisation of the environment")
    parser.add_argument(
        "--quantile", "-q", default=1, type=int, choices=[i for i in range(11)],
        help="The value quantile to use for taking actions")
    parser.add_argument(
        "--mentor", "-m", default="prudent", choices=list(MENTORS.keys()),
        help="The mentor providing queried actions."
    )
    trans_help = "\n".join(
        [f"{k}: {v.__name__}" for k, v in TRANSITIONS.items()])
    parser.add_argument(
        "--trans", "-t", default="0", choices=list(TRANSITIONS.keys()),
        help=f"The mentor providing queried actions.\n{trans_help}")

    parser.add_argument("--num-episodes", "-n", default=0, type=int)
    parser.add_argument("--render", "-r", action="store_true", default=False)

    return parser.parse_args()


if __name__ == "__main__":

    args = get_args()
    env = FiniteStateCliffworld(transition_function=TRANSITIONS[args.trans])

    if args.env_test:
        env_visualisation(env)

    a = FinitePessimisticAgent(
        num_actions=env.num_actions,
        num_states=env.num_states,
        env=env,
        mentor=MENTORS[args.mentor],
        quantile_i=args.quantile,
        gamma=0.99,
        lr=0.5
    )
    
    a.learn(args.num_episodes, render=args.render)
