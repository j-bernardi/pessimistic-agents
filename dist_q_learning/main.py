import argparse
import sys

import matplotlib.pyplot as plt

from env import FiniteStateCliffworld, CartpoleEnv, CartpoleEnv_2
from agents import (
    PessimisticAgent, QTableAgent, QTableMeanIREAgent, QTablePessIREAgent,
    MentorAgent, FinitePessimisticAgent_GLNIRE, ContinuousPessimisticAgent_GLN,
    ContinuousPessimisticAgent_GLN_sigma
)
from mentors import random_mentor, prudent_mentor, random_safe_mentor, cartpole_safe_mentor

from transition_defs import (
    deterministic_uniform_transitions, edge_cliff_reward_slope)

import numpy as np

import jax
print(jax.devices())

MENTORS = {
    "prudent": prudent_mentor,
    "random": random_mentor,
    "random_safe": random_safe_mentor,
    "cartpole_safe": cartpole_safe_mentor,
    "none": None
}

TRANSITIONS = {
    "0": deterministic_uniform_transitions,
    "1": edge_cliff_reward_slope,
    "2": lambda env: edge_cliff_reward_slope(env, standard_dev=None),
}

AGENTS = {
    "pess": PessimisticAgent,
    "q_table": QTableAgent,
    "q_table_ire": QTableMeanIREAgent,
    "q_table_pess_ire": QTablePessIREAgent,
    "mentor": MentorAgent,
    "pess_gln": FinitePessimisticAgent_GLNIRE,
    "continuous_pess_gln": ContinuousPessimisticAgent_GLN
}

SAMPLING_STRATS = ["last_n_steps", "random", "whole", "whole_reset"]

NUM_STEPS = 10
HORIZONS = ["inf", "finite"]  # Finite or infinite horizon
INITS = ["zero", "quantile"]  # Initialise pess Q value to 0. or q


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
        options = []
        for k, v in _dict.items():
            if v is None:
                name = "None"
            elif isinstance(v, str):
                name = v
            elif hasattr(v, "__name__"):
                name = v.__name__
            else:
                raise NotImplementedError(f"Can't handle value {v}")
            options.append(f"{k}: {name}")
        return "\n".join(options)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--env-test", action="store_true",
        help="Run a short visualisation of the environment")
    parser.add_argument(
        "--mentor", "-m", default="none", choices=list(MENTORS.keys()),
        help=f"The mentor providing queried actions.\n{choices_help(MENTORS)}")
    parser.add_argument(
        "--trans", "-t", default="0", choices=list(TRANSITIONS.keys()),
        help=f"The transition function to use.\n"
             f"{choices_help(TRANSITIONS)}")
    parser.add_argument(
        "--agent", "-a", default="q_table", choices=list(AGENTS.keys()),
        help=f"The agent to use.\n{choices_help(AGENTS)}")
    parser.add_argument(
        "--quantile", "-q", default=None, type=int,
        choices=[i for i in range(11)],
        help="The value quantile to use for taking actions")

    parser.add_argument(
        "--init", "-i", choices=INITS, default="zero",  # INITS[0]
        help="Flag whether to init pess q table value to 0. or quantile."
             "Default: 0.")
    parser.add_argument(
        "--unscale-q", action="store_true",
        help="If flagged, Q estimates are for actual discounted Q value"
             " rather than scaled to range [0, 1]")
    parser.add_argument(
        "--horizon", "-o", default="inf", choices=HORIZONS,
        help=f"The Q estimator to use.\n{HORIZONS}")
    parser.add_argument(
        "--sampling-strategy", "-s", default="last_n_steps",
        choices=SAMPLING_STRATS,
        help=f"The Q estimator to use.\n {SAMPLING_STRATS}."
             f"Default: last_n_steps")
    parser.add_argument(
        "--update-freq", default=100, type=int,
        help=f"How often to run the agent update (n steps).")
    parser.add_argument("--num-episodes", "-n", default=0, type=int)
    parser.add_argument(
        "--state-len", "-l", default=7, type=int,
        help=f"The width and height of the grid")
    parser.add_argument(
        "--render", "-r", type=int, default=0, help="render mode 0, 1, 2")
    parser.add_argument(
        "--early-stopping", "-e", default=0, type=int,
        help=f"Number of episodes to have 0 queries to define success.")
    parser.add_argument(
        "--steps-per-ep", default=None, type=int,
        help=f"The number of steps before reporting an episode")
    parser.add_argument("--plot", action="store_true", help="display the plot")

    _args = parser.parse_args(arg_list)

    if "pess" in _args.agent:  # all pessimistic agents
        if _args.quantile is None:
            raise ValueError("Pessimistic agent requires quantile.")
    elif _args.quantile is not None or _args.init != "zero":
        # Invalidate wrong args for non-pessimistic agents
        raise ValueError(
            f"Quantile not required for {_args.agent}."
            f"Init {_args.init} != zero not valid")

    return _args


def run_main(cmd_args):

    args = get_args(cmd_args)
    w = args.state_len
    init = w // 2

    if args.agent == "continuous_pess_gln":
        env = CartpoleEnv_2()
    else:
        env = FiniteStateCliffworld(
            state_shape=(w, w),
            init_agent_pos=(init, init),
            transition_function=TRANSITIONS[args.trans]
        )

    if args.env_test:
        env_visualisation(env)

    agent_init = AGENTS[args.agent]
    if args.agent == "pess_gln":
        agent_kwargs = {"dim_states": 2}
    elif args.agent == "continuous_pess_gln":
        agent_kwargs = {"dim_states": 4}
    else:
        agent_kwargs = {"num_states": env.num_states}

    if "pess" in args.agent:
        agent_kwargs = {
            **agent_kwargs,
            **{"quantile_i": args.quantile, "init_to_zero": args.init == "zero"}
        }
    if args.agent == "pess_gln":
        agent_kwargs = {
            **agent_kwargs, **{"quantile_i": args.quantile}
        }

    if args.num_episodes > 0:
        agent = agent_init(
            num_actions=env.num_actions,
            env=env,
            gamma=0.9,
            mentor=MENTORS[args.mentor],
            sampling_strategy=args.sampling_strategy,
            # 1. for the deterministic env
            lr=1. if str(args.trans) == "2" else 1e-3,
            min_reward=env.min_nonzero_reward,
            eps_max=1.,
            eps_min=0.3,
            horizon_type=args.horizon,
            update_n_steps=args.update_freq,
            batch_size=args.update_freq,
            num_steps=1 if args.horizon == "inf" else NUM_STEPS,
            scale_q_value=not args.unscale_q,
            **agent_kwargs
        )

        learn_kwargs = {}
        if args.steps_per_ep is not None:
            learn_kwargs["steps_per_ep"] = args.steps_per_ep

        success = agent.learn(
            args.num_episodes,
            render=args.render,
            early_stopping=args.early_stopping,
            **learn_kwargs
        )
        print("Finished! Queries per ep:")
        print(agent.mentor_queries_per_ep)
        print(f"Completed {success} after {agent.total_steps} steps")

        if args.plot and args.agent == "pess_gln":
            x = np.linspace(-1, 1, 20)
            y = np.linspace(-1, 1, 20)

            fig1 = plt.figure()
            Q_vals = np.zeros((4, 20, 20))

            for ii in range(4):
                for xi in range(len(x)):
                    for yi in range(len(y)):
                        Q_vals[ii, xi, yi] = agent.q_estimator.estimate(
                            [x[xi], y[yi]], ii)
                fig1.add_subplot(2, 2, ii + 1)
                plt.pcolor(x, y, Q_vals[ii, :, :])
                plt.title(f'action: {ii}')
                plt.colorbar()
            fig2 = plt.figure()

            mentor_Q_vals = np.zeros((20, 20))
            for xi in range(len(x)):
                for yi in range(len(y)):
                    mentor_Q_vals[xi, yi] =\
                        agent.mentor_q_estimator.estimate([x[xi], y[yi]])

            plt.pcolor(x, y, mentor_Q_vals)
            plt.title('Mentor')
            plt.colorbar()

        if args.plot:
            plt.plot(agent.mentor_queries_per_ep)
            # plt.title(a.QEstimators[1].lr)
            plt.title(agent.q_estimator.lr)
            plt.show()

        return agent


if __name__ == "__main__":
    run_main(sys.argv[1:])
