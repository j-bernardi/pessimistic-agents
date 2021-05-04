import sys
import jax
import pprint
import argparse
import numpy as np
import matplotlib.pyplot as plt

from env import FiniteStateCliffworld, CartpoleEnv
from agents import (
    PessimisticAgent, QTableAgent, QTableMeanIREAgent, QTablePessIREAgent,
    MentorAgent, FinitePessimisticAgent_GLNIRE, ContinuousPessimisticAgent_GLN
)
from mentors import (
    random_mentor, prudent_mentor, random_safe_mentor, cartpole_safe_mentor)

from transition_defs import (
    deterministic_uniform_transitions, edge_cliff_reward_slope)

from experiments.teleporter.plotter import print_transitions

print(jax.devices())

MENTORS = {
    "prudent": prudent_mentor,
    "random": random_mentor,
    "random_safe": random_safe_mentor,
    "none": None,
    "cartpole_safe": cartpole_safe_mentor,
    "avoid_state_act": "avoid_state_act_placeholder",
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
        "--action-noise", default=None, type=float, nargs="*",
        help="Min and max range (with optional decay val) for action noise")
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
        help=f"The experience=history sampling strategy to use.\n"
             f"{SAMPLING_STRATS}. Default: last_n_steps")
    parser.add_argument(
        "--update-freq", default=100, type=int,
        help=f"How often to run the agent update (n steps).")
    parser.add_argument(
        "--batch-size", "-b", required=False, type=int,
        help=f"Size of the history sample to update over. Defaults to "
             f"--update-freq. Unused if sampling-strategy `whole`")
    parser.add_argument(
        "--report-every-n", default=500, type=int,
        help="Every report-every-n steps, a progress report is produced for "
             "the agent's last n steps (and render >= 0). Also aggregates "
             "results on this granularity")
    parser.add_argument(
        "--state-len", "-l", default=7, type=int,
        help=f"The width and height of the grid")
    parser.add_argument(
        "--render", "-r", type=int, default=0, help="render mode 0, 1, 2")
    parser.add_argument(
        "--early-stopping", "-e", default=0, type=int,
        help=f"Number of report periods to have 0 queries to define success.")
    parser.add_argument(
        "--n-steps", "-n", default=0, type=int,
        help=f"The number of steps to train for")
    parser.add_argument("--plot", action="store_true", help="display the plot")

    _args = parser.parse_args(arg_list)

    if "pess" in _args.agent:  # all pessimistic agents
        if _args.quantile is None:
            raise ValueError("Pessimistic agent requires quantile.")
    elif _args.quantile is not None or _args.init != "zero":
        # Invalidate wrong args for non-pessimistic agents
        raise ValueError(
            f"Quantile not required for {_args.agent}, and "
            f"init ({_args.init}) != zero not valid")

    if (
            _args.action_noise is not None
            and len(_args.action_noise) not in (2, 3)):
        raise ValueError(f"Must be 2 or 3: {_args.action_noise}")

    if "whole" in _args.sampling_strategy and _args.batch_size is not None:
        raise ValueError()

    return _args


def run_main(cmd_args, env_adjust_kwargs=None):
    """Run the main script given cmd_args, and optional env adjustments

    cmd_args:
    env_adjust_kwargs (Optional[dict]): (see keys in function),
        specifies lists of grid co-ords with actions, probabilities that

    :return:
    """
    print("PASSING", cmd_args)
    args = get_args(cmd_args)
    w = args.state_len
    init = w // 2

    track_positions = []
    if args.agent == "continuous_pess_gln":
        env = CartpoleEnv()
    else:
        if env_adjust_kwargs is None:
            env_adjust_kwargs = {}
        # Mentor only - probability mentor takes the specified action
        AVOID_ACT_PROBS = env_adjust_kwargs.pop("avoid_act_probs", [0.01])
        # Mentor and env - the adjusted (s, a)
        STATES_FROM = env_adjust_kwargs.pop("states_from", [(5, 5)])
        ACTIONS_FROM = env_adjust_kwargs.pop("actions_from", [(-1, 0)])  # 0
        # Env variables only - the probability the state-change happens
        STATES_TO = env_adjust_kwargs.pop("states_to", [(1, 1)])
        REWARDS = env_adjust_kwargs.pop("event_rewards", [None])
        PROBS_ENV_EVENT = env_adjust_kwargs.pop("probs_env_event", [0.01])
        assert not env_adjust_kwargs, (
            f"Unexpected keys remain {env_adjust_kwargs.key()}")

        mentor_avoid_kwargs = {
            "states_from": STATES_FROM,
            "actions_from": ACTIONS_FROM,
            "action_from_probs": AVOID_ACT_PROBS,
        }

        env = FiniteStateCliffworld(
            state_shape=(w, w),
            init_agent_pos=(init, init),
            transition_function=TRANSITIONS[args.trans],
            make_env_adjusts=args.mentor == "avoid_state_act",
            states_from=STATES_FROM,
            actions_from=ACTIONS_FROM,
            states_to=STATES_TO,
            probs_event=PROBS_ENV_EVENT,
            event_rewards=REWARDS,
        )

        def S(s): return env.map_grid_to_int(s)
        def A(a): return env.map_grid_act_to_int(a)

        for st, ac, s_next in zip(STATES_FROM, ACTIONS_FROM, STATES_TO):
            print("STATE", st, ac, s_next)
            track_positions += [
                (S(st), A(ac), S(s_next)),
                # transitions of interest
                (S(st), A(ac), None),
                # transitions TO everywhere else
                (S(st), None, None),  # transitions with all other actions
            ]

    # Select the mentor, adding any kwargs
    if MENTORS[args.mentor] == "avoid_state_act_placeholder":
        def selected_mentor(state, kwargs=None):
            return random_safe_mentor(
                state,
                kwargs={**kwargs, **mentor_avoid_kwargs},
                avoider=True)
    else:
        selected_mentor = MENTORS[args.mentor]

    if args.env_test:
        env_visualisation(env)

    agent_init = AGENTS[args.agent]
    if args.agent == "pess_gln":
        agent_kwargs = {"dim_states": 2}
    elif args.agent == "continuous_pess_gln":
        agent_kwargs = {"dim_states": 4}
    else:
        agent_kwargs = {"num_states": env.num_states}

    if args.action_noise is not None:
        agent_kwargs = {
            **agent_kwargs,
            "eps_a_min": args.action_noise[0],
            "eps_a_max": args.action_noise[1],
        }
        if len(args.action_noise) == 3:
            agent_kwargs["eps_a_decay"] = args.action_noise[2]

    if "pess" in args.agent:
        agent_kwargs = {
            **agent_kwargs,
            **{"quantile_i": args.quantile, "init_to_zero": args.init == "zero"}
        }
    if args.agent == "pess_gln":
        agent_kwargs = {
            **agent_kwargs, **{"quantile_i": args.quantile}
        }

    if args.n_steps > 0:
        agent = agent_init(
            num_actions=env.num_actions,
            env=env,
            gamma=0.99,
            sampling_strategy=args.sampling_strategy,
            # 1. for the deterministic env
            lr=1. if str(args.trans) == "2" else 1e-1,
            mentor=selected_mentor,
            min_reward=env.min_nonzero_reward,
            eps_max=1.,
            eps_min=0.1,
            horizon_type=args.horizon,
            update_n_steps=args.update_freq,
            batch_size=(
                args.update_freq if args.batch_size is None
                else args.batch_size),
            num_steps=1 if args.horizon == "inf" else NUM_STEPS,
            scale_q_value=not args.unscale_q,
            track_transitions=track_positions,
            **agent_kwargs
        )

        learn_kwargs = {}

        success = agent.learn(
            args.n_steps,
            report_every_n=args.report_every_n,
            render=args.render,
            early_stopping=args.early_stopping,
            **learn_kwargs
        )

        print("Finished! Queries per ep:")
        print(agent.mentor_queries_periodic)
        print(f"Completed {success} after {agent.total_steps} steps")
        print("TRANSITIONS")
        print_transitions(agent.transitions)

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
            plt.plot(agent.mentor_queries_periodic)
            # plt.title(a.QEstimators[1].lr)
            plt.title(agent.q_estimator.lr)
            plt.show()

        return agent


if __name__ == "__main__":
    run_main(sys.argv[1:])
