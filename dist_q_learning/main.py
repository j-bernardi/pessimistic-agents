import os
import sys
import jax
import random
import argparse
import numpy as np
import matplotlib.pyplot as plt

# Setting preallocate to false lets memory grow as needed, but increases risk
#  of fragmentation thus hitting out of memory (when not actually OOM)
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
# Mem fraction is harder to use, because it uses fraction of *remaining* mem
# Default is 0.9 - use 90% of *currently available* memory
# os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.45"

from env import FiniteStateCliffworld, ENV_ADJUST_KWARGS_KEYS, CartpoleEnv
from agents import (
    PessimisticAgent, QTableAgent, QTableMeanIREAgent, QTablePessIREAgent,
    MentorAgent, MentorAgentGLN,
    ContinuousPessimisticAgentGLN, ContinuousPessimisticAgentSigmaGLN,
    ContinuousPessimisticAgentBBB,
)
from mentors import (
    random_mentor, prudent_mentor, random_safe_mentor,
    cartpole_safe_mentor_normal, cartpole_safe_mentor)

from transition_defs import (
    deterministic_uniform_transitions, edge_cliff_reward_slope,
    reward_slope_stochastic_trans, generate_every_state_config_dict,
    generate_single_state_config_dict,
)

from experiments.event_experiment.plotter import print_transitions

MENTORS = {
    "prudent": prudent_mentor,
    "random": random_mentor,
    "random_safe": random_safe_mentor,
    "none": None,
    "cartpole_safe": "cartpole_placeholder",
    "avoid_state_act": "avoid_state_act_placeholder",
}

TRANSITIONS = {
    "0": deterministic_uniform_transitions,
    "1": edge_cliff_reward_slope,
    "2": lambda env: edge_cliff_reward_slope(env, standard_dev=None),
    "3": reward_slope_stochastic_trans,
}

EVENT_WRAPPERS = {
    "single_state": generate_single_state_config_dict,
    "every_state": generate_every_state_config_dict,
    "every_state_boost": lambda wid: generate_every_state_config_dict(
        width=wid, boost_rewards=True),
    "every_state_custom": "probs_placeholder",
}

AGENTS = {
    "pess": PessimisticAgent,
    "q_table": QTableAgent,
    "q_table_ire": QTableMeanIREAgent,
    "q_table_pess_ire": QTablePessIREAgent,
    "mentor": MentorAgent,
    "mentor_gln": MentorAgentGLN,
    "continuous_pess_gln": ContinuousPessimisticAgentGLN,
    "continuous_pess_gln_sigma": ContinuousPessimisticAgentSigmaGLN,
    "continuous_pess_bbb": ContinuousPessimisticAgentBBB,
}

SAMPLING_STRATS = ["last_n_steps", "random", "whole", "whole_reset"]

HORIZONS = ["inf", "finite"]  # Finite or infinite horizon
INITS = ["zero", "quantile"]  # Initialise pess Q value to 0. or q


def env_visualisation(_env):
    print("RESET STATE")
    _env.reset()
    _env.render(in_loop=False)

    print("\n\nStep every action")
    for action in range(0, _env.num_actions):
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
    parser.add_argument("--env", "-e", required=True, choices=["cart", "grid"])
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
        "--wrapper", "-w", default=None, nargs="+",
        help=f"The wrapper function to add interesting events to the base "
             f"transition function.\n{choices_help(EVENT_WRAPPERS)}")
    parser.add_argument(
        "--agent", "-a", default="q_table", choices=list(AGENTS.keys()),
        help=f"The agent to use.\n{choices_help(AGENTS)}")
    parser.add_argument(
        "--quantile", "-q", default=None, type=int,
        choices=[i for i in range(11)],
        help="The value quantile to use for taking actions")
    parser.add_argument(
        "--learning-rate", default=None, type=float,
        help="The learning rate for the agent and mentor model")
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
        "--n-horizons", default=10, type=int,
        help=f"The number of horizons to use if using finite horizon.")
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
        "--norm-min-val", default=None, type=int, choices=(0, -1),
        help=f"Min value in state normalisation [min_val, 1]")
    parser.add_argument(
        "--cart-task", default="stand_up", type=str,
        choices=["stand_up", "move_out"], help=f"The task of the cartpole")
    parser.add_argument(
        "--invert-mentor", default=None, action="store_true",
        help=f"Compatible only with move_out cart-task")
    parser.add_argument(
        "--render", "-r", type=int, default=0, help="render mode 0, 1, 2")
    parser.add_argument(
        "--early-stopping", default=0, type=int,
        help=f"Number of report periods to have 0 queries to define success.")
    parser.add_argument(
        "--n-steps", "-n", default=0, type=int,
        help=f"The number of steps to train for")
    parser.add_argument("--plot", action="store_true", help="display the plot")
    parser.add_argument(
        "--debug", action="store_true", help="run in debug mode (printing)")

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

    if _args.wrapper is not None:
        assert _args.wrapper[0] in EVENT_WRAPPERS.keys(), (
            f"{_args.wrapper[0]} not in {EVENT_WRAPPERS.keys()}")
        assert len(_args.wrapper) == 1\
            if _args.wrapper[0] != "every_state_custom" else 3

    if "whole" in _args.sampling_strategy and _args.batch_size is not None:
        raise ValueError()

    return _args


def parse_wrapper(width, args, env_adjust_kwargs):
    """Parse args and kwargs to finalise the env adjustment kwargs

    Returns:
        wrap_env: bool as to whether wrapper is used
        mentor_avoid_kwargs: kwargs for the mentor telling it which
            states to avoid
        env_adjust_kwargs: kwargs for the env setting the events up
    """
    if args.wrapper is not None:
        # Set any adjustments (e.g. unlikely event, etc)
        assert not env_adjust_kwargs, (
            f"Can't have a wrapper and adjust kwargs {env_adjust_kwargs}")
        if args.wrapper[0] != "every_state_custom":
            env_adjust_kwargs = EVENT_WRAPPERS[args.wrapper[0]](width)
        else:
            env_adjust_kwargs = generate_every_state_config_dict(
                width, mentor_prob=float(args.wrapper[1]),
                env_event_prob=float(args.wrapper[2]))
    elif env_adjust_kwargs:
        # Check have all the expected keys
        diff = ENV_ADJUST_KWARGS_KEYS - set(env_adjust_kwargs.keys())
        assert not diff, f"Keys missing: {diff}\n{env_adjust_kwargs}"

    # Parse the env adjust kwargs into those needed for the mentor, in
    # case they're needed for the avoid_state mentor
    if env_adjust_kwargs:
        wrap_env = True
        mentor_avoid_kwargs = {
            k: env_adjust_kwargs[k] for k in (
                "states_from", "actions_from", "avoid_act_probs")}
    else:
        wrap_env = False
        mentor_avoid_kwargs = {}
        env_adjust_kwargs = {}  # ensure a dict, for ** later
    return wrap_env, mentor_avoid_kwargs, env_adjust_kwargs


def parse_trackers(env, env_adjust_kwargs):
    """Track any special states, in the agent

    Returns:
        List of tuples of [state][action][next_state], passed to the
        agent to tell it which states to track.
    """

    track_positions = []
    if env_adjust_kwargs:
        def S(s):
            return env.map_grid_to_int(s)

        def A(a):
            return env.map_grid_act_to_int(a)

        for st, ac, s_next in zip(
                env_adjust_kwargs["states_from"],
                env_adjust_kwargs["actions_from"],
                env_adjust_kwargs["states_to"]):
            track_positions += [
                (S(st), A(ac), S(s_next)),
                # transitions of interest
                (S(st), A(ac), None),
                # transitions TO everywhere else
                (S(st), None, None),  # transitions with all other actions
            ]
        print(f"Tracking {len(track_positions)} transitions")
    else:
        track_positions = []
    return track_positions


def run_main(cmd_args, env_adjust_kwargs=None, seed=None):
    """Run the main script given cmd_args, and optional env adjustments

    cmd_args:
    env_adjust_kwargs (Optional[dict]): (see keys in function),
        specifies lists of grid co-ords with actions, probabilities that

    :return:
    """
    # used to ensure stochastic envs are the same across episodes
    if seed is not None:
        # add tensorflow, jax etc if / when it's used
        np.random.seed(seed)
        random.seed(seed)
    print(f"JAX DEVICES {jax.devices()}")
    print("PASSING", cmd_args)
    args = get_args(cmd_args)
    w = args.state_len
    init = w // 2
    agent_kwargs = {}

    if args.env == "cart":
        env = CartpoleEnv(
            min_val=args.norm_min_val, target=args.cart_task, random_x=False)
    elif args.env == "grid":
        wrap_env, mentor_avoid_kwargs, env_adjust_kwargs =\
            parse_wrapper(w, args, env_adjust_kwargs)

        # Create the (adjusted) env!
        env = FiniteStateCliffworld(
            state_shape=(w, w),
            init_agent_pos=(init, init),
            transition_function=TRANSITIONS[args.trans],
            make_env_adjusts=wrap_env,
            **env_adjust_kwargs
        )

        to_track = parse_trackers(env, env_adjust_kwargs)
        agent_kwargs.update({"track_transitions": to_track})
    else:
        raise ValueError(args.env)

    # Select the mentor, adding any kwargs. Only avoid the above states if we
    # select the corresponding mentor.
    if MENTORS[args.mentor] == "avoid_state_act_placeholder":
        def selected_mentor(state, kwargs=None):
            if kwargs is None:
                kwargs = {}
            return random_safe_mentor(
                state, kwargs={**kwargs, **mentor_avoid_kwargs}, avoider=True)
    elif MENTORS[args.mentor] == "cartpole_placeholder":
        # Handle continuous state scaling.
        # Set inversion on or off; rotates when gets to +/- X, if not None
        agent_kwargs["invert_mentor"] = args.invert_mentor

        def selected_mentor(state, **kwargs):
            if args.norm_min_val is not None:
                assert args.norm_min_val in (0, -1)
                return cartpole_safe_mentor_normal(
                    state,
                    centre_coord=(1. + args.norm_min_val) / 2.,
                    target_centre=args.cart_task == "stand_up",
                    **kwargs)
            else:
                return cartpole_safe_mentor
    else:
        selected_mentor = MENTORS[args.mentor]

    if args.env_test:
        env_visualisation(env)

    agent_init = AGENTS[args.agent]
    if "gln" in args.agent and args.env == "grid":
        agent_kwargs.update({"dim_states": 2})  # gridworld, 2d
        # found to be good for these GLNs
        agent_kwargs.update({
            "lr": args.learning_rate
                if args.learning_rate is not None else 5e-2})
    elif ("gln" in args.agent or "bbb" in args.agent) and args.env == "cart":
        agent_kwargs.update({"dim_states": 4})  # cartpole
        agent_kwargs.update({
            "lr": args.learning_rate
            if args.learning_rate is not None else 5e-2})
    elif args.env == "grid":
        agent_kwargs.update({"num_states": env.num_states})
        agent_kwargs.update({
            "lr": args.learning_rate if args.learning_rate is not None else (
                1. if str(args.trans) == "2" else 0.1)})

    if args.action_noise is not None:
        agent_kwargs.update({
            "eps_a_min": args.action_noise[0],
            "eps_a_max": args.action_noise[1],
        })
        if len(args.action_noise) == 3:
            agent_kwargs.update({"eps_a_decay", args.action_noise[2]})

    if "pess" in args.agent:
        agent_kwargs.update(
            {"quantile_i": args.quantile, "init_to_zero": args.init == "zero"})
    if args.agent == "pess_gln":
        agent_kwargs.update({"quantile_i": args.quantile})

    if args.n_steps > 0:
        agent = agent_init(
            num_actions=env.num_actions,
            env=env,
            gamma=0.95,
            sampling_strategy=args.sampling_strategy,
            mentor=selected_mentor,
            min_reward=env.min_nonzero_reward,
            eps_max=0.2,
            eps_min=0.05,
            horizon_type=args.horizon,
            update_n_steps=args.update_freq,
            batch_size=(
                args.update_freq if args.batch_size is None
                else args.batch_size),
            num_horizons=1 if args.horizon == "inf" else args.n_horizons,
            scale_q_value=not args.unscale_q,
            max_steps=np.inf,
            debug_mode=args.debug,
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

        # print("Finished! Queries per period:")
        # print(agent.mentor_queries_periodic)
        print(f"Completed {success} after {agent.total_steps} steps")
        if hasattr(agent, "transitions") and (
                len(agent.transitions) < 5 or args.plot):
            print("TRANSITIONS (states):", len(agent.transitions))
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
