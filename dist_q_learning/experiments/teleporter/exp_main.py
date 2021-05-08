"""Run from dist_q_learning"""
import os
import copy
import matplotlib.pyplot as plt

from main import run_main
from agents import QUANTILES

from experiments.utils import save_dict_to_pickle, experiment_main
from experiments.teleporter import EXPERIMENT_PATH
from experiments.teleporter.plotter import compare_transitions

# from experiments.teleporter.configs.single_teleport_pad_configs import (
from experiments.teleporter.configs.every_state import (
    all_configs, env_config_dicts, N_REPEATS)


def run_handler(config, results_dir, n_repeats, env_configs):
    """Runs the experiment_main util function with this experiment file"""

    def wrapped_avoider_experiment(fname, **exp_config_kwargs):
        """Make arg signature match the main experiment to share wrapper"""
        return run_event_avoid_experiment(
            fname,
            env_adjust_kwargs_per_repeat=env_configs,
            **exp_config_kwargs)

    experiment_main(
        results_dir=results_dir,
        n_repeats=n_repeats,
        experiment_func=wrapped_avoider_experiment,
        exp_config=config,
        plotting_func=compare_transitions,
        show=False,
    )


def run_event_avoid_experiment(
        results_file, agent, trans, n, steps_per_ep=500, earlystop=0,
        init_zero=False, repeat_n=0, render=-1, update_freq=1,
        sampling_strat="last_n_steps", env_adjust_kwargs_per_repeat=None,
        action_noise=None, horizon="inf", batch_size=None,
        state_len=7,
):
    """
    results_file:
    agent:
    trans:
    n:
    steps_per_ep:
    earlystop:
    init_zero:
    repeat_n: which number repeat this is
    render:
    update_freq:
    sampling_strat:
    env_adjust_kwargs: a list of kwarg-dicts, one per repeat
        (indexed at repeat_n)
    action_noise:
    horizon:
    batch_size:
    state_len:
    """
    repeat_str = f"_repeat_{repeat_n}"
    args = ["--mentor", "avoid_state_act"]
    args += [
        "--trans", trans,
        "--num-episodes", str(n),
        "--steps-per-ep", str(steps_per_ep),
        "--early-stopping", str(earlystop),
        "--render", str(render),
        "--sampling-strategy", sampling_strat,
        "--update-freq", str(update_freq),
        "--horizon", horizon,
        "--state-len", str(state_len),
    ]

    if horizon == "finite":
        args += ["--unscale-q"]
    if batch_size is not None:
        args += ["--batch-size", str(batch_size)]

    quantiles = list(range(len(QUANTILES)))
    pess_agent_args = args + ["--agent", agent]
    if action_noise is not None:
        assert isinstance(action_noise, str), "Comma separated string expected"
        pess_agent_args += ["--action-noise"] + action_noise.split(", ")

    # pessimistic only
    for quant_i in [q for q in quantiles if QUANTILES[q] <= 0.5]:
        # Must copy as we'll be popping
        env_adjust_kwargs = copy.deepcopy(
            env_adjust_kwargs_per_repeat[repeat_n])
        q_i_pess_args = pess_agent_args + ["--quantile", str(quant_i)]
        q_i_pess_args += ["--init", "zero" if init_zero else "quantile"]

        exp_name = f"quant_{quant_i}" + repeat_str
        print("\nRUNNING", exp_name)
        trained_agent = run_main(
            q_i_pess_args, env_adjust_kwargs=env_adjust_kwargs)
        result_i = parse_result(
            quantile_val=QUANTILES[quant_i],
            key=exp_name,
            agent=trained_agent,
            steps=steps_per_ep,
            arg_list=q_i_pess_args
        )
        save_dict_to_pickle(results_file, result_i)
        del trained_agent

    # And run for the q_table agent
    q_table_args = args + ["--agent", "q_table"]
    if action_noise is not None:
        assert isinstance(action_noise, str), "Comma separated string expected"
        q_table_args += ["--action-noise"] + action_noise.split(", ")
    q_table_exp_name = "q_table" + repeat_str
    print("\nRUNNING", q_table_exp_name)
    # Must copy as we'll be popping
    env_adjust_kwargs = copy.deepcopy(env_adjust_kwargs_per_repeat[repeat_n])
    q_table_agent = run_main(q_table_args, env_adjust_kwargs=env_adjust_kwargs)
    q_table_result = parse_result(
        "q_table", q_table_exp_name, q_table_agent, steps_per_ep, q_table_args)
    save_dict_to_pickle(results_file, q_table_result)
    del q_table_agent

    # And run for the mentor as a control
    mentor_args = args + ["--agent", "mentor"]
    mentor_exp_name = "mentor" + repeat_str
    print("\nRUNNING", mentor_exp_name, mentor_args)
    # Must copy as we'll be popping
    env_adjust_kwargs = copy.deepcopy(env_adjust_kwargs_per_repeat[repeat_n])
    mentor_agent_info = run_main(
        mentor_args, env_adjust_kwargs=env_adjust_kwargs)
    mentor_result = parse_result(
        "mentor", mentor_exp_name, mentor_agent_info, steps_per_ep, mentor_args)
    save_dict_to_pickle(results_file, mentor_result)
    del mentor_agent_info


def parse_result(quantile_val, key, agent, steps, arg_list):
    """Take the info from an exp and return a single-item dict"""
    result = {
        key: {
            "quantile_val": quantile_val,
            "steps_per_ep": steps,
            "queries": agent.mentor_queries_per_ep,
            "rewards": agent.rewards_per_ep,
            "failures": agent.failures_per_ep,
            "transitions": agent.transitions,  # ADDED
            "metadata": {
                "args": arg_list,
                "steps_per_ep": steps,
                "min_nonzero": agent.env.min_nonzero_reward,
                "max_r": agent.env.max_r,
            }
        }
    }
    return result


if __name__ == "__main__":
    RESULTS_DIR = os.path.join(EXPERIMENT_PATH, "results")

    ###
    # NUM_EPS = 100
    # STEPS_PER_EP = 200
    # exp_config = {
    #     "agent": "pess",
    #     "trans": "1",
    #     "n": NUM_EPS,
    #     "steps_per_ep": STEPS_PER_EP,
    #     "earlystop": 0,  # hard to know the right place to stop - just do it
    #     "init_zero": True,  # This helps remove failures
    #     # TODO - consider action noise to ensure we explore those states
    #     # "action_noise": "0.01, 0.10, 0.9999",
    #     # 0.01 0.10 0.9999 is an OK start point for 20 * 500 steps (adjust)
    #     "update_freq": 1000,
    #     "sampling_strat": "whole",
    # }
    # all_configs = [exp_config]  # Now imported
    ####

    assert len(env_config_dicts) == N_REPEATS
    for cfg in all_configs:
        run_handler(
            config=cfg,
            results_dir=RESULTS_DIR,
            n_repeats=N_REPEATS,
            env_configs=env_config_dicts,
        )
    plt.show()
