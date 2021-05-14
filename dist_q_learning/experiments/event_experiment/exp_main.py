"""Run from dist_q_learning"""
import os

from main import run_main
from agents import QUANTILES

from experiments.utils import (
    save_dict_to_pickle, experiment_main, parse_experiment_args, parse_result)
from experiments.event_experiment import EXPERIMENT_PATH
from experiments.event_experiment.plotter import compare_transitions

from experiments.event_experiment.configs.every_state import all_configs


def run_event_avoid_experiment(
        results_file, agent, init_zero=True, repeat_n=0, action_noise=None,
        **kwargs):
    repeat_str = f"_repeat_{repeat_n}"
    args = parse_experiment_args(kwargs)

    pess_agent_args = args + ["--agent", agent]
    if action_noise is not None:
        assert isinstance(action_noise, str), "Comma separated string expected"
        pess_agent_args += ["--action-noise"] + action_noise.split(", ")

    # pessimistic only
    # quantiles = [i for i, q in enumerate(QUANTILES) if q <= 0.5]
    quantiles = [0, 1, 4, 5]
    for quant_i in quantiles:
        q_i_pess_args = pess_agent_args + ["--quantile", str(quant_i)]
        q_i_pess_args += ["--init", "zero" if init_zero else "quantile"]

        exp_name = f"quant_{quant_i}" + repeat_str
        print("\nRUNNING", exp_name)
        trained_agent = run_main(q_i_pess_args, seed=repeat_n)
        result_i = parse_result(
            quantile_val=QUANTILES[quant_i],
            key=exp_name,
            agent=trained_agent,
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
    q_table_agent = run_main(q_table_args, seed=repeat_n)
    q_table_result = parse_result(
        "q_table", q_table_exp_name, q_table_agent, q_table_args)
    save_dict_to_pickle(results_file, q_table_result)
    del q_table_agent

    # And run for the mentor as a control
    mentor_args = args + ["--agent", "mentor"]
    mentor_exp_name = "mentor" + repeat_str
    print("\nRUNNING", mentor_exp_name, mentor_args)
    # Must copy as we'll be popping
    mentor_agent_info = run_main(mentor_args, seed=repeat_n)
    mentor_result = parse_result(
        "mentor", mentor_exp_name, mentor_agent_info, mentor_args)
    save_dict_to_pickle(results_file, mentor_result)
    del mentor_agent_info


if __name__ == "__main__":
    RESULTS_DIR = os.path.join(EXPERIMENT_PATH, "results_test")
    N_REPEATS = 7
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

    for cfg in all_configs:
        experiment_main(
            results_dir=RESULTS_DIR,
            n_repeats=N_REPEATS,
            experiment_func=run_event_avoid_experiment,
            exp_config=cfg,
            plotting_func=compare_transitions,
            show=False,
            plot_save_ext="_events"
        )
    # plt.show()
