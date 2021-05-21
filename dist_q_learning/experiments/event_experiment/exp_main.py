"""Run from dist_q_learning"""
import os

from main import run_main
from agents import QUANTILES

from experiments.exp_utils import (
    save_dict_to_pickle, experiment_main, parse_experiment_args, parse_result)
from experiments.event_experiment import EXPERIMENT_PATH
from experiments.event_experiment.plotter import compare_transitions

from experiments.event_experiment.configs.every_state_over_probs import\
    all_configs


def run_event_avoid_experiment(
        results_file, agent, init_zero=True, repeat_n=0, action_noise=None,
        **kwargs):
    repeat_str = f"_repeat_{repeat_n}"
    args = parse_experiment_args(kwargs)
    report_every_n = int(args[args.index("--report-every-n") + 1])

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
            steps_per_report=report_every_n,
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
        quantile_val="q_table", key=q_table_exp_name, agent=q_table_agent,
        steps_per_report=report_every_n, arg_list=q_table_args)
    save_dict_to_pickle(results_file, q_table_result)
    del q_table_agent

    # And run for the mentor as a control
    mentor_args = args + ["--agent", "mentor"]
    mentor_exp_name = "mentor" + repeat_str
    print("\nRUNNING", mentor_exp_name, mentor_args)
    # Must copy as we'll be popping
    mentor_agent_info = run_main(mentor_args, seed=repeat_n)
    mentor_result = parse_result(
        quantile_val="mentor", key=mentor_exp_name, agent=mentor_agent_info,
        steps_per_report=report_every_n, arg_list=mentor_args)
    save_dict_to_pickle(results_file, mentor_result)
    del mentor_agent_info


def run_event_avoid_experiment_over_probs(
        results_file, agent, init_zero=True, repeat_n=0, **kwargs):
    repeat_str = f"_repeat_{repeat_n}"

    # Note 0.25 is the equal-prob (in the centre, not on edges), e.g. as likely
    # to happen as random
    for p in [0.01, 0.02, 0.04, 0.08, 0.12, 0.16, 0.20, 0.25]:
        p_string = str(p).replace(".", "")
        args = parse_experiment_args(kwargs)
        # The key for the experiment: mentor avoid with prob p, bad event
        # never happens (to avoid bad experience variation across envs)
        args += ["--wrapper", "every_state_custom", f"{p}", "0.0"]
        report_every_n = int(args[args.index("--report-every-n") + 1])

        pess_agent_args = args + ["--agent", agent]

        quant_i = 0  # most pessimistic only
        q_i_pess_args = pess_agent_args + ["--quantile", str(quant_i)]
        q_i_pess_args += ["--init", "zero" if init_zero else "quantile"]

        exp_name = f"quant_{quant_i}_{p_string}" + repeat_str
        print("\nRUNNING", exp_name)
        trained_agent = run_main(q_i_pess_args, seed=repeat_n)
        result_i = parse_result(
            quantile_val=QUANTILES[quant_i],
            key=exp_name,
            agent=trained_agent,
            steps_per_report=report_every_n,
            arg_list=q_i_pess_args
        )
        save_dict_to_pickle(results_file, result_i)
        del trained_agent

        # And run for the q_table agent
        q_table_args = args + ["--agent", "q_table"]
        q_table_exp_name = f"q_table_{p_string}" + repeat_str
        print("\nRUNNING", q_table_exp_name)
        # Must copy as we'll be popping
        q_table_agent = run_main(q_table_args, seed=repeat_n)
        q_table_result = parse_result(
            quantile_val="q_table", key=q_table_exp_name, agent=q_table_agent,
            steps_per_report=report_every_n, arg_list=q_table_args)
        save_dict_to_pickle(results_file, q_table_result)
        del q_table_agent

        # And run for the mentor as a control
        mentor_args = args + ["--agent", "mentor"]
        mentor_exp_name = f"mentor_{p_string}" + repeat_str
        print("\nRUNNING", mentor_exp_name)
        # Must copy as we'll be popping
        mentor_agent_info = run_main(mentor_args, seed=repeat_n)
        mentor_result = parse_result(
            quantile_val="mentor", key=mentor_exp_name, agent=mentor_agent_info,
            steps_per_report=report_every_n, arg_list=mentor_args)
        save_dict_to_pickle(results_file, mentor_result)
        del mentor_agent_info


if __name__ == "__main__":
    RESULTS_DIR = os.path.join(EXPERIMENT_PATH, "results_test")
    N_REPEATS = 7

    for cfg in all_configs:
        experiment_main(
            results_dir=RESULTS_DIR,
            n_repeats=N_REPEATS,
            experiment_func=run_event_avoid_experiment_over_probs,
            exp_config=cfg,
            # TODO - plot risky N/7 for each prob run, the 3 next to each other
            #  text can go over the bars for mentor action-taking prob +/-
            plotting_func=compare_transitions,
            show=False,
            plot_save_ext="_events"
        )
    # plt.show()
