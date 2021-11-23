"""Run from dist_q_learning"""
import os

from main import run_main
from agents import QUANTILES

from experiments.exp_utils import (
    save_dict_to_pickle, experiment_main, parse_result, parse_experiment_args)
from experiments.core_experiment import EXPERIMENT_PATH
from experiments.core_experiment.plotter import plot_experiment_separate

from experiments.event_experiment.configs.every_state_gln import all_configs


GLN = True
N_REPEATS = 1


def run_core_experiment(
        results_file, agent, init_zero=False, repeat_n=0, **kwargs):
    repeat_str = f"_repeat_{repeat_n}"

    args = parse_experiment_args(kwargs, gln=GLN)
    report_every_n = int(args[args.index("--report-every-n") + 1])

    pess_agent_args = args + ["--agent", agent]

    # pessimistic only
    # quantiles = [i for i, q in enumerate(QUANTILES) if q <= 0.5]
    quantiles = [1]  # [0, 1, 4, 5]

    for quant_i in quantiles:
        q_i_pess_args = pess_agent_args + ["--quantile", str(quant_i)]
        q_i_pess_args += ["--init", "zero" if init_zero else "quantile"]
        trained_agent = run_main(q_i_pess_args, seed=repeat_n)

        exp_name = f"quant_{quant_i}" + repeat_str
        print("\nRUNNING", exp_name)
        result_i = parse_result(
            quantile_val=QUANTILES[quant_i],
            key=exp_name,
            agent=trained_agent,
            steps_per_report=report_every_n,
            arg_list=pess_agent_args,
            gln=GLN,
        )
        save_dict_to_pickle(results_file, result_i)
        del trained_agent

    # And run for the mentor as a control
    mentor_args = args + ["--agent", "mentor" + ("_gln" if GLN else "")]
    mentor_agent_info = run_main(mentor_args, seed=repeat_n)
    mentor_exp_name = "mentor" + repeat_str
    print("\nRUNNING", mentor_exp_name)
    mentor_result = parse_result(
        quantile_val="mentor",
        key=mentor_exp_name,
        agent=mentor_agent_info,
        steps_per_report=report_every_n,
        arg_list=args,
        gln=GLN
    )
    save_dict_to_pickle(results_file, mentor_result)
    del mentor_agent_info


if __name__ == "__main__":
    results_dir = os.path.join(EXPERIMENT_PATH, "results")
    os.makedirs(results_dir, exist_ok=True)

    for cfg in all_configs:
        print(f"CONFIG {cfg}")
        experiment_main(
            results_dir=results_dir,
            n_repeats=N_REPEATS,
            experiment_func=run_core_experiment,
            exp_config=cfg,
            plotting_func=plot_experiment_separate,
            show=False,
        )
