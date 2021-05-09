"""Run from dist_q_learning"""
import os

from main import run_main
from agents import QUANTILES

from experiments.utils import save_dict_to_pickle, experiment_main
from experiments.core_experiment import EXPERIMENT_PATH
from experiments.core_experiment.plotter import (
    plot_experiment_separate)  # , plot_experiment_together)

from experiments.event_experiment.configs.every_state import all_configs


def run_core_experiment(
        results_file, agent, trans, report_every_n, mentor="random_safe",
        steps=500, earlystop=0, init_zero=False, repeat_n=0, render=-1,
        state_len=7, update_freq=1, batch_size=None,
        sampling_strat="last_n_steps", horizon="inf",
):
    repeat_str = f"_repeat_{repeat_n}"
    args = [
        "--trans", trans,
        "--report-every-n", str(report_every_n),
        "--mentor", mentor,
        "--n-steps", str(steps),
        "--early-stopping", str(earlystop),
        "--render", str(render),
        "--state-len", str(state_len),
        "--update-freq", str(update_freq),
        "--sampling-strategy", sampling_strat,
        "--horizon", horizon,
    ]
    if wrapper is not None:
        args += ["--wrapper", wrapper]

    if horizon == "finite":
        args += ["--unscale-q"]
    if batch_size is not None:
        args += ["--batch-size", str(batch_size)]

    quantiles = list(range(len(QUANTILES)))
    pess_agent_args = args + ["--agent", agent]

    # pessimistic only
    # for quant_i in [q for q in quantiles if QUANTILES[q] <= 0.5]:
    for quant_i in [0, 1, 4, 5]:
        q_i_pess_args = pess_agent_args + ["--quantile", str(quant_i)]
        q_i_pess_args += ["--init", "zero" if init_zero else "quantile"]
        trained_agent = run_main(q_i_pess_args, seed=repeat_n)

        exp_name = f"quant_{quant_i}" + repeat_str
        print("\nRUNNING", exp_name)
        result_i = parse_result(
            exp_name, trained_agent, quant_val=QUANTILES[quant_i],
            steps_per_report=report_every_n, arg_list=pess_agent_args)
        save_dict_to_pickle(results_file, result_i)
        del trained_agent

    # And run for the mentor as a control
    mentor_args = args + ["--agent", "mentor"]
    mentor_agent_info = run_main(mentor_args, seed=repeat_n)
    mentor_exp_name = "mentor" + repeat_str
    print("\nRUNNING", mentor_exp_name)
    mentor_result = parse_result(
        mentor_exp_name, mentor_agent_info, quant_val=-1.,
        steps_per_report=report_every_n, arg_list=args)
    save_dict_to_pickle(results_file, mentor_result)
    del mentor_agent_info


def parse_result(exp_key, agent, quant_val, steps_per_report, arg_list):
    result = {
        exp_key: {
            "quantile_val": quant_val,
            "queries": agent.mentor_queries_periodic,
            "rewards": agent.rewards_periodic,
            "failures": agent.failures_periodic,
            "metadata": {
                "args": arg_list,
                "steps_per_report": steps_per_report,
                "min_nonzero": agent.env.min_nonzero_reward,
                "max_r": agent.env.max_r,
            }
        }
    }
    return result


if __name__ == "__main__":
    results_dir = os.path.join(EXPERIMENT_PATH, "results")
    os.makedirs(results_dir, exist_ok=True)

    N_REPEATS = 7
    # exp_config = {
    #     "agent": "pess",
    #     "trans": "2",  # non-stochastic, sloped reward
    #     "report_every_n": 100,
    #     "steps": 500,
    #     "mentor": "random_safe",  # for exploration
    #     "earlystop": 0,  # hard to know the right place to stop - just do it
    #     "init_zero": True,  # This helps remove failures
    # }

    for cfg in all_configs:
        experiment_main(
            results_dir=results_dir,
            n_repeats=N_REPEATS,
            experiment_func=run_core_experiment,
            exp_config=cfg,
            plotting_func=plot_experiment_separate,
            show=False,
        )
