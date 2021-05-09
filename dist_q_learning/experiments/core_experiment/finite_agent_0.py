"""Run from dist_q_learning"""
import os

from main import run_main
from agents import QUANTILES

from experiments.utils import save_dict_to_pickle, experiment_main
from experiments.core_experiment import EXPERIMENT_PATH
from experiments.core_experiment.plotter import (
    plot_experiment_separate)  # , plot_experiment_together)


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
        "--horizon", horizon
    ]

    if horizon == "finite":
        args += ["--unscale-q"]
    if batch_size is not None:
        args += ["--batch-size", str(batch_size)]

    quantiles = list(range(len(QUANTILES)))
    pess_agent_args = args + ["--agent", agent]

    # pessimistic only
    for quant_i in [q for q in quantiles if QUANTILES[q] <= 0.5]:
        q_i_pess_args = pess_agent_args + ["--quantile", str(quant_i)]
        q_i_pess_args += ["--init", "zero" if init_zero else "quantile"]
        trained_agent = run_main(q_i_pess_args)

        exp_name = f"quant_{quant_i}" + repeat_str
        print("\nRUNNING", exp_name)
        result_i = {
            exp_name: {
                "quantile_val": QUANTILES[quant_i],
                "queries": trained_agent.mentor_queries_periodic,
                "rewards": trained_agent.rewards_periodic,
                "failures": trained_agent.failures_periodic,
                "metadata": {
                    "args": args,
                    "steps_per_report": report_every_n,
                    "min_nonzero": trained_agent.env.min_nonzero_reward,
                    "max_r": trained_agent.env.max_r,
                }
            }
        }
        save_dict_to_pickle(results_file, result_i)
        del trained_agent

    # And run for the mentor as a control
    mentor_args = args + ["--agent", "mentor"]
    mentor_agent_info = run_main(mentor_args)
    mentor_exp_name = "mentor" + repeat_str
    print("\nRUNNING", mentor_exp_name)
    mentor_result = {
        mentor_exp_name: {
            "quantile_val": -1.,
            "queries": mentor_agent_info.mentor_queries_periodic,
            "rewards": mentor_agent_info.rewards_periodic,
            "failures": mentor_agent_info.failures_periodic,
            "metadata": {
                "args": args,
                "steps_per_report": report_every_n,
                "min_nonzero": mentor_agent_info.env.min_nonzero_reward,
                "max_r": mentor_agent_info.env.max_r,
            }
        }
    }
    save_dict_to_pickle(results_file, mentor_result)
    del mentor_agent_info


if __name__ == "__main__":
    results_dir = os.path.join(EXPERIMENT_PATH, "results")
    os.makedirs(results_dir, exist_ok=True)

    N_REPEATS = 7
    exp_config = {
        "agent": "pess",
        "trans": "2",  # non-stochastic, sloped reward
        "report_every_n": 100,
        "steps": 500,
        "mentor": "random_safe",  # for exploration
        "earlystop": 0,  # hard to know the right place to stop - just do it
        "init_zero": True,  # This helps remove failures
    }

    from experiments.teleporter.configs.every_state import all_configs

    experiment_main(
        results_dir, N_REPEATS, run_core_experiment, all_configs[0],
        plot_experiment_separate)
