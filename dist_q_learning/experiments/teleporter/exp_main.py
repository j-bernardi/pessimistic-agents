"""Run from dist_q_learning"""
import os

from main import run_main
from agents import QUANTILES

from experiments.utils import save_dict_to_pickle, experiment_main
from experiments.teleporter import EXPERIMENT_PATH
from experiments.teleporter.plotter import plot_experiment


def run_teleport_experiment(
        results_file, agent, trans, n, mentor, steps=500, earlystop=0,
        init_zero=False, repeat_n=0, render=-1
):
    raise NotImplementedError("Not written for teleporting yet")
    repeat_str = f"_repeat_{repeat_n}"
    args = [
        "--trans", trans, "--num-episodes", str(n), "--mentor", mentor,
        "--steps-per-ep", str(steps), "--early-stopping", str(earlystop),
        "--render", str(render)
    ]

    quantiles = list(range(len(QUANTILES)))
    pess_agent_args = args + ["--agent", agent]

    # pessimistic only
    for quant_i in [q for q in quantiles if QUANTILES[q] <= 0.5]:
        q_i_pess_args = pess_agent_args + ["--quantile", str(quant_i)]
        q_i_pess_args += ["--init-zero"] if init_zero else []
        trained_agent = run_main(q_i_pess_args)

        exp_name = f"quant_{quant_i}" + repeat_str
        print("\nRUNNING", exp_name)
        result_i = {
            exp_name: {
                "quantile_val": QUANTILES[quant_i],
                "steps_per_ep": steps,
                "queries": trained_agent.mentor_queries_per_ep,
                "rewards": trained_agent.rewards_per_ep,
                "failures": trained_agent.failures_per_ep,
                "metadata": {
                    "args": args,
                    "steps_per_ep": steps,
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
            "queries": mentor_agent_info.mentor_queries_per_ep,
            "rewards": mentor_agent_info.rewards_per_ep,
            "failures": mentor_agent_info.failures_per_ep,
            "metadata": {
                "args": args,
                "steps_per_ep": steps,
                "min_nonzero": mentor_agent_info.env.min_nonzero_reward,
                "max_r": mentor_agent_info.env.max_r,
            }
        }
    }
    save_dict_to_pickle(results_file, mentor_result)


if __name__ == "__main__":
    results_dir = os.path.join(EXPERIMENT_PATH, "results")
    N_REPEATS = 7
    exp_config = {
        "agent": "pess",
        "trans": "2",  # non-stochastic, sloped reward
        "n": 100,
        "steps": 200,
        "mentor": "random_safe",  # for exploration
        "earlystop": 0,  # hard to know the right place to stop - just do it
        "init_zero": True,  # This helps remove failures
    }

    experiment_main(
        results_dir, N_REPEATS, run_teleport_experiment, exp_config,
        plot_experiment)
