"""Run from dist_q_learning"""
import os
import pickle

from main import run_main
from agents import QUANTILES

from experiments.core_experiment import EXPERIMENT_PATH
from experiments.core_experiment.plotter import (
    plot_experiment_separate, plot_experiment_together)


def save(filename, new_result):
    """Append an item to results dictionary, cached in pickle file

    Given:
        results = {0: "a", 1: "b"}
        new_result = {2: "c"}
    Result:
        results = {0: "a", 1: "b", 2: "c"}
    """
    # Load
    if os.path.exists(filename):
        with open(filename, 'rb') as f:
            results = pickle.load(f)
    else:
        results = {}
    # Append
    for k in new_result.keys():
        if k in results.keys():
            yes = input(f"Key {k} already in dict - replace?\ny/ n")
            if yes == "y":
                del results[k]
            else:
                raise KeyError(f"{k} already in results {results.keys()}")
    new_results = {**results, **new_result}

    # Save
    with open(filename, 'wb') as fl:
        pickle.dump(new_results, fl, protocol=pickle.HIGHEST_PROTOCOL)


# TODO - add uncertainty bars with 10 repeats
def run_experiment(
        results_file, agent, trans, n, mentor, steps=500, earlystop=0,
        init_zero=False, repeat_n=0
):
    repeat_str = f"_repeat_{repeat_n}"
    args = [
        "--trans", trans, "--num-episodes", str(n), "--mentor", mentor,
        "--steps-per-ep", str(steps), "--early-stopping", str(earlystop),
    ]

    quantiles = list(range(len(QUANTILES)))
    pess_agent_args = args + ["--agent", agent]

    # TEMP (TODO) - pessimistic only
    for quant_i in [q for q in quantiles if QUANTILES[q] <= 0.5]:
        q_i_pess_args = pess_agent_args + ["--quantile", str(quant_i)]
        q_i_pess_args += ["--init-zero"] if init_zero else []
        trained_agent = run_main(q_i_pess_args)

        result_i = {
            f"quant_{quant_i}" + repeat_str: {
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
        save(results_file, result_i)
        del trained_agent

    # And run for the mentor as a control
    mentor_args = args + ["--agent", "mentor"]
    mentor_agent_info = run_main(mentor_args)
    mentor_result = {
        "mentor" + repeat_str: {
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
    save(results_file, mentor_result)


if __name__ == "__main__":
    results_dir = os.path.join(EXPERIMENT_PATH, "results")
    os.makedirs(results_dir, exist_ok=True)

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

    f_name_no_ext = os.path.join(
        results_dir, "_".join([f"{k}_{str(v)}" for k, v in exp_config.items()]))
    dict_loc = f_name_no_ext + ".p"

    if os.path.exists(dict_loc):
        run = input(f"Found {dict_loc}\nOverwrite? y / n / a\n")
    else:
        run = "y"

    if run in ("y", "a"):
        if run == "y" and os.path.exists(dict_loc):
            os.remove(dict_loc)

        for i in range(N_REPEATS):
            run_experiment(dict_loc, repeat_n=i, **exp_config)

    with open(dict_loc, "rb") as f:
        results_dict = pickle.load(f)
    plot_experiment_separate(results_dict, f_name_no_ext + ".png")
