"""Run from dist_q_learning"""
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt

from main import run_main
from agents import QUANTILES

from experiments.core_experiment import RESULTS_FNAME


def save(new_result, filename=RESULTS_FNAME):
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
    new_results = {**results, **new_result}
    # Save
    with open(filename, 'wb') as f:
        pickle.dump(new_results, f, protocol=pickle.HIGHEST_PROTOCOL)


# TODO - add uncertainty bars with 10 repeats
def run_experiment(agent, trans_f, n, mentor, steps=500):
    args = [
        "--trans", trans_f, "--num-episodes", str(n), "--mentor", mentor,
        "--steps-per-ep", str(steps),
    ]

    quantiles = list(range(len(QUANTILES)))
    pess_agent_args = args + ["--agent", agent]
    for quant_i in quantiles:
        q_i_pess_args = pess_agent_args + ["--quantile", str(quant_i)]
        trained_agent = run_main(q_i_pess_args)

        result_i = {
            f"quant_{quant_i}": {
                "quantile_val": QUANTILES[quant_i],
                "steps_per_ep": steps,
                "queries": trained_agent.mentor_queries_per_ep,
                "rewards": trained_agent.rewards_per_ep,
                "failures": trained_agent.failures_per_ep,
            }
        }
        save(result_i)
        del trained_agent

    # And run for the mentor as a control
    mentor_args = args + ["--agent", "mentor"]
    mentor_agent_info = run_main(mentor_args)
    mentor_result = {
        "mentor": {
            "quantile_val": -1.,
            "steps_per_ep": steps,
            "queries": mentor_agent_info.mentor_queries_per_ep,
            "rewards": mentor_agent_info.rewards_per_ep,
            "failures": mentor_agent_info.failures_per_ep,
        }
    }
    save(mentor_result)


def plot_experiment():
    """Double axis plot, queries and rewards"""
    with open(RESULTS_FNAME, 'rb') as f:
        all_results = pickle.load(f)

    fig, ax1 = plt.subplots()
    color = "tab:orange"
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Mentor queries", color=color)
    ax1.tick_params(axis="y", labelcolor=color)

    # instantiate a second axes that shares the same x-axis
    ax2 = ax1.twinx()
    color2 = "tab:blue"
    ax2.set_ylabel("Agent avg rewards/step", color=color2)
    ax2.tick_params(axis="y", labelcolor=color2)

    cmap = plt.get_cmap("tab10")
    legend = []

    for i, exp in enumerate(all_results.keys()):
        num_eps = len(all_results[exp]["queries"])
        xs = list(range(num_eps))
        # Legend = exp
        legend.append(exp)

        # Right axis is rewards
        rewards = np.array(all_results[exp]["rewards"])
        rewards_per_step = rewards / len(rewards)
        ax2.plot(xs, rewards_per_step, color=cmap(i), linestyle="dotted")

        # Left axis is queries and failures
        queries = all_results[exp]["queries"]
        ax1.plot(xs, queries, color=cmap(i), linestyle="dashed")
        failures = all_results[exp]["failures"]
        ax1.plot(xs, failures, color=cmap(i), linestyle="solid")

    plt.legend(legend, loc="center right")
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.show()


if __name__ == "__main__":
    # sloping R transition function
    run_experiment(agent="pess", trans_f="1", n=100, mentor="random_safe")
    # , steps=5)  for speed
    plot_experiment()
