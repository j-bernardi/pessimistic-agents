"""Run from dist_q_learning"""
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt

from main import run_main
from agents import QUANTILES

from experiments.core_experiment import EXPERIMENT_PATH


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
    new_results = {**results, **new_result}
    # Save
    with open(filename, 'wb') as f:
        pickle.dump(new_results, f, protocol=pickle.HIGHEST_PROTOCOL)


# TODO - add uncertainty bars with 10 repeats
def run_experiment(
        results_file, agent, trans, n, mentor, steps=500, earlyStop=0,
        init_zero=False
):
    args = [
        "--trans", trans, "--num-episodes", str(n), "--mentor", mentor,
        "--steps-per-ep", str(steps), "--early-stopping", str(earlyStop),
    ]

    quantiles = list(range(len(QUANTILES)))
    pess_agent_args = args + ["--agent", agent]

    # TEMP (TODO) - pessimistic only
    for quant_i in [q for q in quantiles if QUANTILES[q] <= 0.5]:
        q_i_pess_args = pess_agent_args + ["--quantile", str(quant_i)]
        q_i_pess_args += ["--init-zero"] if init_zero else []
        trained_agent = run_main(q_i_pess_args)

        result_i = {
            f"quant_{quant_i}": {
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
        "mentor": {
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


def set_queries_axis(ax, color="tab:orange", failures=False):
    ax.set_xlabel("Episode")
    label = "Mentor queries" + (", cumulative failures" if failures else "")
    ax.set_ylabel(label, color=color)
    ax.tick_params(axis="y", labelcolor=color)


def set_rewards_axis(ax, color="tab:blue"):
    ax.set_ylabel("Agent avg rewards/step", color=color)
    ax.tick_params(axis="y", labelcolor=color)


def plot_r(xs, exp_dict, ax, color, linestyle="solid"):
    episode_reward_sum = np.array(exp_dict["rewards"])
    rewards_per_step = smooth(
        episode_reward_sum / exp_dict["metadata"]["steps_per_ep"])
    ax.plot(xs, rewards_per_step, color=color, linestyle=linestyle)


def plot_q(xs, exp_dict, ax, color, linestyle="solid"):
    queries = exp_dict["queries"]
    ax.plot(xs, queries, color=color, linestyle=linestyle)


def plot_f(xs, exp_dict, ax, color, linestyle="solid"):
    cumulative_failures = np.cumsum(exp_dict["failures"])
    ax.plot(xs, cumulative_failures, color=color, linestyle=linestyle)


def plot_experiment_together(all_results, save_to=None):
    """Double axis plot, (queries, failures) on left and rewards right

    Args:
        all_results (dict): The dictionary produced by run_experiment.
        save_to (Optional[str]): if not None, saves the experiment plot
            to this location.
    """
    cmap = plt.get_cmap("tab10")
    legend = []

    fig, ax1 = plt.subplots()
    set_queries_axis(ax1, failures=True)

    # instantiate a second axes that shares the same x-axis
    ax2 = ax1.twinx()
    set_rewards_axis(ax2)

    for i, exp in enumerate(all_results.keys()):
        exp_dict = all_results[exp]
        num_eps = len(exp_dict["queries"])
        xs = list(range(num_eps))
        # Legend = exp
        legend.append(exp)

        # Right axis is rewards
        plot_r(xs, exp_dict, ax2, cmap(i), linestyle="dashed")
        # Left axis is queries and failures
        plot_q(xs, exp_dict, ax1, color=cmap(i), linestyle="dotted")
        plot_f(xs, exp_dict, ax1, color=cmap(i), linestyle="solid")

    plt.legend(legend, loc="center right")
    fig.tight_layout()  # otherwise the right y-label is slightly clipped

    if save_to is not None:
        plt.savefig(save_to)
    plt.show()


def plot_experiment_separate(all_results, save_to=None):
    """Triple ax plot, queries, failures, rewards

    Args:
        all_results (dict): the dict saved by run_experiment.
        save_to (Optional[str]): if not None, saves the experiment plot
            to this location.
    """
    cmap = plt.get_cmap("tab10")
    legend = []

    fig, axs = plt.subplots(
        nrows=3, ncols=1, sharex="all", gridspec_kw={'hspace': 0.1}
    )

    axs[0].set_ylabel("Mentor queries")
    axs[1].set_ylabel("Cumulative failures")
    axs[2].set_ylabel("Avg R / step")
    axs[2].set_xlabel("Episode")

    for i, exp in enumerate(all_results.keys()):
        exp_dict = all_results[exp]
        num_eps = len(all_results[exp]["queries"])
        xs = list(range(num_eps))
        # Legend = exp
        legend.append(exp)

        # Left axis is queries and failures
        queries = all_results[exp]["queries"]
        axs[0].plot(xs, queries, color=cmap(i))
        plot_q(xs, exp_dict, axs[0], cmap(i))
        plot_f(xs, exp_dict, axs[1], cmap(i))
        plot_r(xs, exp_dict, axs[2], cmap(i))

    axs[1].legend(legend, loc="center right")

    if save_to is not None:
        plt.savefig(save_to)
    plt.show()


def smooth(vals, rolling=10):
    """Take rolling average to smooth"""
    new_vals = list(vals[:rolling])
    for i in range(rolling, len(vals)):
        new_vals.append(sum(vals[i-rolling:i]) / rolling)
    return np.array(new_vals)


if __name__ == "__main__":
    results_dir = os.path.join(EXPERIMENT_PATH, "results")
    os.makedirs(results_dir, exist_ok=True)
    exp_config = {
        "agent": "pess",
        "trans": "2",
        "n": 100,
        "mentor": "random_safe",
        "earlyStop": 0,
        "steps": 200,
        "init_zero": True,
    }

    f_name_no_ext = os.path.join(
        results_dir, "_".join([f"{k}_{str(v)}" for k, v in exp_config.items()]))
    dict_loc = f_name_no_ext + ".p"

    if os.path.exists(dict_loc):
        run = input(f"Found {dict_loc}\nOverwrite? y / n\n")
    else:
        run = "y"

    if run == "y":
        if os.path.exists(dict_loc):
            os.remove(dict_loc)
        run_experiment(dict_loc, **exp_config)

    with open(dict_loc, "rb") as f:
        results_dict = pickle.load(f)
    plot_experiment_separate(results_dict, f_name_no_ext + ".png")
