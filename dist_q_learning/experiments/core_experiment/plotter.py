import numpy as np
import matplotlib.pyplot as plt


def smooth(vals, rolling=10):
    """Take rolling average to smooth"""
    new_vals = list(vals[:rolling])
    for i in range(rolling, len(vals)):
        new_vals.append(sum(vals[i-rolling:i]) / rolling)
    return np.array(new_vals)


def set_queries_axis(ax, color="tab:orange", failures=False):
    ax.set_xlabel("Steps")
    label = "Mentor queries" + (", cumulative failures" if failures else "")
    ax.set_ylabel(label, color=color)
    ax.tick_params(axis="y", labelcolor=color)


def set_rewards_axis(ax, color="tab:blue"):
    ax.set_ylabel("Agent avg rewards/step", color=color)
    ax.tick_params(axis="y", labelcolor=color)


def plot_r(xs, exp_dict, ax, color, linestyle="solid", alpha=None, norm_by=1.):
    period_reward_sum = np.array(exp_dict["rewards"])
    rewards_per_step = smooth(period_reward_sum / norm_by)
    ax.plot(xs, rewards_per_step, color=color, linestyle=linestyle, alpha=alpha)


def plot_q(xs, exp_dict, ax, color, linestyle="solid", alpha=None, norm_by=1.):
    queries = np.array(exp_dict["queries"]) / norm_by
    ax.plot(xs, queries, color=color, linestyle=linestyle, alpha=alpha)


def plot_f(xs, exp_dict, ax, color, linestyle="solid", alpha=None):
    cumulative_failures = np.cumsum(exp_dict["failures"])
    ax.plot(
        xs, cumulative_failures, color=color, linestyle=linestyle, alpha=alpha)


def plot_experiment_together(all_results, save_to=None, show=True):
    """Double axis plot, (queries, failures) on left and rewards right

    Args:
        all_results (dict): The dictionary produced by run_experiment.
        save_to (Optional[str]): if not None, saves the experiment plot
            to this location.
    """
    cmap = plt.get_cmap("tab10")
    legend = []

    fig, ax1 = plt.subplots(figsize=(10, 10))
    set_queries_axis(ax1, failures=True)

    # instantiate a second axes that shares the same x-axis
    ax2 = ax1.twinx()
    set_rewards_axis(ax2)

    # {"quantile_i": {"n": 0, "rewards": [], ...}
    # Where n is number contributing to mean so far
    mean_dict = {}

    def plot_dict_result(exp_d, color, alpha=None):
        num_reports = len(exp_d["queries"])
        steps_per_report = exp_d["metadata"]["steps_per_report"]
        xs = list(steps_per_report * n for n in range(num_reports))

        # Right axis is rewards
        plot_r(
            xs, exp_d, ax2, color, linestyle="dashed", alpha=alpha,
            norm_by=steps_per_report)
        # Left axis is queries and failures
        plot_q(
            xs, exp_d, ax1, color=color, linestyle="dotted", alpha=alpha,
            norm_by=steps_per_report,
        )
        plot_f(xs, exp_d, ax1, color=color, linestyle="solid", alpha=alpha)

    for exp in all_results.keys():
        # TODO this code is repeated in other plotter
        exp_dict = all_results[exp]
        mean_exp_key = exp.split("_repeat")[0]
        # Find the color
        if "quant" in exp:
            i = int(mean_exp_key.split("_")[-1])  # quantile i
        elif "mentor" in exp:
            i = -1  # hopefully different to quantile i's
        elif "q_table" in exp:
            i = -2
        else:
            raise KeyError("Unexpected experiment key", exp)
        # Plot faded
        plot_dict_result(exp_dict, color=cmap(i), alpha=0.1)

        # UPDATE THE MEAN
        keys = ("queries", "rewards", "failures")
        if mean_exp_key in mean_dict:
            md = mean_dict[mean_exp_key]
            for k in keys:
                md[k] = (
                    md[k] * md["n"] + np.array(exp_dict[k])
                ) / (md[k]["n"] + 1)
            md["n"] += 1
        else:
            mean_dict[mean_exp_key] = {
                **{k: np.array(exp_dict[k]) for k in keys},
                **{"metadata": {
                    "steps_per_report":
                        exp_dict["metadata"]["steps_per_report"]}}
            }
            mean_dict[mean_exp_key]["n"] = 1

    # PLOT THE MEANS
    for k in mean_dict:
        if "quant" in k:
            i = int(k.split("_")[-1])
        elif "mentor" in k:
            i = -1
        elif "q_table" in k:
            i = -2
        else:
            raise KeyError("Unexpected key", k)
        plot_dict_result(mean_dict[k], color=cmap(i), alpha=None)
        legend.append(k)

    leg = plt.legend(legend, loc="center right")
    for line in leg.get_lines():
        line.set_alpha(None)
    fig.tight_layout()  # otherwise the right y-label is slightly clipped

    if save_to is not None:
        plt.savefig(save_to)
    if show:
        plt.show()


def plot_experiment_separate(all_results, save_to=None, show=True):
    """Triple ax plot, queries, failures, rewards

    Args:
        all_results (dict): the dict saved by run_experiment.
        save_to (Optional[str]): if not None, saves the experiment plot
            to this location.
    """
    cmap = plt.get_cmap("tab10")
    legend = []
    fig, axs = plt.subplots(
        nrows=3, ncols=1, sharex="all", gridspec_kw={'hspace': 0.1},
        figsize=(10, 10),
    )

    axs[0].set_ylabel("Mentor query freq / step")
    axs[1].set_ylabel("Cumulative failures")
    axs[2].set_ylabel("Avg R / step")
    axs[2].set_xlabel("Steps")

    def plot_dict_result(exp_d, color, alpha=None):
        num_reports = len(exp_d["queries"])
        steps_per_report = exp_d["metadata"]["steps_per_report"]
        xs = list(steps_per_report * n for n in range(num_reports))
        plot_q(xs, exp_d, axs[0], color, alpha=alpha, norm_by=steps_per_report)
        plot_f(xs, exp_d, axs[1], color, alpha=alpha)
        plot_r(xs, exp_d, axs[2], color, alpha=alpha, norm_by=steps_per_report)

    mean_dict = {}
    for exp in all_results.keys():
        # TODO this code is repeated in other plotter
        exp_dict = all_results[exp]
        mean_exp_key = exp.split("_repeat")[0]
        # Find the color
        if "quant" in exp:
            i = int(mean_exp_key.split("_")[-1])  # quantile i
        elif "mentor" in exp:
            i = -1  # hopefully different to quantile i's
        elif "q_table" in exp:
            i = -2
        else:
            raise KeyError("Unexpected experiment key", exp)
        # Plot faded
        plot_dict_result(exp_dict, color=cmap(i), alpha=0.1)

        # UPDATE THE MEAN
        keys = ("queries", "rewards", "failures")
        if mean_exp_key in mean_dict:
            md = mean_dict[mean_exp_key]
            # Take mean
            for k in keys:
                md[k] = (
                    md[k] * md["n"] + np.array(exp_dict[k])
                ) / (md["n"] + 1)
            md["n"] += 1
        else:
            mean_dict[mean_exp_key] = {
                **{k: np.array(exp_dict[k]) for k in keys},
                **{"metadata": {
                    "steps_per_report":
                        exp_dict["metadata"]["steps_per_report"]}}
            }
            mean_dict[mean_exp_key]["n"] = 1

    # PLOT THE MEANS
    for k in mean_dict:
        if "quant" in k:
            i = int(k.split("_")[-1])
        elif "mentor" in k:
            i = -1
        elif "q_table" in k:
            i = -2
        else:
            raise KeyError("Unexpected key", k)
        plot_dict_result(mean_dict[k], color=cmap(i), alpha=None)
        legend.append(f"{k}_R{mean_dict[k]['n']}")
    leg = axs[1].legend(legend, loc="center right")
    for line in leg.get_lines():
        line.set_alpha(None)

    if save_to is not None:
        plt.savefig(save_to)
    if show:
        plt.show()
