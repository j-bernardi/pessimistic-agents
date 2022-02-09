import numpy as np
import matplotlib
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
    rewards_per_step = rewards_per_step[:len(xs)]
    ax.plot(xs, rewards_per_step, color=color, linestyle=linestyle, alpha=alpha)


def plot_q(xs, exp_dict, ax, color, linestyle="solid", alpha=None, norm_by=1.):
    queries = np.array(exp_dict["queries"]) / norm_by
    # queries = queries[:len(xs)]
    ax.plot(queries, color=color, linestyle=linestyle, alpha=alpha)

def plot_q_vals(xs, exp_dict, ax, color, linestyle="solid", alpha=None, norm_by=1.):
    q_vals = np.array(exp_dict["q_vals"]) / norm_by
    # q_vals = q_vals[:len(xs)]
    ax.plot(q_vals, color=color, linestyle=linestyle, alpha=alpha)

def plot_mentor_q_vals(xs, exp_dict, ax, color, linestyle="solid", alpha=None, norm_by=1.):
    q_vals = np.array(exp_dict["mentor_q_vals"]) / norm_by
    q_vals = q_vals[:len(xs)]
    ax.plot(xs, q_vals, color=color, linestyle=linestyle, alpha=alpha)

def plot_f(xs, exp_dict, ax, color, linestyle="solid", alpha=None):
    cumulative_failures = np.cumsum(exp_dict["failures"])
    cumulative_failures = cumulative_failures[:len(xs)]
    ax.plot(
        xs, cumulative_failures, color=color, linestyle=linestyle, alpha=alpha)


def plot_experiment_separate(all_results, save_to=None, show=True):
    """Triple ax plot, queries, failures, rewards

    Args:
        all_results (dict): the dict saved by run_experiment.
        save_to (Optional[str]): if not None, saves the experiment plot
            to this location.
        show (bool): whether to show the plot at the end
    """
    skip_keys = ()# ("quant_4", "quant_5")
    step_limit = 20000
    cmap = plt.get_cmap("tab10")

    q_table_i, mentor_i = 2, 3  # must be different to quantile 'i's
    legend = []

    font = {"size": 16}
    matplotlib.rc('font', **font)
    fig, axs = plt.subplots(
        nrows=3, ncols=1, sharex="all", gridspec_kw={'hspace': 0.1},
        figsize=(9, 9),
        # figsize=(16/1.2, 9/1.2),
    )

    axs[0].set_ylabel("Queries / step")
    axs[1].set_ylabel("Reward / step")
    axs[2].set_ylabel("Cumulative failures")
    axs[2].set_xlabel("Steps")

    def plot_dict_result(exp_d, color, alpha=None):
        num_reports = len(exp_d["queries"])
        steps_per_report = exp_d["metadata"]["steps_per_report"]
        xs = list(steps_per_report * n for n in range(num_reports))
        if step_limit:
            xs = [x for x in xs if x <= step_limit]
        plot_q(xs, exp_d, axs[0], color, alpha=alpha, norm_by=steps_per_report)
        plot_r(xs, exp_d, axs[1], color, alpha=alpha, norm_by=steps_per_report)
        plot_f(xs, exp_d, axs[2], color, alpha=alpha)

    mean_dict = {}
    for exp in all_results.keys():
        exp_dict = all_results[exp]
        if any(sk in exp for sk in skip_keys):
            continue
        mean_exp_key = exp.split("_repeat")[0]
        # Find the color
        if "quant" in exp:
            i = int(mean_exp_key.split("_")[-1])  # quantile i
        elif "mentor" in exp:
            i = mentor_i
        elif "q_table" in exp:
            i = q_table_i
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
        if any(sk in k for sk in skip_keys):
            continue
        if "quant" in k:
            i = int(k.split("_")[-1])
        elif "mentor" in k:
            i = mentor_i
        elif "q_table" in k:
            i = q_table_i
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



def plot_experiment_separate_with_qs(all_results, save_to=None, show=True):
    """Triple ax plot, queries, failures, rewards

    Args:
        all_results (dict): the dict saved by run_experiment.
        save_to (Optional[str]): if not None, saves the experiment plot
            to this location.
        show (bool): whether to show the plot at the end
    """
    skip_keys = ()# ("quant_4", "quant_5")
    step_limit = 20000
    cmap = plt.get_cmap("tab10")

    q_table_i, mentor_i = 2, 3  # must be different to quantile 'i's
    legend = []

    font = {"size": 16}
    matplotlib.rc('font', **font)
    fig, axs = plt.subplots(
        nrows=5, ncols=1, sharex="all", gridspec_kw={'hspace': 0.1},
        figsize=(9, 16),
        # figsize=(16/1.2, 9/1.2),
    )

    axs[0].set_ylabel("Queries / step")
    axs[1].set_ylabel("Reward / step")
    axs[2].set_ylabel("Cumulative failures")
    axs[2].set_xlabel("Steps")

    axs[3].set_ylabel("Max Agent Q vals")
    axs[4].set_ylabel("Mentor Q vals")


    def plot_dict_result(exp_d, color, alpha=None):
        num_reports = len(exp_d["queries"])
        steps_per_report = exp_d["metadata"]["steps_per_report"]
        xs = list(steps_per_report * n for n in range(num_reports))
        if step_limit:
            xs = [x for x in xs if x <= step_limit]
        plot_q(xs, exp_d, axs[0], color, alpha=alpha, norm_by=steps_per_report)
        plot_r(xs, exp_d, axs[1], color, alpha=alpha, norm_by=steps_per_report)
        plot_f(xs, exp_d, axs[2], color, alpha=alpha)
        plot_q_vals(xs, exp_d, axs[3], color, alpha=alpha)
        plot_mentor_q_vals(xs, exp_d, axs[4], color, alpha=alpha)

    mean_dict = {}
    for exp in all_results.keys():
        exp_dict = all_results[exp]
        if any(sk in exp for sk in skip_keys):
            continue
        mean_exp_key = exp.split("_repeat")[0]
        # Find the color
        if "quant" in exp:
            i = int(mean_exp_key.split("_")[-1])  # quantile i
        elif "mentor" in exp:
            i = mentor_i
        elif "q_table" in exp:
            i = q_table_i
        else:
            raise KeyError("Unexpected experiment key", exp)
        # Plot faded
        plot_dict_result(exp_dict, color=cmap(i), alpha=0.1)

        # UPDATE THE MEAN
        keys = ("queries", "rewards", "failures", "q_vals", "mentor_q_vals")
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
        if any(sk in k for sk in skip_keys):
            continue
        if "quant" in k:
            i = int(k.split("_")[-1])
        elif "mentor" in k:
            i = mentor_i
        elif "q_table" in k:
            i = q_table_i
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
