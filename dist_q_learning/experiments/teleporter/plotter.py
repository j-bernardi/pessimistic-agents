import numpy as np
import matplotlib.pyplot as plt


def compare_transitions(all_results, save_to=None):
    """Double axis plot, (queries, failures) on left and rewards right

    TODO - plot nicely

    Args:
        all_results (dict): The dictionary produced by run_experiment.
        save_to (Optional[str]): if not None, saves the experiment plot
            to this location.
    """
    cmap = plt.get_cmap("tab10")
    legend = []
    fig, ax1 = plt.subplots()
    for k in all_results:
        print("\nEXPERIMENT", k)
        print_transitions(all_results[k]["transitions"])

    # Dict of {exp_name: {teleports: [], state_actions: [], state_visits: []}}
    grouped_dict = {}

    for exp in all_results.keys():
        exp_dict = all_results[exp]
        trans_dict = exp_dict["transitions"]
        group_key = exp.split("_repeat")[0]
        if group_key not in grouped_dict:
            grouped_dict[group_key] = {
                k: [] for k in ("teleports", "state_actions", "state_visits")}

        # Find the color
        if "quant" in exp:
            i = int(group_key.split("_")[-1])  # quantile i
        elif "mentor" in exp:
            i = -2  # hopefully different to quantile i's
        elif "q_table" in exp:
            i = -3  # again, different to quantile i's
        else:
            raise KeyError("Unexpected experiment key", exp)

        grouped_dict[group_key]["teleports"].append(trans_dict[40][0][8])
        grouped_dict[group_key]["state_actions"].append(trans_dict[40][0][None])
        grouped_dict[group_key]["state_visits"].append(
            trans_dict[40][None][None])

    # PLOT THE RESULTS
    legend.append("agent")
    legend.append("mentor")
    tick_locs, tick_labels = [], []
    for x_tick, k in enumerate(grouped_dict.keys()):
        for j, tracked_quantity in enumerate(grouped_dict[k]):
            agent_mentor_arr = np.array(grouped_dict[k][tracked_quantity])
            x_dash = x_tick + (j - 1) * 0.1  # centre on x_tick
            tick_locs.append(x_dash)
            tick_labels.append(k + "_" + tracked_quantity)
            # Plot mean val with stdev
            ax1.errorbar(
                x_dash, np.mean(agent_mentor_arr[:, 0]),
                np.std(agent_mentor_arr[:, 0]), marker="+", color=cmap(j))
            ax1.errorbar(
                x_dash, np.mean(agent_mentor_arr[:, 1]),
                np.std(agent_mentor_arr[:, 1]), marker="^", color=cmap(j))
            # Plot all vals with alphas
            for (agent_n, mentor_n) in agent_mentor_arr:
                ax1.scatter(
                    x_dash, agent_n, marker="+", alpha=0.2, color=cmap(j))
                ax1.scatter(
                    x_dash, mentor_n, marker="^", alpha=0.2, color=cmap(j))
    ax1.set_xticks(tick_locs)
    ax1.set_xticklabels(tick_labels, rotation=90)

    leg = plt.legend(legend, loc="upper right")
    for lh in leg.legendHandles:
        lh.set_alpha(1)
    fig.tight_layout()  # otherwise the right y-label is slightly clipped

    if save_to is not None:
        plt.savefig(save_to)
    plt.show()


def print_transitions(transition_dict):
    if not transition_dict:
        print("Transitions", transition_dict)
        return
    for s in transition_dict:
        print("State", s)
        for a in transition_dict[s]:
            print("\tAction", a)
            for ns, (ag, m) in transition_dict[s][a].items():
                print(f"\t\tTo state {ns}:  - agent: {ag}, mentor: {m}")
    return
