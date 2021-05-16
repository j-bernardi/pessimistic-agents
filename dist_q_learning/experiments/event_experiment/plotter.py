import numpy as np
import matplotlib.pyplot as plt


def compare_transitions(all_results, save_to=None, show=True, cmdline=False):
    """Double axis plot, (queries, failures) on left and rewards right

    Args:
        all_results (dict): The dictionary produced by run_experiment.
        save_to (Optional[str]): if not None, saves the experiment plot
            to this location.
        show (bool): if True, stops the script to show the figure
        cmdline (bool): whether to display data on cmd line
    """
    cmap = plt.get_cmap("tab10")
    legend = []
    fig, ax1 = plt.subplots(figsize=(10, 10))
    ax2 = ax1.twinx()  # set visits on 2nd axis

    if cmdline:
        for k in all_results:
            print("\nEXPERIMENT", k)
            print_transitions(all_results[k]["transitions"])

    # Dict of {exp_name: {events: [], state_actions: [], state_visits: []}}
    grouped_dict = {}

    max_1, max_2 = 0, 0
    for exp in all_results.keys():
        exp_dict = all_results[exp]
        trans_dict = exp_dict["transitions"]
        group_key = exp.split("_repeat")[0]
        if group_key not in grouped_dict:
            grouped_dict[group_key] = {
                k: [] for k in ("events", "state_actions", "state_visits")}

        # Manually collect the results from every state
        state_visits = np.array([0, 0], dtype=int)
        state_actions = np.array([0, 0], dtype=int)
        events = np.array([0, 0], dtype=int)
        for s, a, ns in [
                (s, a, ns) for s in trans_dict
                for a in trans_dict[s]
                for ns in trans_dict[s][a]]:
            num_vect = np.array(trans_dict[s][a][ns])  # [agent_n, mentor_n]
            if a is None and ns is None:
                state_visits += num_vect
            elif ns is None:
                state_actions += num_vect
            else:
                events += num_vect

        grouped_dict[group_key]["state_actions"].append(state_actions)
        grouped_dict[group_key]["events"].append(events)
        grouped_dict[group_key]["state_visits"].append(state_visits)

        # Sort out the plotting axes
        if max(state_actions) > max_1 or max(events) > max_1:
            max_1 = max(max(state_actions), max(events))
        if max(state_visits) > max_2:
            max_2 = max(state_visits)

    ax1.set_ylim(bottom=0, top=1.1 * max_1)
    ax2.set_ylim(bottom=0, top=1.1 * max_2)

    # PLOT THE RESULTS
    legend.append("agent")
    legend.append("mentor")
    tick_locs, tick_labels = [], []
    for x_tick, k in enumerate(grouped_dict.keys()):
        for j, tracked_quantity in enumerate(grouped_dict[k]):
            axis = ax2 if tracked_quantity == "state_visits" else ax1
            agent_mentor_arr = np.array(grouped_dict[k][tracked_quantity])
            x_dash = x_tick + (j - 1) * 0.1  # centre on x_tick
            if tracked_quantity == "events":
                text_x = x_dash - 0.25
            elif tracked_quantity == "state_visits":
                text_x = x_dash + 0.15
            else:
                text_x = x_dash
            tick_locs.append(x_dash)
            tick_labels.append(
                f"{k}_{tracked_quantity}_R{agent_mentor_arr.shape[0]}")

            # Plot mean val with stdev
            agent_mean_val = np.mean(agent_mentor_arr[:, 0])
            axis.errorbar(
                x_dash, agent_mean_val,
                np.std(agent_mentor_arr[:, 0]), marker="+", color=cmap(j))
            axis.annotate(
                f"A: {agent_mean_val:.1f}", (text_x, agent_mean_val),
                color=cmap(j)
            )

            mentor_mean_val = np.mean(agent_mentor_arr[:, 1])
            axis.errorbar(
                x_dash, mentor_mean_val,
                np.std(agent_mentor_arr[:, 1]), marker="^", color=cmap(j))
            axis.annotate(
                f"M: {mentor_mean_val:.1f}", (text_x, mentor_mean_val),
                color=cmap(j))

            # Plot all vals with alphas
            for (agent_n, mentor_n) in agent_mentor_arr:
                axis.scatter(
                    x_dash, agent_n, marker="+", alpha=0.2, color=cmap(j))
                axis.scatter(
                    x_dash, mentor_n, marker="^", alpha=0.2, color=cmap(j))
    ax1.set_xticks(tick_locs)
    ax1.set_xticklabels(tick_labels, rotation=90)

    leg = plt.legend(legend, loc="upper right")
    for lh in leg.legendHandles:
        lh.set_alpha(1)
    fig.tight_layout()  # otherwise the right y-label is slightly clipped

    if save_to is not None:
        plt.savefig(save_to)
    if show:
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
