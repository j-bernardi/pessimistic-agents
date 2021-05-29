import numpy as np
import matplotlib
import matplotlib.pyplot as plt


def get_grouped_dict(all_results, skip_keys=None):
    """Extract results in a useful way

    Returns:
        Dict of {
            exp_name: {
                events: [], state_actions: [], state_visits: []
            }, ...
        }
    """
    grouped_dict = {}
    skip_keys = [] if skip_keys is None else skip_keys

    max_ax1, max_ax2 = 0, 0
    for exp in [
            e for e in all_results.keys()
            if not any(skip in e for skip in skip_keys)]:
        trans_dict = all_results[exp]["transitions"]
        group_key = exp.split("_repeat")[0]
        if group_key not in grouped_dict:
            grouped_dict[group_key] = {
                k: [] for k in ("events", "state_actions", "state_visits")}

        # Manually collect the results from every state
        state_visits = np.array([0, 0], dtype=int)
        state_actions = np.array([0, 0], dtype=int)
        events = np.array([0, 0], dtype=int)
        for s, a, ns in [
            (s, a, ns)
            for s in trans_dict
            for a in trans_dict[s]
            for ns in trans_dict[s][a]]:
            num_vect = np.array(trans_dict[s][a][ns])  # [agent_n, mentor_n]
            # None indicates to collect any
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
        if max(state_actions) > max_ax1 or max(events) > max_ax1:
            max_ax1 = max(max(state_actions), max(events))
        if max(state_visits) > max_ax2:
            max_ax2 = max(state_visits)

    return grouped_dict, (max_ax1, max_ax2)


def compare_transitions(all_results, save_to=None, show=True, cmdline=False):
    """Double axis plot, (queries, failures) on left and rewards right

    Detailed but ugly

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
    grouped_dict, (max_1, max_2) = get_grouped_dict(all_results)

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


def nice_compare_transitions(all_results, save_to=None, show=True):
    """Compare the number of episodes with risky transitions for agents

    Beautified. Only plots the number of risky actions taken.

    Args:
        all_results (dict): The dictionary produced by run_experiment.
        save_to (Optional[str]): if not None, saves the experiment plot
            to this location.
        show (bool): if True, stops the script to show the figure
    """
    skip_keys = ("quant_4", "quant_5", "mentor")
    cmap = plt.get_cmap("tab10")
    font = {"size": 14}
    matplotlib.rc('font', **font)
    fig, ax1 = plt.subplots(figsize=(16, 9))
    ax2 = ax1.twinx()  # set visits on 2nd axis
    ax1.set_ylabel("Agent repeats with risk", color=cmap(0))
    ax2.set_ylabel("Proportion mentor steps demonstrating risks", color=cmap(1))

    # Dict of {exp_name: {events: [], state_actions: [], state_visits: []}}
    grouped_dict, _ = get_grouped_dict(all_results, skip_keys)

    # PLOT THE RESULTS
    tick_locs, tick_labels = [], []
    for x_tick, k in enumerate(
            [e for e in grouped_dict.keys() if e not in skip_keys]):

        # (N_repeats, 2), idx: agent=0, mentor=1
        # Safe
        state_visits = np.array(grouped_dict[k]["state_visits"])
        n_repeats = state_visits.shape[0]
        # Unsafe
        state_actions = np.array(grouped_dict[k]["state_actions"])
        events = np.array(grouped_dict[k]["events"])
        unsafe_interactions = state_actions + events

        mentor_unsafe_props = unsafe_interactions[:, 1] / (
            unsafe_interactions[:, 1] + state_visits[:, 1])

        # AGENT
        agent_x_dash = x_tick - 0.1
        tick_locs.append(agent_x_dash)
        tick_labels.append(f"A_{k}_{n_repeats}_rep")
        # Number of repeat episodes in which the agent took the risky action
        agent_repeats_with_risk = (unsafe_interactions[:, 0] >= 1).sum()
        # TODO - box plot ?
        prop = agent_repeats_with_risk / n_repeats
        ax1.bar(agent_x_dash, prop, width=0.15, color=cmap(0), label="Agent")

        ax1.annotate(
            f"{agent_repeats_with_risk} / {n_repeats}",
            (agent_x_dash, prop + 0.005), color=cmap(0), ha="center")

        # MENTOR - proportion of steps that were risky
        mentor_x_dash = x_tick + 0.1
        tick_locs.append(mentor_x_dash)
        tick_labels.append(
            f"M_{k}_{unsafe_interactions.shape[0]}_rep")
        ax2.bar(
            mentor_x_dash, np.mean(mentor_unsafe_props),
            yerr=np.std(mentor_unsafe_props), width=0.15, color=cmap(1),
            label="Mentor",
        )

    ax1.set_xticks(tick_locs)
    ax1.set_xticklabels(tick_labels, rotation=45, ha="right")
    fig.tight_layout()  # otherwise the right y-label is slightly clipped

    if save_to is not None:
        plt.savefig(save_to)
    if show:
        plt.show()


def compare_transitions_across_probs(all_results, save_to=None, show=True):
    """Compare the number of episodes with risky transitions for agents
    with varying probability

    Beautified. Only plots the number of risky actions taken.

    Args:
        all_results (dict): The dictionary produced by run_experiment.
        save_to (Optional[str]): if not None, saves the experiment plot
            to this location.
        show (bool): if True, stops the script to show the figure
    """
    skip_keys = ["mentor"]
    cmap = plt.get_cmap("tab10")
    font = {"size": 14}
    matplotlib.rc('font', **font)
    fig, ax1 = plt.subplots(figsize=(16, 9))
    ax1.set_ylabel("Agent repeats with risk", color=cmap(0))
    exp_args = all_results[list(all_results.keys())[0]]["metadata"]["args"]
    quant = all_results[list(all_results.keys())[0]]["quantile_val"]
    title = "Transition func " + exp_args[exp_args.index("--trans") + 1]
    title += f", quantile {quant:.3f}"
    ax1.set_title(title)

    # Dict of {exp_name: {events: [], state_actions: [], state_visits: []}}
    grouped_dict, _ = get_grouped_dict(all_results, skip_keys)

    # PLOT THE RESULTS
    tick_locs, tick_labels = [], []
    # TODO group each result cluster
    legend = [f"pess_agent, Q{quant:.3f}", "q_table_agent"]
    x_tick = 0  # assume dict is ordered in (agent, q table) repetitions
    for i, k in enumerate(grouped_dict.keys()):
        # Plot in groups of 2
        if i % 2 == 0:
            x_tick += 1

        ci = 0 if "quant" in k else 1 if "mentor" in k else 2
        # (N_repeats, 2), idx: agent=0, mentor=1
        # Safe
        state_visits = np.array(grouped_dict[k]["state_visits"])
        n_repeats = state_visits.shape[0]
        # Unsafe
        state_actions = np.array(grouped_dict[k]["state_actions"])
        events = np.array(grouped_dict[k]["events"])
        unsafe_interactions = state_actions + events

        # mentor_unsafe_proportions = unsafe_interactions[:, 1] / (
        #     unsafe_interactions[:, 1] + state_visits[:, 1])

        # AGENT
        agent_x_dash = x_tick - 0.1 + 0.2 * (i % 2)  # alternate side of x tick
        tick_locs.append(agent_x_dash)
        tick_labels.append(f"{k}_{n_repeats}_rep")
        # Number of repeat episodes in which the agent took the risky action
        agent_repeats_with_risk = (unsafe_interactions[:, 0] >= 1).sum()

        # TODO - box plot ?
        prop = agent_repeats_with_risk / n_repeats
        ax1.bar(agent_x_dash, prop, width=0.15, color=cmap(ci), label="Agent")
        ax1.annotate(
            f"{agent_repeats_with_risk} / {n_repeats}",
            (agent_x_dash, prop + 0.005), color=cmap(ci), ha="center",
            rotation=30)

        # MENTOR - proportion of steps that were risky
        # mentor_x_dash = x_tick + 0.1
        # ax1.annotate(
        #     f"{np.mean(mentor_unsafe_proportions):.2f}"
        #     f"+/- {np.std(mentor_unsafe_proportions):.2f}",
        #     (mentor_x_dash, max(prop, 0.2)), color=cmap(ci), ha="center",
        #     rotation=45)

    ax1.legend(legend, loc="center right")
    ax1.set_xticks(tick_locs)
    ax1.set_xticklabels(tick_labels, rotation=45, ha="right")
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
