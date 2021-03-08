import numpy as np
import matplotlib.pyplot as plt


def plot_scores(figure_name, scores, title=""):

    plt.figure()

    plt.plot(range(len(scores)), scores)
    plt.xlabel("Epsisode", fontsize=14)
    plt.ylabel("Score", fontsize=14)
    plt.title("Episodic scores " + title)

    plt.savefig(figure_name)


# TODO verify correctness - in a rush
def plot_queries(figure_name, queries, total_steps):
    plt.figure()
    sum_so_far = 0

    cum_series = np.empty((total_steps,))
    last_query_step = queries[0]
    cum_series[0:last_query_step] = 0
    for x in queries[1:]:
        sum_so_far += 1
        cum_series[last_query_step:x] = sum_so_far
        last_query_step = x
    cum_series[last_query_step:] = sum_so_far

    plt.plot(range(total_steps), cum_series)
    plt.xlabel("Timestep")
    plt.ylabel("cumulative queries")

    plt.savefig(figure_name)
