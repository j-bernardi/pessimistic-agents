import os.path
import time
import numpy as np
from sklearn.neighbors import DistanceMetric, NearestNeighbors
from sklearn.cluster import KMeans
from haiku.data_structures import to_immutable_dict
import matplotlib.pyplot as plt

# from q_estimators import QuantileQEstimatorGaussianGLN
from gln_ablation import make_data, make_gln, abs_error

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"


def base_runner(
        gln, n_splits, batch_size, silent=False, uncert=None,
        sample_strat="val"):
    if sample_strat == "whole_hist":
        assert uncert is multi_step_uncert
    n_train, n_val, n_test = n_splits

    x_train, y_train = make_data(n=n_train)
    x_val, y_val = make_data(n=n_val)

    ns, kmeans = [], []

    # TRAIN
    n_steps = n_train // batch_size
    for epoch in range(1):
        print("EPOCH", epoch)
        for i in range(0, n_train, batch_size):
            x_batch = x_train[i:i + batch_size]
            y_batch = y_train[i:i + batch_size]
            gln.predict(x_batch, target=y_batch)
            if n_steps <= 10 or  ((i % (n_steps // 10)) == 0 and i):
                val_losses = []
                for j in range(0, n_val, batch_size):
                    val_preds = gln.predict(x_val[0:batch_size])
                    val_losses.append(
                        abs_error(y_val[0:batch_size], val_preds))

                est_pos = x_val[0:1]  # keep dimensionality
                data_so_far = x_train[:i+batch_size]

                # Get number in the group
                data_plus_pt = np.concatenate((est_pos, data_so_far))
                kmeans_object = KMeans(n_clusters=10)
                kmeans_object.fit_predict(data_plus_pt)
                group_of_pt = kmeans_object.labels_[0]  # first point
                grps, counts = np.unique(
                    kmeans_object.labels_, return_counts=True)
                num_in_group = counts[list(grps).index(group_of_pt)]
                kmeans.append(num_in_group)

                # Measure distances to this point
                dist_metric_obj = DistanceMetric.get_metric("euclidean")
                distances_to_pt = dist_metric_obj.pairwise(data_so_far, est_pos)
                mean, std = np.mean(distances_to_pt), np.std(distances_to_pt)
                if uncert is not None:
                    if sample_strat == "whole_hist":
                        x_uncert_batch = x_train[:i + batch_size]
                        y_uncert_batch = y_train[:i + batch_size]
                        gln.batch_size = i + batch_size
                    elif sample_strat == "val":
                        x_uncert_batch = x_train[1:batch_size]
                        y_uncert_batch = y_train[1:batch_size]
                    elif sample_strat == "nearest":
                        nbrs = NearestNeighbors(
                            n_neighbors=(batch_size - 1),
                            algorithm="ball_tree").fit(x_train[:i + batch_size])
                        _, neighbour_i = nbrs.kneighbors(est_pos)
                        x_uncert_batch = x_train[np.squeeze(neighbour_i)]
                        y_uncert_batch = y_train[np.squeeze(neighbour_i)]
                    else:
                        raise NotImplementedError(sample_strat)
                    if "single_step" in uncert.__name__:
                        gln.batch_size = 1
                    uncert_n, uncert_a, uncert_b = uncert(
                        gln, est_pos, x_uncert_batch, y_uncert_batch)
                    gln.batch_size = batch_size
                    ns.append(uncert_n)
                    uncert_string = (
                        f"\tUncertainty ({uncert.__name__})\n\t\tn: {uncert_n:.2f}"
                        f"\n\t\talpha: {uncert_a:.2f}\n\t\tbeta: {uncert_b:.2f}")
                else:
                    uncert_string = None

                if not silent:
                    print(f"BATCH {i // batch_size} / {n_train // batch_size}")
                    print(f"\tval_loss: {np.mean(val_losses):.4f}")
                    print(f"\tEuclidean distances: {mean:.2f} +/- {std:.2f}")
                    if uncert:
                        print(uncert_string)
                    print(f"\tNum in kmeans group: {num_in_group}")

    # TEST
    x_test, y_test = make_data(n=n_test)
    final_losses = []
    for i in range(0, n_test, batch_size):
        y_test_preds = gln.predict(x_test[i:i + batch_size])
        final_losses.append(
            abs_error(y_test[i:i + batch_size], y_test_preds))
    final_loss = np.mean(final_losses)
    print("TEST LOSS", final_loss)

    return ns, kmeans


def single_step_uncert(gln, x_pos, *unused_batch_args):
    fake_targets = [0., 1.]
    fake_means = np.empty((1, 2))
    current_mean = gln.predict(x_pos)

    def all_equal(params1, params2):
        equals = []
        for k1, k2 in zip(params1, params2):
            for kk1, kk2 in zip(params1[k1], params2[k2]):
                equals.append(np.all(params1[k1][kk1] == params2[k2][kk2]))
        return all(equals)

    current_params = to_immutable_dict(gln.gln_params)
    for j, fake_target in enumerate(fake_targets):
        assert all_equal(gln.gln_params, current_params)
        # Update to fake target
        gln.predict(x_pos, [fake_target])
        assert not all_equal(gln.gln_params, current_params)
        fake_means[:, j] = gln.predict(x_pos)
        # Clean up
        gln.gln_params = to_immutable_dict(current_params)
        assert all_equal(gln.gln_params, current_params)

    return pseudocount(current_mean, fake_means)


def single_step_uncert_about_half(gln, x_pos, *unused_batch_args):
    """Update to 0.5, and [0., 1.], and measure uncertainty there"""
    fake_targets = [0.5, 0., 1.]
    fake_means = np.empty((1, 3))

    current_params = to_immutable_dict(gln.gln_params)
    for j, fake_target in enumerate(fake_targets):
        # Update to fake target
        gln.predict(x_pos, [fake_target])
        fake_means[:, j] = gln.predict(x_pos)
        # Clean up
        gln.gln_params = to_immutable_dict(current_params)

    current_mean = fake_means[:, 0]
    fake_means = fake_means[:, 1:]
    # Can't call pseudocount because of the 2x multiplier here
    diff = 2. * (current_mean[:, None] - fake_means) * np.array([1., -1.])
    diff = np.where(diff == 0., 1e-8, diff)
    n_ais0 = fake_means[:, 0] / diff[:, 0]
    n_ais1 = (1. - fake_means[:, 1]) / diff[:, 1]

    # print(f"n0={n_ais0}, n1={n_ais1}")
    n_ais = np.dstack((n_ais0, n_ais1))
    ns = np.squeeze(np.min(n_ais, axis=-1))
    alpha = np.squeeze(current_mean * ns + 1.)
    beta = np.squeeze((1. - current_mean) * ns + 1.)

    # print(f"n={ns}, alpha={alpha}, beta={beta}")

    return ns, alpha, beta


def single_step_uncert_about_estimate(gln, x_pos, *unused_batch_args):
    """Fake update to current estimate, and [0., 1.]
       and measure uncertainty there"""
    current_est = gln.predict(x_pos)
    fake_targets = [float(current_est[0]), 0., 1.]
    fake_means = np.empty((1, 3))

    current_params = to_immutable_dict(gln.gln_params)
    for j, fake_target in enumerate(fake_targets):
        # Update to fake target
        gln.predict(x_pos, [fake_target])
        fake_means[:, j] = gln.predict(x_pos)
        # Clean up
        gln.gln_params = to_immutable_dict(current_params)

    current_mean = fake_means[:, 0]
    fake_means = fake_means[:, 1:]
    # Can't call pseudocount() because of the 2x multiplier here
    diff = 2. * (current_mean[:, None] - fake_means) * np.array([1., -1.])
    diff = np.where(diff == 0., 1e-8, diff)
    n_ais0 = fake_means[:, 0] / diff[:, 0]
    n_ais1 = (1. - fake_means[:, 1]) / diff[:, 1]

    # print(f"n0={n_ais0}, n1={n_ais1}")
    n_ais = np.dstack((n_ais0, n_ais1))
    ns = np.squeeze(np.min(n_ais, axis=-1))
    alpha = np.squeeze(current_mean * ns + 1.)
    beta = np.squeeze((1. - current_mean) * ns + 1.)

    # print(f"n={ns}, alpha={alpha}, beta={beta}")

    return ns, alpha, beta


def multi_step_uncert(gln, x_pos, x_batch, y_batch):
    fake_targets = [0., 1.]
    fake_means = np.empty((1, 2))
    current_mean = gln.predict(x_pos)

    batch_x_data = np.concatenate((x_batch, x_pos))
    current_params = to_immutable_dict(gln.gln_params)
    for j, fake_target in enumerate(fake_targets):
        # Update to fake target
        batch_y_data = np.concatenate((y_batch, [fake_target]))
        gln.predict(batch_x_data, batch_y_data)
        fake_means[:, j] = gln.predict(x_pos)
        # Clean up
        gln.gln_params = to_immutable_dict(current_params)

    return pseudocount(current_mean, fake_means)


def update_then_batch(gln, x_pos, x_batch, y_batch):
    fake_targets = [0., 1.]
    fake_means = np.empty((1, 2))
    current_mean = gln.predict(x_pos)

    current_params = to_immutable_dict(gln.gln_params)
    for j, fake_target in enumerate(fake_targets):
        # Update to fake target
        gln.predict(x_pos, [fake_target])
        gln.predict(x_batch, y_batch)
        fake_means[:, j] = gln.predict(x_pos)
        # Clean up
        gln.gln_params = to_immutable_dict(current_params)

    return pseudocount(current_mean, fake_means)


def update_then_batch_about_est(gln, x_pos, x_batch, y_batch):
    current_est = gln.predict(x_pos)
    fake_targets = [float(current_est[0]), 0., 1.]
    fake_means = np.empty((1, 3))

    current_params = to_immutable_dict(gln.gln_params)
    for j, fake_target in enumerate(fake_targets):
        # Update to fake target
        store_lr = gln.lr
        gln.update_learning_rate(gln.lr / 8000)
        gln.predict(x_pos, [fake_target])
        gln.update_learning_rate(store_lr)

        gln.predict(x_batch, y_batch)
        fake_means[:, j] = gln.predict(x_pos)
        # Clean up
        gln.gln_params = to_immutable_dict(current_params)

    current_mean = fake_means[:, 0]
    fake_means = fake_means[:, 1:]

    return pseudocount(current_mean, fake_means)


def pseudocount(current_mean, fake_means):
    diff = (current_mean[:, None] - fake_means) * np.array([1., -1.])
    diff = np.where(diff == 0., 1e-8, diff)
    n_ais0 = fake_means[:, 0] / diff[:, 0]
    n_ais1 = (1. - fake_means[:, 1]) / diff[:, 1]

    n_ais = np.dstack((n_ais0, n_ais1))
    ns = np.squeeze(np.min(n_ais, axis=-1))
    alpha = np.squeeze(current_mean * ns + 1.)
    beta = np.squeeze((1. - current_mean) * ns + 1.)

    return ns, alpha, beta


def run_multi(
        multi_n, gln_size, batch_size, lr, n_tuple, uncert_func, sample_strat):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    plt.rcParams.update({'font.size': 14})
    ttl = uncert_func.__name__ if uncert_func is not None else "None"
    ax.set_title(
        f"Pseudocount growth for {ttl}_{sample_strat}")
    ax.set_xlabel("Number in k means")
    ax.set_ylabel("Pseudocounts")
    all_ns, all_ks = [], []
    times = []
    for counter in range(multi_n):
        print(f"\nRUNNING {counter}")
        now = time.time()
        gln = make_gln(gln_size, batch_size, lr)
        ns, ks = base_runner(
            gln, n_tuple, batch_size,
            uncert=uncert_func,
            sample_strat=sample_strat)
        times.append(time.time() - now)
        all_ns.append(ns)
        all_ks.append(ks)
        ax.plot(ks, ns, alpha=0.2, color="blue")

    mean_ks = np.mean(all_ks, axis=0)
    mean_ns = np.mean(all_ns, axis=0)
    ax.plot(mean_ks, mean_ns, alpha=1., color="blue")
    print("AVERAGE TIME", np.mean(times))
    return fig


if __name__ == "__main__":
    # layer sizes, n hyperplanes
    GLN_SIZE = ([64, 64, 32, 1], 4)  # Fairly optimal
    NS = (8000, 8000, 1000)
    BATCH_SIZE = 8000  # 32  # Fairly optimal
    LR = 5e-2 * np.sqrt(8000 / 32)  # Fairly optimal

    func = update_then_batch_about_est
    sample_strat = "nearest"  # val, whole_hist, nearest

    fig = run_multi(
        2, GLN_SIZE, BATCH_SIZE, LR, NS, func, sample_strat=sample_strat)
    ttl = func.__name__ if func is not None else "None"
    basename = f"sandbox/uncert_results/{ttl}_{sample_strat}"
    file_name = f"{basename}.png"
    roll_int = 0
    while os.path.exists(file_name):
        roll_int += 1
        file_name = f"{basename}.{roll_int}.png"
    fig.savefig(file_name)
    plt.show()
