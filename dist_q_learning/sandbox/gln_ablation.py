import itertools
import numpy as np

import glns

X_DIM = 4
A, B = 0.5, 1.


def radial_sine(x):
    """y = A * cos(sqrt(x0^2 + x1^4)) + B * sin(x2 + x3^2)

    Requires final size of x to be 4
    """
    assert x.shape[-1] == 4, x.shape
    raw_radial_sin = (
        A * np.cos(np.sqrt(x[..., 0] ** 2 + x[..., 1] ** 4))
        + B * np.sin(x[..., 2] + x[..., 3] ** 2))
    positive_radial_sin = raw_radial_sin + (A + B)
    normed_radial_sin = positive_radial_sin / (2 * (A + B))
    assert np.all(normed_radial_sin >= 0.) and np.all(normed_radial_sin <= 1.)
    return normed_radial_sin


def make_data(n):
    """Make X Data of shape (N_DATA, X_DIM)

    x_data in range [0, 1]
    y_data in range [0, 1]
    """
    x_all = np.random.rand(n, X_DIM)
    y_all = radial_sine(x_all)

    return x_all, y_all


def abs_error(y_lab, y_pred):
    return np.mean(np.abs(y_lab - y_pred))


def make_gln(size, bs, lr):
    layers, context = size
    return glns.GGLN(
        layer_sizes=layers,
        input_size=X_DIM,
        context_dim=context,
        lr=lr,
        min_sigma_sq=0.1,
        bias_len=3,
        batch_size=bs,
        init_bias_weights=[None, None, None])


def learn_sin(
        gln_size, n_train, n_val, n_test, batch_size, lr, silent=False):
    x_train, y_train = make_data(n=n_train)
    x_val, y_val = make_data(n=n_val)
    gln = make_gln(gln_size, batch_size, lr)

    # TRAIN
    n_steps = n_train // batch_size
    for i in range(0, n_train, batch_size):
        x_batch = x_train[i:i+batch_size]
        y_batch = y_train[i:i+batch_size]
        gln.predict(x_batch, target=y_batch)
        if (i % (n_steps // 10)) == 0:
            val_losses = []
            for j in range(0, n_val, batch_size):
                val_preds = gln.predict(x_val[j:j+batch_size])
                val_losses.append(abs_error(y_val[j:j+batch_size], val_preds))
            if not silent:
                print(f"BATCH {i // batch_size} / {n_train // batch_size} - "
                      f"val loss", np.mean(val_losses))

    # TEST
    x_test, y_test = make_data(n=n_test)
    final_losses = []
    for i in range(0, n_test, batch_size):
        y_test_preds = gln.predict(x_test[i:i+batch_size])
        final_losses.append(abs_error(y_test[i:i+batch_size], y_test_preds))
    final_loss = np.mean(final_losses)
    print("TEST LOSS", final_loss)


if __name__ == "__main__":
    # layer sizes, n hyperplanes
    GLN_SIZES = [([64, 64, 32, 1], 4)]  # Fairly optimal
    N_TRAIN_DATAS = [8000]
    N_VAL_DATAS = [1000]
    N_TEST_DATAS = [1000]
    BATCH_SIZES = [32]  # Fairly optimal
    LR = [5e-2]  # Fairly optimal

    iterate_through = [
        GLN_SIZES, N_TRAIN_DATAS, N_VAL_DATAS, N_TEST_DATAS, BATCH_SIZES, LR,
    ]
    for combination in itertools.product(*iterate_through):
        print("Running", combination)
        learn_sin(*combination, silent=False)
