import os
import pickle


def experiment_main(
        results_dir, n_repeats, experiment_func, exp_config, plotting_func,
        show=True
):
    """Handles result file creation and repeat runs of an experiment

    """
    os.makedirs(results_dir, exist_ok=True)

    f_name_no_ext = os.path.join(
        results_dir, "_".join([f"{k}_{str(v)}" for k, v in exp_config.items()]))
    dict_loc = f_name_no_ext + ".p"

    if os.path.exists(dict_loc):
        run = input(f"Found {dict_loc}\nOverwrite? y / n / a\n")
    else:
        print("No file", dict_loc, "\nrunning")
        run = "y"

    if run in ("y", "a"):
        if run == "y" and os.path.exists(dict_loc):
            os.remove(dict_loc)

        for i in range(n_repeats):
            print("\n\nREPEAT", i, "/", n_repeats)
            experiment_func(dict_loc, repeat_n=i, **exp_config)

    with open(dict_loc, "rb") as f:
        results_dict = pickle.load(f)

    if plotting_func is not None:
        plotting_func(results_dict, f_name_no_ext + ".png", show=show)


def save_dict_to_pickle(filename, new_result):
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
    for k in new_result.keys():
        if k in results.keys():
            yes = input(f"Key {k} already in dict - replace?\ny/ n")
            if yes == "y":
                del results[k]
            else:
                raise KeyError(f"{k} already in results {results.keys()}")
    new_results = {**results, **new_result}

    # Save
    with open(filename, 'wb') as fl:
        pickle.dump(new_results, fl, protocol=pickle.HIGHEST_PROTOCOL)
