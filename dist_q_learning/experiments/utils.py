import copy
import os
import pickle


def experiment_main(
        results_dir, n_repeats, experiment_func, exp_config, plotting_func,
        show=True, plot_save_ext="",
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
        img_loc = f_name_no_ext + plot_save_ext + ".png"
        plotting_func(results_dict, img_loc, show=show)


def parse_experiment_args(kwargs):
    """
    results_file:
    agent:
    trans:
    n:
    steps_per_ep:
    earlystop:
    init_zero:
    repeat_n: which number repeat this is
    render:
    update_freq:
    sampling_strat:
    env_adjust_kwargs: a list of kwarg-dicts, one per repeat
        (indexed at repeat_n)
    action_noise:
    horizon:
    batch_size:
    state_len:
    """
    args = []
    exp_kwargs = copy.deepcopy(kwargs)

    def parse(arg_list, arg_flag, key, required=True, default=None):
        """Add to arg list if the exp kwarg is not None"""
        v = None
        if key in exp_kwargs:
            v = str(exp_kwargs.pop(key))
            if v is not None:
                arg_list += [arg_flag, v]
        elif default is not None:
            arg_list += [arg_flag, default]
        elif required and default is None:
            raise ValueError(f"Missing required kwarg {key}")
        return v

    parse(args, "--mentor", "mentor")  # always required

    parse(args, "--trans", "trans")
    parse(args, "--wrapper", "wrapper", required=False)

    parse(args, "--num-episodes", "n")  # TODO remove
    parse(args, "--steps-per-ep", "steps_per_ep")
    parse(args, "--earlystop", "earlystop", required=False)

    parse(args, "--update-freq", "update_freq", default="1")
    parse(args, "--sampling-strategy", "sampling_strat", default="last_n_steps")
    horizon = parse(args, "--horizon", "horizon", default="inf")
    parse(args, "--batch-size", "batch_size", required=False)
    parse(args, "--state-len", "state_len", default=7)
    parse(args, "--render", "render", default="-1")

    if horizon == "finite":
        args += ["--unscale-q"]

    assert not exp_kwargs, f"Unexpected keys remain {exp_kwargs.keys()}"

    return args


def parse_result(quantile_val, key, agent, arg_list):
    """Take the info from an exp and return a single-item dict"""
    result = {
        key: {
            "quantile_val": quantile_val,
            "queries": agent.mentor_queries_per_ep,
            "rewards": agent.rewards_per_ep,
            "failures": agent.failures_per_ep,
            "transitions": agent.transitions,  # ADDED
            "metadata": {
                "args": arg_list,
                # '"steps_per_ep": steps,
                "min_nonzero": agent.env.min_nonzero_reward,
                "max_r": agent.env.max_r,
            }
        }
    }
    return result


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
