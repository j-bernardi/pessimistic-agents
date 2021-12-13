import copy
import os
import pickle
from pathlib import Path

from utils import upload_blob


def experiment_main(
        results_dir, n_repeats, experiment_func, exp_config, plotting_func,
        show=True, plot_save_ext="", overwrite=None, save=True
):
    """Handles result file creation and repeat runs of an experiment

    Args:
        results_dir (str): a path to the dir to save the result dict
            and image to
        n_repeats (int): number of times to repeat the experiment
            (called below)
        experiment_func (callable): a function taking args
            (dict_loc, repeat_n=i, **exp_config)
            which runs a single repeat of the exeperiment being run.
        exp_config (dict): contains all the arguments to pass to
            experiment_func
        plotting_func (callable): A function taking args
            (results_dict, img_loc, show=show),
            that typically plots the results from the dict.
        show (bool): if true, show every graph after production
        plot_save_ext (str): an extension to give to a plot name. Useful
            when running joint experiments and there'd otherwise be two
            files with the same name.
        overwrite (Optional[bool]): if True, overwrite any result dicts
            found. If false, just read the dict and don't run the
            experiment. If None, ask the user.
        save (bool): passes this to the plotting function
    """
    os.makedirs(results_dir, exist_ok=True)

    def clean_v(val):
        return str(val).replace(
            "(", "").replace(")", "").replace(",", "").replace(" ", "")

    f_name_no_ext = os.path.join(
        results_dir,
        "_".join([f"{k}_{clean_v(v)}" for k, v in exp_config.items()]))
    if len(f_name_no_ext) > 255:
        f_name_no_ext = os.path.join(
            results_dir,
            "_".join([f"{k[0]}_{clean_v(v)}" for k, v in exp_config.items()]))
    f_name_no_ext = f_name_no_ext.replace(" ", "_")
    dict_loc = f_name_no_ext + ".p"
    # Check filename valid
    Path(dict_loc).touch()
    print(f"Future dict loc: {dict_loc}")

    if overwrite and os.path.exists(dict_loc):
        os.remove(dict_loc)
        run = "y"
    elif os.path.exists(dict_loc) and overwrite is False:
        print("Reading existing results", dict_loc)
        run = "n"
    elif os.path.exists(dict_loc):
        run = input(f"Found {dict_loc}\nOverwrite? y / n / a\n")
        if run == "y":
            os.remove(dict_loc)
    else:
        print("No file", dict_loc, "\nrunning fresh")
        run = "y"

    if os.path.exists(dict_loc):
        with open(dict_loc, "rb") as f:
            results_dict = pickle.load(f)
    else:
        results_dict = {"config": exp_config}

    if run in ("y", "a"):
        # Loop over repeat experiments
        for i in range(n_repeats):
            print("\n\nREPEAT", i, "/", n_repeats)
            # Skip until find the last exp that has not been run
            # TODO - this skips whole rounds, not individual experiments
            if any(k.endswith(f"_repeat_{i}") for k in results_dict.keys()):
                print(f"Found repeat {i} in dict {dict_loc}")
                continue
            experiment_func(dict_loc, repeat_n=i, **exp_config)

    with open(dict_loc, "rb") as f:
        results_dict = pickle.load(f)

    if plotting_func is not None:
        if not save:
            img_loc = None
        elif plot_save_ext is not None:
            img_loc = f_name_no_ext + plot_save_ext + ".png"
        else:
            img_loc = f_name_no_ext + ".png"

        plotting_func(results_dict, save_to=img_loc, show=show)

        try:
            upload_blob(img_loc, os.path.basename(img_loc), overwrite=False)
        except Exception as e:
            print(f"Upload of image {img_loc} failed\n{e}")


def parse_experiment_args(kwargs, gln=False):
    """Parse a dict of kwargs into arguments compatible with main.py

    results_file:
    agent:
    trans:
    report_every_n:
    steps:
    earlystop:
    init_zero:
    repeat_n: which number repeat this is
    render:
    update_freq:
    sampling_strat:
    lr:
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
            v = exp_kwargs.pop(key)
            if v is not None:
                arg_list += (
                    [arg_flag]
                    + str(v).replace(
                        "(", "").replace(")", "").replace(",", "").split(" "))
        elif default is not None:
            arg_list += [arg_flag, default]
        elif required and default is None:
            raise ValueError(f"Missing required kwarg {key}")
        return v

    parse(args, "--mentor", "mentor")  # always required

    if not gln:
        args += ["--env", "grid"]
        parse(args, "--trans", "trans")
        parse(args, "--wrapper", "wrapper", required=False)
        parse(args, "--state-len", "state_len", default=7)
    else:
        args += ["--env", "cart"]
        args += ["--cart-task", "move_out"]
        args += ["--disable-gui", "--norm-min-val", "-1"]
        args += ["--knock-cart"]  # always run knocking experiment
        parse(args, "--burnin-n", "burnin_n")
        parse(args, "--scaling", "scaling", required=False)
        parse(args, "--min-sigma", "min_sigma", required=False)

    parse(args, "--quantile", "quantile")
    parse(args, "--gamma", "gamma", required=False)
    parse(args, "--report-every-n", "report_every_n")
    parse(args, "--n-steps", "steps")
    parse(args, "--earlystop", "earlystop", required=False)

    parse(args, "--update-freq", "update_freq", default="1")
    parse(args, "--sampling-strategy", "sampling_strat", default="last_n_steps")
    parse(args, "--learning-rate", "learning_rate")
    parse(args, "--learning-rate-step", "learning_rate_step", required=False)
    horizon = parse(args, "--horizon", "horizon", default="inf")
    parse(args, "--batch-size", "batch_size", required=False)
    parse(args, "--render", "render", default="-1")

    if horizon == "finite":
        args += ["--unscale-q"]

    assert not exp_kwargs, f"Unexpected keys remain {exp_kwargs.keys()}"

    return args


def parse_result(
        quantile_val, key, agent, steps_per_report, arg_list, gln=False):
    """Take the info from an exp and return a single-item dict"""
    result = {
        key: {
            "quantile_val": quantile_val,
            "queries": agent.mentor_queries_periodic,
            "rewards": agent.rewards_periodic,
            "failures": agent.failures_periodic,
            "metadata": {
                "args": arg_list,
                "steps_per_report": steps_per_report,
                "min_nonzero": agent.env.min_nonzero_reward,
                "max_r": agent.env.max_r,
            }
        }
    }
    if not gln:
        result[key]["transitions"] = agent.transitions
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
    print("Saving to", filename)
    with open(filename, 'wb') as fl:
        pickle.dump(new_results, fl, protocol=pickle.HIGHEST_PROTOCOL)

    try:
        upload_blob(filename, os.path.basename(filename), overwrite=True)
    except Exception as e:
        print(f"Dict upload of {filename} failed\n{e}")
