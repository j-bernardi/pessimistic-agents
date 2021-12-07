"""Run from dist_q_learning"""
import os
import jax
import argparse
from multiprocessing import Pool

from main import run_main
from agents import QUANTILES

from experiments.exp_utils import (
    save_dict_to_pickle, experiment_main, parse_result, parse_experiment_args)
from experiments.core_experiment import EXPERIMENT_PATH
from experiments.core_experiment.plotter import plot_experiment_separate

from experiments.event_experiment.configs.every_state_gln import all_configs


GLN = True
N_REPEATS = 1


def run_core_experiment(
        results_file, agent, init_zero=False, repeat_n=0, device_id=0,
        **kwargs):
    repeat_str = f"_repeat_{repeat_n}"

    args = parse_experiment_args(kwargs, gln=GLN)
    report_every_n = int(args[args.index("--report-every-n") + 1])
    quantile_val_index = args.index("--quantile") + 1
    quant_i = args[quantile_val_index]
    if quant_i == "mentor":
        del args[quantile_val_index]
        args.remove("--quantile")  # not needed for mentor
        # And run for the mentor as a control
        full_args = args + ["--agent", "mentor" + ("_gln" if GLN else "")]
        exp_name = "mentor" + repeat_str
    else:
        quant_i = int(quant_i)
        # pessimistic only
        full_args = args + ["--agent", agent]
        full_args += ["--init", "zero" if init_zero else "quantile"]
        exp_name = f"quant_{quant_i}" + repeat_str

    print("\nRUNNING", exp_name)
    trained_agent = run_main(full_args, seed=repeat_n, device_id=device_id)
    result_dict = parse_result(
        quantile_val=QUANTILES[quant_i] if quant_i != "mentor" else "mentor",
        key=exp_name,
        agent=trained_agent,
        steps_per_report=report_every_n,
        arg_list=full_args,
        gln=GLN,
    )
    save_dict_to_pickle(results_file, result_dict)
    del trained_agent


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--device-id", "-i", required=False, type=int)
    parser.add_argument(
        "--multiprocess", "-m", required=False, action="store_true",
        help="only available for Jax")
    parser.add_argument(
        "--config-num", "-c", required=True, type=int, nargs="+")
    all_args = parser.parse_args()

    assert all_args.device_id or all_args.multiprocess
    assert not (all_args.device_id and all_args.multiprocess)
    return all_args


if __name__ == "__main__":
    args = parse_args()
    results_dir = os.path.join(EXPERIMENT_PATH, "results")
    os.makedirs(results_dir, exist_ok=True)
    # for Jax experiments
    if args.multiprocess:
        print(f"CONFIGs {args.config_num}")
        n_devices = len(jax.devices())
        assert n_devices >= len(args.config_num), (
            f"{n_devices} devices available")
        map_args = []
        for dev_id, config_id in enumerate(args.config_num):
            map_args.append((
                results_dir,
                N_REPEATS,
                run_core_experiment,
                all_configs[config_id],
                plot_experiment_separate,
                False,
                "",
                False,
                True,
                dev_id
            ))
        with Pool() as p:
            p.starmap(experiment_main, map_args)
    else:
        print(f"DEVICE {args.device_id}, CONFIG {args.config_num}")
        for c in args.config_num:
            experiment_main(
                results_dir=results_dir,
                n_repeats=N_REPEATS,
                experiment_func=run_core_experiment,
                exp_config=all_configs[c],
                plotting_func=plot_experiment_separate,
                show=False,
                device_id=args.device_id,
            )
