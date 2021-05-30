import os
import matplotlib.pyplot as plt

from main import run_main

from experiments.exp_utils import (
    save_dict_to_pickle, experiment_main, parse_result, parse_experiment_args)
from experiments.core_experiment import EXPERIMENT_PATH
from experiments.core_experiment.plotter import plot_horizon_experiment

# Fixed for all experiments
base_exp = {
    "agent": "q_table",
    "mentor": "random_safe",
    "report_every_n": 100,
    "steps": 40000,
    "init_zero": True,
    "state_len": 9,
    "sampling_strat": "random",
    "batch_size": 10,  # Train on every data point twice, on average
    "update_freq": 10,  # Only update table every 10 steps
    "learning_rate": 0.1,
}

# 1: Stochastic reward, 2: fully deterministic,
# 3: stochastic reward and transition
episodic = ["every_n"]  # ["never", "every_n"]
trans = ["3"]

all_configs = []
for t in trans:
    for e in episodic:
        config_params = {"trans": t, "reset": e}
        all_configs.append({**config_params, **base_exp})


def run_horizon_experiment(
        results_file, agent, init_zero=False, repeat_n=0, **kwargs):
    repeat_str = f"_repeat_{repeat_n}"

    args = parse_experiment_args(kwargs)
    args += ["--agent", agent]  # not added by parser (for other exps)
    report_every_n = int(args[args.index("--report-every-n") + 1])

    # pessimistic only
    # quantiles = [i for i, q in enumerate(QUANTILES) if q <= 0.5]
    for horizon in [2, 4, 6, 10]:
        horizon_args = args + [
            "--horizon", "finite", "--n-horizons", str(horizon), "--unscale-q"]
        horizon_args += ["--init", "zero" if init_zero else "quantile"]
        trained_agent = run_main(horizon_args, seed=repeat_n)

        exp_name = f"horizon_{horizon}" + repeat_str
        print("\nRUNNING", exp_name)
        result_i = parse_result(
            quantile_val=-1, key=exp_name, agent=trained_agent,
            steps_per_report=report_every_n, arg_list=horizon_args)
        save_dict_to_pickle(results_file, result_i)
        del trained_agent

    # And run for the infinite agent
    # unscale q for fair comparison
    control_args = args + ["--horizon", "inf", "--unscale-q"]
    control_info = run_main(control_args, seed=repeat_n)
    control_exp_name = "inf" + repeat_str
    print("\nRUNNING", control_exp_name)
    mentor_result = parse_result(
        quantile_val="mentor", key=control_exp_name, agent=control_info,
        steps_per_report=report_every_n, arg_list=control_args)
    save_dict_to_pickle(results_file, mentor_result)
    del control_info


if __name__ == "__main__":
    results_dir = os.path.join(EXPERIMENT_PATH, "results_finite_horizon")
    os.makedirs(results_dir, exist_ok=True)

    N_REPEATS = 7

    for cfg in all_configs:
        experiment_main(
            results_dir=results_dir,
            n_repeats=N_REPEATS,
            experiment_func=run_horizon_experiment,
            exp_config=cfg,
            plotting_func=plot_horizon_experiment,
            show=False,
        )

    plt.show()
