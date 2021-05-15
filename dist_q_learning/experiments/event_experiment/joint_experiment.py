"""Runs the event experiment and core experiment together"""
import os

from experiments.utils import experiment_main
from experiments import EXPERIMENTS_DIR

from experiments.event_experiment.exp_main import run_event_avoid_experiment
from experiments.event_experiment.plotter import compare_transitions

from experiments.core_experiment.finite_agent_0 import run_core_experiment
from experiments.core_experiment.plotter import plot_experiment_separate

# CHANGE config
from experiments.event_experiment.configs.every_state import all_configs

if __name__ == "__main__":
    RESULTS_DIR = os.path.join(EXPERIMENTS_DIR, "all_results")
    os.makedirs(RESULTS_DIR, exist_ok=True)
    N_REPEATS = 17

    for cfg in all_configs:
        experiment_main(
            results_dir=RESULTS_DIR,
            n_repeats=N_REPEATS,
            experiment_func=run_event_avoid_experiment,
            exp_config=cfg,
            plotting_func=compare_transitions,
            show=False,
            plot_save_ext="_events"  # essential - so as not to overwrite image
        )

        experiment_main(
            results_dir=RESULTS_DIR,
            n_repeats=N_REPEATS,
            experiment_func=run_core_experiment,
            exp_config=cfg,
            plotting_func=plot_experiment_separate,
            show=False,
            overwrite=False,  # Uses results from above without re-running
        )