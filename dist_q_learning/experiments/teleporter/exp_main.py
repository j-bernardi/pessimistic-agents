"""Run from dist_q_learning"""
import os

from main import run_main
from agents import QUANTILES

from experiments.utils import save_dict_to_pickle, experiment_main
from experiments.teleporter import EXPERIMENT_PATH
from experiments.teleporter.plotter import compare_transitions


def run_teleport_experiment(
        results_file, agent, trans, n, steps_per_ep=500, earlystop=0,
        init_zero=False, repeat_n=0, render=-1, teleporter_kwargs=None
):
    repeat_str = f"_repeat_{repeat_n}"
    args = ["--mentor", "avoid_teleport"]
    args += [
        "--trans", trans,
        "--num-episodes", str(n),
        "--steps-per-ep", str(steps_per_ep),
        "--early-stopping", str(earlystop),
        "--render", str(render)
    ]

    quantiles = list(range(len(QUANTILES)))
    pess_agent_args = args + ["--agent", agent]

    # pessimistic only
    for quant_i in [q for q in quantiles if QUANTILES[q] <= 0.5]:
        q_i_pess_args = pess_agent_args + ["--quantile", str(quant_i)]
        q_i_pess_args += ["--init", "zero" if init_zero else "quantile"]
        trained_agent = run_main(q_i_pess_args)

        exp_name = f"quant_{quant_i}" + repeat_str
        print("\nRUNNING", exp_name)
        result_i = parse_result(
            quantile_val=QUANTILES[quant_i],
            key=exp_name,
            agent=trained_agent,
            steps=steps_per_ep,
            arg_list=q_i_pess_args
        )
        save_dict_to_pickle(results_file, result_i)
        del trained_agent

    # And run for the q_table agent
    q_table_args = args + ["--agent", "q_table"]
    q_table_agent = run_main(q_table_args)
    q_table_exp_name = "q_table" + repeat_str
    print("\nRUNNING", q_table_exp_name)
    q_table_result = parse_result(
        "q_table", q_table_exp_name, q_table_agent, steps_per_ep, q_table_args)
    save_dict_to_pickle(results_file, q_table_result)
    del q_table_agent

    # And run for the mentor as a control
    mentor_args = args + ["--agent", "mentor"]
    mentor_agent_info = run_main(mentor_args)
    mentor_exp_name = "mentor" + repeat_str
    print("\nRUNNING", mentor_exp_name, mentor_args)
    mentor_result = parse_result(
        "mentor", mentor_exp_name, mentor_agent_info, steps_per_ep, mentor_args)
    save_dict_to_pickle(results_file, mentor_result)


def parse_result(quantile_val, key, agent, steps, arg_list):
    """Take the info from an exp and return a single-item dict"""
    result = {
        key: {
            "quantile_val": quantile_val,
            "steps_per_ep": steps,
            "queries": agent.mentor_queries_per_ep,
            "rewards": agent.rewards_per_ep,
            "failures": agent.failures_per_ep,
            "transitions": agent.transitions,  # ADDED
            "metadata": {
                "args": arg_list,
                "steps_per_ep": steps,
                "min_nonzero": agent.env.min_nonzero_reward,
                "max_r": agent.env.max_r,
            }
        }
    }
    return result


if __name__ == "__main__":
    results_dir = os.path.join(EXPERIMENT_PATH, "results")

    N_REPEATS = 7
    NUM_EPS = 100
    STEPS_PER_EP = 200

    exp_config = {
        "agent": "pess",
        "trans": "2",  # non-stochastic, sloped reward
        "n": NUM_EPS,
        "steps_per_ep": STEPS_PER_EP,
        "earlystop": 0,  # hard to know the right place to stop - just do it
        "init_zero": True,  # This helps remove failures
        # TODO - consider action noise to ensure we explore those states
        # 0.01 0.10 0.9999 is an OK start point for 20 * 500 steps (adjust)
    }

    teleport_config = {
        # Mentor only
        "avoid_act_prob": 0.01,
        # Mentor and env
        "state_from": (5, 5),
        "action_from": (-1, 0),  # 0
        # Env variables only
        "state_to": (1, 1),
        "prob_env_teleport": 0.01,
    }

    def wrapped_teleport_experiment(fname, **exp_config_kwargs):
        """Make arg signature match the main experiment to share wrapper"""
        return run_teleport_experiment(
            fname,
            teleporter_kwargs=teleport_config,
            **exp_config_kwargs
        )

    experiment_main(
        results_dir=results_dir,
        n_repeats=N_REPEATS,
        experiment_func=wrapped_teleport_experiment,
        exp_config=exp_config,
        plotting_func=compare_transitions
    )
