import random
import numpy as np

N_REPEATS = 7
WIDTH = 5

base_exp = {  # Fixed for all experiments
    "agent": "pess",
    "n": 1,  # TODO - 25 or something
    "steps_per_ep": 2,  # TODO 200
    "init_zero": True,  # This helps remove failures
    "state_len": WIDTH,
}


def generate_env_config_dict(width):
    """Generate a randomly-disastrous action for every state

    Intended that the mentor avoids this action, and the pessimistic
    agent more often avoids it than a q_table.
    """
    env_config_dict = {
        k: [] for k in (
            "avoid_act_probs", "states_from", "actions_from", "states_to",
            "probs_env_event")
    }
    for s in range(width ** 2):
        all_actions = [(-1, 0), (+1, 0), (0, -1), (0, +1)]
        y_coord = int(s % width)
        x_coord = int((s - y_coord) // width)
        if x_coord in (0, width - 1) or y_coord in (0, width - 1):
            continue  # don't change probability of disaster (edge states)
        state_tuple = (x_coord, y_coord)
        safe_actions = []
        for act in all_actions:
            new_pos = np.array(state_tuple) + np.array(act)
            if np.all(1 <= new_pos) and np.all(new_pos <= width - 2):
                safe_actions.append(act)
        # Mentor only
        env_config_dict["avoid_act_probs"].append(0.01),
        # Mentor and env
        env_config_dict["states_from"].append(state_tuple),
        env_config_dict["actions_from"].append(random.choice(safe_actions)),
        # Env variables only
        env_config_dict["states_to"].append((0, 0)),
        env_config_dict["probs_env_event"].append(0.01),
        # "event_rewards": [Optional, default to 0],
    return env_config_dict


# Each repeat experiment across agents should use the same environment.
env_config_dicts = [generate_env_config_dict(WIDTH) for _ in range(N_REPEATS)]

# 8 Experiments
strategy = [("random", 10)]
trans = ["2"]  # stochastic reward env TODO - try stochastic transitions too?
horizons = ["finite"]  # ["inf", "finite"] NOTE - already ran inf

all_configs = []
for strat, freq in strategy:
    for t in trans:
        for h in horizons:
            config_params = {
                "trans": t,
                "update_freq": freq,  # batch size defaults to this too
                "sampling_strat": strat,
                "horizon": h,
            }
            all_configs.append({**base_exp, **config_params})
