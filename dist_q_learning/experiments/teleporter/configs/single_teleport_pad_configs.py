N_REPEATS = 7

# These don't change
base_exp = {
    "agent": "pess",
    "n": 100,
    "steps_per_ep": 200,
    "earlystop": 0,  # hard to know the right place to stop - just do it
    "init_zero": True,  # This helps remove failures
}

# For now, this is the config we're examining (single teleporter pad -
# check it's avoided)
env_config_dict = {
    # Mentor only
    "avoid_act_probs": [0.01],
    # Mentor and env
    "states_from": [(5, 5)],
    "actions_from": [(-1, 0)],  # 0
    # Env variables only
    "states_to": [(1, 1)],
    "probs_env_event": [0.01],
    # "event_rewards": [Optional],
}

# 8 Experiments
strategy = [("whole", 1000), ("random", 1)]
trans = ["1", "2"]
horizons = ["finite"]  # ["inf", "finite"] NOTE - already ran inf

all_configs = []
for strat, freq in strategy:
    for t in trans:
        for h in horizons:
            new_config = {
                "trans": t,
                "update_freq": freq,
                "sampling_strat": strat,
                "horizon": h,
            }
            if strat == "random":
                new_config["batch_size"] = 10  # fix BS to 10 from history
            all_configs.append({**base_exp, **new_config})
