from transition_defs import generate_every_state_config_dict

N_REPEATS = 7
WIDTH = 5

base_exp = {  # Fixed for all experiments
    "agent": "pess",
    "n": 1,  # 25
    "steps_per_ep": 2,  # 200
    "init_zero": True,  # This helps remove failures
    "state_len": WIDTH,
}

# Each repeat experiment across agents should use the same environment.
env_config_dicts = [
    generate_every_state_config_dict(WIDTH) for _ in range(N_REPEATS)]

# 8 Experiments
strategy = [("random", 10)]
trans = ["3"]
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
