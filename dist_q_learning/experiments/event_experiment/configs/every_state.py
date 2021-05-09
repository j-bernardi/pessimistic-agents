WIDTH = 5

# Fixed for all experiments
base_exp = {
    "agent": "pess",
    "n": 25,
    "steps_per_ep": 200,
    "init_zero": True,  # This helps remove failures
    "state_len": WIDTH,
    "wrapper": "every_state",  # the key for the experiment
}

# 8 Experiments
strategy = [("random", 10)]  # seems a good sample
trans = ["1", "2"]  # stochastic and deterministic underlying reward
horizons = ["finite", "inf"]

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
