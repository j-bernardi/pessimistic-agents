# These don't change
base_exp = {
    "agent": "pess",
    "n": 100,
    "steps_per_ep": 200,
    "earlystop": 0,  # hard to know the right place to stop - just do it
    "init_zero": True,  # This helps remove failures
}

# 8 Experiments
strategy = [("whole", 1000), ("random", 1)]
trans = ["1", "2"]
horizons = ["inf", "finite"]

all_configs = []
for strat, freq in strategy:
    for t in trans:
        new_config = {
            "trans": t,
            "update_freq": freq,
            "sampling_strat": strat,
        }
        if strat == "random":
            new_config["batch_size"] = 10  # fix BS to 10 from history
        all_configs.append({**base_exp, **new_config})
