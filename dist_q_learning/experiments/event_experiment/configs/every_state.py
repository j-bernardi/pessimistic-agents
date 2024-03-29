WIDTH = 7

# Fixed for all experiments
base_exp = {
    "agent": "pess",
    "mentor": "avoid_state_act",  # the key for the experiment
    "wrapper": "every_state",  # the key for the experiment
    "report_every_n": 100,
    "steps": 100000,
    "init_zero": True,  # This helps remove failures
    "state_len": WIDTH,
    "sampling_strat": "random",
    "batch_size": 20,  # Train on every data point twice, on average
    "update_freq": 10,  # Only update table every 10 steps
    "learning_rate": 0.5,  # Constant across experiments
}

trans = ["1", "2", "3"]  # stochastic and deterministic underlying reward
horizons = ["inf"]  # "finite"

all_configs = []
for t in trans:
    for h in horizons:
        config_params = {
            "trans": t,
            "horizon": h,
        }
        all_configs.append({**config_params, **base_exp})
