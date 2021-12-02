WIDTH = 7

# Fixed for all experiments
base_exp = {
    "agent": "continuous_pess_gln",
    "mentor": "cartpole_sweep",
    "report_every_n": 32,
    "steps": 150000,
    "burnin_n": 5000,
    "sampling_strat": "random",
    "batch_size": 64,
    "update_freq": 32,
    "init_zero": False,  # Not implemented for glns
}

gamma = [0.9, 0.95, 0.99]
lrs = [0.1, 0.08, 0.05]
lr_steps = [(50, 0.97), (80, 0.98), (100, 0.99), (120, 0.98)]
quantiles = ["0", "1", "2", "3", "4", "5"]

mentor_run = {
    "quantile": "mentor",
    "learning_rate": 1.,
}

all_configs = [{**mentor_run, **base_exp}]
for lr in lrs:
    for q in quantiles:
        for lr_step in lr_steps:
            config_params = {
                "quantile": q,
                "learning_rate": lr,
                "learning_rate_step": lr_step,
            }
            all_configs.append({**config_params, **base_exp})
