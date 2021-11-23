WIDTH = 7

# Fixed for all experiments
base_exp = {
    "agent": "continuous_pess_gln",
    "mentor": "cartpole_sweep",
    "report_every_n": 32,
    "steps": 256,  # 150000,
    "sampling_strat": "random",
    "batch_size": 64,
    "update_freq": 32,
    "init_zero": False,  # Not implemented for glns
}

lrs = [0.09]
# TODO add learning rate step here and to parsing

all_configs = []
for lr in lrs:
        config_params = {
            "learning_rate": lr,
        }
        all_configs.append({**config_params, **base_exp})
