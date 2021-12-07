import random
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

gammas = [0.9, 0.95, 0.99]
lrs = [0.1, 0.08, 0.05]
lr_steps = [(50, 0.97), (80, 0.98), (100, 0.99), (120, 0.98)]
quantiles = ["0", "1", "2", "3", "4", "5"]
# "ire_scale", "ire_alpha", "q_scale", "q_alpha"
scaling = [
    (2., 2., 8., 2.),  # original
    (1., 1., 1., 1.), (1., 1.3, 1., 1.3), (2., 1., 4., 1.),  # raw ish
    (2., 1., 8., 1.), (1., 2., 1., 2.),  # multiply only or square only
    (2., 1.3, 8., 1.3), (2., 1.8, 8., 1.8),  # some other alpha
]

mentor_run = {
    "quantile": "mentor",
    "learning_rate": 1.,
}

all_configs = [{**mentor_run, **base_exp}]
for lr in lrs:
    for q in quantiles:
        for lr_step in lr_steps:
            for gamma in gammas:
                for scale in scaling:
                    config_params = {
                        "quantile": q,
                        "learning_rate": lr,
                        "learning_rate_step": lr_step,
                        "scaling": scaling,
                        "gamma": gamma,
                        "min_sigma": min_sigs
                    }
                    all_configs.append({**config_params, **base_exp})
random.shuffle(all_configs)
