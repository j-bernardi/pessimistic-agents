WIDTH = 7

# Fixed for all experiments
base_exp = {
    "agent": "continuous_pess_mcd",
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
dropout_rates = [0.1, 0.5, 0.9]
hidden_sizes_options = [[1024,16], [89,128,5]]
use_gaussians = [1]
weight_decays = [50e-6]
baserate_breadths = [0.08, 0.001]
n_samples = [10, 100]


mentor_run = {
    "quantile": "mentor",
    "learning_rate": 1.,
}

all_configs = [{**mentor_run, **base_exp}]
for lr in lrs:
    for q in quantiles:
        for lr_step in lr_steps:
            for gamma in gammas:
                for dropout_rate in dropout_rates:
                    for hidden_sizes in hidden_sizes_options:
                        for n_sample in n_samples:
                            for use_gaussian in use_gaussians:
                                if use_gaussian == 0:
                                    config_params = {
                                        "gamma": gamma,
                                        "quantile": q,
                                        "learning_rate": lr,
                                        "learning_rate_step": lr_step,
                                        "dropout_rate": dropout_rate,
                                        "hidden_sizes": hidden_sizes,
                                        "n_samples": n_sample,
                                        "use_gaussian": use_gaussian,
                                        "weight_decay": None,
                                        "baserate_breadth": None,            
                                    }
                                    all_configs.append({**config_params, **base_exp})

                                else:
                                    for weight_decay in weight_decays:
                                        for baserate_breadth in baserate_breadths:
                                            config_params = {
                                                "gamma": gamma,
                                                "quantile": q,
                                                "learning_rate": lr,
                                                "learning_rate_step": lr_step,
                                                "dropout_rate": dropout_rate,
                                                "hidden_sizes": hidden_sizes,
                                                "n_samples": n_sample,
                                                "use_gaussian": use_gaussian,
                                                "weight_decay": weight_decays,
                                                "baserate_breadth": baserate_breadth,

                                            }
                                            all_configs.append({**config_params, **base_exp})


