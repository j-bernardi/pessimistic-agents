
WIDTH = 7

# Fixed for all experiments
base_exp = {
    "agent": "continuous_pess_mcd",
    "mentor": "cartpole_sweep",
    "report_every_n": 32,
    "steps": 1500,
    "burnin_n": 5000,
    "sampling_strat": "random",
    "batch_size": 64,
    "update_freq": 32,
    "init_zero": False,  # Not implemented for glns
}

gammas = [0.9, 0.95, 0.99]
lrs = [0.1, 0.08, 0.05]
lr_steps = [(50, 0.97), (80, 0.98), (100, 0.99), (120, 0.98)]
# quantiles = ["0", "1", "2", "3", "4", "5"]
quantiles = ["0", "3",  "5"]

dropout_rates = [0.1, 0.5, 0.9]
hidden_sizes_options = [[100],[8,8,8,8],[64,64]]
use_gaussians = [0]
weight_decays = [1e-6, 1e-5]
baserate_breadths = [0.08, 0.001]
n_samples = [100,10]


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
                                                "weight_decay": weight_decay,
                                                "baserate_breadth": baserate_breadth,

                                            }
                                            all_configs.append({**config_params, **base_exp})


