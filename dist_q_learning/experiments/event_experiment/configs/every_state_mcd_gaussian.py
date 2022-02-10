
WIDTH = 7

# Fixed for all experiments
base_exp = {
    "agent": "continuous_pess_mcd",
    "mentor": "cartpole_sweep",
    "report_every_n": 32,
    "steps": 1500,
    "burnin_n": 5000,
    "sampling_strat": "random",
    "batch_size": 512,
    "update_freq": 32,
    "init_zero": False,  # Not implemented for glns
}

gammas = [0.99]
lrs = [0.01, 0.0001]
lr_steps = [(50, 0.97), (80, 0.98), (100, 0.99), (120, 0.98)]
# lr_steps = [(50, 0.97), (100, 0.99), (100, 1.)]
lr_steps = [(100, 0.99)]


# quantiles = ["0", "1", "2", "3", "4", "5"]
quantiles = ["3"]

dropout_rates = [0.5, 0.9]
hidden_sizes_options = [[100], [128,128], [32,64,256]]
use_gaussians = [1]
weight_decays = [1e-6]
baserate_breadths = [10, 1, 0.1]
n_samples = [10]


mentor_run = {
    "quantile": "mentor",
    "learning_rate": 0.01,
}

all_configs = [{**mentor_run, **base_exp}]
all_configs = [{**base_exp}]

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


