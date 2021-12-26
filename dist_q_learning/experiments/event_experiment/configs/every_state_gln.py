import random

from experiments.to_run import set_args_of_interest
from experiments.exp_utils import args_to_name

# Fixed for all experiments
base_exp = {
    "agent": "continuous_pess_gln",
    "mentor": "cartpole_sweep",
    "report_every_n": 32,
    "steps": 300000,
    "burnin_n": 5000,
    "sampling_strat": "random",
    "batch_size": 64,
    "update_freq": 32,
    "init_zero": False,  # Not implemented for glns
}

min_sigs = [0.0001, 0.001, 0.01, 0.1]
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
                    for min_sig in min_sigs:
                        config_params = {
                            "quantile": q,
                            "learning_rate": lr,
                            "learning_rate_step": lr_step,
                            "scaling": scale,
                            "gamma": gamma,
                            "min_sigma": min_sig,
                        }
                        all_configs.append({**config_params, **base_exp})
random.Random(0).shuffle(all_configs)

for i, c in enumerate(all_configs):
    names = tuple(args_to_name(c, short=short) for short in (True, False))
    if any(n in set_args_of_interest for n in names):
        print(f"\nConfig of interest {names[0]}\ncounter={i}")
