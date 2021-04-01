# pessimistic-agents
A repository for code for empirical investigations of pessimistic agents

# Setup

## Supported conda env

With anaconda

```bash
conda create -f conda_env_cpu.yml
```

# Experiments

## `pessimistic_prior`

Apply pessimism approximation to current RL agents.

## `dist_q_learning`

Learn from distributions of the Q-value estimate (using a pessimistic quantile)

See `dist_q_learning/README.md`

