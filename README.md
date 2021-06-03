# Pessimistic Agents

[Pessimistic Agents](https://arxiv.org/abs/2006.08753)
are ask-for-help reinforcement learning agents that offer guarantees of:

1. Eventually outperforming the mentor
2. Eventually stopping querying the mentor
3. Never causing unprecedented events to happen, with arbitrary probability

In this repository, we investigate their behaviour in the faithful setting, and explore approximations that allow them
to be used in real-world RL problems.

Overview - see individual README.md files for more detail.

---

## Distributional Q Learning - dist_q_learning/

We introduce a tractable implementation of Pessimistic Agents. Approximate the Bayesian world and mentor models
as a distribution over epistemic uncertainty of Q values. By using a pessimistic (low) quantily, we demonstrate the
expected behaviour and safety results for a pessimistic agent.

| Work | Status  | 
| ------------- | ------------- |
| Finite state Q Table proof of concept | ![DONE](https://via.placeholder.com/100x40/008000/FFFFFF?text=DONE)     | 
| Continuous deep Q learning implementation | ![WIP](https://via.placeholder.com/100x40/FF7B00/FFFFFFF?text=WIP) |

---
## Faithful implementation - cliffworld/

Implement and investigate a faithful representation of a Bayesian Pessimistic Agent.

| Work | Status  | 
| ------------- | ------------- |
| Environment | ![DONE](https://via.placeholder.com/100x40/008000/FFFFFF?text=DONE)     | 
| Agent | ![HOLD](https://via.placeholder.com/100x40/A83500/FFFFFFF?text=On+Hold) |

On hold, some progress made in implementing the environment and mentor models.

---

## Pessimistic RL - pessimistic_prior/

Apply pessimism approximation to neural network based, deep Q learning RL agents.

| Work | Status  | 
| ------------- | ------------- |
| DQN proof of concept | ![HOLD](https://via.placeholder.com/100x40/A83500/FFFFFFF?text=On+Hold) |

-----
# Setup

## Supported conda env

With anaconda

```bash
conda env create -f torch_env_cpu.yml
```