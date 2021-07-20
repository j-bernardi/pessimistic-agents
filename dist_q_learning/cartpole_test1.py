import gym
import numpy.random as npr
import numpy as np

from env import CartpoleEnv

# cartpole policy from here: https://towardsdatascience.com/how-to-beat-the-cartpole-game-in-5-lines-5ab4e738c93f
def theta_omega_policy1(obs):
    theta, w = obs[2:4]
    if abs(theta + 0e-3*obs[0]) < 0.03:
        return 0 if w < 0 else 1
    else:
        return 0 if theta + 0e-3*obs[0] < 0 else 1

def theta_omega_policy(state):
    x = state[0] / 9.6
    v = state[1] / 20
    theta = state[2] / 0.836
    v_target = max(min(-x * 0.5, 0.01), -0.01)
    theta_target = max(min(- (v - v_target) * 4, 0.2), -0.2)
    w = state[3] / 2
    w_target = max(min(- (theta - theta_target) * 0.9, 0.1), -0.1)
    return 0 if w < w_target else 1 

# env = gym.make('CartPole-v1')
# env._max_episode_steps=np.inf

env = CartpoleEnv()
state = env.reset()
print(state)
observation, reward, done, info = env.step(env.gym_env.action_space.sample())  # take a random action

# observation, reward, done, info = env.step(env.action_space.sample())  # take a random action
steps = []
for ex  in range(100):
    states = np.zeros((4, 20000))
    state = env.reset()

    print(state)
    # break
    for i in range(20000):
        # env.render()
        if npr.rand() < 0.01:
            observation = npr.randn(4)
        observation, reward, done, info = env.step(theta_omega_policy(observation)) # take
        # print(reward)
        # if done:
        #     print('done')
        if abs(observation[0]) >1:
            # print(observation)
            # print(i)
            steps.append(i)
            break

        states[:, i] = observation
print(np.mean(steps))
# env.close()
print()
print(np.max(states,1))
print(np.mean(states,1))
print(np.std(states,1))

