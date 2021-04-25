import gym
import numpy.random as npr

# cartpole policy from here: https://towardsdatascience.com/how-to-beat-the-cartpole-game-in-5-lines-5ab4e738c93f
def theta_omega_policy(obs):
    theta, w = obs[2:4]
    if abs(theta) < 0.03:
        return 0 if w < 0 else 1
    else:
        return 0 if theta < 0 else 1

env = gym.make('CartPole-v1')
state = env.reset()
print(state)
observation, reward, done, info = env.step(env.action_space.sample())  # take a random action
for _ in range(1000):
    env.render()
    if npr.rand() < 0.2:
        observation = npr.randn(4)
    observation, reward, done, info = env.step(theta_omega_policy(observation)) # take
    # print(reward)
    print(observation)
    print(_)
    print(info)
    print(done)
env.close()


