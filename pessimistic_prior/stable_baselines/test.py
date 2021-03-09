import gym

from stable_baselines3 import DQN

env = gym.make('CartPole-v1')

model = DQN('MlpPolicy', env, verbose=1)
model.learn(total_timesteps=10)

obs = env.reset()
for i in range(100):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    env.render()
    if done:
      obs = env.reset()

env.close()
