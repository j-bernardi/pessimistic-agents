import gym

from collections import Counter

from stable_baselines3 import PDQN

env = gym.make('CartPole-v1')

timesteps = 100000 # 100000
quick_args = {}
# 	# buffer_size: int = 1000000,
#     "learning_starts": 500,
#     "target_update_interval": 100,
#     "exploration_fraction": 0.1,
#     "exploration_initial_eps": 0.5,
#     "exploration_final_eps": 0.01,
# }


model = PDQN('MlpPessPolicy', env, verbose=1, **quick_args)


model.learn(total_timesteps=timesteps)

print("Queries after train")
print(len(model.queries), "/", timesteps)  # , Counter(model.queries))

queries = []

obs = env.reset()
for i in range(100):
    action, _states, queried = model.pess_predict(obs, deterministic=True)
    if queried:
    	queries.append(i)
    obs, reward, done, info = env.step(action[0])
    env.render()
    if done:
      obs = env.reset()

env.close()

print("Queries after test")
print(len(queries), "/", 100)  # , Counter(queries))
