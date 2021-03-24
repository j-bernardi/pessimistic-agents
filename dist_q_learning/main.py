from env import FiniteStateCliffworld
from agents import FinitePessimisticAgent
from mentors import prudent_mentor

if __name__ == "__main__":

    env = FiniteStateCliffworld()

    print("RESET STATE")
    env.reset()
    env.render()

    print("\n\nStep every action")
    for action in range(0, 4):
        print("TAKE ACTION", action)
        returned = env.step(action)
        print("Return tuple: ", returned)
        env.render()
    print("\n\nStep off the edge")
    rew, done = None, False
    while not done:
        obs, rew, done, _ = env.step(0)
        env.render()
        print("Reward, done:", rew, done)
    assert rew == -0.

    a = FinitePessimisticAgent(env.num_actions, env.num_states, env, prudent_mentor, 1, 0.99)
    
    a.learn(5,render=False)
