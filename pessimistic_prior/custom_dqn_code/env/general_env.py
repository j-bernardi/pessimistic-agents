import gym, time
import numpy as np

from gym import spaces, envs


class Env(object):
    """A general class that wraps an environment.
    Environments should inherit from this class.
    """

    def __init__(self, selected_env='MountainCar-v0'):
        
        self.env_name = selected_env
        self.kwargs = None
        # The environment
        self.env = gym.make(selected_env)

        self.min_rew = self.env.reward_range[0]
        self.max_rew = self.env.reward_range[1]

        # Some meta parameters
        self.observation_space = self.env.observation_space.shape[0]
        self.action_space = self.env.action_space.n

    def get_spaces(self, registry=True):
        """Print out the action and observation spaces for viewing."""
        if registry:
            print("Registry:\n", envs.registry.all())
        print("Action space:", self.env.action_space)
        #> Discrete(2)
        print("Obs space:", self.env.observation_space)
        #> Box(4,)

    def do_random_runs(self, episodes, steps, verbose=False, wait=0.0):
        """Run some episodes with random actions, stopping on 
        actual failure / win conditions. Just for viewing.
        """
        for i_episode in range(episodes):
            observation = self.env.reset()
            print("Episode {}".format(i_episode+1))
            for t in range(steps):
                self.env.render()
                if verbose:
                    print(observation)
                # take a random action
                action = self.env.action_space.sample()
                observation, reward, done, info = self.env.step(action)
                time.sleep(wait)
                if done:
                    print("Episode finished after {} timesteps".format(t+1))
                    break
        self.env.close()

    def get_score(self, state, next_state, reward, step):
        """Returns the score for an episode"""
        raise NotImplementedException("Should be overridden")

    def reward_on_step(self, state, next_state, reward, done):
        """Returns the reward for a step within an episode"""
        raise NotImplementedException("Should be overridden")
    
    def check_solved_on_done(self, state, episodes, scores, verbose=False):
        """Checks if the task was completed by this episode"""
        raise NotImplementedException("This method should be overridden")
