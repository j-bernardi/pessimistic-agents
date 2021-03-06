import os
import sys
import random
import pickle
import datetime
import itertools

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense

class StandardAgent():

    def __init__(self, env_wrapper, model_name, experiment_name, saving=True):

        self.env_wrapper = env_wrapper
        self.state_size = env_wrapper.observation_space
        self.action_size = env_wrapper.action_space

        self.model_name = model_name 
        experiment_dir = "_".join((model_name, experiment_name))

        self.experiment_dir = (
            "saved_models" + os.sep + experiment_dir + os.sep)
        self.model_location = self.experiment_dir + "model.h5"
        self.dict_location =  self.experiment_dir + "status.p"

        self.default_saves = ["scores", "total_t", "elapsed_time", ]

        self.saving = saving
        if self.saving:
            os.makedirs(self.experiment_dir, exist_ok=True)

    def build_model(self, output_size=None, model_name=None):
        """
        Returns a standard 24x48xaction_space tanh activated dense network. 
        If you don't care what network you're using, this is a good start for
        solving something like cartpole in <1000 epsiodes
        """

        if not output_size:
            output_size = self.action_size
        if not model_name:
            model_name = self.model_name

        tf.keras.backend.set_floatx('float64')

        model = tf.keras.Sequential(name=model_name)
        model.add(Dense(24, input_dim=self.state_size, activation='tanh'))
        model.add(Dense(48, activation='tanh'))
        model.add(Dense(output_size, activation='linear'))

        model.build()

        return model

    def show(self, show_episodes=1, verbose=True, render=True):
        env = self.env_wrapper.env
        for episode in range(show_episodes):
            state = env.reset()
            # Take steps until failure / win
            for step in itertools.count():
                if render:
                    env.render()
                action = self.act(self.model, state, epsilon=self.epsilon)
                observation, reward, done, _ = env.step(action)
                state_next = observation
                # Custom reward if required by env wrapper
                reward = self.env_wrapper.reward_on_step(
                    state, state_next, reward, done, step)
                state = observation
                
                print(
                    f"\rEpisode {episode + 1}/{show_episodes} - steps {step}",
                    end="")
                sys.stdout.flush()
                if done:
                    break

            # Calculate a (optionally custom) score for this episode
            score = self.env_wrapper.get_score(state, state_next, reward, step)

            print(f"\rEpisode {episode + 1}/{show_episodes} "
                  f"- steps {step} - score {score}/"
                  f"{self.env_wrapper.score_target}")

    def act(self, action_model, state, epsilon=None):
        """
        action_model:
          The model that takes state and returns the distribution across 
          actions to be maximised
        state: 
          A (1, state_shape) or (state_shape, ) tensor corresponding to 
          the action_model's input size
        epsilon: 
          The probability of taking a random step, rather than the model's 
          most valuable step. Should be between 0 and 1
        Returns
          A single action, either random or the model's best predicted action
        """
        if epsilon and (epsilon < 0. or epsilon > 1.):
            raise ValueError(
                f"Epsilon is a probability. You passed {epsilon}")
        if (state.shape != (self.state_size,) 
                and state.shape != (1, self.state_size)):
            raise NotImplementedError(
                "Not intended for use on batch state; returns integer")

        # If in exploration
        if epsilon and np.random.rand() <= epsilon:
            return random.randrange(self.action_size)

        if state.ndim == 1:
            state = tf.reshape(state, (1, self.state_size))

        return tf.math.argmax(action_model(state), axis=-1).numpy()[0]

    def report_step(self, step, batch, max_batches):

        print(
            f"\rIter {batch + 1}/{max_batches} ({self.total_episodes}) "
            f"- steps {step} ({self.total_t + 1})",
            end="")
        sys.stdout.flush()

        self.total_t += 1

    def handle_episode_end(self, state, state_next, 
        reward, step, max_episodes, verbose=False):

        solved, agent_score = self.env_wrapper.check_solved_on_done(
            self.scores, verbose=verbose)

        if self.total_episodes % 25 == 0 or solved:
            print(f"\rEpisode {self.total_episodes + 1}/{max_episodes} "
                  f"- steps {step} - score {int(agent_score)}/"
                  f"{int(self.env_wrapper.score_target)}")
            if self.saving:
                self.save_state()

        self.total_episodes += 1

        if solved:
            self.solved_on = (self.total_episodes, self.total_t)

        return solved

    def save_state_to_dict(self, append_dict={}):

        model_dict = {}

        for key in (
                "model_location", "scores", "total_t", 
                "total_episodes", "elapsed_time"):
            model_dict[key] = getattr(self, key)

        model_dict["trained_episodes"] = len(self.scores)

        for k, v in append_dict.items():
            model_dict[k] = v

        with open(self.dict_location, 'wb') as md:
            pickle.dump(model_dict, md)

        return model_dict

    def load_state_from_dict(self):

        model_dict = self.return_state_dict()

        # Initialise standard state
        self.scores = model_dict.get("scores", [])
        self.total_t = model_dict.get("total_t", 0)
        self.total_episodes = model_dict.get("total_episodes", 0)
        self.elapsed_time = model_dict.get(
            "elapsed_time", datetime.timedelta(0))
        
        # Set any other state found in the dictionary
        for k, v in model_dict.items():
            if k in self.default_saves:
                continue
            else:
                setattr(self, k, v)

        return model_dict

    def return_state_dict(self):
        """Open the model dict to view what models we have."""
        
        if os.path.exists(self.dict_location):
            with open(self.dict_location, 'rb') as md:
                model_dict = pickle.load(md)
        else:
            print("Model dict file does not exist for viewing, yet")
            model_dict = {}

        return model_dict

    def save_state(self):
        raise NotImplementedError(
            "To be implemented by the inheriting agent.")

    def load_state(self):
        raise NotImplementedError(
            "To be implemented by the inheriting agent.")
