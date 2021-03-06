import os
import sys
import gym
import shutil
import numpy as np

from . import Env

CUSTOM_CARTPOLE_FILE = os.path.join("env", "custom_cartpole_to_copy.py")


def register_custom_env(entry_point, force=False):
    """Copies required files to gym for custom env

    Args:
        entry_point: colon-seperated path to gym file and name
            of the custom env object
        force: whether to force-update the copied files in the
            gym env, or return if present

    """
    gym_base_dir, _ = os.path.split(gym.__file__)
    control_dir = os.path.join(gym_base_dir, "envs", "classic_control")
    control_init = os.path.join(control_dir, "__init__.py")

    if ":" not in entry_point:
        raise ValueError("Expecting path like gym:path:CustomObj")
    entry_location, custom_env_name = (
        entry_point.split(':')[:-1], entry_point.split(":")[-1])

    custom_file = "custom_cartpole.py"
    import_line = (
        f"from {'.'.join(entry_location)}.{custom_file.replace('.py', '')} "
        f"import {custom_env_name}")

    with open(control_init, "r") as f:
        init_text = f.read()

    if import_line in init_text and not force:
        print("Loading existing env import")
        return

    user_input = input(
        f"First time run detected.\nAbout to register custom env to"
        f"\n{control_dir}\nContinue? (y/n) ")

    if user_input.lower() != "y":
        print("Stopping...")
        sys.exit()

    # Place required env files in gym/gym/envs/classic_control
    print("Copying", CUSTOM_CARTPOLE_FILE, "to", control_dir)
    try:
        _ = shutil.copy(
            CUSTOM_CARTPOLE_FILE, os.path.join(control_dir, custom_file))
    except shutil.SameFileError:
        print("Copy failed - same file")

    # add the import string to the init
    orig_copy = control_init + ".orig"
    if not os.path.exists(orig_copy):
        shutil.copy(control_init, orig_copy)

    if import_line not in init_text:
        with open(control_init, "a") as f_init:
            f_init.write(f"\n# Added automatically:\n{import_line}\n")


class CustomCartPole(Env):

    def __init__(
            self, cp_id='CustomCartPole-v1',
            angle_threshold=12., x_threshold=2.4,
            max_episode_steps=500,
    ):

        super().__init__(selected_env='CartPole-v1')

        self.angle_threshold = angle_threshold
        self.x_threshold = x_threshold
        self.max_episode_steps = max_episode_steps

        entry_point = 'gym.envs.classic_control:CustomCartPoleEnv'
        register_custom_env(entry_point)

        # Lazy singleton implementation - gym throws error if already registered
        try:
            gym.envs.register(
                id=cp_id,
                entry_point=entry_point,
                max_episode_steps=max_episode_steps,
                kwargs={
                    'x_threshold': x_threshold,
                    'angle_threshold': angle_threshold}
            )
        except gym.error.Error as gym_error:
            if "cannot re-register" in str(gym_error).lower():
                print(f"Env already registered: {cp_id}")
            else:
                raise gym_error

        # Override the initialised env
        self.env = gym.make(cp_id)


class CartPoleStandUp(CustomCartPole):
    
    def __init__(self, angle_threshold=12., score_target=195., episodes_threshold=100, max_episode_steps=500, reward_on_fail=-1.):
        
        self.score_target = score_target
        self.episodes_threshold = episodes_threshold
        self.reward_on_fail = reward_on_fail

        super().__init__(
            angle_threshold=angle_threshold,
            max_episode_steps=max_episode_steps)
        self.kwargs = locals()

    def get_score(self, state, next_state, reward_list, step_number):
        """
        This task's reward is simply how many steps it has survived.
        However some implementations provide a reward list.
        Consider summing this however NOTE: 
          this makes the finishing criterion harder, because the 
          final step has a negative reward
        """
        return step_number

    def reward_on_step(self, state, next_state, reward, done, step, **kwargs):
        """The reward per step is the default reward for cartpole
        (1 for a complete step) and -1 if it failed.
        """
        if done and step < self.max_episode_steps - 1:
            # Done before got to the end
            return self.reward_on_fail
        elif done:
            # It's okay to hit max steps
            return 0.
        else:
            # It's good to stay up
            return reward

    def check_solved_on_done(self, scores, verbose=False):
        """The task is solved if the average score over the last self.episodes_threshold 
        episodes averaged to be over the score threshold.
        """
        if len(scores) < 2:
            return False, 0
        solved = False
        if len(scores) < self.episodes_threshold:
            up_to = len(scores)
        else:
            up_to = self.episodes_threshold
        
        score = np.mean(scores[-up_to:])
        
        if (len(scores) >= self.episodes_threshold 
            and score > self.score_target):
                solved = True
        
        return solved, score
