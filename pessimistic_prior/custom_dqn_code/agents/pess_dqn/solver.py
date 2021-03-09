import os
import pprint
import datetime
import numpy as np
import tensorflow as tf

from tensorflow.keras.optimizers import Adam
from collections import deque

from standard_agent import StandardAgent
from utils import get_batch_from_memory
from agents.mentors import random_mentor


class PessDQNSolver(StandardAgent):
    """
    A standard dqn_solver, inpired by:
      https://gym.openai.com/evaluations/eval_EIcM1ZBnQW2LBaFN6FY65g/
    Implements a simple DNN that predicts values.
    """

    def __init__(
        self,
        experiment_name,
        env_wrapper,
        mentor=random_mentor,
        memory_len=100000,
        gamma=0.99,  # 1. can work
        batch_size=64,
        n_cycles=128,
        pessimism=10.,
        epsilon=1.,
        epsilon_min=0.01,
        epsilon_decay=0.995,
        learning_rate=0.01,
        learning_rate_decay=0.01,
        rollout_steps=10000,
        model_name="pess_dqn",
        saving=True):

        super(PessDQNSolver, self).__init__(
            env_wrapper,
            model_name,
            experiment_name,
            saving=saving)

        self.mentor = mentor
        self.queries = []

        # Training
        self.batch_size = batch_size
        self.n_cycles = n_cycles

        self.memory = deque(maxlen=memory_len)
        self.mentor_memory = deque(maxlen=memory_len)
        self.solved_on = None

        self.gamma = gamma
        self.pessimism = pessimism
        print("PESSIMISM", self.pessimism)
        self.epsilon = epsilon  # exploration rate
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        # Diffs: pessimsitic is num_actions + 1? But not in qfunc
        self.pessimistic_model = self.build_model()

        # TODO why extra action?
        self.unbiased_model = self.build_model(
            output_size=self.action_size + 1)

        self.optimizer = Adam(
            lr=learning_rate, 
            decay=learning_rate_decay)

        self.load_state()

        # self.rollout_memory(rollout_steps - len(self.memory))

    def act(self, state, epsilon=None):
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
            return np.random.randint(0, high=self.action_size), False
        if state.ndim == 1:
            state = tf.reshape(state, (1, self.state_size))

        # Final dim is the world model pred?
        unbiased_preds = self.unbiased_model(state)  # output = actions + 1
        # print("PREDS", unbiased_preds.shape)
        mentor_value = unbiased_preds[0, self.env_wrapper.action_space]
        unbiased_value = tf.reduce_max(unbiased_preds)
        pess_preds = self.pessimistic_model(state)
        pess_value = tf.reduce_max(pess_preds)

        # print(
        #     pess_value, unbiased_value, "\t",
        #     pess_value - unbiased_value)  # , mentor_value)
        mentor_act = mentor_value > pess_value + 0.01 or self.total_t < 1000  \
            # total_timesteps / 10
        if self.pessimism < 0:  # Optimistic
            mentor_act = False
        if mentor_act:
            self.queries.append(self.total_t)
            action = self.mentor(state, self.env_wrapper.action_space)
        else:
            action = tf.math.argmax(pess_preds[0], axis=-1).numpy()

        #################################

        return action, mentor_act

    def rollout_memory(self, rollout_steps, verbose=False, render=False):

        raise NotImplementedError("Not reimplimented for pessimism")

        if rollout_steps <= 0:
            return
        env = self.env_wrapper.env
        state = env.reset()
        for step in range(rollout_steps):
            if render:
                env.render()

            action = self.act(self.model, state, epsilon=1.)  # Max random
            observation, reward, done, _ = env.step(action)
            state_next = observation
            
            # Custom reward if required by env wrapper
            reward = self.env_wrapper.reward_on_step(
                state, state_next, reward, done, step)

            self.memory.append(
                (state, np.int32(action), reward, state_next, done)
            )
            state = observation

            if done:
                state = env.reset()
                # OR env_wrapper.get_score(state, state_next, reward, step)
        print(f"Rolled out {len(self.memory)}")

    def solve(self, max_iters, verbose=False, render=False):
        start_time = datetime.datetime.now()
        env = self.env_wrapper.env
        state = env.reset()
        success_steps = 0
        
        for iteration in range(max_iters):
            for step in range(self.n_cycles):
                if render:
                    env.render()

                action, mentor_act = self.act(state, epsilon=self.epsilon)
                action = np.int32(action)
                observation, reward, done, _ = env.step(action)
                state_next = observation
                
                # Custom reward if required by env wrapper
                reward = self.env_wrapper.reward_on_step(
                    state, state_next, reward, done, step)

                self.memory.append(
                    (state, action, reward, state_next, done))
                # Store transition in the replay buffer.
                self.mentor_memory.append(
                    (state, action, reward, state_next, done))
                if mentor_act:
                    self.mentor_memory.append(
                        (state, np.int32(self.env_wrapper.action_space),
                         reward, state_next, done))

                state = observation

                self.report_step(step, iteration, max_iters)
                if done:
                    state = env.reset()
                    # OR env_wrapper.get_score(state, state_next, reward, step)
                    self.scores.append(success_steps)
                    success_steps = 0
                else:
                    success_steps += 1

            score = step

            solved = self.handle_episode_end(
                state, state_next, reward, 
                step, max_iters, verbose=verbose)

            if solved:
                break

        self.elapsed_time += (datetime.datetime.now() - start_time)
        return solved

    def learn(self):
        """
        Updated the agent's decision network based
        on a sample of previous decisions it has seen.
        Here, we combine the target and action networks.

        pair:
            self.memory, self.pessimistic_model
            self.mentor_memory, self.unbiased_model

        """
        if len(self.memory) < self.batch_size:
            return

        args_tuple = get_batch_from_memory(self.memory, self.batch_size)
        mentor_args_tuple = get_batch_from_memory(
            self.mentor_memory, self.batch_size)

        pess_loss = self.take_training_step(args_tuple, pess=True)
        mentor_loss = self.take_training_step(mentor_args_tuple, pess=False)

        if self.epsilon is not None and self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    @tf.function
    def take_training_step(self, args, pess):

        sts, a, r, n_sts, d = args
        preds = self.pessimistic_model(n_sts) if pess\
            else self.unbiased_model(n_sts)

        future_q_pred = tf.math.reduce_max(preds, axis=-1)
        future_q_pred = tf.where(
            d, tf.zeros((1,), dtype=tf.dtypes.float64), future_q_pred)

        q_targets = tf.cast(r, tf.float64) + self.gamma * future_q_pred

        loss_value, grads = self.squared_diff_loss_at_a(
            self.pessimistic_model if pess else self.unbiased_model,
            sts, a, future_q_pred, q_targets
        )

        # TODO - unbiased model?
        if pess:
            self.optimizer.apply_gradients(
                zip(grads, self.pessimistic_model.trainable_variables))
        else:
            self.optimizer.apply_gradients(
                zip(grads, self.unbiased_model.trainable_variables))

        return loss_value

    @tf.function
    def squared_diff_loss_at_a(self, model, sts, a, q_t, q_next):
        """
        A squared difference loss function

        Diffs the Q model's predicted values for a state with 
        the actual reward + predicted values for the next state

        Applies pessimism
        """
        with tf.GradientTape() as tape:
            q_s = model(sts)  # Q(st)
            # Take only predicted value of the action taken for Q(st|at)
            gather_indices = tf.range(a.shape[0]) * tf.shape(q_s)[-1] + a
            q_s_a = tf.gather(tf.reshape(q_s, [-1]), gather_indices)

            error = q_s_a - q_next
            # TODO should be huber loss?
            error = tf.reduce_mean(tf.square(error))

            if self.pessimism == 0.:
                pess_errors = 0.  # weights = 1
            elif self.pessimism < 0.:
                # optimistic update
                raise NotImplementedError()
            elif self.pessimism > 0.:
                assert self.env_wrapper.max_rew == 0 or self.gamma < 1, (
                    "optimistic agent must have gamma < 1 or max_rew = 0")
                if self.env_wrapper.max_rew == 0:
                    max_value = 0
                else:
                    # TODO - verify 1- gamma - gets big?
                    max_value = self.env_wrapper.max_rew / (1. - self.gamma)
                unpleasant_surprise = (
                        (max_value - tf.stop_gradient(q_t))
                        / (self.env_wrapper.max_rew - self.env_wrapper.min_rew)
                )
                # TODO what is timestep_ph (global, or within the episode?)
                #  Just a decreasing effect over time?
                #  Also why negative - this REDUCES the loss!

                # TODO use self.t_total for timestep division?
                pess_errors = (-self.pessimism) * tf.reduce_mean(
                    tf.square(unpleasant_surprise), axis=-1)  # / timestep_ph
                # weighted_error = tf.reduce_mean(
                #     importance_weights_ph * (pess_errors + errors))
            else:
                assert False, "unexpected"

            weighted_error = tf.reduce_mean(pess_errors + error)

            # Q(st|at) diff Q(st+1)
            # losses = tf.math.squared_difference(error)
            # reduced_loss = tf.math.reduce_mean(losses)
        # TODO and unbiased model update?
        grad = tape.gradient(
            weighted_error, model.trainable_variables)
        return weighted_error, grad

    def save_state(self):
        """
        Called at the end of saving-episodes.

        Save a (trained) model with its weights to a specified file.
        Passes the required information to add to the pickle dict for the 
         model.
        """

        add_to_save = {
            "epsilon": self.epsilon,
            "memory": self.memory,
            "queries": self.queries,
            "optimizer_config": self.optimizer.get_config(),
            }

        self.save_state_to_dict(append_dict=add_to_save)

        # TODO save unbiased model
        self.pessimistic_model.save(self.model_location)

    def load_state(self):
        """Load a model with the specified name"""

        model_dict = self.load_state_from_dict()

        print("Loading weights from", self.model_location + "...", end="")
        if os.path.exists(self.model_location):
            # TODO load unbiased model
            self.pessimistic_model = tf.keras.models.load_model(self.model_location)
            self.optimizer = self.optimizer.from_config(self.optimizer_config)
            del model_dict["optimizer_config"], self.optimizer_config
            print(" Loaded.")
        else:
            print(" Model not yet saved at loaction.")

        if "memory" in model_dict:
            del model_dict["memory"]

        print("Loaded state:")
        pprint.pprint(model_dict, depth=1)
