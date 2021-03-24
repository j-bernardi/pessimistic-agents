import numpy as np
from estimators import ImmediateRewardEstimator, plot_beta, QEstimator, MentorQEstimator
from collections import deque

QUANTILES = [2**k/(1 + 2**k ) for k in range(-5, 5)]

class FinitePessimisticAgent():

    def __init__(self, num_actions, num_states, env, mentor,quantile_i, gamma, eps_max=1., eps_min=0.05, batch_size=64):
        self.quantile_i = quantile_i
        self.num_actions = num_actions
        self.num_states = num_states
        self.env = env
        self.gamma = gamma
        self.mentor = mentor
        self.eps_max = eps_max
        self.eps_min = eps_min
        self.batch_size = batch_size

        self.IREs = [ImmediateRewardEstimator(action_i) for action_i in range(num_actions)]
  
        self.history = deque(maxlen=10000)
        self.mentor_history = deque(maxlen=10000)

        self.QEstimators = [QEstimator(q, self.IREs, gamma, num_states, num_actions) for q in QUANTILES]
        self.MentorQEstimator = MentorQEstimator(num_states, num_actions, gamma)

    def learn(self, num_eps, steps_per_ep=500, update_n_steps=100, render=True):
        
        total_steps = 0
        for ep in range(num_eps):
            state = self.env.reset()
            state=int(state)
            
            for step in range(steps_per_ep):
                
                values = [self.QEstimators[self.quantile_i].estimate(state, action_i) for action_i in range(self.num_actions)]
                proposed_action = int(np.argmax(values))

                mentor_value = self.MentorQEstimator.estimate(state)
                if mentor_value > values[proposed_action] + self.epsilon():
                    action = self.mentor(self.env.map_int_to_grid(state), kwargs={'state_shape': self.env.state_shape})
                    mentor_acted = True
                    # print('called mentor')
                else:
                    action = proposed_action
                    mentor_acted = False

                next_state, reward, done, _ = self.env.step(action)
                next_state = int(next_state)
                if render:
                    self.env.render()

                if mentor_acted:
                    self.mentor_history.append((state, action, reward, next_state))
                
                self.history.append((state, action, reward, next_state))

                total_steps += 1

                if total_steps%update_n_steps==0:
                    self.update_estimators()
                
                state = next_state
                if done:
                    print('failed')
                    break
    def update_estimators(self):
        rand_ints = np.random.randint(low=0, high=len(self.mentor_history),size=self.batch_size)
        mentor_history_samples = [self.mentor_history[i] for i in rand_ints]

        rand_ints = np.random.randint(low=0, high=len(self.history),size=self.batch_size)
        history_samples = [self.history[i] for i in rand_ints]
        
        self.MentorQEstimator.update(mentor_history_samples)

        for IRE_index, IRE in enumerate(self.IREs):
            IRE.update([(s, r) for s, a, r, _ in history_samples if IRE_index==a])
        
        for qestimator in self.QEstimators:
            qestimator.update(history_samples)

        

    def epsilon(self):

        if self.eps_max > self.eps_min:
            self.eps_max *= 0.999

        return self.eps_max * np.random.rand()

                





