"""
This part of code is the Q learning brain, which is a brain of the agent.
All decisions are made in here.

View more on my tutorial page: https://morvanzhou.github.io/tutorials/
"""

import numpy as np
from collections import defaultdict


class QLearningTable:
    def __init__(self, n_actions, learning_rate=0.1, reward_decay=0.9, epsilon_start=0.9, epsilon_end=0.99, epsilon_decay=300):
        self.n_actions = n_actions
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.sample_count = 0
        self.q_table = defaultdict(lambda: np.zeros(n_actions))

    def choose_action(self, state):
        # action selection
        self.sample_count += 1
        epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * \
            np.exp(-self.sample_count / self.epsilon_decay)
        if np.random.uniform() < epsilon:
            # choose best action
            state_action = self.q_table[state]
            # some actions may have the same value, randomly choose on in these actions

            action = np.random.choice(np.argwhere(
                state_action == np.max(state_action)).squeeze(1))

        else:
            # choose random action
            action = np.random.choice(self.n_actions)
        return action

    def learn(self, state, action, reward, next_state, done):
        q_predict = self.q_table[state][action]
        if not done:
            q_target = reward + self.gamma * \
                np.max(self.q_table[next_state]
                       )  # next state is not terminal
        else:
            q_target = reward  # next state is terminal
        self.q_table[state][action] += self.lr * \
            (q_target - q_predict)  # update

    def predict(self, state):
        state_action = self.q_table[state]
        action = np.random.choice(np.argwhere(
            state_action == np.max(state_action)).squeeze(1))
        return action
