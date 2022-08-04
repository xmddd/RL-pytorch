"""
This part of code is the Q learning brain, which is a brain of the agent.
All decisions are made in here.

View more on my tutorial page: https://morvanzhou.github.io/tutorials/
"""

import numpy as np
from collections import defaultdict
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random


class MLP(nn.Module):
    def __init__(self, n_states, n_actions, hidden_dim=128):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(n_states, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, n_actions)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.pos = 0

    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.pos] = (state, action, reward, next_state, done)
        self.pos = (self.pos+1) % self.capacity

    def sample(self, batch_size):
        if batch_size < len(self.buffer):
            batch = random.sample(self.buffer, batch_size)
        else:
            batch = random.sample(self.buffer, len(self.buffer))
        state, action, reward, next_state, done = zip(*batch)
        return state, action, reward, next_state, done


class DQN(nn.Module):
    def __init__(self, n_states, n_actions, capacity=100000, batch_size=64, learning_rate=0.0001, reward_decay=0.9, epsilon_start=0.9, epsilon_end=0.99, epsilon_decay=300):
        super().__init__()
        self.n_actions = n_actions
        self.n_states = n_states
        self.gamma = reward_decay
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.sample_count = 0
        # self.q_table = defaultdict(lambda: np.zeros(n_actions))
        self.policy_net = MLP(n_states, n_actions)
        self.target_net = MLP(n_states, n_actions)
        # copy parameters from policy to target
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.memory = ReplayBuffer(capacity)
        self.batch_size = batch_size
        self.optimizer = optim.Adam(
            self.policy_net.parameters(), lr=learning_rate)

    def choose_action(self, state):
        # action selection
        self.sample_count += 1
        epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * \
            np.exp(-self.sample_count / self.epsilon_decay)
        if np.random.uniform() < epsilon:
            # choose best action
            with torch.no_grad():
                state = torch.tensor(state, dtype=torch.float).unsqueeze(0)
                state_action = self.policy_net(state)
                # some actions may have the same value, randomly choose on in these actions
                # print(state_action)
                action = state_action.max(1).indices.item()
        else:
            # choose random action
            action = np.random.choice(self.n_actions)
        return action

    def learn(self):
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = self.memory.sample(
            self.batch_size)
        state_batch = torch.tensor(state_batch, dtype=torch.float)
        action_batch = torch.tensor(action_batch).unsqueeze(1)
        reward_batch = torch.tensor(reward_batch, dtype=torch.float)
        next_state_batch = torch.tensor(next_state_batch, dtype=torch.float)
        done_batch = torch.tensor(done_batch, dtype=torch.float)

        # choose the q_value w.r.t the action by gather function
        # [[q[s1,a1],q[s1,a2],...],
        #  [q[s2,a1],q[s2,a2],...],
        #  [q[s3,a1],q[s3,a2],...],...]
        # while s1,s2,... in state_batch
        # -->
        # [[q[s1,a(s1)]],
        #  [q[s2,a(s2)]],
        #  [q[s3,a(s3)]],]
        # while a(si) is the action in action_batch w.r.t the state_action
        # and use squeeze to reduce dim
        q_values = self.policy_net(state_batch).gather(
            dim=1, index=action_batch).squeeze(1)
        next_q_values = self.target_net(next_state_batch).max(1).values
        expected_q_values = reward_batch + \
            self.gamma * next_q_values * (1 - done_batch)
        loss = nn.MSELoss()(q_values, expected_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
