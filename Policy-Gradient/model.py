import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import torch.optim as optim
import random
import numpy as np


class MLP(nn.Module):
    def __init__(self, n_states, n_actions, hidden_dim=128):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(n_states, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, n_actions)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.softmax(self.fc3(x))
        return x

class PG():
    def __init__(self, n_states, n_actions, gamma=0.95, learning_rate=0.001):
        self.n_states = n_states
        self.n_actions = n_actions
        self.gamma = gamma
        self.trajectory = []
        self.policy_net = MLP(n_states, n_actions)
        self.optimizer = optim.Adam(
            self.policy_net.parameters(), lr=learning_rate)

    def choose_action(self, state):
        state = torch.tensor(state, dtype=torch.float)
        # print(state)
        pr_actions = self.policy_net(state)
        # print(pr_actions)
        m = Categorical(pr_actions)
        action = m.sample().item()
        return (action, pr_actions[action])

    def learn(self):
        vt = self.calculate_vt()  # td_error
        self.optimizer.zero_grad()
        for i in range(len(self.trajectory)):
            state = self.trajectory[i]["state"]
            state = torch.tensor(state, dtype=torch.float)
            action = self.trajectory[i]["action"]
            action = torch.tensor(action, dtype=torch.float)
            # reward = self.trajectory["reward"]
            pr_actions = self.policy_net(state)
            m = Categorical(pr_actions)
            loss = - m.log_prob(action) * vt[i]
            loss.backward()
        self.optimizer.step()

    def calculate_vt(self):
        discounted_reward = np.zeros(len(self.trajectory))
        running_add = 0
        for i in reversed(range(len(self.trajectory))):
            running_add = running_add * self.gamma + \
                self.trajectory[i]["reward"]
            discounted_reward[i] = running_add
        discounted_reward -= np.mean(discounted_reward)
        discounted_reward /= np.std(discounted_reward)
        return discounted_reward
