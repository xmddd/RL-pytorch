import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import torch.optim as optim
import random
import numpy as np


class Actor(nn.Module):
    def __init__(self, n_states, n_actions, hidden_dim=128):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(n_states, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, n_actions)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.softmax(self.fc3(x))
        return x


class Critic(nn.Module):
    def __init__(self, n_states, n_actions, hidden_dim=128):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(n_states, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class A2C():
    def __init__(self, n_states, n_actions, gamma=0.99, learning_rate=0.001):
        self.n_states = n_states
        self.n_actions = n_actions
        self.gamma = gamma
        self.trajectory = []
        self.policy_net = Actor(n_states, n_actions)
        self.value_net = Critic(n_states, n_actions)
        self.policy_optimizer = optim.Adam(
            self.policy_net.parameters(), lr=learning_rate)
        self.value_optimizer = optim.Adam(
            self.value_net.parameters(), lr=learning_rate)

    def choose_action(self, state):
        state = torch.tensor(state, dtype=torch.float)
        # print(state)
        pr_actions = self.policy_net(state)
        # print(pr_actions)
        m = Categorical(pr_actions)
        action = m.sample().item()
        return (action, pr_actions[action])

    def learn(self, state, action, next_state, reward, done):
        self.value_optimizer.zero_grad()
        self.policy_optimizer.zero_grad()

        # critic learn
        state = torch.tensor(state, dtype=torch.float)
        v_eval = self.value_net(state)
        next_state = torch.tensor(next_state, dtype=torch.float)
        if not done:
            v_reality = reward + self.gamma * self.value_net(next_state)
        else:
            v_reality = reward
        # the smaller td_error is, the more accurate v estimates
        td_error = v_reality - v_eval
        critic_loss = torch.square(td_error)

        # actro learn
        action = torch.tensor(action, dtype=torch.float)
        pr_actions = self.policy_net(state)
        m = Categorical(pr_actions)
        actor_loss = - m.log_prob(action) * td_error

        critic_loss.backward(retain_graph=True)
        actor_loss.backward()

        self.value_optimizer.step()
        self.policy_optimizer.step()
