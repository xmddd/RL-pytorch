import numpy as np
from collections import defaultdict
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
from torch.distributions import Categorical


class Actor(nn.Module):
    """
    output an action which reault in the max Q
    """

    def __init__(self, n_states, n_actions, hidden_dim=256):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(n_states, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, n_actions)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return F.softmax(self.fc3(x))


class Critic(nn.Module):
    """
    output Q based on state and action
    """

    def __init__(self, n_states, hidden_dim=256):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(n_states, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class memory():
    def __init__(self):
        self.buffer = []

    def push(self, state, action, reward, log_pr):
        self.buffer.append((state, action, reward, log_pr))

    def get(self):
        state, action, reward, log_pr = zip(*self.buffer)
        return state, action, reward, log_pr

    def clear(self):
        self.buffer = []


class PPO(nn.Module):
    def __init__(self, n_states, n_actions, critic_learning_rate=1e-3, actor_learning_rate=1e-3, reward_decay=0.99, actor_episodes=10, critic_episodes=10, epsilon=0.1):
        super().__init__()
        self.n_actions = n_actions
        self.n_states = n_states
        self.gamma = reward_decay
        self.epsilon = epsilon
        self.memory = memory()
        self.critic = Critic(n_states)
        self.actor = Actor(n_states, n_actions)
        self.actor_old = Actor(n_states, n_actions)
        self.actor_old.load_state_dict(self.actor.state_dict())
        self.critic_optimizer = optim.Adam(
            self.critic.parameters(), lr=critic_learning_rate)
        self.actor_optimizer = optim.Adam(
            self.actor.parameters(), lr=actor_learning_rate)
        self.actor_episodes = actor_episodes
        self.critic_episodes = critic_episodes
        # self.batch_size = batch_size

    def choose_action_old(self, state):
        """
        choose action according to the old policy
        """
        state = torch.tensor(state, dtype=torch.float)
        # print(state)
        pr_actions = self.actor_old(state)
        # print(pr_actions)
        m = Categorical(pr_actions)
        action = m.sample()
        log_pr = m.log_prob(action)
        return action.detach().item(), log_pr.detach().item()

    def learn(self, next_state, done):
        # extract states into a tensor
        state_batch, action_batch, reward_batch, log_pr_batch = self.memory.get()
        state_batch = torch.tensor(state_batch, dtype=torch.float)
        action_batch = torch.tensor(action_batch, dtype=torch.float)
        log_pr_batch = torch.tensor(log_pr_batch, dtype=torch.float)

        discounted_reward = torch.zeros(len(self.memory.buffer))
        if done:
            running_add = 0
        else:
            running_add = self.critic(torch.tensor(next_state)).item()
        for i in reversed(range(len(self.memory.buffer))):
            running_add = running_add * self.gamma + reward_batch[i]
            discounted_reward[i] = running_add
        # discounted_reward -= self.critic(state_batch).unsqueeze()
        # discounted_reward /= torch.std(discounted_reward)

        # critic learn M episodes
        v_reality = discounted_reward
        for i in range(self.critic_episodes):
            v_eval = self.critic(state_batch).squeeze(-1)
            td_error = v_reality - v_eval
            critic_loss = nn.MSELoss()(v_eval, v_reality)
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

        adv = td_error.detach()

        # actor learn B episodes
        for i in range(self.actor_episodes):
            pr_action = self.actor(state_batch)
            m = Categorical(pr_action)
            new_log_pr_batch = m.log_prob(action_batch)

            ratio = torch.exp(new_log_pr_batch - log_pr_batch.detach())
            surr1 = ratio * adv
            surr2 = torch.clamp(ratio, 1 - self.epsilon, 1+self.epsilon) * adv
            actor_loss = - torch.min(surr1, surr2)
            actor_loss = actor_loss.mean()
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

        # copy parameters
        for old_param, param in zip(self.actor_old.parameters(), self.actor.parameters()):
            old_param.data.copy_(param.data)

        self.memory.clear()
