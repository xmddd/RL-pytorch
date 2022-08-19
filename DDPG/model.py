import numpy as np
from collections import defaultdict
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random


class Critic(nn.Module):
    """
    output Q based on state and action
    """

    def __init__(self, n_states, n_actions, hidden_dim=256):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(n_states + n_actions, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)

        init_w = 3e-3
        self.fc3.weight.data.uniform_(-init_w, init_w)
        self.fc3.bias.data.uniform_(-init_w, init_w)

    def forward(self, x, y):
        x = torch.cat([x, y], 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class Actor(nn.Module):
    """
    output an action which reault in the max Q
    """

    def __init__(self, n_states, n_actions, hidden_dim=256):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(n_states, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, n_actions)

        init_w = 3e-3
        self.fc3.weight.data.uniform_(-init_w, init_w)
        self.fc3.bias.data.uniform_(-init_w, init_w)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return torch.tanh(self.fc3(x))


class ReplayBuffer:
    def __init__(self):
        self.buffer = []

    def push(self, state, action, pr, val, reward, done):
        self.buffer[self.pos] = (state, action, pr, val, reward, done)

    def sample(self, batch_size):
        if batch_size < len(self.buffer):
            batch = random.sample(self.buffer, batch_size)
        else:
            batch = random.sample(self.buffer, len(self.buffer))
        state, action, pr, val, reward, done = zip(*batch)
        return state, action, pr, val, reward, done
    def clear(self):
        self.buffer = []

class DDPG(nn.Module):
    def __init__(self, n_states, n_actions, capacity=8000, batch_size=128, critic_learning_rate=1e-3, actor_learning_rate=1e-4, reward_decay=0.99, soft_tau=1e-2):
        super().__init__()
        self.n_actions = n_actions
        self.n_states = n_states
        self.gamma = reward_decay
        self.soft_tau = soft_tau
        self.sample_count = 0
        self.critic_net = Critic(n_states, n_actions)
        self.critic_target_net = Critic(n_states, n_actions)
        self.actor_net = Actor(n_states, n_actions)
        self.actor_target_net = Actor(n_states, n_actions)
        # copy parameters from critic to critic_target
        self.critic_target_net.load_state_dict(self.critic_net.state_dict())
        # copy parameters from actor to actor_target
        self.actor_target_net.load_state_dict(self.actor_net.state_dict())
        self.memory = ReplayBuffer(capacity)
        self.batch_size = batch_size
        self.critic_optimizer = optim.Adam(
            self.critic_net.parameters(), lr=critic_learning_rate)
        self.actor_optimizer = optim.Adam(
            self.actor_net.parameters(), lr=actor_learning_rate)

    def choose_action(self, state):
        # action selection
        self.sample_count += 1
        # choose best action
        state = torch.tensor(state, dtype=torch.float).unsqueeze(0)
        action = self.actor_net(state)
        # print(state_action)
        return action.detach().numpy()[0]

    def learn(self):
        if len(self.memory.buffer) < self.batch_size:
            return
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = self.memory.sample(
            self.batch_size)
        # print("state_batch", state_batch)
        # print("action_batch", action_batch)
        # print("reward_batch", reward_batch)
        # print("done_batch", done_batch)
        state_batch = torch.tensor(np.array(state_batch), dtype=torch.float)
        action_batch = torch.tensor(np.array(action_batch), dtype=torch.float)
        reward_batch = torch.tensor(
            reward_batch, dtype=torch.float).unsqueeze(1)
        next_state_batch = torch.tensor(
            np.array(next_state_batch), dtype=torch.float)
        done_batch = torch.tensor(
            done_batch, dtype=torch.float).unsqueeze(1)
        # print("state_batch", state_batch.shape)
        # print("action_batch", action_batch.shape)
        # print("reward_batch", reward_batch.shape)
        # print("done_batch", done_batch.shape)

        """
        state_batch = [s1,s2,...]
        action_batch = [a1,a2,...]
        concat state and action then put them into critic target network
        """
        # Q_expected = reward_batch + self.gamma * (1-done_batch) * self.critic_target_net(
        #     next_state_batch, self.actor_target_net(next_state_batch).detach())
        # Q_estimated = self.critic_net(state_batch, action_batch)
        # critic_loss = nn.MSELoss()(Q_expected.detach(), Q_estimated)
        # actor_loss = - \
        #     torch.mean(self.critic_net(
        #         state_batch, self.actor_net(state_batch)))
        actor_loss = self.critic_net(state_batch, self.actor_net(state_batch))
        actor_loss = -actor_loss.mean()
        next_action = self.actor_target_net(next_state_batch)
        target_value = self.critic_target_net(next_state_batch, next_action.detach())
        expected_value = reward_batch + (1.0 - done_batch) * self.gamma * target_value
        expected_value = torch.clamp(expected_value, -np.inf, np.inf)

        value = self.critic_net(state_batch, action_batch)
        critic_loss = nn.MSELoss()(value, expected_value.detach())

        # print(critic_loss, actor_loss)
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # soft update
        for target_param, param in zip(self.critic_target_net.parameters(), self.critic_net.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - self.soft_tau) +
                param.data * self.soft_tau
            )
        for target_param, param in zip(self.actor_target_net.parameters(), self.actor_net.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - self.soft_tau) +
                param.data * self.soft_tau
            )
