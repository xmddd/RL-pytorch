import torch.nn as nn
from torch.distributions import Categorical
import torch
import gym
env = gym.make('CartPole-v0')
env.reset()

# # for i in range(1000):
print(env.action_space.sample())
print(env.step(env.action_space.sample()))
# # env.render()
# print(torch.tensor(state).unsqueeze(0))

# a = torch.tensor([])
# a = torch.cat((a, torch.tensor(state).unsqueeze(0)), 0)
# a = torch.cat((a, torch.tensor(state).unsqueeze(0)), 0)
# print(a)

# print(torch.zeros(5))

# print(env.observation_space)


# m = Categorical(torch.tensor([[0.1, 0.2, 0.3, 0.4],[0.8,0.1,0.1,0]]))
# a = m.sample()
# print(a, m.log_prob(a))

# a = [(1,2),(3,4)]
# b,c= zip(*a)
# print(b,c)
a = torch.FloatTensor([1])
b = torch.FloatTensor()
print(nn.MSELoss()(a, b))
print(a.size())
print(b.size())