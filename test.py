import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

a = torch.tensor([[1,2,3],[4,5,6]],dtype=float)

# b = a.gather(dim = 1, index = torch.tensor([[1],[2]]))

# print(b.squeeze())

a = torch.tensor([1,2,3],dtype=float)
# b = torch.tensor([2,3,4],dtype=float).unsqueeze(1)
# loss = nn.MSELoss()(a,b)
# print(loss)

b=F.softmax(a)
print(b)

c = [{"a":1,"b":2},{"a":3,"b":4}]
print(c[1]["a"])

from torch.distributions import Categorical

m = Categorical(a)
action = m.log_prob(torch.tensor(0.5))
print(action)