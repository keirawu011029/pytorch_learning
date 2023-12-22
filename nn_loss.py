import torch
from torch import nn


inputs = torch.tensor([1, 2, 3], dtype=float)
targets = torch.tensor([1, 2, 5],  dtype=float)

# inputs = torch.reshape(inputs, (1, 1, 1, 3))
# targets = torch.reshape(targets, (1, 1, 1, 3))

loss = nn.L1Loss(reduction="mean")
result = loss(inputs, targets)

loss_mes = nn.MSELoss()
result_mes = loss_mes(inputs, targets)

print(result)
print(result_mes)



x = torch.tensor([0.1, 0.2, 0.3])
print(x.shape)
y = torch.tensor(1)
print(y.shape)
loss_corss = nn.CrossEntropyLoss()
result_cross = loss_corss(x, y)
print(result_cross)