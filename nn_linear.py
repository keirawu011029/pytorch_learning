import torchvision
from torch.utils.data import DataLoader
from torch import nn
from torch.nn import Linear
import torch


dataset = torchvision.datasets.CIFAR10(
    "CIFAR",
    train=False,
    transform=torchvision.transforms.ToTensor(),
    download=True)

dataloder = DataLoader(dataset, batch_size=64)


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = Linear(196608, 10)

    def forward(self, input):
        output = self.linear1(input)
        return output


model = Model()

for data in dataloder:
    imgs, targets = data
    print(imgs.shape)
    # output = torch.reshape(imgs, (1, 1, 1, -1))
    output = torch.flatten(imgs) # 将张量展平为一维张量
    print(output.shape)
    output = model(output)
    print(output.shape)
