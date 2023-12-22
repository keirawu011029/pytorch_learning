import torch
import torch.utils.data
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


dataset = torchvision.datasets.CIFAR10(
    "CIFAR",
    train=False,
    transform=torchvision.transforms.ToTensor(),
    download=True)

dataloader = DataLoader(dataset, batch_size=3)


class Model(nn.Module):
    def __init__(self):
        super().__init__()

        self.model = nn.Sequential(
            nn.Conv2d(3, 32, 5, padding=2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 32, 5, padding=2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 5, padding=2),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(1024, 64),
            nn.Linear(64, 10)
        )

       
    def forward(self, x):
        x = self.model(x)
        return x
    
loss = nn.CrossEntropyLoss()
model = Model()
#1. 定义一个优化器
optim = torch.optim.SGD(model.parameters(), lr=0.01)
for epoch in range(20):
    running_loss = 0.0
    for data in dataloader:
        imgs, targets = data
        output = model(imgs)
        result_loss = loss(output,targets)
        #2. 把网络中每个可以调节的参数的梯度调为0
        optim.zero_grad()
        #3. 调用损失函数的反向传播，计算出每个参数的梯度
        result_loss.backward()
        #4. 调用优化器对每个参数进行调优
        optim.step()
        running_loss += result_loss
    print(running_loss)