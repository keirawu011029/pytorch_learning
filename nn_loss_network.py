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
for data in dataloader:
    imgs, targets = data
    output = model(imgs)
    result_loss = loss(output,targets)
    print(result_loss)
    result_loss.backward()
    print("ok")