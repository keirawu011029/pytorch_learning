from torch import nn
from torch.nn import MaxPool2d
import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter



dataset = torchvision.datasets.CIFAR10("CIFAR", train=False, download=True, transform=torchvision.transforms.ToTensor())
dataloader = DataLoader(dataset, batch_size=64)
 
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.maxpool1 = MaxPool2d(kernel_size=3,ceil_mode=True)

    def forward(self, input):
        output = self.maxpool1(input)
        return output

model=Model()

writer = SummaryWriter("logs")
step = 0


for data in dataloader:
    imgs, target = data
    writer.add_images("input", imgs, step)
    output = model(imgs)
    writer.add_images("output", output, step)
    step += 1

writer.close()