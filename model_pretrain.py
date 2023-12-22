import torchvision
from torch import nn


vgg16_true = torchvision.models.vgg16(weights="DEFAULT")
vgg16_false = torchvision.models.vgg16()

train_data = torchvision.datasets.CIFAR10("CIFAR", train=True, transform=torchvision.transforms.ToTensor(), download=True)
# print(vgg16_true)
vgg16_true.classifier.add_module("ad_linear", nn.Linear(1000, 10))
print(vgg16_true)

vgg16_false.classifier[6] = nn.Linear(4096, 10)
print(vgg16_false)