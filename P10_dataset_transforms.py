import torchvision
from torch.utils.tensorboard import SummaryWriter


dataset_trasnforms = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
])

train_set = torchvision.datasets.CIFAR10(root="./CIFAR", train=True, transform=dataset_trasnforms, download=True)
test_set = torchvision.datasets.CIFAR10(root="./CIFAR", train=False, transform=dataset_trasnforms, download=True)


# print(test_set[0])
# print(test_set.classes)

# img, target = test_set[0]
# print(img, target)
# print(test_set.classes[target])
# img.show()

img, target = train_set[0]
print(type(img))
print(train_set.classes[target])

writer = SummaryWriter("./logs")
for i in range(10):
    img, target = train_set[i]
    writer.add_image("img", img, i)

