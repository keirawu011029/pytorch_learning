import torch
from PIL import Image
import torchvision
from torch import nn



device = torch.device("cuda:0" if torch.cuda.is_available else "cpu")

image_path = "./imgs/dog.png"
image = Image.open(image_path)
print(image)
image = image.convert('RGB')

transform = torchvision.transforms.Compose([torchvision.transforms.Resize((32,32)),
                                            torchvision.transforms.ToTensor()])

image = transform(image)
print(image.shape)


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 32, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 32, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64*4*4, 64),
            nn.Linear(64, 10)
        )
    
    def forward(self, x):
        x = self.model(x)
        return x
    
model = torch.load("./model/model_9.pth")
# model.to(device)
# print(model)

image = torch.reshape(image, (1, 3, 32, 32))
image = image.to(device)
model.eval()
with torch.no_grad():
    output = model(image)
print(output)

print(output.argmax(1))


# import torch
# from PIL import Image
# import torchvision
# from torch import nn



# # device = torch.device("cuda:0" if torch.cuda.is_available else "cpu")

# image_path = "./imgs/dog.png"
# image = Image.open(image_path)
# print(image)
# image = image.convert('RGB')

# transform = torchvision.transforms.Compose([torchvision.transforms.Resize((32,32)),
#                                             torchvision.transforms.ToTensor()])

# image = transform(image)
# print(image.shape)


# class Model(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.model = nn.Sequential(
#             nn.Conv2d(3, 32, 5, 1, 2),
#             nn.MaxPool2d(2),
#             nn.Conv2d(32, 32, 5, 1, 2),
#             nn.MaxPool2d(2),
#             nn.Conv2d(32, 64, 5, 1, 2),
#             nn.MaxPool2d(2),
#             nn.Flatten(),
#             nn.Linear(64*4*4, 64),
#             nn.Linear(64, 10)
#         )
    
#     def forward(self, x):
#         x = self.model(x)
#         return x
    
# model = torch.load("./model/model_9.pth", map_location=torch.device('cpu'))
# # model.to(device)
# # print(model)

# image = torch.reshape(image, (1, 3, 32, 32))
# # image = image.to(device)
# model.eval()
# with torch.no_grad():
#     output = model(image)
# print(output)

# print(output.argmax(1))