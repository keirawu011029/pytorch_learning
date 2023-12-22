import torch
import torchvision


# 对于陷阱1
from model_save import *

# 方式1->保存方式1，加载模型
# model = torch.load("vgg16_method1.pth")
# print(model)


# 方式2加载模型->保存方式2
vgg16 = torchvision.models.vgg16()
vgg16.load_state_dict(torch.load("vgg16_method2.pth"))
# model2 = torch.load("vgg16_method2.pth")
print(vgg16)


## 陷阱1
model = torch.load("model_method1.pth")
print(model)
