import torchvision
from torch.utils.data import DataLoader
from torch import nn
import torch
from model import *
from torch.utils.tensorboard import SummaryWriter
import os



# 准备数据集
train_data = torchvision.datasets.CIFAR10("CIFAR", train=True, transform=torchvision.transforms.ToTensor(), download=True)

test_data = torchvision.datasets.CIFAR10("CIFAR", train=False, transform=torchvision.transforms.ToTensor(), download=True)

# length长度
train_data_size = len(train_data)
test_data_size = len(test_data)
print("训练数据集的长度：{}".format(train_data_size))
print("测试数据集的长度：{}".format(test_data_size))


# 用dataloader加载数据集
train_dataloader = DataLoader(train_data, batch_size=64)
test_dataloader = DataLoader(test_data, batch_size=64)

# 创建网络模型
model = Model()

# 损失函数
loss_fn = nn.CrossEntropyLoss()

# 优化器
learning_rate = 1e-2
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# 设置训练网络的一些参数
# 记录训练的次数
total_train_step = 0
# 记录测试的次数
total_test_step = 0
# 训练的轮数
epoch = 10


# 添加tensorboard
writer = SummaryWriter("logs")

for i in range(epoch):
    print("---------第 {} 轮训练开始--------".format(i+1))

    # 训练步骤开始
    model.train()
    for data in train_dataloader:
        imgs, targets = data
        output = model(imgs)
        loss = loss_fn(output, targets)

        # 优化器优化模型
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_train_step += 1
        if total_train_step % 100 == 0:
            print("训练次数：{}, loss: {}".format(total_train_step, loss.item()))
            writer.add_scalar("train_loss", loss.item(), total_train_step)

    # 测试步骤开始
    model.eval()
    total_test_loss = 0
    total_accuracy = 0
    with torch.no_grad():
        for data in test_dataloader:
            imgs, targets = data
            outputs = model(imgs)
            loss = loss_fn(outputs, targets)
            total_test_loss = total_test_loss + loss
            accracy = (outputs.argmax(1) == targets).sum()
            total_accuracy = total_accuracy + accracy

    print("整体测试集上的loss：{}".format(total_test_loss))
    print("整体测试集上的正确率：{}".format(total_accuracy/test_data_size))
    writer.add_scalar("test_loss", total_test_loss, total_test_step)
    writer.add_scalar("test_accuracy", total_accuracy/test_data_size, total_test_step)
    total_test_step +=1

    save_path = "./model"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    torch.save(model, os.path.join(save_path, "model_{}.pth".format(i)))
    # torch.save(model.state_dict(), os.path.join(save_path, "model_{}.pth".format(i)))
    print("模型已保存")

writer.close()