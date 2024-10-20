# -*- coding: utf-8 -*-
# @Time     : 2024/10/16 21:06
# @Author   : wangDx
# @File     : train.py
# @describe : 训练网络模型
import os
import torch
from model import AlexNet
from torch.nn import CrossEntropyLoss
import torch.optim as optim
import torch.utils.data as Data
from torchvision import transforms
from torchvision.datasets import FashionMNIST

# 1.是否使用GPU进行训练数据
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# 2.设置超参数
EPOCHS = 25  # 训练轮次
BATCH_SIZE = 64  # 一轮训练批量大小
LR = 0.001  # 学习率

# 3.构建pipline 对图像做处理
pipeline = transforms.Compose([
    transforms.Resize(size=224),  # dataset中数据为227x227 需要裁剪为224x224
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))  # 正则化 降低模型复杂度
])

# 4.加载FashionMNIST训练数据集
train_data = FashionMNIST(root="./data/train", train=True, transform=pipeline, download=True)
# 配置训练数据加载器
train_loader = Data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)

# 5.将模型加载到设备上
model = AlexNet().to(DEVICE)

# 6.定义损失函数和优化器
loss_fn = CrossEntropyLoss()  # 交叉熵函数损失
option = optim.Adam(model.parameters(), lr=LR)  # Adam 梯度下降法

# 5.定义网络的训练过程
for epoch in range(EPOCHS):
    model.train()
    for step, (b_x, b_y) in enumerate(train_loader):
        b_x = b_x.to(DEVICE)
        b_y = b_y.to(DEVICE)

        output = model(b_x.float())  # 前向传播过程，输入为一个batch，输出为一个batch中对应的预测
        loss = loss_fn(output, b_y.long())  # 计算每一个batch的损失函数

        option.zero_grad()  # 将梯度初始化为0
        loss.backward()  # 反向传播计算
        option.step()  # 根据网络反向传播的梯度信息来更新网络的参数，以起到降低loss函数计算值的作用

        if step % 500 == 0:
            print("Train Epoch : {} \t Loss:{:.6f}".format(epoch, loss.item()))

# 当没有models文件夹时,要创建文件夹
if not os.path.isdir("models"):
    os.mkdir("models")
# 将训练好的模型保存到models文件夹下，方便后面测试时直接调用
torch.save(model, './models/model.pth')

print("Model finished training")
