# -*- coding: utf-8 -*-
# @Time     : 2024/10/16 21:06
# @Author   : wangDx
# @File     : model.py
# @describe : 搭建网络模型
from torch import nn


# 定义一个卷积神经网络
class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet, self).__init__()

        # 第一个卷积层，使用96个11*11的卷积核，步幅为4，池化层为最大池化
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=96, kernel_size=11, stride=4),  # in:3x227x227  out:96x55x55
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),  # in:96x55x55  out:96x27x27
        )

        # 第二个卷积层，使用256个5*5的卷积核，步幅为1，padding为2，池化层为最大池化
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5, stride=1, padding=2),  # in:96x27x27  out:256x27x27
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),  # in:256x27x27  out:256x13x13
        )

        # 连续3个卷积之后再进行池化，前两个卷积操作不改变通道数
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),

            nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),

            nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2)  # in:256x13x13  out:256x6x6
        )

        # 这里全连接层的单元个数比LeNet中的大数倍
        self.fc = nn.Sequential(
            nn.Linear(256 * 5 * 5, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),  # 使用Dropout来缓解过拟合

            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),  # 使用Dropout来缓解过拟合

            nn.Linear(4096, 10),
        )

    # 前向传播路径
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        return x

print("Model Create Finished!")
