# -*- coding: utf-8 -*-
# @Time     : 2024/10/16 21:04
# @Author   : wangDx
# @File     : data_test.py
# @describe : 处理测试数据集
import torch.utils.data as Data
from torchvision import transforms
from torchvision.datasets import FashionMNIST


# 处理测试集数据
def test_data_process():
    test_data = FashionMNIST(root="./data/test",  # 数据路径
                             train=False,  # 不使用训练数据集
                             transform=transforms.Compose([transforms.Resize(size=224), transforms.ToTensor()]),  # 把PIL.Image或者numpy.array数据类型转变为torch.FloatTensor类型
                                                                                                                  # 尺寸为Channel * Height * Width，数值范围缩小为[0.0, 1.0]
                             download=True,  # 如果前面数据已经下载，这里不再需要重复下载
                             )
    test_loader = Data.DataLoader(dataset=test_data,  # 传入的数据集
                                   batch_size=1,  # 每个Batch中含有的样本数量
                                   shuffle=True,
                                   )

    # 获得一个Batch的数据
    for step, (b_x, b_y) in enumerate(test_loader):
        if step > 0:
            break
    batch_x = b_x.squeeze().numpy()  # 将四维张量移除第1维，并转换成Numpy数组
    batch_y = b_y.numpy()  # 将张量转换成Numpy数组
    print("The size of batch in test data:", batch_x.shape)

    return test_loader

test_data_process()
