# -*- coding: utf-8 -*-
# @Time     : 2024/10/16 21:07
# @Author   : wangDx
# @File     : test.py
# @describe : 测试

# 测试模型
def test_model(model, testdataloader, device):
    '''
    :param model: 网络模型
    :param testdataloader: 测试数据集
    :param device: 运行设备
    '''

	# 初始化参数
    test_corrects = 0.0
    test_num = 0
    test_acc = 0.0
    # 只进行前向传播计算，不计算梯度，从而节省内存，加快运行速度
    with torch.no_grad():
        for test_data_x, test_data_y in testdataloader:
            test_data_x = test_data_x.to(device)
            test_data_y = test_data_y.to(device)
            model.eval()  # 设置模型为评估模式，不启用Batch Normalization和Dropout
            output = model(test_data_x)  # 前向传播过程，输入为测试数据集，输出为对每个样本的预测
            pre_lab = torch.argmax(output, 1)  # 查找每一行中最大值对应的行标
            test_corrects += torch.sum(pre_lab == test_data_y.data)  # 如果预测正确，则准确度val_corrects加1
            test_num += test_data_x.size(0)  # 当前用于训练的样本数量

    test_acc = test_corrects.double().item() / test_num  # 计算在测试集上的分类准确率
    print("test accuracy:", test_acc)

