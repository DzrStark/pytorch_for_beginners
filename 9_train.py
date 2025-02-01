#GPU训练法一：1.网络 2.损失函数 3.训练和测试的数据+.cuda()
#GPU训练法二：1.device=torch.device("cuda") 2.net=net.to(device)
import torch
from torch import nn
import torchvision
from torch.utils.data import DataLoader

#准备数据集
train_data = torchvision.datasets.CIFAR10("./CIFAR10",True,torchvision.transforms.ToTensor())
test_data = torchvision.datasets.CIFAR10("./CIFAR10",False,torchvision.transforms.ToTensor())

#获得数据集长度
train_data_size=len(train_data)
test_data_size=len(test_data)
print("训练集的长度为:{}".format(train_data_size))
print("测试集的长度为:{}".format(test_data_size))

#使用dataloader
train_dataloader = DataLoader(train_data,64)
test_dataloader = DataLoader(test_data,64)

#搭建网络
from model import *
net = Mynet()
net = net.cuda()

#定义损失函数
loss_func = nn.CrossEntropyLoss()
loss_func = loss_func.cuda()

#定义优化器
optim = torch.optim.SGD(net.parameters(),lr=0.01)

#常用参数的设置
train_cnt = 0
test_cnt = 0
epoch = 10

#进入训练
for i in range(epoch):
    print("-----这是第{}轮数据处理------".format(i+1))

    for data in train_dataloader:
        img, target = data
        img = img.cuda()
        target = target.cuda()
        out=net(img)
        loss=loss_func(out,target)

        optim.zero_grad()
        loss.backward()
        optim.step()

        train_cnt = train_cnt+1
        if train_cnt%100==0:
            print("这是第{}次训练,损失函数为：{}".format(train_cnt,loss.item()))

#进入测试
    loss_test_sum=0
    with torch.no_grad():
        for data in test_dataloader:
            img, target = data
            img = img.cuda()
            target = target.cuda()
            out=net(img)
            loss=loss_func(out,target)
            loss_test_sum = loss_test_sum+loss

        print("整体测试集上的loss:{}".format(loss_test_sum))
