# 掌握从torchvision.dataset下载常用数据集
import torchvision

trans=torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
train_set=torchvision.datasets.CIFAR10(root="./CIFAR10", train=True,transform=trans,download=True )
test_set=torchvision.datasets.CIFAR10(root="./CIFAR10",train=False,transform=trans,download=True)

print(test_set[0])