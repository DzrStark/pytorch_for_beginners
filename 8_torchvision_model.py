#学会使用torchvision中的models并修改
import torch
import torchvision
from torch import nn

#使用models
vgg16=torchvision.models.vgg16(pretrained=True )
print(vgg16)

#给models添加步骤
vgg16.classifier.add_module("7",nn.Linear(1000,10,bias=True))
print(vgg16)

#修改models的步骤
vgg16.classifier[6]=nn.Linear(4096,1000,True)
print(vgg16)

#学会保存并使用自己的models
#方法一：保存网络+参数 
torch.save(vgg16,"vgg16_method1.pth")
model=torch.load("vgg16_method1.pth")
#方法二：保存模型参数（官方推荐）
torch.save(vgg16.state_dict(),"vgg16_method2.pth")
vgg16.load_state_dict(torch.load("vgg16_method2.pth"))