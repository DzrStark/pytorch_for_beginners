import torch.optim
import torchvision
from torch import nn
from torch.utils.data import DataLoader

dataset=torchvision.datasets.CIFAR10("./CIFAR10",False,torchvision.transforms.ToTensor(),download=True)
dataloader=DataLoader(dataset,1)
class Mynet(nn.Module):
    def __init__(self):
        super(Mynet,self).__init__()
        self.conv1 = nn.Conv2d(3,32,5,padding=2)
        self.maxpool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(32,32,5,padding=2)
        self.maxpool2 = nn.MaxPool2d(2)
        self.conv3 = nn.Conv2d(32,64,5,padding=2)
        self.maxpool3 = nn.MaxPool2d(2)
        self.flat = nn.Flatten()
        self.lin1 = nn.Linear(1024,64)
        self.lin2 = nn.Linear(64,10)

    def forward(self,input):
        x = self.conv1(input)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.maxpool2(x)
        x = self.conv3(x)
        x = self.maxpool3(x)
        x = self.flat(x)
        x = self.lin1(x)
        x = self.lin2(x)
        return x

net=Mynet()
loss=nn.CrossEntropyLoss()
optim=torch.optim.SGD(net.parameters(),lr=0.01)

for data in dataloader:
    img,label=data
    output=net(img)
    myloss=loss(output,label)
    optim.zero_grad()
    myloss.backward()
    optim.step()
