import torch
from torch import nn

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

if __name__=='__main__':
    net=Mynet()
    input=torch.ones((64,3,32,32))
    output=net(input)
    print(output.shape)