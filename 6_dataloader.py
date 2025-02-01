import torchvision
from torch.utils.data import DataLoader

test_set=torchvision.datasets.CIFAR10(root="./CIFAR10", train=False, transform=torchvision.transforms.ToTensor())

test_loader=DataLoader(dataset=test_set,batch_size=64,shuffle=True,num_workers=0)
