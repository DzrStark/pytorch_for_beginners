from PIL import Image
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter

# ToTensor:输入PILimage\nparray，输出Tensor
img_path="G:\\Python_for_beginners\\Pytorch_for_beginners\\hymenoptera_data\\hymenoptera_data\\train\\ants\\0013035.jpg"
img=Image.open(img_path)
tensor_tran=transforms.ToTensor()
img_tensor=tensor_tran(img)

# Normalize:输入Tensor，输出Tensor(output= (input - mean) / std)
norm=transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
img_norm=norm.forward(img_tensor)

# Resize:输入PILimage，输出PILimage
print(img)
resz=transforms.Resize((512,512))
img_resz=resz.forward(img)
print(img_resz)

# Compose([transforms1,transforms2,...])注意类型匹配
comp=transforms.Compose([resz,tensor_tran])
img_comp=comp(img)
print(type(img_comp))

# RandomCrop:输入PILimage，输出PILimage
radm=transforms.RandomCrop((512,523))
comp2=transforms.Compose([radm,tensor_tran])
img_comp2=comp2(img)
print(type(img_comp2))