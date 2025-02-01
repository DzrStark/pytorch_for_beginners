from torch.utils.tensorboard import SummaryWriter
import numpy as np
from PIL import Image
mywriter=SummaryWriter("log")

# add_scalar(tag(name), scalar_value(y), lobal_step(x) )
for i in range(100):
    mywriter.add_scalar("y=x^2",i*i,i)

# add_image(tag(name), img_tensor, global_step, dataformats="CHW")
img_path="G:\\Python_for_beginners\\Pytorch_for_beginners\\hymenoptera_data\\hymenoptera_data\\train\\ants\\0013035.jpg"
img=Image.open(img_path)
img_array=np.array(img)
mywriter.add_image("train",img_array,1,dataformats='HWC')
mywriter.close()

# 用法：打开终端，直接输入 tensorboard --logdir "G:\Python_for_beginners\Pytorch_for_beginners\log"（绝对路径）
