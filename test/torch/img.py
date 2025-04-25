# tensorboard的使用
# tensorboard --logdir=logs  在控制台输入打开tensorboard

# 显示图像
from torch.utils.tensorboard import SummaryWriter
from PIL import Image
import numpy as np

writer = SummaryWriter("logs")
image_path = "./train/ants_image/0013035.jpg"

img = Image.open(image_path)
img_array = np.array(img)

writer.add_image("test",img_array,1,dataformats='HWC')


# 卷积神经网络
import torch

class MyMoudle(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        output = input+1
        return output

net = MyMoudle()
x = torch.tensor(1.0)
res = net.forward(x)
print(res)