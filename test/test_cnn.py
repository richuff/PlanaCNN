from torch.utils.tensorboard import SummaryWriter
from PIL import Image
import numpy as np

writer = SummaryWriter("logs")
image_path = "../train/ants_image/0013035.jpg"

img = Image.open(image_path)
img_array = np.array(img)

writer.add_image("test",img_array,1,dataformats='HWC')
for i in range(100):
    writer.add_scalar("y=2x", i, i)

writer.close()

# 损失函数和反向传播
from torch import nn
import torch

inputs = torch.tensor([1,2,3],dtype=torch.float32)
targets = torch.tensor([1,2,5],dtype=torch.float32)

inputs = torch.reshape(inputs,(1,1,1,3))
targets = torch.reshape(targets,(1,1,1,3))

loss = nn.L1Loss()
result = loss(inputs,targets)
print(result)