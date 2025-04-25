# 非线性激活Relu
import torch
import torchvision
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

train_set = torchvision.datasets.CIFAR10(root='./data', train=False, transform=torchvision.transforms.ToTensor())
test_loader = DataLoader(dataset=train_set,batch_size=64,shuffle=True,num_workers=0,drop_last=True)

writer = SummaryWriter("p1")
input = torch.tensor([[1,-0.5],[-1,3]])
input_reshape = torch.reshape(input,(-1,1,2,2))

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.relu = nn.ReLU()
        self.sigmod1 = nn.Sigmoid()

    def forward(self,input):
        output_value = self.sigmod1(input)
        return output_value
net = Net()
step = 0
for data in test_loader:
    imgs,targets = data
    output_value = net(imgs)
    writer.add_images("input",imgs,step)
    writer.add_images("output",output_value,step)
    step+=1

writer.close()