# Sequential的使用
import torch
import torchvision
from torch import nn
from torch.nn import Conv2d, MaxPool2d, Flatten, Linear, Sequential
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.module = Sequential(
            Conv2d(3, 32, kernel_size=5, padding=2),
            Conv2d(32, 32, kernel_size=5, padding=2),
            MaxPool2d(2),
            MaxPool2d(2),
            Conv2d(32, 64, kernel_size=5, padding=2),
            MaxPool2d(2),
            Flatten(),
            Linear(1024, 64),
            Linear(64, 10)
        )

    def forward(self, x):
        x = self.module(x)
        return x


net = Net()
input = torch.ones((64, 3, 32, 32))
output = net(input)

writer = SummaryWriter("./p1")
writer.add_graph(net, input)
writer.close()