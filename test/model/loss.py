# Loss Function的使用
import torch
import torchvision
from torch import nn
from torch.nn import Conv2d, MaxPool2d, Flatten, Linear, Sequential
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

datasets = torchvision.datasets.CIFAR10(root='./data', train=False, transform=torchvision.transforms.ToTensor())
data_loader = DataLoader(datasets, batch_size=1)


class Net(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
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

optimizer = torch.optim.SGD(net.parameters(), lr=0.001)
cross_loss = nn.CrossEntropyLoss()
for data in data_loader:
    imgs, targets = data
    output = net(imgs)
    cross_loss(output, targets)
    print(output)
# writer = SummaryWriter("./p1")
# writer.add_graph(net,input)
# writer.close()