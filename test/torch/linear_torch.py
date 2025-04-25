# 线性层linear
import torchvision
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

train_set = torchvision.datasets.CIFAR10(root='./data', train=False, transform=torchvision.transforms.ToTensor())
test_loader = DataLoader(dataset=train_set, batch_size=64, shuffle=True, num_workers=0, drop_last=True)


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(196608, 10)

    def forward(self, input):
        output_value = self.linear(input)
        return output_value


net = Net()
for data in test_loader:
    imgs, targets = data
    print(imgs.shape)
    imgs = torch.flatten(imgs)
    output_value = net(imgs)
    print(output_value.shape)