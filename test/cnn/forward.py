# 卷积层的使用
import torch
import torchvision
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

train_set = torchvision.datasets.CIFAR10(root='./data', train=False, transform=torchvision.transforms.ToTensor())
test_loader = DataLoader(dataset=train_set, batch_size=64, shuffle=True, num_workers=0, drop_last=True)


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=3, stride=1, padding=0)

    def forward(self, input_value):
        input_value = self.conv1(input_value)
        return input_value


writer = SummaryWriter("p1")
net = Net()
# Net(
#   (conv1): Conv2d(3, 6, kernel_size=(3, 3), stride=(1, 1))
# )
result_list = []
step = 0
for data in test_loader:
    imgs, targets = data
    output_value = net(imgs)
    result_list.append(output_value)
    writer.add_images("input", imgs, step)
    output_value = torch.reshape(output_value, (-1, 3, 30, 30))
    writer.add_images("output", output_value, step)
    step += 1
print(len(result_list))
writer.close()