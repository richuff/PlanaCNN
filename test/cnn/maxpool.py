# 最大池化的使用
import torchvision
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

train_set = torchvision.datasets.CIFAR10(root='./data', train=False, transform=torchvision.transforms.ToTensor())
test_loader = DataLoader(dataset=train_set, batch_size=64, shuffle=True, num_workers=0, drop_last=True)


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.pool = nn.MaxPool2d(kernel_size=3, ceil_mode=False)

    def forward(self, input_value):
        input_value = self.pool(input_value)
        return input_value


writer = SummaryWriter("p1")
net = Net()
result_list = []
step = 0
for data in test_loader:
    imgs, targets = data
    output_value = net(imgs)
    result_list.append(output_value)
    writer.add_images("input", imgs, step)
    writer.add_images("output", output_value, step)
    step += 1
print(len(result_list))
writer.close()