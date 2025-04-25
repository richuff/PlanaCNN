import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np

input_size = 28  # 输入的图像的大小
num_classes = 10  # 标签的种类数
num_epochs = 3  # 训练的循环事件
batch_size = 64  # 每个批次的大小

# 加载数据集
train_datasets = datasets.MNIST(root='./data', train=True, transform=transforms.ToTensor(), download=True)
test_datasets = datasets.MNIST(root='./data', train=False, transform=transforms.ToTensor())

# 构建batch数据
train_loader = torch.utils.data.DataLoader(dataset=train_datasets, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_datasets, batch_size=batch_size, shuffle=False)

# 卷积网络模块的构建
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.out = nn.Linear(32 * 7 * 7, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        output = self.out(x)
        return output

def accuracy(predictions, labels):
    pred = torch.max(predictions, 1)[1]
    rights = pred.eq(labels.view_as(pred)).sum()
    return rights.item(), len(labels)

if __name__ == '__main__':
    # 实例化
    net = CNN()
    # 损失函数
    criterion = nn.CrossEntropyLoss()
    # 优化器
    optimizer = optim.Adam(net.parameters(), lr=0.001)

    for epoch in range(num_epochs):
        train_right = []

        for batch_idx, (data, target) in enumerate(train_loader):
            net.train()
            output = net(data)
            loss = criterion(output, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            right = accuracy(output, target)
            train_right.append(right)

            if batch_idx % 100 == 0:
                net.eval()
                val_rights = []

                for data, target in test_loader:
                    output = net(data)
                    right = accuracy(output, target)
                    val_rights.append(right)

                train_r = (sum([tup[0] for tup in train_right]), sum([tup[1] for tup in train_right]))
                val_r = (sum([tup[0] for tup in val_rights]), sum([tup[1] for tup in val_rights]))

                print(f"Epoch {epoch + 1}, Batch {batch_idx}, Train Accuracy: {train_r[0] / train_r[1]:.4f}, Val Accuracy: {val_r[0] / val_r[1]:.4f}")