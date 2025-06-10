# 使用GPU训练
# 模型训练套路
import torchvision
from torch.utils.data import DataLoader
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import torch

learning_rate = 1e-3
writer = SummaryWriter("logs")
# 准备数据集
train_data = torchvision.datasets.CIFAR10(root="./data", train=True, transform=torchvision.transforms.ToTensor())
test_data = torchvision.datasets.CIFAR10(root="./data", train=False, transform=torchvision.transforms.ToTensor())

# length长度
train_data_length = len(train_data)
test_data_length = len(test_data)
print("训练集的长度为{}，测试集的长度为{}".format(train_data_length, test_data_length))

# 使用DataLoader来加载数据
train_loader = DataLoader(train_data, batch_size=64)
test_loader = DataLoader(train_data, batch_size=64)


# 建立神经网络
class Net(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = nn.Sequential(
            nn.Conv2d(3, 32, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 32, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64 * 16, 64),
            nn.Linear(64, 10),
        )

    def forward(self, x):
        x = self.model(x)
        return x


net = Net()
net.cuda()
# 定义损失函数
loss_func = nn.CrossEntropyLoss()
loss_func = loss_func.cuda()
# 定义优化器
optimzer = torch.optim.SGD(net.parameters(), lr=learning_rate)

# 设置训练网络的一些参数
total_train_step = 0
total_test_step = 0
# 训练的轮数
epoch = 1
for i in range(1, epoch + 1):
    print("第{}轮训练".format(i))

    # 训练步骤开始
    net.train()
    for data in train_loader:
        imgs, targets = data
        imgs = imgs.cuda()
        targets = targets.cuda()
        output = net(imgs)
        # 损失值
        loss = loss_func(output, targets)
        # 优化调优
        optimzer.zero_grad()
        loss.backward()
        optimzer.step()
        total_train_step += 1
        if total_train_step % 100 == 0:
            print("训练次数: {},loss: {}".format(total_train_step, loss.item()))
            writer.add_scalar("train_loss", loss.item(), total_train_step)
    # 测试调优
    net.eval()
    total_test_loss = 0
    total_accuracy = 0
    with torch.no_grad():
        for data in test_loader:
            imgs, targets = data
            imgs = imgs.cuda()
            targets = targets.cuda()
            output = net(imgs)
            loss = loss_func(output, targets)
            total_test_loss += loss.item()
            accuracy = (output.argmax(1) == targets).sum()
            total_accuracy += accuracy

    print("整体上的损失值为Loss: {}".format(total_test_loss.item()))
    print("整体上的测试准确率为Accuracy: {}".format(total_accuracy / test_data_length))
    writer.add_scalar("test_loss", total_test_loss, total_test_step)
    writer.add_scalar("test_accuracy", total_accuracy / test_data_length, total_test_step)
    total_test_step += 1

    torch.save(net, "richu_{}.pth".format(i))
    # torch.save(net.state_dict(),"richu_{}.pth".format(i))
    print("模型已保存")

writer.close()