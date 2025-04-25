import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST
import matplotlib.pyplot as plt


class Net(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(28*28, 64)
        self.fc2 = torch.nn.Linear(64, 64)
        self.fc3 = torch.nn.Linear(64, 64)
        self.fc4 = torch.nn.Linear(64, 10)

    def forward(self, x):
        x = torch.nn.functional.relu(self.fc1(x))
        x = torch.nn.functional.relu(self.fc2(x))
        x = torch.nn.functional.relu(self.fc3(x))
        x = torch.nn.functional.log_softmax(self.fc4(x), dim=1)
        return x


def get_data_loader(is_train):
    to_tensor = transforms.Compose([transforms.ToTensor()])
    data_set = MNIST("", is_train, transform=to_tensor)
    return DataLoader(data_set, batch_size=15, shuffle=True)


def evaluate(test_data, net):
    n_correct = 0
    n_total = 0
    with torch.no_grad():
        for (x, y) in test_data:
            outputs = net.forward(x.view(-1, 28*28))
            for i, output in enumerate(outputs):
                if torch.argmax(output) == y[i]:
                    n_correct += 1
                n_total += 1
    return n_correct / n_total


def main():

    train_data = get_data_loader(is_train=True)
    test_data = get_data_loader(is_train=False)
    net = Net()

    print("initial accuracy:", evaluate(test_data, net))
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
    for epoch in range(4):
        for (x, y) in train_data:
            net.zero_grad()
            output = net.forward(x.view(-1, 28*28))
            loss = torch.nn.functional.nll_loss(output, y)
            loss.backward()
            optimizer.step()
        print("epoch", epoch, "accuracy:", evaluate(test_data, net))

    for (n, (x, _)) in enumerate(test_data):
        if n > 3:
            break
        predict = torch.argmax(net.forward(x[0].view(-1, 28*28)))
        plt.figure(n)
        plt.imshow(x[0].view(28, 28))
        plt.title("prediction: " + str(int(predict)))
    plt.show()

#
# if __name__ == "__main__":
#     main()
# t=[]
# for _ in range(40):
#     t.append(1)
#     i = 0
#     dn = 0
#     while i <= 15:
#         if dn == 0:
#             dn = dn + 8
#         else:
#             dn = dn + 9
#         while t[dn] == 0:
#             dn += 1
#         if dn > 29:
#             dn = dn - 30
#         t[dn] = 0
#         dn += 1
#         i = i + 1
#     i = 0
#     while i <= 29:
#         print(t[i], end='')
#         i += 1

# import turtle as t
# t.pensize(4)
# t.hideturtle()
# t.colormode(255)
# t.color((255, 155, 192), "pink")
# t.setup(840, 500)
# t.speed(20)
# # 鼻子
# t.pu()
# t.goto(-100, 100)
# t.pd()
# t.seth(-30)
# t.begin_fill()
# a = 0.4
# for i in range(120):
#     if 0 <= i < 30 or 60 <= i < 90:
#         a = a + 0.08
#         t.lt(3)  # 向左转3度
#         t.fd(a)  # 向前走a的步长
#     else:
#         a = a - 0.08
#         t.lt(3)
#         t.fd(a)
# t.end_fill()
# t.pu()
# t.seth(90)
# t.fd(25)
# t.seth(0)
# t.fd(10)
# t.pd()
# t.pencolor(255, 155, 192)
# t.seth(10)
# t.begin_fill()
# t.circle(5)
# t.color(160, 82, 45)
# t.end_fill()
# t.pu()
# t.seth(0)
# t.fd(20)
# t.pd()
# t.pencolor(255, 155, 192)
# t.seth(10)
# t.begin_fill()
# t.circle(5)
# t.color(160, 82, 45)
# t.end_fill()
# # 头
# t.color((255, 155, 192), "pink")
# t.pu()
# t.seth(90)
# t.fd(41)
# t.seth(0)
# t.fd(0)
# t.pd()
# t.begin_fill()
# t.seth(180)
# t.circle(300, -30)
# t.circle(100, -60)
# t.circle(80, -100)
# t.circle(150, -20)
# t.circle(60, -95)
# t.seth(161)
# t.circle(-300, 15)
# t.pu()
# t.goto(-100, 100)
# t.pd()
# t.seth(-30)
# a = 0.4
# for i in range(60):
#     if 0 <= i < 30 or 60 <= i < 90:
#         a = a + 0.08
#         t.lt(3)  # 向左转3度
#         t.fd(a)  # 向前走a的步长
#     else:
#         a = a - 0.08
#         t.lt(3)
#         t.fd(a)
# t.end_fill()
# # 耳朵
# t.color((255, 155, 192), "pink")
# t.pu()
# t.seth(90)
# t.fd(-7)
# t.seth(0)
# t.fd(70)
# t.pd()
# t.begin_fill()
# t.seth(100)
# t.circle(-50, 50)
# t.circle(-10, 120)
# t.circle(-50, 54)
# t.end_fill()
# t.pu()
# t.seth(90)
# t.fd(-12)
# t.seth(0)
# t.fd(30)
# t.pd()
# t.begin_fill()
# t.seth(100)
# t.circle(-50, 50)
# t.circle(-10, 120)
# t.circle(-50, 56)
# t.end_fill()
# # 眼睛
# t.color((255, 155, 192), "white")
# t.pu()
# t.seth(90)
# t.fd(-20)
# t.seth(0)
# t.fd(-95)
# t.pd()
# t.begin_fill()
# t.circle(15)
# t.end_fill()
# t.color("black")
# t.pu()
# t.seth(90)
# t.fd(12)
# t.seth(0)
# t.fd(-3)
# t.pd()
# t.begin_fill()
# t.circle(3)
# t.end_fill()
# t.color((255, 155, 192), "white")
# t.pu()
# t.seth(90)
# t.fd(-25)
# t.seth(0)
# t.fd(40)
# t.pd()
# t.begin_fill()
# t.circle(15)
# t.end_fill()
# t.color("black")
# t.pu()
# t.seth(90)
# t.fd(12)
# t.seth(0)
# t.fd(-3)
# t.pd()
# t.begin_fill()
# t.circle(3)
# t.end_fill()
# # 腮
# t.color((255, 155, 192))
# t.pu()
# t.seth(90)
# t.fd(-95)
# t.seth(0)
# t.fd(65)
# t.pd()
# t.begin_fill()
# t.circle(30)
# t.end_fill()
# # 嘴
# t.color(239, 69, 19)
# t.pu()
# t.seth(90)
# t.fd(15)
# t.seth(0)
# t.fd(-100)
# t.pd()
# t.seth(-80)
# t.circle(30, 40)
# t.circle(40, 80)
# # 身体
# t.color("red", (255, 99, 71))
# t.pu()
# t.seth(90)
# t.fd(-20)
# t.seth(0)
# t.fd(-78)
# t.pd()
# t.begin_fill()
# t.seth(-130)
# t.circle(100, 10)
# t.circle(300, 30)
# t.seth(0)
# t.fd(230)
# t.seth(90)
# t.circle(300, 30)
# t.circle(100, 3)
# t.color((255, 155, 192), (255, 100, 100))
# t.seth(-135)
# t.circle(-80, 63)
# t.circle(-150, 24)
# t.end_fill()
# # 手
# t.color((255, 155, 192))
# t.pu()
# t.seth(90)
# t.fd(-40)
# t.seth(0)
# t.fd(-27)
# t.pd()
# t.seth(-160)
# t.circle(300, 15)
# t.pu()
# t.seth(90)
# t.fd(15)
# t.seth(0)
# t.fd(0)
# t.pd()
# t.seth(-10)
# t.circle(-20, 90)
# t.pu()
# t.seth(90)
# t.fd(30)
# t.seth(0)
# t.fd(237)
# t.pd()
# t.seth(-20)
# t.circle(-300, 15)
# t.pu()
# t.seth(90)
# t.fd(20)
# t.seth(0)
# t.fd(0)
# t.pd()
# t.seth(-170)
# t.circle(20, 90)
# # 脚
# t.pensize(10)
# t.color((240, 128, 128))
# t.pu()
# t.seth(90)
# t.fd(-75)
# t.seth(0)
# t.fd(-180)
# t.pd()
# t.seth(-90)
# t.fd(40)
# t.seth(-180)
# t.color("black")
# t.pensize(15)
# t.fd(20)
# t.pensize(10)
# t.color((240, 128, 128))
# t.pu()
# t.seth(90)
# t.fd(40)
# t.seth(0)
# t.fd(90)
# t.pd()
# t.seth(-90)
# t.fd(40)
# t.seth(-180)
# t.color("black")
# t.pensize(15)
# t.fd(20)
# # 尾巴
# t.pensize(4)
# t.color((255, 155, 192))
# t.pu()
# t.seth(90)
# t.fd(70)
# t.seth(0)
# t.fd(95)
# t.pd()
# t.seth(0)
# t.circle(70, 20)
# t.circle(10, 330)
# t.circle(70, 30)
# t.exitonclick()

# import turtle as t
#
# t.pensize(1)
# t.setup(1000,1000,0,0)
# t.goto(1000, 1000)
# t.exitonclick()

