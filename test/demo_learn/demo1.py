import numpy as np

x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]
import sklearn
# pip install scikit-learn -i https://pypi.tuna.tsinghua.edu.cn/simple/
# 预测，得到的是y预测值

def forward(x):
    return x * w

# 损失函数，平方来表示，，，，cost为误差


def loss(x, y):
    y_pred = forward(x)
    return (y_pred - y) ** 2


for w in np.arange(0.0, 4.1, 0.1):
    l_sum = 0
    for x_val, y_val in zip(x_data, y_data):
        y_pred_val = forward(x_val)
        loss_val = loss(x_val, y_val)
        print('\t', x_val, y_val, y_pred_val, loss_val)

while True:
    x = eval(input())
    print(w)


