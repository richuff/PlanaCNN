# class Person():
#     def __call__(self, name):
#         print(name)
#
#     def hello(self, name):
#         print('Hello' + name)
#
#
# person = Person()
# person("zhangsan")
# person.hello("zhangsan")
# import torchvision
# from torch.utils.tensorboard import SummaryWriter
#
# trans_totensor = torchvision.transforms.Compose([
#     torchvision.transforms.ToTensor()
# ])
#
# train_set = torchvision.datasets.CIFAR10(root='./data', train=True, transform=trans_totensor)
# test_set = torchvision.datasets.CIFAR10(root='./data', train=False, transform=trans_totensor)
#
# writer = SummaryWriter("p1")
# for i in range(10):
#     img,target = train_set[i]
#     writer.add_image("test",img,)
# img,target = train_set[0]
# print(img)
# print(target)
#
# img.show()

import torchvision
import torch.nn as nn

vgg16_false = torchvision.models.vgg16(weights='pretrained')
vgg16_true = torchvision.models.vgg16(eights='pretrained')

train_data = torchvision.datasets.CIFAR10(root='../data', train=True, transform=torchvision.transforms.ToTensor())

vgg16_true.add_module('add_linear',nn.Linear(1000, 10))
print(vgg16_true)
