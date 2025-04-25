# transforms的使用
from PIL import Image
from torchvision import  transforms
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter("logs")

img_path = "D:\\MYproject\\djangoProject\\DjangoProject\\train\\ants_image\\5650366_e22b7e1065.jpg"
img = Image.open(img_path)
tensor_trans = transforms.ToTensor()
tensor_img = tensor_trans(img)
writer.add_image("test", tensor_img)

#Normalize 归一化
trans_norm = transforms.Normalize([1, 3, 5],[2,4,6])
img_norm = trans_norm(tensor_img)
writer.add_image("norm", img_norm)

writer.close()


#Normalize 归一化
from torchvision import  transforms

trans_norm = transforms.Normalize([1, 3, 5],[2,4,6])
img_norm = trans_norm(tensor_img)
writer.add_image("norm", img_norm)


# Resize
from torchvision import  transforms

image_path = "./train/ants_image/0013035.jpg"
img = Image.open(image_path)

print(img.size)
trans_resize = transforms.Resize((256, 256))
# img PIL -->resize  img PIL
img_resize = trans_resize(img)
img_resize_tensor = tensor_trans(img_resize)
writer.add_image("resize", img_resize_tensor,0)
print(img_resize_tensor)


# torchvision的datasets的使用
import torchvision
from torch.utils.tensorboard import SummaryWriter

trans_totensor = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor()
])

train_set = torchvision.datasets.CIFAR10(root='./data', train=True, transform=trans_totensor,download=True)
test_set = torchvision.datasets.CIFAR10(root='./data', train=False, transform=trans_totensor,download=True)

writer = SummaryWriter("p1")
for i in range(10):
    img,target = train_set[i]
    writer.add_image("test",img,i)
writer.close()


# DataLoder
from torch.utils.data import DataLoader
import torchvision
from torch.utils.tensorboard import SummaryWriter

train_set = torchvision.datasets.CIFAR10(root='./data', train=False, transform=torchvision.transforms.ToTensor())

test_loader = DataLoader(dataset=train_set,batch_size=64,shuffle=True,num_workers=0,drop_last=True)
writer = SummaryWriter("p1")

step = 0
for data in test_loader:
    imgs ,targets = data
    writer.add_images("test",imgs,step)
    step+=1
writer.close()