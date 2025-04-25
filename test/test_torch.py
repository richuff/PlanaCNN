from PIL import Image
from torchvision.transforms import transforms
import torch.nn as nn
import torch

img_path = './test/dog.jpg'
image = Image.open(img_path)

transform = transforms.Compose([transforms.Resize((32,32)),transforms.ToTensor()])

imgage = transform(image)
#建立神经网络
class Net(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = nn.Sequential(
            nn.Conv2d(3,32,5,1,2),
            nn.MaxPool2d(2),
            nn.Conv2d(32,32,5,1,2),
            nn.MaxPool2d(2),
            nn.Conv2d(32,64,5,1,2),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64*16,64),
            nn.Linear(64,10),
        )
    def forward(self,x):
        x = self.model(x)
        return x

model = torch.load("./richu_1.pth",weights_only=False)
image = torch.reshape(imgage, (1,3,32,32))
model.eval()
with torch.no_grad():
    output = model(image)
print(output)