# transforms的使用
from PIL import Image
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter("logs")

img_path = "D:\\MYproject\\djangoProject\\DjangoProject\\train\\ants_image\\5650366_e22b7e1065.jpg"
img = Image.open(img_path)
tensor_trans = transforms.ToTensor()
tensor_img = tensor_trans(img)
writer.add_image("ToTensor", tensor_img)

# Normalize 归一化
trans_norm = transforms.Normalize([1, 3, 5], [2, 4, 6])
img_norm = trans_norm(tensor_img)
writer.add_image("Norm", img_norm, 0)

# Resize
print(img.size)
trans_resize = transforms.Resize((512, 512))
# img PIL -->resize  img PIL
img_resize = trans_resize(img)
img_resize_tensor = tensor_trans(img_resize)
writer.add_image("resize", img_resize_tensor, 0)

# Compose - resize
trans_tensor_2 = transforms.Resize(1024)
trans_compose = transforms.Compose([trans_tensor_2, tensor_trans])
img_resize_2 = trans_compose(img)
writer.add_image("resize", img_resize_2, 1)

writer.close()
