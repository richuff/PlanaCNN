# 模型的保存和模型的加载
import torchvision

vgg16 = torchvision.models.vgg16(pretrained=False)
# 保存方式一
torch.save(vgg16, "vgg16_method1.pth")
# 保存方式二 (官方推荐)
torch.save(vgg16.state_dict(), "vgg16_method2.pth")
# 加载保存方式二的模型
vgg16 = torchvision.models.vgg16(pretrained=False)
vgg16.load_state_dict(torch.load("vgg16_method2.pth"))
print(vgg16)