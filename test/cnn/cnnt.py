from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter("logs")


class Mydata(Dataset):
    def __init__(self, root_dir, label_dir):
        self.root_dir = root_dir
        self.label_dir = label_dir
        self.path = os.path.join(self.root_dir, self.label_dir)
        self.img_path_list = os.listdir(self.path)

    def __getitem__(self, index):
        img_name = self.img_path_list[index]
        img_path = os.path.join(self.path, img_name)
        img = Image.open(img_path)
        writer.add_image(img_name, img)
        return img, self.label_dir[index]


root_dir = '../../train'
label_dir1 = 'ants_image'
label_dir2 = 'bees_image'

ant_datasets = Mydata(root_dir, label_dir1)
bees_datasets = Mydata(root_dir, label_dir2)

train_datasets = ant_datasets + bees_datasets
