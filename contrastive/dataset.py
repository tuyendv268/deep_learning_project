import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torch.optim as optim

import torchvision
from torchvision import transforms

from PIL import Image

class ContrastiveTransformations(object):
    def __init__(self, mode=None):
        self.transformations = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomResizedCrop(size=256),
            transforms.RandomApply([transforms.ColorJitter(
                brightness=0.5,
                contrast=0.5,
                saturation=0.5,
                hue=0.1
            )], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            # transforms.GaussianBlur(kernel_size=9),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

        self.processing = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(256),
            # transforms.GaussianBlur(kernel_size=9),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

        self.mode = mode

    def __call__(self, x):
        if self.mode == 'contrastive':
            return ([self.transformations(x), self.transformations(x)])
        else:
            return ([self.processing(x), self.transformations(x)])

class ImageDataset(data.Dataset):
    def __init__(self, datas, config, mode=None):
        super(ImageDataset, self).__init__()
        
        self.datas = datas
        self.label_names = config["labels"]
        self.max_width = config["max_width"]
        self.max_height = config["max_height"]

        self.transforms = ContrastiveTransformations(mode)

    def __len__(self):
        return self.datas.shape[0]
    
    def _parse_sample(self, image_path, label):
        label = torch.tensor(label).unsqueeze(-1)

        image = Image.open(image_path).convert('RGB')
        image, image_aug = self.transforms(image)

        return image.float(), image_aug.float(), label
    
    def __getitem__(self, index):
        image_path = self.datas["id"][index]
        label = self.datas[self.label_names].iloc[index].tolist()
        return self._parse_sample(image_path, label)


if __name__ == '__main__':
    import yaml
    from yaml.loader import SafeLoader
    import pandas as pd
    import matplotlib.pyplot as plt

    with open("config.yml", "r") as f:
        config = yaml.load(f, Loader=SafeLoader)
    
    df = pd.read_csv(config["train_path"])

    img_dataset = ImageDataset(
        datas=df,
        config=config["dataset"])

    img, img_aug, labels = img_dataset[0]
    
    plt.imshow(torch.add(img.permute(1,2,0), 0.5))
    plt.show()

    plt.imshow(torch.add(img_aug.permute(1,2,0), 0.5))
    plt.show()

