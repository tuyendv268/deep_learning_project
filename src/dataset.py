from torch.utils.data import Dataset
import torch
from torchvision.io import read_image
from torchvision.utils import make_grid
from torchvision import transforms
import torchvision
import PIL
from imgaug import augmenters as iaa
import imgaug as ia
from torchvision.io import read_image
import numpy as np
import imageio

class ImgAugTransform:
    def __init__(self):
        self.aug = iaa.Sequential([
            # iaa.Scale((256, 256)),
            iaa.Sometimes(0.25, iaa.GaussianBlur(sigma=(0, 3.0))),
            iaa.Fliplr(0.5),
            iaa.Affine(rotate=(-20, 20), mode='symmetric'),
            iaa.Sometimes(0.25,
                          iaa.OneOf([iaa.Dropout(p=(0, 0.1)),
                                     iaa.CoarseDropout(0.1, size_percent=0.5)])),
            iaa.AddToHueAndSaturation(value=(-10, 10), per_channel=True)
        ], random_order=True)
      
    def __call__(self, img):
        img = np.array(img)
        return self.aug.augment_image(img)

class ImageDataset(Dataset):
    def __init__(self, datas, config) -> None:
        super(ImageDataset, self).__init__()
        
        self.datas = datas
        self.label_names = config["labels"]
        self.max_width = config["max_width"]
        self.max_height = config["max_height"]
        self.aug_img = ImgAugTransform()
        
    def __len__(self):
        return self.datas.shape[0]
    
    def _parse_sample(self, image_path, label):
        image = read_image(image_path)
        image = image[:, 0:self.max_height, 0:self.max_width]
        label = torch.tensor(label).unsqueeze(-1)
        
        image = self.aug_img(image.permute(1, 2, 0))
        image = torch.tensor(image)
        image = image.permute(2, 0, 1)

        return image.float(), label
    
    def __getitem__(self, index) -> None:
        image_path = self.datas["id"][index]

        label = self.datas[self.label_names].iloc[index].tolist()
        
        return self._parse_sample(image_path, label)