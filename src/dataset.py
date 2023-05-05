from torch.utils.data import Dataset
import torch
from torchvision.io import read_image

class ImageDataset(Dataset):
    def __init__(self, datas, config) -> None:
        super(ImageDataset, self).__init__()
        
        self.datas = datas
        self.label_names = config["labels"]
        self.max_width = config["max_width"]
        self.max_height = config["max_height"]
        
    def __len__(self):
        return self.datas.shape[0]
    
    def _parse_sample(self, image_path, label):
        image = read_image(image_path)
        image = image[:, 0:self.max_height, 0:self.max_width]
        label = torch.tensor(label).unsqueeze(-1)
        
        return image.float(), label
    
    def __getitem__(self, index) -> None:
        image_path = self.datas["id"][index]
        label = self.datas[self.label_names].iloc[index].tolist()
        
        return self._parse_sample(image_path, label)