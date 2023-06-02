import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torch.optim as optim

import torchvision
from torchvision import transforms

class ResNet50(nn.Module):
    def __init__(self, config):
        super(ResNet50, self).__init__()

        self.feature_dim = config['feature_dim']

        self.conv_net = torchvision.models.resnet50(num_classes=4*self.feature_dim)

        self.conv_net.fc = nn.Sequential(
            self.conv_net.fc,
            nn.ReLU(inplace=True),
            nn.Linear(4*self.feature_dim, self.feature_dim)
        )

    def forward(self, images, images_aug=None):
        if images_aug is not None:
            images = torch.cat((images, images_aug), dim=0)
            
        features = self.conv_net(images)
        return features