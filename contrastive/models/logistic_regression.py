import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torch.optim as optim

import torchvision
from torchvision import transforms

class LogisticRegression(nn.Module):
    def __init__(self, config):
        super(LogisticRegression, self).__init__()
        self.feature_dim = config['feature_dim']
        self.num_labels = config['num_labels']

        self.linear = nn.Linear(self.feature_dim, self.num_labels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, features):
        return self.sigmoid(self.linear(features))