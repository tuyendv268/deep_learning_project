import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torch.optim as optim

import torchvision
from torchvision import transforms

class ContrastiveLoss(nn.Module):
    def __init__(self, config, logging=None):
        super(ContrastiveLoss, self).__init__()
        self.temperature = config['temperature']

    def forward(self, features, mode='train'):
        cosine_sim = F.cosine_similarity(features[:,None,:], features[None,:,:], dim=-1)

        self_mask = torch.eye(cosine_sim.shape[0], dtype=torch.bool, device=cosine_sim.device)
        cosine_sim.masked_fill_(self_mask, -9e15)

        pos_mask = self_mask.roll(shifts=cosine_sim.shape[0] // 2, dims=0)
        cosine_sim = cosine_sim / self.temperature

        nll = -cosine_sim[pos_mask] + torch.logsumexp(cosine_sim, dim=-1)
        nll = nll.mean()

        comb_sim = torch.cat([
                cosine_sim[pos_mask][:,None],  # First position positive example
                cosine_sim.masked_fill(pos_mask, -9e15)
        ], dim=-1)
        sim_argsort = comb_sim.argsort(dim=-1, descending=True).argmin(dim=-1)

        return nll, sim_argsort


    

        

