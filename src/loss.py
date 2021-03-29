import torch.nn.functional as F
import torch
from torch import nn

class ContrastiveLoss(nn.Module):
    
    def __init__(self, margin):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.eps = 1e-9

    def forward(self, z1, z2, target):
        distances = (z2 - z1).pow(2).sum(1)
        losses = 0.5 * (target.float() * distances + (1 + -1 * target).float() * F.relu(self.margin - (distances + self.eps).sqrt()).pow(2))
        return losses.mean()

class SpectralNetLoss(nn.Module):
    
    def __init__(self):
        super(SpectralNetLoss, self).__init__()
        
    def forward(self, Y, W):
        Yd = torch.cdist(Y, Y, p=2, compute_mode='use_mm_for_euclid_dist_if_necessary')**2
        return torch.sum(W*Yd)/(W.shape[0])

