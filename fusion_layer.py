import torch
import torch.nn as nn
import torch.nn.functional as F

# 128 + 128 -> 128
class Fusion(nn.Module):
    def __init__(self, d_feat=128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(2*d_feat, 256), nn.ReLU(),
                                 nn.Linear(256, d_feat), nn.ReLU())
    def forward(self, z_txt, z_img):
        return self.net(torch.cat([z_txt, z_img], dim=1))
