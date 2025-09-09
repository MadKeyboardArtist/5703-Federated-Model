# IGNORE this file

import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self, d_feat):
        super().__init__()
        self.conv = nn.Sequential(
            # for [1, 32, 32] input images
            # [1, 32, 32] -> [32, 16, 16]
            nn.Conv2d(1,32,3,padding=1), nn.ReLU(), nn.MaxPool2d(2),
            # [32,16, 16] -> [64, 8,  8 ] 
            nn.Conv2d(32,64,3,padding=1), nn.ReLU(), nn.MaxPool2d(2))
        
        self.proj = nn.Linear(64*8*8, d_feat)

        self.d_feat = d_feat

    def forward(self, x):
        # flatten [batch, 64, 8, 8] -> [batch, 4096]
        h = self.conv(x).flatten(1)
        # [batch, 4096] -> [batch, d_feat]
        return F.relu(self.proj(h))

    '''
    def forward(self, x):
        h = self.conv(x).flatten(1)
        z = F.relu(self.proj(h))
        
        # add an empty text embedding as head
        empty_text = torch.zeros(z.size(0), self.d_feature, device=z.device, dtype=z.dtype)
        z_extended = torch.cat([empty_text, z], dim=1)
        
        return z_extended
    '''
