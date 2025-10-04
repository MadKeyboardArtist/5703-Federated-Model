# IGNORE this file

import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self, d_feat):
        super().__init__()
        self.conv = nn.Sequential(
            # for [3, 32, 32] input images
            # [3, 32, 32] -> [32, 16, 16]
            nn.Conv2d(3,32,3,padding=1), nn.ReLU(), nn.MaxPool2d(2),
            # [32,16, 16] -> [64, 8,  8 ] 
            nn.Conv2d(32,64,3,padding=1), nn.ReLU(), nn.MaxPool2d(2))
        
        self.proj = nn.Linear(64*8*8, d_feat)

        self.d_feat = d_feat

    def forward(self, x):
        # flatten [batch, 64, 8, 8] -> [batch, 4096]
        h = self.conv(x).flatten(1)
        # [batch, 4096] -> [batch, d_feat]
        return F.relu(self.proj(h))

class Encoder(nn.Module):
    def __init__(self, d_feat):
        super().__init__()
        # Expect 3-channel input to match your tfms (RGB + ImageNet mean/std)
        self.conv = nn.Sequential(
            nn.Conv2d(3 , 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            # Make it size-agnostic: force feature map to 8x8 regardless of IMG_SIZE
            # nn.AdaptiveAvgPool2d((8, 8))
        )
        # self.proj = nn.Linear(64 * 8 * 8, d_feat)
        self.proj = nn.LazyLinear(d_feat)  # infers input features on first forward
        self.d_feat = d_feat

    def forward(self, x):
        h = self.conv(x).flatten(1)          # [B, 64, 8, 8] -> [B, 4096]
        return F.relu(self.proj(h))          # [B, 4096] -> [B, d_feat]

class Encoder(nn.Module):
    def __init__(self, d_feat):
        super().__init__()
        # expects 3-channel input (matches your tfms with ImageNet mean/std)
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((8, 8))   # always produce [B, 64, 8, 8]
        )
        self.proj = nn.Linear(64 * 8 * 8, d_feat)

    def forward(self, x):
        h = self.conv(x).flatten(1)          # [B, 64, 8, 8] -> [B, 4096]
        return F.relu(self.proj(h))          # [B, 4096] -> [B, d_feat]

    '''
    def forward(self, x):
        h = self.conv(x).flatten(1)
        z = F.relu(self.proj(h))
        
        # add an empty text embedding as head
        empty_text = torch.zeros(z.size(0), self.d_feature, device=z.device, dtype=z.dtype)
        z_extended = torch.cat([empty_text, z], dim=1)
        
        return z_extended
    '''
