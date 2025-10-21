import torch
import torch.nn as nn
import torch.nn.functional as F

from FederatedModel.model_config import D_EMBEDDING, D_FUSION

# 128 + 128 -> 128
class Fusion(nn.Module):
    def __init__(self, d_embedding = D_EMBEDDING, d_fusion = D_FUSION):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2 * d_embedding, 256), 
            nn.ReLU(),
            nn.Linear(256, d_fusion), 
            nn.ReLU()
            )
    def forward(self, z_txt, z_img):
        return self.net(torch.cat([z_txt, z_img], dim = 1))
