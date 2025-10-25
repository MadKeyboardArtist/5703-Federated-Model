import torch
import torch.nn as nn
from torchvision import models

'''
self.tabular_enc = TextEncoder (d_tabular, d_embedding)
self.image_enc   = ImageEncoder(d_embedding)
self.fusion      = FusionLayer (d_embedding, d_fusion)
'''

class ImageEncoder(nn.Module):
    def __init__(self, out_dim = 256, pretrained = True, freeze_backbone = True):
        super().__init__()
        m = models.efficientnet_b0(
            weights=models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        )
        feat_dim = 1280  
        self.backbone = nn.Sequential(
            m.features,   
            m.avgpool,    
            nn.Flatten(), 
        )
        
        self.proj = nn.Linear(feat_dim, out_dim)
        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

    
    def forward(self, x, return_normed = True):
        feats = self.backbone(x)        # [B,1280]
        z = self.proj(feats)            # [B,256]
        if return_normed:
            z = nn.functional.normalize(z, dim=1)
        return z