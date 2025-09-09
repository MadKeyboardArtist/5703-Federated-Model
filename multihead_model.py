# KAISON

import torch
import torch.nn as nn
import torch.nn.functional as F

# Model components:
from text_encoder  import Encoder as TextEncoder
from image_encoder import Encoder as ImageEncoder
from fusion_layer  import Fusion  as FusionLayer


# other components:
# 1. text file reader -> 1. samples + n_classes, 2. image place holder
# 2. image folder reader -> 1. resized samples + n_classes, 2. text place holder
#    (will influence the impelementation of image_encoder)
# 3. ...

from config import D_TEXT, D_EMBEDDING, D_FUSION


class Head(nn.Module):
    def __init__(self, d_fusion = D_FUSION, n_classes = None): 
        super().__init__()
        self.fc = nn.Linear(d_fusion, n_classes)
        
    def forward(self, z): 
        return self.fc(z)

class MultiHeadModel(nn.Module):
    def __init__(self, 
                 d_text = D_TEXT, 
                 d_embedding = D_EMBEDDING, 
                 d_fusion = D_FUSION, 
                 n_classes = None
                 ):
        
        super().__init__()
        self.txt_enc = TextEncoder(d_text, d_embedding)
        self.img_enc = ImageEncoder(d_embedding)
        self.fusion  = FusionLayer(d_fusion)
        self.head = Head(d_fusion, n_classes) if n_classes else None

    def forward(self, *, x_text = None, x_img = None):
        assert self.head is not None, "No head defined"
        zt = self.txt_enc(x_text)
        zi = self.img_enc(x_img)
        z  = self.fusion(zt, zi)
        return self.head(z)


