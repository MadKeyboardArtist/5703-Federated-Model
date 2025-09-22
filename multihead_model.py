
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


from config import D_TABULAR, D_EMBEDDING, D_FUSION


class Head(nn.Module):
    def __init__(self, d_fusion = D_FUSION, n_classes = None): 
        super().__init__()
        self.fc = nn.Linear(d_fusion, n_classes)
        
    def forward(self, z): 
        return self.fc(z)


class MultiHeadModel(nn.Module):
    def __init__(self, 
                 d_tabular = D_TABULAR, 
                 d_embedding = D_EMBEDDING, 
                 d_fusion = D_FUSION,
                 n_tabular_classes = None,
                 n_image_classes = None,
                 n_multi_classes = None
                 ):
        
        super().__init__()
        self.tabular_enc = TextEncoder(d_tabular, d_embedding)
        self.image_enc   = ImageEncoder(d_embedding)
        self.fusion  = FusionLayer(d_fusion)
        
        # multi heads
        self.head_tabular = Head(d_embedding, n_tabular_classes) if n_tabular_classes is not None else None
        
        self.head_image   = Head(d_embedding, n_image_classes)   if n_image_classes   is not None else None
        self.head_multi   = Head(d_fusion,    n_multi_classes)   if n_multi_classes   is not None else None

    def forward(self, *, x_text = None, x_img = None):        
        zt = self.tabular_enc(x_text)
        zi = self.image_enc(x_img)
        z  = self.fusion(zt, zi)
        return self.head(z)
    
    def forward(self, *, task_type, x_text = None, x_img = None):
        if task_type == "tabular":
            zt = self.tabular_enc(x_text)
            z = self.head_tabular(zt)
            return z
        
        elif task_type == "image":
            zi = self.image_enc(x_img)
            z = self.head_image(zi)
            return z
        
        elif task_type == "multi":
            zt = self.tabular_enc(x_text)
            zi = self.image_enc(x_img)
            z  = self.fusion(zt, zi)
            return self.head_multi(z)
        
        else:
            raise ValueError("task_type must be 'tabular' | 'image' | 'multi'")


