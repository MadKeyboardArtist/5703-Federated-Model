
import torch
import torch.nn as nn
import torch.nn.functional as F


# Model components:
# MLP
# from model_components.text_encoder  import Encoder as TextEncoder
from model_components.MLP_Wayne import MLPEncoder as TextEncoder

# CNN
from model_components.image_encoder import Encoder as ImageEncoder

# Fusion
from model_components.fusion_layer  import Fusion  as FusionLayer


# other components:
# 1. text file reader -> 1. samples + n_classes, 2. image place holder
# 2. image folder reader -> 1. resized samples + n_classes, 2. text place holder
#    (will influence the impelementation of image_encoder)
# 3. ...


from config import D_TABULAR, D_EMBEDDING, D_FUSION


class Head(nn.Module):
    def __init__(self, d_embedding = D_EMBEDDING, n_classes = None): 
        super().__init__()
        self.fc = nn.Linear(d_embedding, n_classes)
        
    def forward(self, z): 
        return self.fc(z)

class SharedEncoders(nn.Module):
    def __init__(self, 
                 d_tabular   = D_TABULAR, 
                 d_embedding = D_EMBEDDING, 
                 d_fusion    = D_FUSION
                 ):
        
        super().__init__()
        self.tabular_enc = TextEncoder (d_tabular, d_embedding)
        self.image_enc   = ImageEncoder(d_embedding)
        self.fusion      = FusionLayer (d_embedding, d_fusion)

    def forward_tabular(self, x):
        return self.tabular_enc(x)

    def forward_image(self, x):
        return self.image_enc(x)

    def forward_multi(self, x_text, x_img):
        zt = self.tabular_enc(x_text)
        zi = self.image_enc(x_img)
        z  = self.fusion(zt, zi)
        return self.fusion(z)

# local model - tabular site
class TabularClientModel(nn.Module):
    def __init__(self, shared_encoders: SharedEncoders, n_classes = None):
        super().__init__()
        self.enc = shared_encoders
        self.head = Head(d_embedding = D_EMBEDDING, 
                         n_classes = n_classes)

    def forward(self, x_text):
        z = self.enc.forward_tabular(x_text)
        return self.head(z)
    
class ImageClientModel(nn.Module):
    def __init__(self, shared_encoders: SharedEncoders, n_classes = None):
        super().__init__()
        self.enc = shared_encoders
        self.head = Head(d_embedding = D_EMBEDDING, 
                         n_classes = n_classes)

    def forward(self, x_image):
        z = self.enc.forward_image(x_image)
        return self.head(z)

class MultiClientModel(nn.Module):
    def __init__(self, shared_encoders: SharedEncoders, n_classes = None):
        super().__init__()
        self.enc = shared_encoders
        self.head = Head(d_fusion = D_FUSION, 
                         n_classes = n_classes)

    def forward(self, x_text, x_image):
        z = self.enc.forward_multi(x_text, x_image)
        return self.head(z)
