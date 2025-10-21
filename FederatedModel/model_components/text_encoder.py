import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self, d_in, d_embedding):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, 256), 
            nn.ReLU(),
            nn.Linear(256, d_embedding), 
            nn.ReLU()
            )
    
    def forward(self, x):  
        return self.net(x)
    
    '''
    def forward(self, x):
        z = self.net(x)
        # add an empty image embedding as tail
        empty_image = torch.zeros(z.size(0), self.d_feat, device=z.device, dtype=z.dtype)
        z_extended = torch.cat([z, empty_image], dim=1)
        return z_extended
    '''