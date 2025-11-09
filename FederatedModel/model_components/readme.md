model_components

purpose
Self-contained PyTorch modules used by the federated multi-head model. Each file defines a torch.nn.Module with a simple, consistent contract so encoders can be swapped without touching training code.

files
* EfficientNet_REX.py
  ImageEncoder implementation based on EfficientNet-style backbone. forward(x_img) -> embedding of size D_EMBEDDING.
* image_encoder.py
  Simple CNN ImageEncoder alternative. forward(x_img) -> D_EMBEDDING.
* text_encoder.py
  MLP encoder for tabular/EHR features. forward(x_tab) -> D_EMBEDDING.
* MLP_Wayne.py
  Alternate MLP encoder with a different block design. forward(x_tab) -> D_EMBEDDING.
* fusion_layer.py
  Fusion module to combine tabular and image embeddings. forward(z_tab, z_img) -> D_FUSION.

shape contracts
* tabular encoder input: [batch, D_TABULAR] -> [batch, D_EMBEDDING]
* image encoder input:   [batch, 3, H, W]   -> [batch, D_EMBEDDING] (H=W after resize)
* fusion layer input:    (z_tab: [batch, D_EMBEDDING], z_img: [batch, D_EMBEDDING])
  output: [batch, D_FUSION]

shared dims (from FederatedModel/model_config.py)
* D_TABULAR: number of tabular features after preprocessing
* D_EMBEDDING: encoder output size (keep the same across encoders)
* D_FUSION: fused representation size (often equals D_EMBEDDING)

swap guidance
1. implement your new module here following the contracts above
2. update imports in FederatedModel/federated_multihead_model.py to use it
3. ensure D_EMBEDDING and D_FUSION match your module outputs

quick example
```python
from FederatedModel.model_components.text_encoder import Encoder as TextEncoder
from FederatedModel.model_components.EfficientNet_REX import ImageEncoder
from FederatedModel.model_components.fusion_layer import Fusion
from FederatedModel.model_config import D_TABULAR, D_EMBEDDING, D_FUSION
import torch

tab_enc = TextEncoder(D_TABULAR, D_EMBEDDING)
img_enc = ImageEncoder(D_EMBEDDING)
fuse    = Fusion(D_EMBEDDING, D_FUSION)

x_tab = torch.randn(8, D_TABULAR)
x_img = torch.randn(8, 3, 224, 224)

z_tab = tab_enc(x_tab)       # [8, D_EMBEDDING]
z_img = img_enc(x_img)       # [8, D_EMBEDDING]
z_fus = fuse(z_tab, z_img)   # [8, D_FUSION]
```
