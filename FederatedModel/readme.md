FederatedModel

Overview
This package implements the model side of the “global encoder + local head” framework. It defines reusable encoders for tabular and image data, a simple fusion layer for multi-modal settings, and thin client models that attach a site-specific head to the shared encoders.

The default wiring in federated_multihead_model.py uses:
* TextEncoder from model_components/text_encoder.py for tabular (MLP)
* ImageEncoder from model_components/EfficientNet_REX.py for images (EfficientNet-style)
* Fusion from model_components/fusion_layer.py for combining modalities

Alternative components (e.g., model_components/MLP_Wayne.py or model_components/image_encoder.py) can be swapped by changing the imports in federated_multihead_model.py.

Folder structure
* model_components/
  * EfficientNet_REX.py        (ImageEncoder module)
  * image_encoder.py           (basic CNN alternative)
  * text_encoder.py            (MLP for tabular features)
  * MLP_Wayne.py               (alternate MLP encoder)
  * fusion_layer.py            (Fusion module to combine embeddings)
* federated_multihead_model.py (top-level module wiring encoders + heads)
* model_config.py              (dimension constants shared across modules)
All files inside model_components define a torch.nn.Module.


Model configuration (model_config.py)
* D_TABULAR = 16
  Number of tabular input features (e.g., 8 core features + 8 masks).
* D_IMG = 224
  Target image side length for transforms (square input).
* D_EMBEDDING = 128
  Size of the feature embedding produced by individual encoders (keep consistent across encoders).
* D_FUSION = D_EMBEDDING
  Embedding size used after fusion in multi-modal settings.
Adjust these constants if you change the encoders or preprocessing.


Core classes (federated_multihead_model.py)
1. Head
A single linear layer mapping from an embedding to the site’s label space.
* init: Head(d_embedding=D_EMBEDDING, n_classes)
* forward(z): returns logits of shape [N, n_classes]

2. SharedEncoders
Holds the shared modules used across sites.
* tabular_enc = TextEncoder(D_TABULAR, D_EMBEDDING)
* image_enc   = ImageEncoder(D_EMBEDDING)
* fusion      = Fusion(D_EMBEDDING, D_FUSION)

Convenience forwards:
* forward_tabular(x_tab) -> z_tab of size D_EMBEDDING
* forward_image(x_img)   -> z_img of size D_EMBEDDING
* forward_multi(x_tab, x_img) -> z_fused of size D_FUSION
  (encodes tabular and image separately, then applies Fusion)

3. Client models
Thin wrappers that attach a site head to the shared encoders.
TabularClientModel(shared_encoders, n_classes)
* forward(x_tab): enc.forward_tabular(x_tab) -> Head -> logits
ImageClientModel(shared_encoders, n_classes)
* forward(x_img): enc.forward_image(x_img) -> Head -> logits
MultiClientModel(shared_encoders, n_classes)
* forward(x_tab, x_img): enc.forward_multi(...) -> Head -> logits
  Intended for fusion output of size D_FUSION; the attached head should map from that dimension to n_classes.


Typical usage

```python
import torch
from FederatedModel.federated_multihead_model import SharedEncoders, TabularClientModel, ImageClientModel
from FederatedModel.model_config import D_TABULAR, D_EMBEDDING, D_FUSION

# build shared encoders
enc = SharedEncoders(d_tabular=D_TABULAR, d_embedding=D_EMBEDDING, d_fusion=D_FUSION)

# site models
tabular_site = TabularClientModel(shared_encoders=enc, n_classes=3)
image_site   = ImageClientModel(shared_encoders=enc, n_classes=5)

# forward examples
x_tab = torch.randn(32, D_TABULAR)   # tabular batch
x_img = torch.randn(32, 3, 224, 224) # image batch (example shape)

logits_tab = tabular_site(x_tab)     # [32, 3]
logits_img = image_site(x_img)       # [32, 5]
```


Swapping components
To try a different encoder:
1. Implement a new torch.nn.Module inside model_components with the same input/output contract:

   * Tabular/Text encoder: forward(x) -> tensor of size D_EMBEDDING
   * Image encoder: forward(x) -> tensor of size D_EMBEDDING
   * Fusion: forward(z_tab, z_img) -> tensor of size D_FUSION
2. Update the imports at the top of federated_multihead_model.py to point to your new modules.
3. Make sure D_EMBEDDING / D_FUSION in model_config.py match the new encoders’ output sizes.


Notes and tips
* Keep D_EMBEDDING consistent across encoders; client heads assume a fixed embedding size.
* If you change D_TABULAR, update preprocessing to emit that many features.
* Fusion is optional for single-modality sites; only the relevant encoder path is used.
* All client models expose a simple forward that returns logits; training code is responsible for loss, metrics, and optimisation.
