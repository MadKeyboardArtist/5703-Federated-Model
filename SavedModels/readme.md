SavedModels directory

Purpose
This folder stores model checkpoints produced during federated training and evaluation. 
It separates the server-side global encoder weights from site-specific local heads, and keeps multiple “best” and “latest” snapshots for safe resumption and comparison.

Layout
current_best_local_heads/
- Per-site heads that currently pair with the best global encoders in the ongoing run (best so far this session).
newest_local_heads/
- Most recently trained heads from the latest global round (not necessarily the best).
overall_best_local_heads/
- Canonical, across-runs best heads per site (highest validation metric ever observed).


Checkpoint meaning
best_global_encoders.pth
- State dict of the aggregated global encoders only (no local heads). 
- This is the server model you broadcast to sites. 
- It typically contains the image encoder, tabular encoder, and fusion block (all initialized even without training).

<site_name>.pth (e.g., image_3.pth, tabular_4.pth)
- State dict of a single site’s local head. 
- Each head maps the shared encoder representation to that site’s label space (e.g., 2-class, 3-class, 5-class). 
- Heads are only compatible with the encoder representation size they were trained against.


Quick usage examples (PyTorch)

1. Load the best global encoders and one site head for evaluation

```python
import torch

enc = build_global_encoders()        # construct modules with the same architecture
enc.load_state_dict(torch.load("SavedModels/overall_best_local_heads/best_global_encoders.pth", map_location="cpu"))

head = build_local_head_for_site("image_3")  # use the correct input dim and n_classes for site image_3
head.load_state_dict(torch.load("SavedModels/overall_best_local_heads/image_3.pth", map_location="cpu"))

enc.eval(); head.eval()
# logits = head(enc(x_batch))
```

2. Resume site training from latest heads after a new global round

```python
head = build_local_head_for_site("tabular_4")
head.load_state_dict(torch.load("SavedModels/newest_local_heads/tabular_4.pth", map_location="cpu"))
# continue training head with the current global encoders
```

3. Safely replace current best with newest if the metric improved

```python
def maybe_update_best(newest_path, best_path, improved: bool):
    if improved:
        torch.save(torch.load(newest_path, map_location="cpu"), best_path)
```

Compatibility notes and common pitfalls
1. Shape mismatch (e.g., “mat1 and mat2 shapes cannot be multiplied”) usually means the head’s expected representation size does not match the loaded global encoders. Rebuild heads with the correct input size or reload the matching encoder checkpoint.
2. Heads are site-specific. Do not reuse a head across sites with different label spaces.
3. Keep the encoder architecture constant across rounds; changing it will invalidate all saved heads.
4. If you maintain multiple experiments, consider naming runs (e.g., run_2025-11-09) and placing their checkpoints in subfolders to avoid confusion.