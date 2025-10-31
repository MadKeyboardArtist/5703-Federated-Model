# outside libraries
import torch
import torch.nn as nn
import torch.nn.functional as F
import os, copy
import random
import numpy as np
import pandas as pd
import math

from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import datasets, transforms

# self-built files
# model configs
from FederatedModel.federated_multihead_model import SharedEncoders, ImageClientModel
from FederatedModel.model_config import D_TABULAR, D_EMBEDDING, D_FUSION 

# training configs
from SiteTrainingFunctions.training_config import VAL_RATIO, EPOCHS, BATCH, LR, WD 


# 0. Configs
device = "cuda" if torch.cuda.is_available() else "cpu"

# 1. build local model: global encoders + local heads:
def build_local_model (global_state, n_classes, head_path):
    # 1. define the complete model structure
    # solved with import
    
    # 2. assemble local model
    # 2.1 Initialize local encoders
    local_encoder = SharedEncoders(
        d_tabular   = D_TABULAR, 
        d_embedding = D_EMBEDDING, 
        d_fusion    = D_FUSION
        )
    # 2.2 Load the weights
    local_encoder.load_state_dict(global_state, strict = False)

    # 2.3 build complete local model
    local_model = ImageClientModel(
        shared_encoders = local_encoder, 
        n_classes = n_classes
        )
    
    # 2.4 Load the saved local head weights
    if os.path.exists(head_path):
        head_state = torch.load(head_path, map_location = "cpu", weights_only=True)
        local_model.head.load_state_dict(head_state, strict = False)
    
    return local_model

# 2. build Dataset for dataloader (Image: returns dict for consistency)
class ImageFolderDict(datasets.ImageFolder):
    def __getitem__(self, index):
        img, label = super().__getitem__(index)
        return {"img": img, "label": torch.tensor(label, dtype=torch.long)}

def build_loaders(train_set_path, val_set_path, tfms, batch_size = BATCH, workers=4):
    train_ds = ImageFolderDict(train_set_path, transform = tfms)
    val_ds   = ImageFolderDict(val_set_path,   transform = tfms)
    
    train_loader = DataLoader(train_ds, batch_size = batch_size, shuffle = True,  
                              num_workers = workers, pin_memory = True)
    val_loader   = DataLoader(val_ds,   batch_size = batch_size, shuffle = False, 
                              num_workers = workers, pin_memory = True)
    
    return train_loader, val_loader

# 3. Training 1 epoch function
def train_one_epoch(model, loader, optimizer, device, log_every=50):
    model.train()
    total, correct, loss_sum = 0, 0, 0.0

    for i, batch in enumerate(loader, 1):
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(device, non_blocking = True)

        optimizer.zero_grad(set_to_none = True)

        logits = model(batch["img"])
        loss = F.cross_entropy(logits, batch["label"])
        
        loss.backward()
        optimizer.step()

        bs = batch["label"].size(0) # might change for the final batch

        loss_sum += loss.item() * bs
        total    += bs
        correct  += (logits.argmax(1) == batch["label"]).sum().item()
        
    # calculate results (weighted avg loss and acc)
    run_loss = loss_sum/max(total,1)
    run_acc  = correct /max(total,1)

    return run_loss, run_acc, total

# 3.1 simple training for 1 epoch
def train_one_epoch_simple(model, loader, optimizer):
    model.train()
    total = 0

    for i, batch in enumerate(loader, 1):
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(device, non_blocking = True)

        optimizer.zero_grad(set_to_none = True)

        logits = model(batch["img"])
        loss = F.cross_entropy(logits, batch["label"])
        
        loss.backward()
        optimizer.step()

        bs = batch["label"].size(0) # might change for the final batch
        total += bs

    return total

# 4. evaluation function (on validation)
@torch.no_grad()
def evaluation_in_training(model, loader, device):
    # 1. build the model
    pass
    # 2. build dataloder and load dataset
    pass
    # 3. calculate acc    
    model.eval()
    total, correct, loss_sum = 0, 0, 0.0

    for batch in loader:
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(device, non_blocking=True)

        logits = model(batch["img"])
        loss = F.cross_entropy(logits, batch["label"])

        bs = batch["label"].size(0)
        loss_sum += loss.item() * bs
        total    += bs
        correct  += (logits.argmax(1) == batch["label"]).sum().item()

    # calculate results (weighted avg loss and acc)
    run_loss = loss_sum/max(total,1)
    run_acc  = correct / max(total,1)

    return run_loss, run_acc, total


# 5. operate training
def training(site_name,
             global_state, 
             freeze_global,
             train_set_path, 
             val_set_path, 
             tsfm,
             n_classes, 
             newest_head_path, 
             current_best_head_path
             ):
    # 0.
    print("{:s}: image site training START".format(site_name))
    perf_track = []

    # 1. Build model
    client_model = build_local_model(global_state, n_classes, newest_head_path)    
    model = client_model.to(device)

    # 2. freeze_global or not
    if freeze_global:
        # 2.1 Freeze the encoder (shared global part)
        for param in model.enc.parameters():
            param.requires_grad = False
        # 2.2. Set optimizer to only optimize the local head
        optimizer = torch.optim.Adam (model.head.parameters(), lr = LR, weight_decay = WD) 
    else:
        optimizer = torch.optim.Adam (model.parameters(), lr = LR, weight_decay = WD) 

    # LR, WD could be different
    # design experinments if needed

    # 3. build dataloder and load dataset
    train_loader, val_loader = build_loaders(train_set_path, 
                                             val_set_path, 
                                             tsfm, 
                                             batch_size = BATCH, 
                                             workers=4)
    
    # 4. train each epoch
    sp_count = 0
    min_loss = float("inf")

    for epoch in range(1, EPOCHS + 1):
        # t_loss, t_acc, total = train_one_epoch(model, train_loader, optimizer, device, log_every=50)
        total = train_one_epoch_simple(model, train_loader, optimizer)
        v_loss, v_acc, v_sp = evaluation_in_training(model, val_loader, device)
        perf_track.append(v_acc)

        '''
        print(f"train: loss={t_loss:.4f}, acc={t_acc:.4f} | "
              f"val: loss={v_loss:.4f}, acc={v_acc:.4f}")
        '''
        sp_count += total
        
        # loss based check point:
        # save the model when it's more confident and better calibrated (i.e., lower cross-entropy or MSE).
        # Sensitive to probability confidence, not just correctness.
        # Use when:
        # 1. metric is cross-entropy, MSE, MAE, etc.
        # 2. You care about calibrated probabilities (e.g., for risk estimation or decision making)
        # 3. Accuracy is high but model still makes overconfident wrong guesses

        # examples:
        # | Metric     | Save model when... | Focus                    | Typical for                 |
        # | ---------- | ------------------ | ------------------------ | --------------------------- |
        # | `accuracy` | Accuracy improves  | % correct predictions    | Classification              |
        # | `loss`     | Loss decreases     | Confidence + correctness | Classification + Regression |

        # | Model version | Accuracy | Loss | Saved if using...          |
        # | ------------- | -------- | ---- | ---------------------------|
        # | Model A       | 0.91     | 0.45 | saved    ; `acc` or `loss` |
        # | Model B       | 0.91     | 0.38 | not saved; Only `loss`     |
        # | Model C       | 0.92     | 0.46 | saved    ; Only `acc`      |

        float_v_loss = float(v_loss)
        float_v_acc  = float(v_acc)

        if float_v_loss < min_loss - 1e-6:
            min_loss = float_v_loss
            # 1. record the current best local head 
            torch.save(model.head.state_dict(), current_best_head_path)
            # 2. record ckpt results
            ckpt_eva_results = {
                       "val_loss": v_loss,
                       "val_acc" : v_acc,
                       "num_samples": v_sp
                       }
            # print(f" [best updated] loss: {min_loss:.4f}")
            print(f" [best updated] acc: {float_v_acc:.4f}")    
    print("{:s}: image site training DONE".format(site_name))

    # 4. Return only image encoder weights
    updated_image = model.enc.image_enc.state_dict()

    # record the newest head for the next federated training round
    torch.save(model.head.state_dict(), newest_head_path)

    # 4 return values:
    # - updated_image: 1 trained image encoder
    # - sample_count: trained sample count in this site
    # - ckpt_eva_results: 1 dict, of best performence over epochs
    # - performance_track: list of acc after each epoch
    return updated_image, sp_count, ckpt_eva_results, perf_track