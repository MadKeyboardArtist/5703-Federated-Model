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

def store_predictions_in_df(n_classes, true_label, pred_label, prob_label):
    if n_classes == 2: # binary
        df = pd.DataFrame({
            "y_true": true_label,
            "y_pred": pred_label,
            "y_prob": prob_label
        })

    else: # multi-class tasks, store probs as list columns
        df = pd.DataFrame({
            "y_true": true_label,
            "y_pred": pred_label,
            "y_prob": list(prob_label)
        })
    return df


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

# 3. predict labels
@torch.no_grad()
def make_predictions(
    site_name,
    global_state,
    test_set_path,
    tsfm,
    n_classes,
    best_head_path
    ):
    
    # 1. Build local model
    model = build_local_model(global_state, n_classes, best_head_path).to(device)
    model.eval()

    # 2. Build test DataLoader
    test_dataset = ImageFolderDict(test_set_path, transform = tsfm)
    test_loader  = DataLoader(test_dataset, batch_size = BATCH, shuffle = False)

    # 3. Predict
    all_labels, all_preds, all_probs = [], [], []

    for batch in test_loader:
        imgs   = batch["img"].to(device)
        labels = batch["label"].to(device)

        logits = model(imgs)
        probs  = torch.softmax(logits, dim=1)
        preds  = logits.argmax(dim=1)

        all_labels.append(labels.cpu().numpy())
        all_preds.append(preds.cpu().numpy())

        if n_classes == 2:
            all_probs.append(probs[:, 1].cpu().numpy())  # probability of positive class
        else:
            all_probs.append(probs.cpu().numpy())

    # Combine results 
    y_true = np.concatenate(all_labels)
    y_pred = np.concatenate(all_preds)
    y_prob = np.concatenate(all_probs, axis=0)

    # 4. store results in pd.df
    # 4. store results in pd.df
    result_df = store_predictions_in_df(n_classes, y_true, y_pred, y_prob)
    print(f"[{site_name}] Image Site Prediction: Samples={len(result_df)}  "
          f"Predictions collected (shape={y_prob.shape})")

    return result_df
    
