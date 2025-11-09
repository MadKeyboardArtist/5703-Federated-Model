
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import random
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader

# model configs
from FederatedModel.federated_multihead_model import SharedEncoders, TabularClientModel
from FederatedModel.model_config import D_TABULAR, D_EMBEDDING, D_FUSION 

# training configs
from SiteTrainingFunctions.training_config import BATCH_TABULAR

####################################################
# 0. Configs
LOG_EVERY = 50
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 0. helpers
def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False
set_seed(42)

def accuracy_from_logits(logits, labels):
    return (logits.argmax(1) == labels).float().mean().item()

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


######################################################
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
    local_model = TabularClientModel(
        shared_encoders = local_encoder, 
        n_classes = n_classes
        )
    
    # 2.4 Load the saved local head weights
    if os.path.exists(head_path):
        head_state = torch.load(head_path, map_location = "cpu", weights_only=True)
        local_model.head.load_state_dict(head_state, strict = False)
    
    return local_model


# 2. build Dataset for dataloader
from SiteTrainingFunctions.tabular_site_training import TabularOnlyDataset

########
'''
class TabularOnlyDataset(Dataset):
    # Only table features + labels are read
    def __init__(self, csv_path, label_col):
        df = pd.read_csv(csv_path)
        assert label_col in df.columns
        self.label_col = label_col

        drop_cols = {label_col}
        feat_cols = [c for c in df.columns if (c not in drop_cols) and pd.api.types.is_numeric_dtype(df[c])]
        
        self.X = df[feat_cols].astype(np.float32).values
        self.y = df[label_col].astype(int).values.astype(np.int64)

    def __len__(self): return len(self.y)

    def __getitem__(self, i):
        ehr   = torch.from_numpy(self.X[i])
        label = torch.tensor(self.y[i], dtype = torch.long)
        return {"ehr": ehr, "label": label}
'''
########

# 3. predict labels
@torch.no_grad()
def make_predictions(
    site_name,
    global_state,
    test_set_path, 
    labelcol, 
    n_classes, 
    best_head_path
    ):

    # Evaluate one site using its own local head and the global encoder.
    # Returns: (mean_loss, mean_acc)

    # 1. Build complete local model
    model = build_local_model(global_state, n_classes, best_head_path).to(device)
    model.eval()

    # 2. Build dataloader 
    ds = TabularOnlyDataset(test_set_path, labelcol)
    loader = DataLoader(ds, batch_size = BATCH_TABULAR, shuffle=False)

    # 3. Predict
    all_labels, all_preds, all_probs = [], [], []

    for batch in loader:
        ehr   = batch["ehr"].to(device)
        label = batch["label"].to(device)

        logits = model(ehr)

        # probabilities
        prob = torch.softmax(logits, dim=1)
        preds = logits.argmax(dim=1)

        all_labels.append(label.cpu().numpy())
        all_preds.append(preds.cpu().numpy())

        '''if n_classes == 2:
            all_probs.append(prob[:, 1].cpu().numpy())  # positive class prob
        else:
            all_probs.append(prob.cpu().numpy())'''

        if n_classes == 2:
            all_probs.append(prob[:, 1].detach().cpu().numpy())  # positive class prob
        else:
            all_probs.append(prob.detach().cpu().numpy())

    y_true = np.concatenate(all_labels)
    y_pred = np.concatenate(all_preds)
    y_prob = np.concatenate(all_probs, axis=0)

    # 4. store results in pd.df
    result_df = store_predictions_in_df(n_classes, y_true, y_pred, y_prob)
    print(f"[{site_name}] Tabular Site Prediction: Samples={len(result_df)}  "
          f"Predictions collected (shape={y_prob.shape})")

    return result_df

