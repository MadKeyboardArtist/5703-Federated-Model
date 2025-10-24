
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
from SiteTrainingFunctions.training_config import BATCH

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

    def __len__(self): 
        return len(self.y)

    def __getitem__(self, i):
        ehr   = torch.from_numpy(self.X[i])
        label = torch.tensor(self.y[i], dtype = torch.long)
        return {"ehr": ehr, "label": label}

# 3. operate evaluation
def final_evaluation(
    site_name,
    global_state,
    test_set_path, 
    labelcol, 
    n_classes, 
    best_head_path
    ):
    # Evaluate one site using its own local head and the global encoder.
    # Returns: (mean_loss, mean_acc)

    # 1. Prepare device
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    # 3. Build complete local model
    model = build_local_model(global_state, n_classes, best_head_path).to(device)
    model.eval()

    # 4. Build dataloader 
    ds = TabularOnlyDataset(test_set_path, labelcol)
    loader = DataLoader(ds, batch_size = BATCH, shuffle=False)

    # 5. Evaluate
    total_loss, total_acc, total_count = 0.0, 0.0, 0

    for batch in loader:
        ehr   = batch["ehr"].to(device)
        label = batch["label"].to(device)

        logits = model(ehr)
        loss   = F.cross_entropy(logits, label)
        preds  = logits.argmax(dim=1)
        acc    = (preds == label).float().mean()

        total_loss  += loss.item() * len(label)
        total_acc   += acc.item()  * len(label)
        total_count += len(label)

    mean_loss = total_loss / total_count
    mean_acc  = total_acc / total_count

    print(f"[Site Evaluation] Test samples={total_count}  Loss={mean_loss:.4f}  Acc={mean_acc:.4f}")
    return mean_loss, mean_acc


