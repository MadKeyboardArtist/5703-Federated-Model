
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import random
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader

from federated_multihead_model import SharedEncoders, TabularClientModel
from config import D_TABULAR, D_EMBEDDING, D_FUSION # model configs
from config import VAL_RATIO, EPOCHS, BATCH, LR, WD # training configs

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
        head_state = torch.load(head_path, map_location = "cpu")
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

    def __len__(self): return len(self.y)

    def __getitem__(self, i):
        ehr   = torch.from_numpy(self.X[i])
        label = torch.tensor(self.y[i], dtype = torch.long)
        return {"ehr": ehr, "label": label}

# 3. evaluation function (on validation)
@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    tot_l = tot_a = n = 0

    for batch in loader:
        for k, v in batch.items():
            if isinstance(v, torch.Tensor): batch[k]=v.to(device)

        logits = model(batch["ehr"])
        loss   = F.cross_entropy(logits, batch["label"])        
        acc    = accuracy_from_logits(logits, batch["label"])

        tot_l += loss.item()
        tot_a += acc
        n += 1

    return tot_l/max(1,n), tot_a/max(1,n)

# 4. training for 1 epoch
def train_one_epoch(model, loader, opt):
    model.train()
    run_l = run_a = total_samples = 0

    for i, batch in enumerate(loader, 1):
        for k,v in batch.items():
            if isinstance(v, torch.Tensor): 
                batch[k] = v.to(device)

        opt.zero_grad(set_to_none=True)

        logits = model(batch["ehr"])
        loss   = F.cross_entropy(logits, batch["label"])
        
        loss.backward()
        opt.step()

        acc = accuracy_from_logits(logits.detach(), batch["label"])
        bs = batch["label"].size(0) # might change for the final batch

        run_l += loss.item()
        run_a += acc * bs # weighted acc
        total_samples += bs

        # calculate results
        run_loss = run_l / max(1, i)
        run_acc  = run_a / max(1, total_samples)

    return run_loss, run_acc, total_samples

# 5. operate training
def training(global_state, csvpath, labelcol, n_classes, head_path):
    # 1. Build model
    client_model = build_local_model(global_state, n_classes, head_path)
    
    model = client_model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr = LR, weight_decay = WD)

    # 2. build dataloder and load dataset
    full = TabularOnlyDataset(csvpath, labelcol)

    N = len(full)
    idx = np.arange(N)
    np.random.shuffle(idx)

    cut = int(N*(1.0-VAL_RATIO))
    tr_idx, va_idx = idx[:cut], idx[cut:]

    train_ds = torch.utils.data.Subset(full, tr_idx)
    val_ds   = torch.utils.data.Subset(full, va_idx)

    train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True )
    val_loader   = DataLoader(val_ds,   batch_size=BATCH, shuffle=False)

    # d_text = next(iter(train_loader))["ehr"].shape[1]  # Table feature dimensions

    # 3. train
    os.makedirs("runs/exp_final", exist_ok = True)
    best_path="runs/exp_final/best.pth"

    best = -1.0
    sample_count = 0
    best_global_state = global_state


    for epoch in range(1, EPOCHS+1):        
        tr_l,tr_a, tr_sp = train_one_epoch(model, train_loader, optimizer)
        va_l,va_a = evaluate(model, val_loader)
        '''
        print(f"\nEpoch {epoch}")
        print(f" -> train loss={tr_l:.4f} acc={tr_a:.4f}")
        print(f" ->   val loss={va_l:.4f} acc={va_a:.4f}")
        '''
        sample_count += tr_sp

        float_va_a = float(va_a)
        if float_va_a > best:
            best = float_va_a
            # save HEAD ONLY, for privacy-preserving            
            torch.save(model.head.state_dict(), head_path)
            # record the global encoder for returning
            best_global_state = model.enc.state_dict()
            
            # optional saving
            torch.save({"epoch":epoch,"model":model.state_dict()}, best_path)
            print(f" [best updated] {best:.4f} -> {best_path}")

    # 4. Return only tabular encoder weights
    # 4.1 filter, only keeps tabular_encoder, drop image encoder and fusion
    # 4.2 To return paras in the form of:
    # "tabular_enc.net.0.weight"
    # "tabular_enc.net.0.bias"
    # instead of:
    # "net.0.weight"
    # "net.0.bias" when using: model.enc.tabular_enc.state_dict()
    # this is because the FedAng calculation in server need the prefix (like: "tabular_enc", "inage_enc", ...)
    updated_state = {}
    for key, value in best_global_state.items():
        if key.startswith("tabular_enc"):
            updated_state[key] = value    

    # Return the filtered encoder state and the sample count
    return updated_state, sample_count

