
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

    def __len__(self): return len(self.y)

    def __getitem__(self, i):
        ehr   = torch.from_numpy(self.X[i])
        label = torch.tensor(self.y[i], dtype = torch.long)
        return {"ehr": ehr, "label": label}

# 3. evaluation function (on validation)
def evaluation_in_training (model, val_loader):
    # 1. build the model
    pass
    # 2. build dataloder and load dataset
    pass
    # 3. calculate acc
    model.eval()
    tot_l = tot_a = n = 0

    for batch in val_loader:
        for k, v in batch.items():
            if isinstance(v, torch.Tensor): batch[k]=v.to(device)

        logits = model(batch["ehr"])
        loss   = F.cross_entropy(logits, batch["label"])        
        acc    = accuracy_from_logits(logits, batch["label"])

        tot_l += loss.item()
        tot_a += acc
        n += 1

    # results calculation
    site_loss = float(tot_l/max(1,n))
    site_acc  = float(tot_a/max(1,n))
    site_sample_num = n

    return site_loss, site_acc, site_sample_num

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

# 4.1 simple training for 1 epoch
def train_one_epoch_simple(model, loader, opt):
    model.train()
    total_samples = 0

    for _, batch in enumerate(loader, 1):
        for k,v in batch.items():
            if isinstance(v, torch.Tensor): 
                batch[k] = v.to(device)

        opt.zero_grad(set_to_none=True)

        logits = model(batch["ehr"])
        loss   = F.cross_entropy(logits, batch["label"])
        
        loss.backward()
        opt.step()

        bs = batch["label"].size(0) # might change for the final batch
        total_samples += bs

    return total_samples

# 5. operate training
def training(site_name,
             global_state, 
             freeze_global,
             train_set_path, 
             val_set_path, 
             labelcol, 
             n_classes, 
             newest_head_path, 
             current_best_head_path):
    # 0.
    print("{:s}: tabular site training START".format(site_name))

    # 1. Build model
    client_model = build_local_model(global_state, n_classes, newest_head_path)
    
    model = client_model.to(device)
    
    # 2. freeze_global or not
    if freeze_global == True:
        # 2. Freeze the encoder (shared global part)
        for param in model.enc.parameters():
            param.requires_grad = False
        # 3. Set optimizer to only optimize the local head
        optimizer = torch.optim.AdamW(model.head.parameters(), lr=LR, weight_decay=WD)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr = LR, weight_decay = WD)

    # 3. build dataloder and load dataset
    # Build datasets directly from the .csv
    train_ds = TabularOnlyDataset(train_set_path, labelcol)
    val_ds   = TabularOnlyDataset(val_set_path,   labelcol)

    # wrap with DataLoader
    train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH, shuffle=False)

    # 4. train
    best = -1.0 # best acc
    sample_count = 0
    best_head_state = None

    for epoch in range(1, EPOCHS + 1):        
        #tr_l,tr_a, tr_sp = train_one_epoch_simple(model, train_loader, optimizer)
        tr_sp = train_one_epoch_simple(model, train_loader, optimizer)
        va_l, va_a, va_sp = evaluation_in_training(model, val_loader)
        '''
        print(f"\nEpoch {epoch}")
        print(f" -> train loss={tr_l:.4f} acc={tr_a:.4f}")
        print(f" ->   val loss={va_l:.4f} acc={va_a:.4f}")
        '''
        sample_count += tr_sp

        float_va_a = float(va_a)
        # acc based check point:
        # save the model when it's more accurate on validation set (i.e., higher % of correct predictions).
        # Common in classification tasks.
        # Use when:
        # 1. evaluation metric is classification accuracy
        # 2. the goal is to maximize correct predictions, regardless of confidence
        # 3. it is okay with a slightly higher loss as long as accuracy improves
        if float_va_a > best:
            best = float_va_a
            # 1. keep this HEAD for outcome record
            # HEAD ONLY, for privacy-preserving            
            best_head_state = model.head.state_dict()
            # safe to save. (will be replaced if applying re-training)
            torch.save(best_head_state, current_best_head_path)
            
            # 2. record the best ckpt results (epoch - level)
            ckpt_eva_results = {
                       "val_loss": va_l,
                       "val_acc" : va_a,
                       "num_samples": va_sp
                       }
            print(f" [best updated] acc: {best:.4f}")

    print("{:s}: tabular site training DONE".format(site_name))

    # 5. Return only tabular encoder weights
    # sent back the newest global encodeers, instead of the best
    # Because FedAvg is about aggregating gradients/weight updates, not cherry-picking local checkpoints.
    # If each site sent back different “best” encoders (picked at different epochs), 
    # the updates would be inconsistent 
    # and the optimization wouldn’t converge properly.
    updated_state = model.enc.state_dict() # the final global state

    # 6. record the newest local heads for the next federated training round
    torch.save(model.head.state_dict(), newest_head_path) # the final local state

    # 7. Return the encoder state, sample count, and best ckpt
    return updated_state, sample_count, ckpt_eva_results


def training_fed_prox(
        site_name,
        global_state,           
        freeze_global,
        train_set_path, 
        val_set_path, 
        labelcol, 
        n_classes, 
        newest_head_path, 
        current_best_head_path,
        fed_prox_agg = False,
        mu = 0.05
        ):
    
    # 0.
    print(f"{site_name}: tabular site training START (FedProx μ={mu})")

    # 1. Build model
    client_model = build_local_model(global_state, n_classes, newest_head_path)
    model = client_model.to(device)

    # 2. freeze_global or not
    if freeze_global:
        for p in model.enc.parameters():
            p.requires_grad = False
        optimizer = torch.optim.AdamW(model.head.parameters(), lr=LR, weight_decay=WD)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)

    # 3. Data
    train_ds = TabularOnlyDataset(train_set_path, labelcol)
    val_ds   = TabularOnlyDataset(val_set_path, labelcol)
    train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=BATCH, shuffle=False)

    # 4. Prepare FedProx anchor
    prox_anchor = {k: v.detach().clone().to(device) for k, v in global_state.items()}

    best = -1.0
    sample_count = 0
    best_head_state = None

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_samples = 0

        for batch in train_loader:
            for k, v in batch.items():
                if torch.is_tensor(v):
                    batch[k] = v.to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(batch["ehr"])
            loss = F.cross_entropy(logits, batch["label"])

            # ===== FedProx proximal term =====
            if fed_prox_agg:
                prox_loss = torch.zeros(1, device=device)
                for name, param in model.named_parameters():
                    if not param.requires_grad: 
                        continue
                    # optional: skip head parameters (usually "head" in name)
                    if "head" in name.lower():
                        continue
                    if name in prox_anchor:
                        diff = param - prox_anchor[name]
                        prox_loss += (diff * diff).sum()
                loss = loss + (mu / 2.0) * prox_loss
            # =================================

            loss.backward()
            optimizer.step()

            total_samples += batch["label"].size(0)

        # Validation
        va_l, va_a, va_sp = evaluation_in_training(model, val_loader)
        sample_count += total_samples

        if float(va_a) > best:
            best = float(va_a)
            best_head_state = model.head.state_dict()
            torch.save(best_head_state, current_best_head_path)
            ckpt_eva_results = {"val_loss": va_l, "val_acc": va_a, "num_samples": va_sp}
            print(f" [best updated] acc: {best:.4f}")

    print(f"{site_name}: tabular site training DONE (FedProx)")

    # 5. Return updated encoder
    updated_state = model.enc.state_dict()
    torch.save(model.head.state_dict(), newest_head_path)

    return updated_state, sample_count, ckpt_eva_results
