
import torch
import torch.nn as nn
import torch.nn.functional as F


from multihead_model import MultiHeadModel
from config import D_TABULAR, D_EMBEDDING, D_FUSION


import os, random, numpy as np, pandas as pd
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# ==== Config ====
CSV_PATH  = "tabular_dataset/diabetes_012_ready_to_model.csv"  # CSV
LABEL_COL = "Diabetes_012"                                     # Label column name

VAL_RATIO = 0.2
EPOCHS    = 5
BATCH     = 64
LR        = 3e-4
WD        = 1e-4
LOG_EVERY = 50

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False
set_seed(42)

# === from config.py ===
from config import D_EMBEDDING, D_FUSION, D_TABULAR
try:
    import tabular_info
    N_CLASSES = tabular_info.n_tabular_classes
except:
    N_CLASSES = 2

#Dataset
class TabularOnlyDataset(Dataset):
    """
    Only table features + labels are read.
    """
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


# DataLoader
full = TabularOnlyDataset(CSV_PATH, LABEL_COL)

N = len(full)
idx = np.arange(N)
np.random.shuffle(idx)

cut = int(N*(1.0-VAL_RATIO))
tr_idx, va_idx = idx[:cut], idx[cut:]

train_ds = torch.utils.data.Subset(full, tr_idx)
val_ds   = torch.utils.data.Subset(full, va_idx)

train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True )
val_loader   = DataLoader(val_ds,   batch_size=BATCH, shuffle=False)

d_text = next(iter(train_loader))["ehr"].shape[1]  # Table feature dimensions


def ensure_img(batch, img_shape=(1, 32, 32)):
    if batch["img"] is None:
        B = batch["ehr"].size(0)
        batch["img"] = torch.zeros(B, *img_shape, dtype=batch["ehr"].dtype, device=batch["ehr"].device)
    return batch


model = MultiHeadModel(
    d_tabular   = D_TABULAR,
    d_embedding = D_EMBEDDING,
    d_fusion    = D_FUSION,
    n_tabular_classes = N_CLASSES,
    n_image_classes   = None,
    n_multi_classes   = None
).to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
# loss_fn   = nn.CrossEntropyLoss()

'''
logits = model(task_type="tabular", x_text = batch_x)
loss = F.cross_entropy(logits, batch_y)
'''



def accuracy_from_logits(logits, labels):
    return (logits.argmax(1) == labels).float().mean().item()

@torch.no_grad()
def evaluate(model, loader):
    model.eval(); tot_l=tot_a=n=0
    for batch in loader:
        for k,v in batch.items():
            if isinstance(v, torch.Tensor): batch[k]=v.to(device)
        batch = ensure_img(batch)  # Placeholder
        '''
        logits = model(batch["ehr"], batch["img"])
        loss   = loss_fn(logits, batch["label"])
        '''
        logits = model(task_type="tabular", x_text = batch["ehr"])
        loss   = F.cross_entropy(logits, batch["label"])
        
        acc    = accuracy_from_logits(logits, batch["label"])
        tot_l += loss.item(); tot_a += acc; n += 1
    return tot_l/max(1,n), tot_a/max(1,n)

def train_one_epoch(model, loader, opt):
    model.train()
    run_l=run_a=n=0
    for i,batch in enumerate(loader,1):
        for k,v in batch.items():
            if isinstance(v, torch.Tensor): batch[k]=v.to(device)
        # batch = ensure_img(batch)  # Placeholder

        opt.zero_grad(set_to_none=True)
        '''
        logits = model(batch["ehr"], batch["img"])
        loss   = loss_fn(logits, batch["label"])
        '''
        logits = model(task_type="tabular", x_text = batch["ehr"])
        loss   = F.cross_entropy(logits, batch["label"])
        
        loss.backward()
        opt.step()

        acc = accuracy_from_logits(logits.detach(), batch["label"])
        run_l += loss.item(); run_a += acc; n += 1
        if i % LOG_EVERY == 0:
            print(f"step {i:>5d}: loss={run_l/n:.4f} acc={run_a/n:.4f}")
    return run_l/max(1,n), run_a/max(1,n)



os.makedirs("runs/exp_final", exist_ok=True)
best=-1.0; best_path="runs/exp_final/best.pth"

for epoch in range(1, EPOCHS+1):
    print(f"\nEpoch {epoch}")
    tr_l,tr_a = train_one_epoch(model, train_loader, optimizer)
    va_l,va_a = evaluate(model, val_loader)
    print(f" -> train loss={tr_l:.4f} acc={tr_a:.4f}")
    print(f" ->   val loss={va_l:.4f} acc={va_a:.4f}")
    if va_a > best:
        best = va_a
        torch.save({"epoch":epoch,"model":model.state_dict()}, best_path)
        print(f" [best updated] {best:.4f} -> {best_path}")



