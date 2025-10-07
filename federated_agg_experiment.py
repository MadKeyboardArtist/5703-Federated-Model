# federated_agg_experiment.py
from multihead_model import MultiHeadModel

import os, json, time, random
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms


#General experimental setup

NUM_CLIENTS = 3
ALPHA = 0.5          # Dirichlet non-IID 
ROUNDS = 5
LOCAL_EPOCHS = 1
LR = 1e-3
BATCH = 32
VAL_BATCH = 64
USE_DUMMY_TABULAR = True   # image branch


FEDPROX_MU = 0.001
FADAM_ETA = 0.1
FADAM_BETA1, FADAM_BETA2 = 0.9, 0.99
FADAM_EPS = 1e-8

# path
TRAIN_DIR = "image_dataset/split/train"
VAL_DIR   = "image_dataset/split/val"

#Visual Preprocessing
IMAGENET_MEAN=[0.485,0.456,0.406]; IMAGENET_STD=[0.229,0.224,0.225]
IMG_SIZE = 224
img_tfms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(10),
    transforms.ColorJitter(0.1,0.1,0.1,0.05),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])


# tool

def seed_everything(seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def dirichlet_split(indices, num_clients=5, alpha=1.0):
    indices = np.array(indices)
    np.random.shuffle(indices)
    props = np.random.dirichlet([alpha]*num_clients)
    cuts = (np.cumsum(props)*len(indices)).astype(int)[:-1]
    parts = np.split(indices, cuts)
    return [p.tolist() for p in parts]


# data Packaging

class MMImageTabular(Dataset):
    """ImageFolder +（可选）表格向量；本实验仅用 image。"""
    def __init__(self, image_folder: datasets.ImageFolder, indices, tab_lookup=None, dummy_dim: int = 21):
        self.base = image_folder
        self.indices = list(indices)
        self.tab_lookup = tab_lookup
        self.dummy_dim = dummy_dim

    def __len__(self): return len(self.indices)

    def __getitem__(self, i):
        idx = self.indices[i]
        img, y = self.base[idx]
        if self.tab_lookup is not None:
            xt = torch.tensor(self.tab_lookup[idx], dtype=torch.float32)
        else:
            xt = torch.zeros(self.dummy_dim, dtype=torch.float32)
        return img, xt, y




def build_model_fn(n_classes: int):
    #  n_image_classes
    model = MultiHeadModel(n_image_classes=n_classes)

    # Freeze all, then enable head_image (you may also unfreeze image_enc for comparison).
    for p in model.parameters(): p.requires_grad = False
    if hasattr(model, "head_image") and isinstance(model.head_image, torch.nn.Module):
        for p in model.head_image.parameters(): p.requires_grad = True

    enc_flag = "✔" if hasattr(model, "image_enc") and any(p.requires_grad for p in model.image_enc.parameters()) else "✖"
    print(f"🔧 trainable params: head_image | encoder: image_enc {enc_flag}")
    return model

def build_client_loaders_fn(num_clients=5, alpha=1.0, batch_size=32):
    img_ds = datasets.ImageFolder(TRAIN_DIR, transform=img_tfms)
    base_indices = np.arange(len(img_ds)).tolist()
    client_indices_list = dirichlet_split(base_indices, num_clients=num_clients, alpha=alpha)

    client_loaders = []
    for idxs in client_indices_list:
        ds = MMImageTabular(img_ds, idxs, tab_lookup=None, dummy_dim=21)
        dl = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=False)  # macOS更稳
        client_loaders.append(dl)
    n_classes = len(img_ds.classes)
    return client_loaders, n_classes

def build_val_loader_fn(batch_size=64):
    img_val = datasets.ImageFolder(VAL_DIR, transform=transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ]))
    val_indices = np.arange(len(img_val)).tolist()
    ds = MMImageTabular(img_val, val_indices, tab_lookup=None)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False)
    return dl, len(img_val.classes)


# Training, Evaluation, and Aggregation

def local_train(model, loader, device, epochs=1, lr=1e-3, strategy="fedavg", global_state=None, mu=0.0):
    model = model.to(device)
    model.train()
    opt = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    total_loss, n_seen = 0.0, 0

    global_params = None
    if strategy == "fedprox" and global_state is not None:
        global_params = {k: v.to(device) for k, v in global_state.items()}

    for _ in range(epochs):
        for imgs, tabs, ys in loader:
            imgs, ys = imgs.to(device), ys.to(device)
            logits = model(task_type="image", x_img=imgs)  # 只用 image 分支
            ce = F.cross_entropy(logits, ys)

            prox = 0.0
            if global_params is not None:
                for name, p in model.named_parameters():
                    if p.requires_grad and name in global_params:
                        prox = prox + 0.5 * mu * torch.norm(p - global_params[name])**2

            loss = ce + prox
            opt.zero_grad()
            loss.backward()
            opt.step()

            bs = ys.size(0)
            total_loss += loss.item() * bs
            n_seen += bs

    avg_loss = total_loss / max(1, n_seen)
    cpu_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    return cpu_state, n_seen, avg_loss

@torch.no_grad()
def evaluate(model, loader, device):
    model = model.to(device)
    model.eval()
    total, correct, total_loss = 0, 0, 0.0
    for imgs, tabs, ys in loader:
        imgs, ys = imgs.to(device), ys.to(device)
        logits = model(task_type="image", x_img=imgs)
        loss = F.cross_entropy(logits, ys)
        pred = logits.argmax(dim=1)
        correct += (pred == ys).sum().item()
        total += ys.size(0)
        total_loss += loss.item() * ys.size(0)
    return {"acc": correct / max(1,total), "loss": total_loss / max(1,total), "n": total}

def fedavg_aggregate_states(client_states, client_weights):
    total = float(sum(client_weights))
    agg = {k: torch.zeros_like(v) for k, v in client_states[0].items()}
    for st, w in zip(client_states, client_weights):
        scale = float(w) / max(1.0, total)
        for k, v in st.items():
            agg[k] += v * scale
    return agg

def server_opt_step_fedadam(global_model, agg_state, opt_state, eta=0.1, beta1=0.9, beta2=0.99, eps=1e-8):
    if opt_state.get("t", None) is None:
        opt_state["m"] = {k: torch.zeros_like(v) for k, v in agg_state.items()}
        opt_state["v"] = {k: torch.zeros_like(v) for k, v in agg_state.items()}
        opt_state["t"] = 0

    global_state = global_model.state_dict()
    m, v, t = opt_state["m"], opt_state["v"], opt_state["t"] + 1

    new_state = {}
    for k in global_state.keys():
        g = agg_state[k] - global_state[k]  # FedOpt 伪梯度：期望的移动方向
        m[k] = beta1 * m[k] + (1 - beta1) * g
        v[k] = beta2 * v[k] + (1 - beta2) * (g * g)
        m_hat = m[k] / (1 - (beta1 ** t))
        v_hat = v[k] / (1 - (beta2 ** t))
        new_state[k] = global_state[k] + eta * m_hat / (torch.sqrt(v_hat) + eps)

    opt_state["t"] = t
    global_model.load_state_dict(new_state)
    return opt_state


# single experiment

def run_experiment(strategy: str,
                   num_clients=NUM_CLIENTS,
                   alpha=ALPHA,
                   rounds=ROUNDS,
                   local_epochs=LOCAL_EPOCHS,
                   lr=LR,
                   mu=FEDPROX_MU,
                   eta=FADAM_ETA,
                   beta1=FADAM_BETA1,
                   beta2=FADAM_BETA2,
                   eps=FADAM_EPS,
                   seed=42):

    seed_everything(seed)
    os.makedirs("runs", exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n===== ▶ Strategy: {strategy} | device: {device} =====")

    client_loaders, n_classes_a = build_client_loaders_fn(num_clients=num_clients, alpha=alpha, batch_size=BATCH)
    val_loader, n_classes_b = build_val_loader_fn(batch_size=VAL_BATCH)
    assert n_classes_a == n_classes_b, "train/val 类别数不一致"
    n_classes = n_classes_a

    global_model = build_model_fn(n_classes)
    print(f"✅ Clients: {len(client_loaders)}; val size: {len(val_loader.dataset)}")

    metrics_path = os.path.join("runs", f"{strategy}_metrics.jsonl")
    open(metrics_path, "w").close()

    init_metric = evaluate(global_model, val_loader, device)
    print(f"🌱 Init | val_acc={init_metric['acc']:.4f}  val_loss={init_metric['loss']:.4f}")
    with open(metrics_path, "a") as f:
        f.write(json.dumps({"round": 0, **init_metric}) + "\n")

    server_state = {}  # FedAdam

    for r in range(1, rounds + 1):
        t0 = time.time()
        client_states, client_weights, client_train_losses = [], [], []

        global_cpu = {k: v.detach().cpu() for k, v in global_model.state_dict().items()}

        for dl in client_loaders:
            cm = build_model_fn(n_classes)
            cm.load_state_dict(global_model.state_dict(), strict=True)
            state, n_i, tr_loss = local_train(
                cm, dl, device,
                epochs=local_epochs, lr=lr,
                strategy=strategy,
                global_state=global_cpu if strategy == "fedprox" else None,
                mu=mu
            )
            client_states.append(state)
            client_weights.append(n_i)
            client_train_losses.append(tr_loss)

        agg = fedavg_aggregate_states(client_states, client_weights)

        if strategy == "fedadam":
            server_state = server_opt_step_fedadam(global_model, agg, server_state,
                                                   eta=eta, beta1=beta1, beta2=beta2, eps=eps)
        else:
            global_model.load_state_dict(agg)

        metric = evaluate(global_model, val_loader, device)
        dt = time.time() - t0
        print(f"🌀 Round {r:02d} [{strategy}] | train_loss={np.mean(client_train_losses):.4f} "
              f"| val_acc={metric['acc']:.4f} val_loss={metric['loss']:.4f} | time={dt:.1f}s")

        with open(metrics_path, "a") as f:
            f.write(json.dumps({"round": r, "train_loss": float(np.mean(client_train_losses)), **metric, "time": dt}) + "\n")

    torch.save(global_model.state_dict(), os.path.join("runs", f"{strategy}_final.pth"))
    print(f"💾 Saved to runs/{strategy}_final.pth")
    print(f"📈 Metrics log -> {metrics_path}")


# main program

if __name__ == "__main__":
    print("🚀 Start federated aggregation benchmark (FedAvg / FedProx / FedAdam)")

    
    strategies = ["fedavg", "fedprox", "fedadam"]

    for st in strategies:
        run_experiment(
            strategy=st,
            num_clients=NUM_CLIENTS,
            alpha=ALPHA,
            rounds=ROUNDS,
            local_epochs=LOCAL_EPOCHS,
            lr=LR,
            mu=FEDPROX_MU,
            eta=FADAM_ETA,
            beta1=FADAM_BETA1,
            beta2=FADAM_BETA2,
            eps=FADAM_EPS,
            seed=42
        )

    print("✅ All strategies finished.")
