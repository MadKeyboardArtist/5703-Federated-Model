# federated_agg_experiment.py (simple version)

# ===== imports & fixed paths =====
import os, sys, json, time, random
from collections import defaultdict
from typing import Dict, List, Tuple, Any

BASE_DIR = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(BASE_DIR, "past_files"))  # 让 past_files 可导入
try:
    from past_files.multihead_model import MultiHeadModel
except Exception:
    from past_files.multihead_model import MultiClientModel as MultiHeadModel

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms

# ==================== Config ====================
NUM_CLIENTS = 3
ALPHA = 0.5
ROUNDS = 5
LOCAL_EPOCHS = 1
LR = 1e-3
BATCH = 32
VAL_BATCH = 64

FEDPROX_MU = 0.001
FADAM_ETA = 0.1
FADAM_BETA1, FADAM_BETA2 = 0.9, 0.99
FADAM_EPS = 1e-8

TRAIN_DIR = os.path.join(BASE_DIR, "image_dataset", "image_dataset_1", "train")
VAL_DIR   = os.path.join(BASE_DIR, "image_dataset", "image_dataset_1", "val")

# ==================== Transforms ====================
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

# ==================== Utils ====================
def seed_everything(seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def dirichlet_split(indices, num_clients=5, alpha=1.0):
    indices = np.array(indices); np.random.shuffle(indices)
    props = np.random.dirichlet([alpha]*num_clients)
    cuts = (np.cumsum(props)*len(indices)).astype(int)[:-1]
    return [p.tolist() for p in np.split(indices, cuts)]

# ==================== Data ====================
class MMImageTabular(Dataset):
    def __init__(self, image_folder: datasets.ImageFolder, indices, tab_lookup=None, dummy_dim: int = 21):
        self.base = image_folder; self.indices = list(indices)
        self.tab_lookup = tab_lookup; self.dummy_dim = dummy_dim
    def __len__(self): return len(self.indices)
    def __getitem__(self, i):
        idx = self.indices[i]
        img, y = self.base[idx]
        xt = torch.zeros(self.dummy_dim, dtype=torch.float32)
        return img, xt, y

def build_model_fn(n_classes: int):
    model = MultiHeadModel(n_image_classes=n_classes)
    for p in model.parameters(): p.requires_grad = False
    if hasattr(model, "head_image"): 
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
        dl = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=False)
        client_loaders.append(dl)
    return client_loaders, len(img_ds.classes)

def build_val_loader_fn(batch_size=64):
    img_val = datasets.ImageFolder(VAL_DIR, transform=transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ]))
    indices = np.arange(len(img_val)).tolist()
    ds = MMImageTabular(img_val, indices, tab_lookup=None)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False)
    return dl, len(img_val.classes)

# ==================== Train & Eval ====================
def local_train(model, loader, device, epochs=1, lr=1e-3, strategy="fedavg", global_state=None, mu=0.0):
    model = model.to(device); model.train()
    opt = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    total_loss, n_seen = 0.0, 0
    global_params = {k: v.to(device) for k, v in global_state.items()} if (strategy=="fedprox" and global_state) else None

    for _ in range(epochs):
        for imgs, tabs, ys in loader:
            imgs, ys = imgs.to(device), ys.to(device)
            logits = model(task_type="image", x_img=imgs)
            ce = F.cross_entropy(logits, ys)
            prox = 0.0
            if global_params is not None:
                for name, p in model.named_parameters():
                    if p.requires_grad and name in global_params:
                        prox += 0.5 * mu * torch.norm(p - global_params[name])**2
            loss = ce + prox
            opt.zero_grad(); loss.backward(); opt.step()
            bs = ys.size(0); total_loss += loss.item() * bs; n_seen += bs

    cpu_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    return cpu_state, n_seen, (total_loss / max(1, n_seen))

@torch.no_grad()
def evaluate(model, loader, device):
    model = model.to(device); model.eval()
    total, correct, total_loss = 0, 0, 0.0
    for imgs, tabs, ys in loader:
        imgs, ys = imgs.to(device), ys.to(device)
        logits = model(task_type="image", x_img=imgs)
        loss = F.cross_entropy(logits, ys)
        pred = logits.argmax(dim=1)
        correct += (pred == ys).sum().item()
        total += ys.size(0); total_loss += loss.item() * ys.size(0)
    return {"acc": correct / max(1,total), "loss": total_loss / max(1,total), "n": total}

# ==================== Aggregations ====================
StateDict = Dict[str, torch.Tensor]

def _selective_fedavg(client_states: List[StateDict], client_weights: List[float]) -> StateDict:
    grouped = defaultdict(list)
    for st, w in zip(client_states, client_weights):
        for k, v in st.items(): grouped[k].append((v, float(w)))
    out: StateDict = {}
    for k, ups in grouped.items():
        tw = sum(w for _, w in ups); acc = None
        for v, w in ups:
            scaled = v * (w / max(1.0, tw))
            acc = scaled if acc is None else acc + scaled
        out[k] = acc
    return out

def aggregate_fedavg(global_model: torch.nn.Module, client_states: List[StateDict],
                     client_weights: List[float], server_state: Dict[str, Any] | None = None, **_) -> Tuple[StateDict, Dict[str, Any]]:
    try:
        total = float(sum(client_weights))
        keys = client_states[0].keys()
        agg = {k: torch.zeros_like(client_states[0][k]) for k in keys}
        for st, w in zip(client_states, client_weights):
            scale = float(w) / max(1.0, total)
            for k in keys: agg[k] += st[k] * scale
        return agg, (server_state or {})
    except Exception:
        return _selective_fedavg(client_states, client_weights), (server_state or {})

def aggregate_fedprox(global_model, client_states, client_weights, server_state=None, **_):
    return aggregate_fedavg(global_model, client_states, client_weights, server_state)

def aggregate_fedadam(global_model, client_states, client_weights, server_state=None,
                      eta=0.1, beta1=0.9, beta2=0.99, eps=1e-8, **_) -> Tuple[StateDict, Dict[str, Any]]:
    if server_state is None: server_state = {}
    target, _ = aggregate_fedavg(global_model, client_states, client_weights, server_state)
    if server_state.get("t") is None:
        server_state["m"] = {k: torch.zeros_like(v) for k, v in target.items()}
        server_state["v"] = {k: torch.zeros_like(v) for k, v in target.items()}
        server_state["t"] = 0
    cur = global_model.state_dict(); m, v = server_state["m"], server_state["v"]; t = server_state["t"] + 1
    new_state: StateDict = {}
    for k in target.keys():
        g = target[k] - cur[k]
        m[k] = beta1 * m[k] + (1 - beta1) * g
        v[k] = beta2 * v[k] + (1 - beta2) * (g * g)
        m_hat = m[k] / (1 - (beta1 ** t)); v_hat = v[k] / (1 - (beta2 ** t))
        new_state[k] = cur[k] + eta * m_hat / (torch.sqrt(v_hat) + eps)
    server_state["t"] = t
    return new_state, server_state

AGG_FNS = {"fedavg": aggregate_fedavg, "fedprox": aggregate_fedprox, "fedadam": aggregate_fedadam}

# ==================== Runner ====================
def run_experiment(strategy: str, num_clients=NUM_CLIENTS, alpha=ALPHA, rounds=ROUNDS,
                   local_epochs=LOCAL_EPOCHS, lr=LR, mu=FEDPROX_MU,
                   eta=FADAM_ETA, beta1=FADAM_BETA1, beta2=FADAM_BETA2, eps=FADAM_EPS, seed=42):
    seed_everything(seed); os.makedirs("runs", exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n===== ▶ Strategy: {strategy} | device: {device} =====")

    client_loaders, n_classes_a = build_client_loaders_fn(num_clients=num_clients, alpha=alpha, batch_size=BATCH)
    val_loader, n_classes_b = build_val_loader_fn(batch_size=VAL_BATCH); assert n_classes_a == n_classes_b
    n_classes = n_classes_a

    global_model = build_model_fn(n_classes)
    print(f"✅ Clients: {len(client_loaders)}; val size: {len(val_loader.dataset)}")

    metrics_path = os.path.join("runs", f"{strategy}_metrics.jsonl"); open(metrics_path, "w").close()
    init_metric = evaluate(global_model, val_loader, device)
    print(f"🌱 Init | val_acc={init_metric['acc']:.4f}  val_loss={init_metric['loss']:.4f}")
    with open(metrics_path, "a") as f: f.write(json.dumps({"round": 0, **init_metric}) + "\n")

    server_state: Dict[str, Any] = {}
    for r in range(1, rounds + 1):
        t0 = time.time()
        client_states, client_weights, client_train_losses = [], [], []
        global_cpu = {k: v.detach().cpu() for k, v in global_model.state_dict().items()}

        for dl in client_loaders:
            cm = build_model_fn(n_classes); cm.load_state_dict(global_model.state_dict(), strict=True)
            st, n_i, tr_loss = local_train(cm, dl, device, epochs=local_epochs, lr=lr,
                                           strategy=strategy, global_state=global_cpu if strategy=="fedprox" else None, mu=mu)
            client_states.append(st); client_weights.append(n_i); client_train_losses.append(tr_loss)

        next_state, server_state = AGG_FNS[strategy](global_model, client_states, client_weights,
                                                     server_state=server_state, eta=eta, beta1=beta1, beta2=beta2, eps=eps)
        global_model.load_state_dict(next_state)

        metric = evaluate(global_model, val_loader, device); dt = time.time() - t0
        print(f"🌀 Round {r:02d} [{strategy}] | train_loss={np.mean(client_train_losses):.4f} "
              f"| val_acc={metric['acc']:.4f} val_loss={metric['loss']:.4f} | time={dt:.1f}s")
        with open(metrics_path, "a") as f:
            f.write(json.dumps({"round": r, "train_loss": float(np.mean(client_train_losses)), **metric, "time": dt}) + "\n")

    torch.save(global_model.state_dict(), os.path.join("runs", f"{strategy}_final.pth"))
    print(f"💾 Saved to runs/{strategy}_final.pth"); print(f"📈 Metrics log -> {metrics_path}")

# ==================== Main ====================
if __name__ == "__main__":
    print("🚀 Start federated aggregation benchmark (FedAvg / FedProx / FedAdam)")
    for st in ["fedavg", "fedprox", "fedadam"]:
        run_experiment(strategy=st)
    print("✅ All strategies finished.")
