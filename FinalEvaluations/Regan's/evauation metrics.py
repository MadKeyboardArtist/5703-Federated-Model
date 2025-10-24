
import os, json, numpy as np, torch
import torch.nn.functional as F

from evaluator_binary import evaluate_binary, compute_threshold
from evaluator_multiclass import evaluate_multiclass

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")



@torch.no_grad()
def collect_preds(model, loader, modality, device=DEVICE):
    model.eval()
    y_true, y_prob, sites = [], [], []

    for batch in loader:
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(device, non_blocking=True)

        if modality == "image":
            logits = model(task_type="image", x_img=batch["img"])
        elif modality == "tabular":
            logits = model(task_type="tabular", x_tab=batch["ehr"])
        elif modality == "multi":
            logits = model(task_type="multi", x_tab=batch["ehr"], x_img=batch["img"])
        else:
            raise ValueError("modality must be 'image' or 'tabular' or 'multi'")

        # Probability: Compatible with single/double/multiple logits
        prob = torch.sigmoid(logits).view(-1, 1) if logits.shape[-1] == 1 else F.softmax(logits, dim=-1)

        y_true.extend(batch["label"].long().cpu().numpy().tolist())

        sid = batch.get("client_id", None)
        if sid is None:
            sites.extend(["global"] * len(batch["label"]))
        elif isinstance(sid, torch.Tensor):
            sites.extend(sid.detach().cpu().numpy().tolist())
        else:
            sites.extend(list(sid))

        y_prob.extend(prob.detach().cpu().numpy().tolist())

    return np.array(y_true, int), np.array(y_prob, float), np.array(sites)



def macro_micro(vals, weights=None):
    arr = np.array(vals, float)
    macro = float(arr.mean()) if len(arr) else float("nan")
    micro = float(np.average(arr, weights=weights)) if (weights is not None and len(arr)) else macro
    std   = float(arr.std()) if len(arr) else float("nan")
    rng   = float(arr.max() - arr.min()) if len(arr) else float("nan")
    return {"macro": macro, "micro": micro, "std": std, "range": rng}

def bootstrap_ci(stat_fn, y_true, y_prob, n_boot=1000, alpha=0.05, seed=2025):
    rng = np.random.default_rng(seed)
    n   = len(y_true)
    stats = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        yt, yp = y_true[idx], y_prob[idx]
        try:
            s = stat_fn(yt, yp)
            if not np.isnan(s):
                stats.append(s)
        except Exception:
            # For instance, when resampling a binary classification yields a single category, AUROC/AP will report an error; this is skipped here.
            continue
    low, high = np.quantile(stats, [alpha/2, 1-alpha/2])
    return float(low), float(high)



def eval_binary_and_save(model, loader, modality, run_dir, split="test",
                         recall_target=0.80, do_ci=True):
    from sklearn.metrics import roc_auc_score, average_precision_score

    y_true, y_prob2, sites = collect_preds(model, loader, modality)
    y_prob = y_prob2[:, 0] if y_prob2.shape[1] == 1 else y_prob2[:, 1]  # Positive probability

    th_youden = compute_threshold(y_true, y_prob, method="youden")
    rep_youden, _ = evaluate_binary(y_true, y_prob, threshold=th_youden)

    th_rec   = compute_threshold(y_true, y_prob, method=f"fix_recall={recall_target}")
    rep_rec, _ = evaluate_binary(y_true, y_prob, threshold=th_rec)

    rep_050, _ = evaluate_binary(y_true, y_prob, threshold=0.5)

    # per-site
    per_site, sizes = {}, {}
    for s in np.unique(sites):
        m = (sites == s)
        if m.sum() < 2: continue
        th_s = compute_threshold(y_true[m], y_prob[m], method="youden")
        rep_s, _ = evaluate_binary(y_true[m], y_prob[m], threshold=th_s)
        per_site[str(s)] = rep_s
        sizes[str(s)]    = int(m.sum())

    aurocs = [per_site[k]["AUROC"] for k in per_site]
    auprcs = [per_site[k]["AUPRC"] for k in per_site]
    w      = np.array([sizes[k] for k in per_site], float) if len(per_site) else None

    agg = {
        "AUROC": macro_micro(aurocs, w),
        "AUPRC": macro_micro(auprcs, w),
        "sizes": sizes
    }

    ci = None
    if do_ci:
        ci = {
            "AUROC": bootstrap_ci(lambda yt, yp: roc_auc_score(yt, yp), y_true, y_prob),
            "AUPRC": bootstrap_ci(lambda yt, yp: average_precision_score(yt, yp), y_true, y_prob)
        }

    os.makedirs(os.path.join(run_dir, "metrics"), exist_ok=True)
    out = {
        "SPLIT": split,
        "TASK": "binary",
        "GLOBAL": {
            "youden": {"overall": rep_youden, "threshold": float(th_youden)},
            f"recall@{recall_target:.2f}": {"overall": rep_rec, "threshold": float(th_rec)},
            "thr=0.5": {"overall": rep_050, "threshold": 0.5}
        },
        "PER_SITE": per_site,
        "AGGREGATION": agg,
        "CI": ci
    }
    save_path = os.path.join(run_dir, "metrics", f"{split}_binary_summary.json")
    with open(save_path, "w") as f: json.dump(out, f, indent=2)
    print("Saved →", save_path)
    return out



def eval_multiclass_and_save(model, loader, modality, run_dir, split="test",
                             num_classes=5, do_ci=True):
    y_true, y_probC, sites = collect_preds(model, loader, modality)
    overall, per_class = evaluate_multiclass(y_true, y_probC, num_classes=num_classes, topk=5)

    per_site, sizes = {}, {}
    for s in np.unique(sites):
        m = (sites == s)
        if m.sum() < 2: continue
        ov_s, _ = evaluate_multiclass(y_true[m], y_probC[m], num_classes=num_classes, topk=5)
        per_site[str(s)] = ov_s
        sizes[str(s)]    = int(m.sum())

    auroc_macro_vals = [per_site[k]["AUROC_macro"] for k in per_site]
    auprc_macro_vals = [per_site[k]["AUPRC_macro"] for k in per_site]
    w = np.array([sizes[k] for k in per_site], float) if len(per_site) else None

    agg = {
        "AUROC_macro": macro_micro(auroc_macro_vals, w),
        "AUPRC_macro": macro_micro(auprc_macro_vals, w),
        "sizes": sizes
    }

    ci = None
    if do_ci:
        rng = np.random.default_rng(2025)
        n = len(y_true)
        pred = y_probC.argmax(axis=1)
        stats = []
        for _ in range(1000):
            idx = rng.integers(0, n, n)
            stats.append((pred[idx] == y_true[idx]).mean())
        low, high = np.quantile(stats, [0.025, 0.975])
        ci = {"Top1": (float(low), float(high))}

    os.makedirs(os.path.join(run_dir, "metrics"), exist_ok=True)
    out = {
        "SPLIT": split,
        "TASK": f"{num_classes}-class",
        "GLOBAL": {"overall": overall, "per_class": per_class},
        "PER_SITE": per_site,
        "AGGREGATION": agg,
        "CI": ci
    }
    save_path = os.path.join(run_dir, "metrics", f"{split}_{num_classes}class_summary.json")
    with open(save_path, "w") as f: json.dump(out, f, indent=2)
    print("Saved →", save_path)
    return out



#Evaluation Model Wrapper + Load Global Encoder

import torch, torch.nn as nn, os
from FederatedModel.federated_multihead_model import SharedEncoders, Head
from FederatedModel.model_config import D_TABULAR, D_EMBEDDING, D_FUSION

class FederatedEvalModel(nn.Module):
    def __init__(self, encoders: SharedEncoders):
        super().__init__()
        self.encoders = encoders
        self.heads = nn.ModuleDict()   # {site: ModuleDict({ "tabular": Head(...), "image": Head(...) })}
        self.active_site = None

    def set_site(self, site_id: str):
        assert site_id in self.heads, f"Unknown site_id={site_id}"
        self.active_site = site_id

    def forward(self, task_type: str, x_tab=None, x_img=None):
        assert self.active_site is not None, "Call set_site(site_id) before forward"
        hd = self.heads[self.active_site]
        if task_type == "tabular":
            z = self.encoders.forward_tabular(x_tab); head = hd["tabular"]
        elif task_type == "image":
            z = self.encoders.forward_image(x_img);   head = hd["image"]
        else:
            raise ValueError("modality must be 'tabular' or 'image'")
        return head(z)

def load_global_model(ckpt_path: str, device=DEVICE):
    enc = SharedEncoders(D_TABULAR, D_EMBEDDING, D_FUSION)
    model = FederatedEvalModel(enc)
    sd = torch.load(ckpt_path, map_location="cpu")
    sd = sd.get("model", sd)  # Compatible with {"model": state_dict} or directly state_dict
    model.load_state_dict(sd, strict=False)   # Permit loading encoders only
    model.to(device).eval()
    return model




# Mount the "overall optimal header" for each site

def attach_best_head(model, site_name, modality, n_classes, head_ckpt_path):
    # The input dimensions for both tabular and image head are D_EMBEDDING.
    d_in = D_EMBEDDING
    try:
        head = Head(d_in=d_in, n_classes=n_classes)
    except TypeError:
        head = Head(d_embedding=d_in, n_classes=n_classes)

    if site_name not in model.heads:
        model.heads[site_name] = nn.ModuleDict()
    model.heads[site_name][modality] = head

    if head_ckpt_path and os.path.exists(head_ckpt_path):
        payload = torch.load(head_ckpt_path, map_location="cpu")
        state = payload.get("state_dict", payload) if isinstance(payload, dict) else payload
        model.heads[site_name][modality].load_state_dict(state, strict=False)



#

import glob, os

def discover_sites_from_heads():
    sites=[]
    for p in glob.glob("overall_best_local_heads/*.pth"):
        name = os.path.splitext(os.path.basename(p))[0]   # e.g., 'tabular_1' or 'image_1'
        if name.startswith("tabular"):
            sites.append({
                "name": name, "modality":"tabular", "n_classes":2,
                "test_path":"tabular_dataset/diabetes_012_test.csv",
                "label_col":"Diabetes_012", "best_head":p
            })
        elif name.startswith("image"):
            # 这里按你的图像数据实际情况填：
            sites.append({
                "name": name, "modality":"image", "n_classes":5,
                "test_path":"image_dataset/test_meta.csv",
                "label_col":"label", "best_head":p
            })
    return sites

sites = discover_sites_from_heads()
assert len(sites)>0, "overall_best_local_heads 里没发现任何 head 文件"




