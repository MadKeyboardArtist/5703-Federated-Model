import numpy as np
from sklearn.metrics import (roc_auc_score, average_precision_score, accuracy_score,
    f1_score, precision_score, recall_score, confusion_matrix, log_loss)

def brier_score(y_true, y_prob):
    y_true = np.asarray(y_true).astype(float); y_prob = np.asarray(y_prob).astype(float)
    return float(np.mean((y_prob - y_true) ** 2))

def ece_binary(y_true, y_prob, n_bins=15):
    y_true = np.asarray(y_true).astype(int); y_prob = np.asarray(y_prob).astype(float)
    bins = np.linspace(0, 1, n_bins+1); ece = 0.0
    for i in range(n_bins):
        idx = (y_prob >= bins[i]) & (y_prob < bins[i+1])
        if idx.sum() == 0: continue
        conf = y_prob[idx].mean()
        acc  = ((y_prob[idx] >= 0.5).astype(int) == y_true[idx]).mean()
        ece += abs(acc - conf) * (idx.sum()/len(y_true))
    return float(ece)

def compute_threshold(y_true, y_prob, method="youden"):
    from sklearn.metrics import roc_curve
    fpr, tpr, thr = roc_curve(y_true, y_prob)
    if method == "youden":
        j = tpr - fpr; return float(thr[np.argmax(j)])
    elif method.startswith("fix_recall="):
        tgt = float(method.split("=")[1]); idx = np.argmin(np.abs(tpr - tgt)); return float(thr[idx])
    return 0.5

def evaluate_binary(y_true, y_prob, threshold=0.5):
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float).clip(1e-7, 1-1e-7)
    y_pred = (y_prob >= threshold).astype(int)

    cm  = confusion_matrix(y_true, y_pred, labels=[0,1])
    tn, fp, fn, tp = cm.ravel()
    tpr = tp / max(tp+fn, 1); tnr = tn / max(tn+fp, 1)
    bal = (tpr + tnr) / 2.0
    denom = np.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))
    mcc = float((tp*tn - fp*fn)/denom) if denom > 0 else 0.0

    overall = dict(
        AUROC = float(roc_auc_score(y_true, y_prob)),
        AUPRC = float(average_precision_score(y_true, y_prob)),
        Accuracy = float(accuracy_score(y_true, y_pred)),
        F1_macro = float(f1_score(y_true, y_pred, average="macro")),
        F1_weighted = float(f1_score(y_true, y_pred, average="weighted")),
        Precision_macro = float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        Recall_macro = float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        BalancedAcc = float(bal),
        MCC = mcc,
        LogLoss = float(log_loss(y_true, y_prob)),
        Brier = brier_score(y_true, y_prob),
        ECE = ece_binary(y_true, y_prob),
        Threshold = float(threshold),
        N_pos = int(y_true.sum()),
        N_neg = int((1 - y_true).sum()),
        CM = cm.tolist()
    )
    per_class = []
    for cls in [0,1]:
        p = precision_score(y_true, y_pred, labels=[cls], average=None, zero_division=0)[0]
        r = recall_score(y_true, y_pred, labels=[cls], average=None, zero_division=0)[0]
        f = f1_score(y_true, y_pred, labels=[cls], average=None, zero_division=0)[0]
        per_class.append(dict(class_id=int(cls), precision=float(p), recall=float(r), f1=float(f),
                              support=int(np.sum(y_true==cls))))
    return overall, per_class
