import numpy as np
from sklearn.metrics import (accuracy_score, top_k_accuracy_score, f1_score, precision_score,
    recall_score, confusion_matrix, log_loss, roc_auc_score, average_precision_score, matthews_corrcoef)
from sklearn.preprocessing import label_binarize

def ece_multiclass(y_true, y_prob, n_bins=15):
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float).clip(1e-7, 1-1e-7)
    conf = y_prob.max(axis=1); pred = y_prob.argmax(axis=1)
    bins = np.linspace(0,1,n_bins+1); ece=0.0
    for i in range(n_bins):
        if i < n_bins-1:
            idx = (conf>=bins[i]) & (conf<bins[i+1])
        else:
            idx = (conf>=bins[i]) & (conf<=bins[i+1])
        if idx.sum()==0: continue
        ece += abs((pred[idx]==y_true[idx]).mean() - conf[idx].mean()) * (idx.sum()/len(y_true))
    return float(ece)

def evaluate_multiclass(y_true, y_prob, num_classes, topk=5):
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    y_pred = y_prob.argmax(axis=1)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    recalls = np.diag(cm) / np.maximum(cm.sum(axis=1), 1)
    balanced = float(np.nanmean(np.nan_to_num(recalls)))
    mcc = float(matthews_corrcoef(y_true, y_pred))

    # OvR AUC/PR: Fall back to nan when class is missing
    from sklearn.preprocessing import label_binarize
    y_bin = label_binarize(y_true, classes=list(range(num_classes)))
    try:
        auroc_macro = float(roc_auc_score(y_bin, y_prob, average="macro", multi_class="ovr"))
        auroc_micro = float(roc_auc_score(y_bin, y_prob, average="micro", multi_class="ovr"))
    except Exception:
        auroc_macro = float("nan"); auroc_micro = float("nan")
    try:
        auprc_macro = float(average_precision_score(y_bin, y_prob, average="macro"))
        auprc_micro = float(average_precision_score(y_bin, y_prob, average="micro"))
    except Exception:
        auprc_macro = float("nan"); auprc_micro = float("nan")

    overall = dict(
        Top1 = float(accuracy_score(y_true, y_pred)),
        Top5 = float(top_k_accuracy_score(y_true, y_prob, k=min(topk, num_classes))),
        AUROC_macro = auroc_macro, AUROC_micro = auroc_micro,
        AUPRC_macro = auprc_macro, AUPRC_micro = auprc_micro,
        F1_macro = float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        F1_weighted = float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        BalancedAcc = balanced, MCC = mcc,
        LogLoss = float(log_loss(y_true, y_prob)),
        ECE = ece_multiclass(y_true, y_prob),
        N_per_class = {int(c): int(np.sum(y_true==c)) for c in range(num_classes)},
        CM = cm.tolist()
    )

    prec = precision_score(y_true, y_pred, average=None, labels=list(range(num_classes)), zero_division=0)
    rec  = recall_score(y_true, y_pred,   average=None, labels=list(range(num_classes)), zero_division=0)
    f1   = f1_score(y_true, y_pred,       average=None, labels=list(range(num_classes)), zero_division=0)
    per_class = [dict(class_id=int(c), precision=float(prec[c]), recall=float(rec[c]),
                      f1=float(f1[c]), support=int(np.sum(y_true==c)))
                 for c in range(num_classes)]
    return overall, per_class

