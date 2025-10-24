import numpy as np
import pandas as pd
from sklearn.metrics import (
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
    accuracy_score,
    f1_score
)

# Operate site evaluation metrics with predict df
# cm
# AUROC, AUPRC
# Accuracy, F1

def site_evaluation_cm(df: pd.DataFrame) -> np.ndarray:
    """
    Compute confusion matrix from DataFrame of predictions.
    Works for binary or multi-class tasks.
    """
    y_true = np.array(df["y_true"])
    y_pred = np.array(df["y_pred"])
    cm = confusion_matrix(y_true, y_pred)
    return cm


def site_evaluation_auroc(df: pd.DataFrame) -> float:
    """
    Compute AUROC from DataFrame of predictions.
    For binary tasks: uses single-column y_prob.
    For multi-class: uses one-hot probability vectors stored in list form.
    """
    y_true = np.array(df["y_true"])

    # Binary
    if len(np.unique(y_true)) == 2:
        y_prob = np.array(df["y_prob"])
        return roc_auc_score(y_true, y_prob)

    # Multi-class
    y_prob = np.stack(df["y_prob"].values)
    return roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro")


def site_evaluation_auprc(df: pd.DataFrame) -> float:
    """
    Compute AUPRC (average precision score).
    """
    y_true = np.array(df["y_true"])

    # Binary
    if len(np.unique(y_true)) == 2:
        y_prob = np.array(df["y_prob"])
        return average_precision_score(y_true, y_prob)

    # Multi-class
    y_prob = np.stack(df["y_prob"].values)
    return average_precision_score(y_true, y_prob, average="macro")


def site_evaluation_acc(df: pd.DataFrame) -> float:
    """
    Compute overall accuracy from predictions.
    """
    y_true = np.array(df["y_true"])
    y_pred = np.array(df["y_pred"])
    return accuracy_score(y_true, y_pred)


def site_evaluation_f1(df: pd.DataFrame) -> float:
    """
    Compute F1 score.
    Binary → normal f1_score
    Multi-class → macro F1
    """
    y_true = np.array(df["y_true"])
    y_pred = np.array(df["y_pred"])

    if len(np.unique(y_true)) == 2:
        return f1_score(y_true, y_pred)
    else:
        return f1_score(y_true, y_pred, average="macro")
