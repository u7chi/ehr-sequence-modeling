"""Binary-classification metrics for imbalanced CAD prediction."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> dict[str, float]:
    """y_prob is P(class=1). AUPRC is primary (imbalanced, ~12% positive)."""
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)

    out = {}
    # guard against a degenerate batch with a single class present
    if len(np.unique(y_true)) < 2:
        out["auroc"] = float("nan")
        out["auprc"] = float("nan")
    else:
        out["auroc"] = float(roc_auc_score(y_true, y_prob))
        out["auprc"] = float(average_precision_score(y_true, y_prob))

    pred = (y_prob >= 0.5).astype(int)
    out["f1"] = float(f1_score(y_true, pred, zero_division=0))          # binary (positive class)
    out["f1_best"], out["thr_best"] = _best_f1(y_true, y_prob)
    # graph-project parity (its dict): macro-F1 @0.5 + acc/precision/recall
    out["f1_macro"] = float(f1_score(y_true, pred, average="macro", zero_division=0))
    out["acc"] = float(accuracy_score(y_true, pred))
    out["precision"] = float(precision_score(y_true, pred, zero_division=0))
    out["recall"] = float(recall_score(y_true, pred, zero_division=0))
    out["pos_rate"] = float(y_true.mean())
    return out


def _best_f1(y_true: np.ndarray, y_prob: np.ndarray) -> tuple[float, float]:
    if len(np.unique(y_true)) < 2:
        return float("nan"), 0.5
    prec, rec, thr = precision_recall_curve(y_true, y_prob)
    f1 = 2 * prec * rec / (prec + rec + 1e-12)
    i = int(np.nanargmax(f1))
    best_thr = float(thr[i]) if i < len(thr) else 0.5
    return float(f1[i]), best_thr
