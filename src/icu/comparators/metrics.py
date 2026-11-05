"""Shared metric computation helpers for all comparator baselines."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score


def compute_ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    """Compute Expected Calibration Error (ECE)."""
    if y_true.size == 0 or y_prob.size == 0:
        return float("nan")
    if np.unique(y_true).size < 2:
        return float("nan")

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_indices = np.digitize(y_prob, bins, right=False) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)

    ece = 0.0
    for bin_idx in range(n_bins):
        mask = bin_indices == bin_idx
        if not np.any(mask):
            continue
        bin_acc = float(np.mean(y_true[mask]))
        bin_conf = float(np.mean(y_prob[mask]))
        bin_weight = float(np.sum(mask)) / float(y_true.shape[0])
        ece += bin_weight * abs(bin_acc - bin_conf)
    return float(ece)


def precision_at_recall(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    lead_hours: np.ndarray | None = None,
    target_recall: float = 0.50,
) -> float:
    """Compute precision at a fixed recall level."""
    if y_true.size == 0 or y_prob.size == 0:
        return float("nan")
    if np.unique(y_true).size < 2:
        return float("nan")

    sorted_indices = np.argsort(y_prob)[::-1]
    y_sorted = y_true[sorted_indices]
    n_pos = int(np.sum(y_true))
    if n_pos == 0:
        return float("nan")

    target_hits = max(1, int(round(n_pos * target_recall)))
    cum_hits = 0
    for k in range(1, y_sorted.shape[0] + 1):
        cum_hits += int(y_sorted[k - 1])
        if cum_hits >= target_hits:
            precision = float(cum_hits) / float(k)
            return float(precision)

    return float(cum_hits) / float(y_sorted.shape[0])


def compute_binary_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    lead_hours: np.ndarray | None = None,
) -> dict:
    """Compute the shared metric set for all supervised baselines."""
    result: dict = {
        "count": int(y_true.shape[0]),
        "auroc": float("nan"),
        "auprc": float("nan"),
        "brier": float("nan"),
        "ece": float("nan"),
        "lead_time_precision_at_recall_050": float("nan"),
    }

    if y_true.size == 0:
        return result

    try:
        result["positive_ratio"] = float(np.mean(y_true))
    except Exception:
        pass

    try:
        if np.unique(y_true).size > 1:
            result["auroc"] = float(roc_auc_score(y_true, y_prob))
            result["auprc"] = float(average_precision_score(y_true, y_prob))
    except Exception:
        pass

    try:
        result["brier"] = float(brier_score_loss(y_true, y_prob))
    except Exception:
        pass

    try:
        result["ece"] = float(compute_ece(y_true, y_prob, n_bins=10))
    except Exception:
        pass

    if lead_hours is not None:
        try:
            result["lead_time_precision_at_recall_050"] = float(
                precision_at_recall(y_true, y_prob, lead_hours=lead_hours, target_recall=0.50)
            )
        except Exception:
            pass

    return result
