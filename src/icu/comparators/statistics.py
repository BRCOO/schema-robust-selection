from __future__ import annotations

import math
from typing import Iterable

import numpy as np
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score


def _safe_metric(metric: str, y_true: np.ndarray, y_prob: np.ndarray) -> float:
    if y_true.size == 0:
        return float("nan")
    if metric == "auroc":
        if np.unique(y_true).size < 2:
            return float("nan")
        return float(roc_auc_score(y_true, y_prob))
    if metric == "auprc":
        if np.unique(y_true).size < 2:
            return float("nan")
        return float(average_precision_score(y_true, y_prob))
    if metric == "brier":
        return float(brier_score_loss(y_true, y_prob))
    raise ValueError(f"unsupported metric: {metric}")


def bootstrap_confidence_interval(
    values: Iterable[float],
    n_bootstrap: int = 2000,
    alpha: float = 0.05,
    seed: int = 42,
) -> dict:
    arr = np.asarray(list(values), dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {
            "n": 0,
            "mean": float("nan"),
            "std": float("nan"),
            "low": float("nan"),
            "high": float("nan"),
        }
    if arr.size == 1:
        x = float(arr[0])
        return {"n": 1, "mean": x, "std": 0.0, "low": x, "high": x}

    rng = np.random.default_rng(int(seed))
    sample_size = int(arr.size)
    boot_means = []
    for _ in range(max(100, int(n_bootstrap))):
        idx = rng.integers(0, sample_size, size=sample_size)
        boot_means.append(float(np.mean(arr[idx])))
    low_q = 100.0 * (float(alpha) / 2.0)
    high_q = 100.0 * (1.0 - float(alpha) / 2.0)
    return {
        "n": sample_size,
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr, ddof=1)),
        "low": float(np.percentile(boot_means, low_q)),
        "high": float(np.percentile(boot_means, high_q)),
    }


def aggregate_seed_metrics(
    rows: list[dict],
    metric_key: str,
    n_bootstrap: int = 2000,
    alpha: float = 0.05,
    seed: int = 42,
) -> dict:
    values = []
    for row in rows:
        value = row.get(metric_key)
        if value is None:
            continue
        try:
            x = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(x):
            values.append(x)
    ci = bootstrap_confidence_interval(values, n_bootstrap=n_bootstrap, alpha=alpha, seed=seed)
    return {
        "n": ci["n"],
        "mean": ci["mean"],
        "std": ci["std"],
        "ci_low": ci["low"],
        "ci_high": ci["high"],
    }


def paired_bootstrap_pvalue(
    y_true: Iterable[int],
    y_prob_a: Iterable[float],
    y_prob_b: Iterable[float],
    metric: str = "auroc",
    n_bootstrap: int = 3000,
    seed: int = 42,
) -> dict:
    y = np.asarray(list(y_true), dtype=int)
    prob_a = np.asarray(list(y_prob_a), dtype=float)
    prob_b = np.asarray(list(y_prob_b), dtype=float)
    if not (y.shape[0] == prob_a.shape[0] == prob_b.shape[0]):
        raise ValueError("paired inputs must have same length")
    if y.shape[0] == 0:
        return {
            "metric": metric,
            "n": 0,
            "observed_delta": float("nan"),
            "p_value": float("nan"),
        }

    observed_a = _safe_metric(metric, y, prob_a)
    observed_b = _safe_metric(metric, y, prob_b)
    observed_delta = float(observed_a - observed_b)
    if not math.isfinite(observed_delta):
        return {
            "metric": metric,
            "n": int(y.shape[0]),
            "observed_delta": float("nan"),
            "p_value": float("nan"),
        }

    rng = np.random.default_rng(int(seed))
    deltas = []
    n = int(y.shape[0])
    for _ in range(max(200, int(n_bootstrap))):
        idx = rng.integers(0, n, size=n)
        boot_y = y[idx]
        if np.unique(boot_y).size < 2 and metric in ("auroc", "auprc"):
            continue
        delta = _safe_metric(metric, boot_y, prob_a[idx]) - _safe_metric(metric, boot_y, prob_b[idx])
        if math.isfinite(delta):
            deltas.append(float(delta))
    if not deltas:
        return {
            "metric": metric,
            "n": n,
            "observed_delta": observed_delta,
            "p_value": float("nan"),
        }

    deltas_arr = np.asarray(deltas, dtype=float)
    if observed_delta >= 0:
        p = float(np.mean(deltas_arr <= 0.0))
    else:
        p = float(np.mean(deltas_arr >= 0.0))
    p_two_sided = min(1.0, 2.0 * p)
    return {
        "metric": metric,
        "n": n,
        "observed_delta": observed_delta,
        "p_value": p_two_sided,
    }


def multiple_comparison_correction(p_values: Iterable[float], method: str = "holm") -> list[float]:
    values = list(p_values)
    n = len(values)
    if n == 0:
        return []

    finite_items = []
    corrected = [float("nan")] * n
    for idx, value in enumerate(values):
        try:
            p = float(value)
        except (TypeError, ValueError):
            continue
        if not math.isfinite(p):
            continue
        p = min(1.0, max(0.0, p))
        finite_items.append((idx, p))

    m = len(finite_items)
    if m == 0:
        return corrected

    sorted_items = sorted(finite_items, key=lambda x: x[1])

    if method == "holm":
        running_max = 0.0
        for rank, (orig_idx, p) in enumerate(sorted_items, start=1):
            adj = (m - rank + 1) * p
            adj = min(1.0, max(running_max, adj))
            running_max = adj
            corrected[orig_idx] = float(adj)
        return corrected

    if method == "fdr_bh":
        adjusted_sorted = [0.0] * m
        for rank, (_, p) in enumerate(sorted_items, start=1):
            adjusted_sorted[rank - 1] = min(1.0, (m / float(rank)) * p)
        for i in range(m - 2, -1, -1):
            adjusted_sorted[i] = min(adjusted_sorted[i], adjusted_sorted[i + 1])
        for (orig_idx, _), adj in zip(sorted_items, adjusted_sorted):
            corrected[orig_idx] = float(adj)
        return corrected

    if method == "bonferroni":
        for orig_idx, p in finite_items:
            corrected[orig_idx] = float(min(1.0, p * m))
        return corrected

    raise ValueError(f"unsupported correction method: {method}")
