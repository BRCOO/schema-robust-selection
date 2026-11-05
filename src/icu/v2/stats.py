"""Uncertainty estimates for v2 utility metrics (AUPRC, AUROC).

v1 bootstrapped episodes, which treats the many windows of one patient as independent
draws and gives intervals that are too narrow. ``patient_cluster_bootstrap`` resamples
patients (clusters) with replacement and keeps every episode of each drawn patient;
``episode_bootstrap`` keeps the v1 resampling unit only for side-by-side comparison.

Resamples that contain a single class (where both metrics are undefined) are skipped and
counted, not redrawn. Intervals are percentile intervals of the bootstrap distribution.
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import ArrayLike
from sklearn.metrics import average_precision_score, roc_auc_score

METRICS = ("auprc", "auroc")


def _check_metric(metric: str) -> None:
    if metric not in METRICS:
        raise ValueError(f"metric must be one of {METRICS}, got {metric!r}")


def _check_bootstrap_args(n_boot: int, ci: float) -> None:
    if isinstance(n_boot, bool) or int(n_boot) != n_boot or n_boot < 1:
        raise ValueError(f"n_boot must be a positive integer, got {n_boot!r}")
    if not 0.0 < float(ci) < 1.0:
        raise ValueError(f"ci must be in (0, 1), got {ci!r}")


def _labels_and_scores(y: ArrayLike, *score_vectors: ArrayLike) -> tuple[np.ndarray, list[np.ndarray]]:
    labels = np.asarray(y)
    if labels.ndim != 1 or labels.size == 0:
        raise ValueError("y must be a non-empty 1-D array of 0/1 labels")
    if not np.isin(labels, (0, 1)).all():
        raise ValueError("y must contain only 0/1 labels")
    labels = labels.astype(np.int64)
    scores = []
    for vector in score_vectors:
        values = np.asarray(vector, dtype=float)
        if values.shape != labels.shape:
            raise ValueError(f"scores have shape {values.shape}, labels have shape {labels.shape}")
        if not np.isfinite(values).all():
            raise ValueError("scores must be finite")
        scores.append(values)
    return labels, scores


def _require_both_classes(labels: np.ndarray) -> None:
    n_pos = int(labels.sum())
    if n_pos == 0 or n_pos == labels.size:
        raise ValueError("AUPRC and AUROC need both classes; y contains only one class")


def _score(labels: np.ndarray, scores: np.ndarray, metric: str) -> float:
    if metric == "auprc":
        return float(average_precision_score(labels, scores))
    return float(roc_auc_score(labels, scores))


def _group_codes(groups: ArrayLike, n: int) -> np.ndarray:
    values = np.asarray(groups)
    if values.ndim != 1 or values.size != n:
        raise ValueError(f"groups must have one entry per episode ({n}), got shape {values.shape}")
    # np.unique sorts, so codes do not depend on the input row order.
    return np.unique(values, return_inverse=True)[1].reshape(-1)


def _bootstrap(
    labels: np.ndarray,
    score_vectors: list[np.ndarray],
    codes: np.ndarray,
    metric: str,
    n_boot: int,
    seed: int,
) -> tuple[np.ndarray, int, int]:
    """Cluster bootstrap; returns (values[n_used, n_vectors], n_skipped, n_groups)."""
    counts = np.bincount(codes)
    positives = np.bincount(codes, weights=labels).astype(np.int64)
    order = np.argsort(codes, kind="stable")
    starts = np.cumsum(counts) - counts
    n_groups = int(counts.size)
    rng = np.random.default_rng(int(seed))
    rows: list[list[float]] = []
    skipped = 0
    for _ in range(int(n_boot)):
        picked = rng.integers(0, n_groups, size=n_groups)
        sizes = counts[picked]
        total = int(sizes.sum())
        n_pos = int(positives[picked].sum())
        if n_pos == 0 or n_pos == total:
            skipped += 1
            continue
        # Concatenate the episodes of every drawn group (with multiplicity).
        offsets = np.repeat(starts[picked] - (np.cumsum(sizes) - sizes), sizes) + np.arange(total)
        index = order[offsets]
        sampled = labels[index]
        rows.append([_score(sampled, vector[index], metric) for vector in score_vectors])
    values = np.asarray(rows, dtype=float).reshape(len(rows), len(score_vectors))
    return values, skipped, n_groups


def _interval(values: np.ndarray, ci: float) -> tuple[float, float]:
    if values.size == 0:
        return math.nan, math.nan
    tail = (1.0 - float(ci)) / 2.0
    low, high = np.quantile(values, [tail, 1.0 - tail])
    return float(low), float(high)


def metric_value(y: ArrayLike, scores: ArrayLike, metric: str = "auprc") -> float:
    """AUPRC (average precision) or AUROC from sklearn; raises ValueError on one class."""
    _check_metric(metric)
    labels, (values,) = _labels_and_scores(y, scores)
    _require_both_classes(labels)
    return _score(labels, values, metric)


def _single_bootstrap(
    y: ArrayLike,
    scores: ArrayLike,
    groups: ArrayLike | None,
    metric: str,
    n_boot: int,
    seed: int,
    ci: float,
) -> dict:
    _check_metric(metric)
    _check_bootstrap_args(n_boot, ci)
    labels, (values,) = _labels_and_scores(y, scores)
    _require_both_classes(labels)
    codes = np.arange(labels.size) if groups is None else _group_codes(groups, labels.size)
    boot, skipped, n_groups = _bootstrap(labels, [values], codes, metric, n_boot, seed)
    low, high = _interval(boot[:, 0], ci)
    return {
        "metric": metric,
        "point": _score(labels, values, metric),
        "low": low,
        "high": high,
        "ci": float(ci),
        "n_boot": int(n_boot),
        "n_boot_used": int(boot.shape[0]),
        "n_skipped_single_class": skipped,
        "n_groups": n_groups,
        "n_episodes": int(labels.size),
        "seed": int(seed),
        "resample_unit": "episode" if groups is None else "cluster",
    }


def patient_cluster_bootstrap(
    y: ArrayLike,
    scores: ArrayLike,
    groups: ArrayLike,
    metric: str = "auprc",
    n_boot: int = 2000,
    seed: int = 0,
    ci: float = 0.95,
) -> dict:
    """Percentile CI from resampling groups (patients) with replacement.

    Each resample draws ``n_groups`` group ids with replacement and concatenates all of
    their episodes. Single-class resamples are skipped and counted, so
    ``n_boot_used + n_skipped_single_class == n_boot``; with no usable resample the
    bounds are NaN.
    """
    return _single_bootstrap(y, scores, groups, metric, n_boot, seed, ci)


def episode_bootstrap(
    y: ArrayLike,
    scores: ArrayLike,
    metric: str = "auprc",
    n_boot: int = 2000,
    seed: int = 0,
    ci: float = 0.95,
) -> dict:
    """v1 resampling unit (episodes), kept only to compare against the cluster bootstrap.

    Identical to ``patient_cluster_bootstrap`` with one group per episode.
    """
    return _single_bootstrap(y, scores, None, metric, n_boot, seed, ci)


def paired_cluster_bootstrap_diff(
    y: ArrayLike,
    scores_a: ArrayLike,
    scores_b: ArrayLike,
    groups: ArrayLike,
    metric: str = "auprc",
    n_boot: int = 2000,
    seed: int = 0,
    ci: float = 0.95,
) -> dict:
    """CI of metric(a) - metric(b) with both methods scored on the same patient resamples.

    ``frac_le_zero`` is the share of usable resamples where a does not beat b (a one-sided
    bootstrap check of "a > b"; NaN when no resample is usable).
    """
    _check_metric(metric)
    _check_bootstrap_args(n_boot, ci)
    labels, (values_a, values_b) = _labels_and_scores(y, scores_a, scores_b)
    _require_both_classes(labels)
    codes = _group_codes(groups, labels.size)
    boot, skipped, n_groups = _bootstrap(labels, [values_a, values_b], codes, metric, n_boot, seed)
    diffs = boot[:, 0] - boot[:, 1]
    low, high = _interval(diffs, ci)
    point_a = _score(labels, values_a, metric)
    point_b = _score(labels, values_b, metric)
    return {
        "metric": metric,
        "point_diff": point_a - point_b,
        "point_a": point_a,
        "point_b": point_b,
        "low": low,
        "high": high,
        "frac_le_zero": float(np.mean(diffs <= 0.0)) if diffs.size else math.nan,
        "ci": float(ci),
        "n_boot": int(n_boot),
        "n_boot_used": int(diffs.size),
        "n_skipped_single_class": skipped,
        "n_groups": n_groups,
        "n_episodes": int(labels.size),
        "seed": int(seed),
        "resample_unit": "cluster",
    }
