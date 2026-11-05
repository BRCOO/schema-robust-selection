"""Leakage probes for episode windows (v2 protocol).

Single-feature probes use each per-episode feature summary directly as a score: AUROC
and AUPRC are computed without fitting anything. A summary that separates positives from
controls on its own is a leakage suspect (v1: the last-row ``vasopressor_on``).
:func:`multivariate_probe` catches combinations single features miss: a small L2
logistic regression and a depth-3 tree fitted on the train split, scored on the test split.

Episode files are opened only by constructed path (``shock_episode_<episode_id>.json``,
``.npy`` and ``_missing_mask.npy``); the episode directory is never listed.

CLI (light: about three small file reads per episode, sequential)::

    PYTHONDONTWRITEBYTECODE=1 python -m src.icu.v2.leakage_audit \\
        --manifest docs/experiments/icde_comparator_split_manifest.json \\
        --episode-dir data/processed/mimic-iv-full-cw3-gap0-real/mimic-iv \\
        --out-json artifacts/v2/leakage_audit/locked500_audit.json
"""

from __future__ import annotations

import argparse
import json
import math
import warnings
from collections import Counter
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Collection, Hashable, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from src.icu.v2.task import (
    CONTROL,
    POSITIVE,
    TaskSpec,
    apply_task,
    build_stay_anchor_index,
    episode_label,
    episode_type,
    summarize_exclusions,
    transform_episode,
)

WINDOW_KEY = "__window__"
SUMMARY_STATS = ("last", "mean", "max", "observed_last")
DEFAULT_SEPARABILITY_THRESHOLD = 0.80
EPISODE_PREFIX = "shock_episode_"

# Alternatives the team may pick instead of the default spec (examples, not decisions).
CANDIDATE_SPECS: dict[str, TaskSpec] = {
    "initiation_only": TaskSpec(anchor_kinds=("initiation",)),
    "initiation_first_anchor_washout24h": TaskSpec(
        anchor_kinds=("initiation",), first_anchor_per_stay_only=True, initiation_washout_hours=24.0
    ),
    "gap_2h": TaskSpec(prediction_gap_hours=2.0),
    "control_buffer_24h": TaskSpec(control_min_hours_from_anchor=24.0),
}

# Multivariate probe: column views and the logistic-regression C grid.
MULTIVARIATE_VIEWS = ("all", "observation_only", "values_only")
OBSERVATION_STATS = ("observed_last",)
LR_C_GRID = (0.01, 0.1, 1.0, 10.0)


# ------------------------------------------------------------------------------ summaries


def summarize_episode(
    values: np.ndarray,
    mask: np.ndarray,
    feature_names: Sequence[str],
) -> dict[str, dict[str, float]]:
    """Per-feature summaries of one window.

    ``mask`` is True where a value was missing (imputed), as in v1 ``_missing_mask.npy``;
    v1 zero-fill features (presence flags, urine) are never marked missing. Per feature:
    ``last`` (value in the last observed row, NaN if never observed), ``mean`` and ``max``
    over all rows (the filled matrix a model consumes), and ``observed_last`` (1.0 if the
    last row is observed). ``WINDOW_KEY`` holds ``{"n_rows": window length}``.
    """
    values = np.asarray(values, dtype=float)
    mask = np.asarray(mask, dtype=bool)
    if values.ndim != 2 or mask.shape != values.shape:
        raise ValueError(f"values {values.shape} and mask {mask.shape} must be equal 2-D shapes")
    if len(feature_names) != values.shape[1]:
        raise ValueError(f"{len(feature_names)} feature names for {values.shape[1]} columns")
    n_rows = values.shape[0]
    summaries: dict[str, dict[str, float]] = {}
    for j, name in enumerate(feature_names):
        if str(name) == WINDOW_KEY:
            raise ValueError(f"feature name {WINDOW_KEY!r} is reserved")
        column = values[:, j]
        finite = np.isfinite(column)
        observed = finite & ~mask[:, j]
        observed_rows = np.flatnonzero(observed)
        summaries[str(name)] = {
            "last": float(column[observed_rows[-1]]) if observed_rows.size else math.nan,
            "mean": float(column[finite].mean()) if finite.any() else math.nan,
            "max": float(column[finite].max()) if finite.any() else math.nan,
            "observed_last": float(observed[-1]) if n_rows else math.nan,
        }
    summaries[WINDOW_KEY] = {"n_rows": float(n_rows)}
    return summaries


# --------------------------------------------------------------------------------- probes


def _nan_lowest(raw: np.ndarray) -> np.ndarray:
    scores = np.array(raw, dtype=float, copy=True)
    scores[np.isnan(scores)] = -np.inf
    return scores


def _auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Mann-Whitney AUROC with average ranks for ties."""
    n_pos = int(labels.sum())
    n_neg = int(labels.size - n_pos)
    ranks = rankdata(scores)
    return float((ranks[labels == 1].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def _average_precision(scores: np.ndarray, labels: np.ndarray) -> float:
    """Step-wise average precision; tied scores form one threshold (as sklearn)."""
    order = np.argsort(-scores, kind="mergesort")
    sorted_scores = scores[order]
    sorted_labels = labels[order]
    last_of_tie = np.r_[np.flatnonzero(sorted_scores[1:] != sorted_scores[:-1]), sorted_labels.size - 1]
    true_pos = np.cumsum(sorted_labels)[last_of_tie]
    precision = true_pos / (last_of_tie + 1.0)
    recall = true_pos / true_pos[-1]
    return float(np.sum(np.diff(np.r_[0.0, recall]) * precision))


def _nanmean(values: np.ndarray) -> float:
    finite = values[~np.isnan(values)]
    return float(finite.mean()) if finite.size else math.nan


def single_feature_probes(
    summaries_by_episode: Mapping[str, Mapping[str, Mapping[str, float]]],
    labels: Mapping[str, int],
) -> list[dict[str, Any]]:
    """Score every (feature, summary) pair directly; no model is fitted.

    NaN scores count as the lowest value. Each row has ``auroc``, ``auprc`` (raw score),
    ``separability = max(auroc, 1 - auroc)``, ``direction``, ``auprc_oriented`` (score
    flipped when lower values indicate positives), ``prevalence``, counts, NaN counts per
    class (a ``last`` probe can separate on "was it measured at all"), and the class means
    ``mean_pos`` / ``mean_neg`` over non-NaN values. Sorted by separability, highest first.
    """
    episode_ids = sorted(summaries_by_episode)
    missing = [episode_id for episode_id in episode_ids if episode_id not in labels]
    if missing:
        raise ValueError(f"{len(missing)} episodes have no label, e.g. {missing[0]!r}")
    y = np.array([int(labels[episode_id]) for episode_id in episode_ids], dtype=int)
    if not set(np.unique(y).tolist()) <= {0, 1}:
        raise ValueError("labels must be 0 or 1")
    n_pos = int(y.sum())
    n_neg = int(y.size - n_pos)
    if n_pos == 0 or n_neg == 0:
        raise ValueError(f"probes need both classes (positives={n_pos}, controls={n_neg})")

    keys = sorted(
        {
            (feature, stat)
            for summary in summaries_by_episode.values()
            for feature, stats in summary.items()
            for stat in stats
        }
    )
    rows: list[dict[str, Any]] = []
    for feature, stat in keys:
        raw = np.array(
            [summaries_by_episode[eid].get(feature, {}).get(stat, math.nan) for eid in episode_ids],
            dtype=float,
        )
        scores = _nan_lowest(raw)
        auroc = _auroc(scores, y)
        if auroc > 0.5 + 1e-12:
            direction, oriented = "higher_in_positive", scores
        elif auroc < 0.5 - 1e-12:
            direction, oriented = "lower_in_positive", -scores
        else:
            direction, oriented = "none", scores
        rows.append(
            {
                "feature": feature,
                "summary": stat,
                "auroc": auroc,
                "auprc": _average_precision(scores, y),
                "separability": max(auroc, 1.0 - auroc),
                "direction": direction,
                "auprc_oriented": _average_precision(oriented, y),
                "prevalence": n_pos / y.size,
                "n": int(y.size),
                "n_pos": n_pos,
                "n_neg": n_neg,
                "n_nan": int(np.isnan(raw).sum()),
                "n_nan_pos": int(np.isnan(raw[y == 1]).sum()),
                "n_nan_neg": int(np.isnan(raw[y == 0]).sum()),
                "mean_pos": _nanmean(raw[y == 1]),
                "mean_neg": _nanmean(raw[y == 0]),
            }
        )
    rows.sort(key=lambda row: (-row["separability"], row["feature"], row["summary"]))
    return rows


def flag_leaks(
    rows: Sequence[Mapping[str, Any]],
    separability_threshold: float = DEFAULT_SEPARABILITY_THRESHOLD,
) -> list[dict[str, Any]]:
    """Probe rows with ``separability >= threshold``, in input order, marked as suspected leaks."""
    flagged: list[dict[str, Any]] = []
    for row in rows:
        value = row.get("separability")
        if value is None or not math.isfinite(float(value)):
            continue
        if float(value) >= separability_threshold - 1e-12:
            flagged.append({**row, "flag": "suspected_leak"})
    return flagged


# --------------------------------------------------------------------- multivariate probe


def _design_matrix(
    summaries_by_episode: Mapping[str, Mapping[str, Mapping[str, float]]],
    episode_ids: Sequence[str],
    n_train: int,
    feature_subset: Collection[str] | None,
    view: str,
) -> tuple[np.ndarray, list[str]]:
    """Flat matrix (NaN kept) of the summaries selected by ``view``, plus missingness
    indicators for value columns with NaN among the first ``n_train`` rows."""
    allowed = None if feature_subset is None else {str(feature) for feature in feature_subset}
    keys = sorted(
        {
            (feature, stat)
            for episode_id in episode_ids
            for feature, stats in summaries_by_episode[episode_id].items()
            for stat in stats
        }
    )
    names: list[str] = []
    columns: list[np.ndarray] = []
    for feature, stat in keys:
        if allowed is not None and feature != WINDOW_KEY and feature not in allowed:
            continue
        column = np.array(
            [float(summaries_by_episode[eid].get(feature, {}).get(stat, math.nan)) for eid in episode_ids],
            dtype=float,
        )
        is_observation = feature != WINDOW_KEY and stat in OBSERVATION_STATS
        is_value = feature != WINDOW_KEY and not is_observation
        if (is_observation and view in ("all", "observation_only")) or (
            not is_observation and view in ("all", "values_only")
        ):
            names.append(f"{feature}.{stat}")
            columns.append(column)
        if is_value and view in ("all", "observation_only") and np.isnan(column[:n_train]).any():
            names.append(f"{feature}.{stat}.missing")
            columns.append(np.isnan(column).astype(float))
    matrix = np.column_stack(columns) if columns else np.zeros((len(episode_ids), 0))
    return matrix, names


def _cv_splits(
    y: np.ndarray,
    groups: np.ndarray | None,
    n_folds: int,
    seed: int,
) -> tuple[list[tuple[np.ndarray, np.ndarray]], str]:
    """Stratified folds of the train rows, grouped when ``groups`` is given."""
    n_splits = int(min(n_folds, np.bincount(y, minlength=2).min()))
    if groups is not None:
        n_splits = min(n_splits, len(set(groups.tolist())))
    if n_splits < 2:
        raise ValueError("too few train episodes per class for cross-validation")
    placeholder = np.zeros(y.size)
    if groups is not None:
        splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        return list(splitter.split(placeholder, y, groups)), f"StratifiedGroupKFold({n_splits})"
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return list(splitter.split(placeholder, y)), f"StratifiedKFold({n_splits})"


def _lr_pipeline(c: float) -> Pipeline:
    """Train-median imputation, standardization, L2 logistic regression (sklearn default
    penalty), balanced class weights."""
    return Pipeline(
        [
            ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("scale", StandardScaler()),
            ("model", LogisticRegression(C=float(c), class_weight="balanced", max_iter=5000)),
        ]
    )


def _tree_pipeline(seed: int) -> Pipeline:
    """Train-median imputation and a depth-3 tree with at least 5 episodes per leaf."""
    return Pipeline(
        [
            ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("model", DecisionTreeClassifier(max_depth=3, min_samples_leaf=5, random_state=seed)),
        ]
    )


def _positive_fraction(value: np.ndarray) -> float:
    counts = np.asarray(value, dtype=float).reshape(-1)
    return float(counts[1] / counts.sum()) if counts.size > 1 and counts.sum() > 0 else math.nan


def _tree_splits(tree: DecisionTreeClassifier, names: Sequence[str]) -> list[dict[str, Any]]:
    """Internal nodes, largest impurity decrease first; ``left`` is ``feature <= threshold``."""
    structure = tree.tree_
    depth = np.zeros(structure.node_count, dtype=int)
    total = float(structure.weighted_n_node_samples[0])
    splits: list[dict[str, Any]] = []
    for node in range(structure.node_count):  # parents precede their children
        left, right = structure.children_left[node], structure.children_right[node]
        if left == -1:
            continue
        depth[left] = depth[right] = depth[node] + 1
        decrease = (
            structure.weighted_n_node_samples[node] * structure.impurity[node]
            - structure.weighted_n_node_samples[left] * structure.impurity[left]
            - structure.weighted_n_node_samples[right] * structure.impurity[right]
        ) / total
        splits.append(
            {
                "depth": int(depth[node]),
                "feature": names[structure.feature[node]],
                "threshold": float(structure.threshold[node]),
                "n_samples": int(structure.n_node_samples[node]),
                "pos_fraction": _positive_fraction(structure.value[node]),
                "left_pos_fraction": _positive_fraction(structure.value[left]),
                "right_pos_fraction": _positive_fraction(structure.value[right]),
                "impurity_decrease": float(decrease),
            }
        )
    splits.sort(key=lambda split: -split["impurity_decrease"])
    return splits


def _bootstrap_ci(
    y: np.ndarray,
    scores: np.ndarray,
    groups: np.ndarray | None,
    n_bootstrap: int,
    seed: int,
) -> dict[str, Any]:
    """Percentile 95 % intervals of test AUROC/AUPRC; resamples groups when given."""
    if groups is None:
        clusters = [np.array([row]) for row in range(y.size)]
    else:
        index: dict[Any, list[int]] = {}
        for row, group in enumerate(groups.tolist()):
            index.setdefault(group, []).append(row)
        clusters = [np.array(rows) for rows in index.values()]
    rng = np.random.default_rng(seed)
    aurocs: list[float] = []
    auprcs: list[float] = []
    for _ in range(int(n_bootstrap)):
        rows = np.concatenate([clusters[pick] for pick in rng.integers(0, len(clusters), size=len(clusters))])
        sample = y[rows]
        if sample.min() == sample.max():
            continue
        aurocs.append(_auroc(scores[rows], sample))
        auprcs.append(_average_precision(scores[rows], sample))
    if not aurocs:
        return {"test_auroc_ci95": None, "test_auprc_ci95": None, "bootstrap_valid": 0}
    return {
        "test_auroc_ci95": [float(np.percentile(aurocs, 2.5)), float(np.percentile(aurocs, 97.5))],
        "test_auprc_ci95": [float(np.percentile(auprcs, 2.5)), float(np.percentile(auprcs, 97.5))],
        "bootstrap_valid": len(aurocs),
    }


def _test_metrics(
    y: np.ndarray,
    scores: np.ndarray,
    groups: np.ndarray | None,
    n_bootstrap: int,
    seed: int,
) -> dict[str, Any]:
    if np.unique(y).size < 2:
        return {"test_auroc": None, "test_auprc": None, "note": "test split has a single class"}
    metrics: dict[str, Any] = {"test_auroc": _auroc(scores, y), "test_auprc": _average_precision(scores, y)}
    if n_bootstrap > 0:
        metrics.update(_bootstrap_ci(y, scores, groups, n_bootstrap, seed))
    return metrics


def multivariate_probe(
    summaries_by_episode: Mapping[str, Mapping[str, Mapping[str, float]]],
    labels: Mapping[str, int],
    train_ids: Sequence[str],
    test_ids: Sequence[str],
    feature_subset: Collection[str] | None = None,
    *,
    view: str = "all",
    groups: Mapping[str, Hashable] | None = None,
    c_grid: Sequence[float] = LR_C_GRID,
    n_folds: int = 5,
    seed: int = 0,
    n_bootstrap: int = 1000,
) -> dict[str, Any]:
    """Fit two small models on the train episodes and score the test episodes.

    Columns are the per-feature summaries (``last``, ``mean``, ``max``, ``observed_last``)
    and the window ``n_rows``, restricted to the base features in ``feature_subset`` if
    given. ``view`` picks columns: ``"all"``; ``"observation_only"`` (``observed_last``
    plus missingness indicators, i.e. whether a feature was measured in the window);
    ``"values_only"`` (``last``/``mean``/``max`` and ``n_rows``, no indicators). NaN is
    imputed with the train median; a ``<column>.missing`` indicator is added for each value
    column with NaN in train. Columns that are all-NaN or constant on train are dropped.

    Models: L2 logistic regression (``class_weight="balanced"``, standardized on train, C
    from ``c_grid`` by mean AUROC over ``n_folds`` stratified folds of the train set only,
    grouped by ``groups`` when given; imputation and scaling refit in every fold) and a
    depth-3 decision tree (``min_samples_leaf=5``). Returns test AUROC/AUPRC per model with
    bootstrap 95 % intervals (groups resampled when given), test prevalence, the top-10
    standardized LR coefficients by absolute value, and the tree's splits.
    """
    if view not in MULTIVARIATE_VIEWS:
        raise ValueError(f"view must be one of {MULTIVARIATE_VIEWS}, got {view!r}")
    train_ids = list(dict.fromkeys(str(episode_id) for episode_id in train_ids))
    test_ids = list(dict.fromkeys(str(episode_id) for episode_id in test_ids))
    if not train_ids or not test_ids:
        raise ValueError("train_ids and test_ids must both be non-empty")
    overlap = set(train_ids) & set(test_ids)
    if overlap:
        raise ValueError(f"{len(overlap)} episodes are in both train and test, e.g. {sorted(overlap)[0]!r}")
    episode_ids = train_ids + test_ids
    unknown = [eid for eid in episode_ids if eid not in summaries_by_episode or eid not in labels]
    if unknown:
        raise ValueError(f"{len(unknown)} episodes lack a summary or label, e.g. {unknown[0]!r}")
    if groups is not None:
        ungrouped = [eid for eid in episode_ids if eid not in groups]
        if ungrouped:
            raise ValueError(f"{len(ungrouped)} episodes have no group, e.g. {ungrouped[0]!r}")
    n_train = len(train_ids)
    y = np.array([int(labels[eid]) for eid in episode_ids], dtype=int)
    if not set(np.unique(y).tolist()) <= {0, 1}:
        raise ValueError("labels must be 0 or 1")
    y_train, y_test = y[:n_train], y[n_train:]
    if np.unique(y_train).size < 2:
        raise ValueError("the train episodes need both classes")

    matrix, names = _design_matrix(summaries_by_episode, episode_ids, n_train, feature_subset, view)
    keep: list[int] = []
    for j in range(matrix.shape[1]):
        column = matrix[:n_train, j]
        observed = column[~np.isnan(column)]
        if observed.size and np.ptp(np.where(np.isnan(column), np.median(observed), column)) > 0:
            keep.append(j)
    n_dropped = matrix.shape[1] - len(keep)
    matrix = matrix[:, keep]
    names = [names[j] for j in keep]
    if not names:
        raise ValueError(f"no usable columns for view {view!r}")
    x_train, x_test = matrix[:n_train], matrix[n_train:]
    group_array = None if groups is None else np.array([groups[eid] for eid in episode_ids], dtype=object)
    train_groups = None if group_array is None else group_array[:n_train]
    test_groups = None if group_array is None else group_array[n_train:]

    splits, splitter = _cv_splits(y_train, train_groups, n_folds, seed)
    mean_auroc_by_c: dict[float, float] = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        for c in c_grid:
            fold_scores = []
            for fit_rows, held_rows in splits:
                if np.unique(y_train[fit_rows]).size < 2 or np.unique(y_train[held_rows]).size < 2:
                    continue
                fold_model = _lr_pipeline(c).fit(x_train[fit_rows], y_train[fit_rows])
                fold_scores.append(_auroc(fold_model.predict_proba(x_train[held_rows])[:, 1], y_train[held_rows]))
            mean_auroc_by_c[float(c)] = float(np.mean(fold_scores)) if fold_scores else math.nan
        valid = {c: score for c, score in mean_auroc_by_c.items() if math.isfinite(score)}
        if not valid:
            raise ValueError("no cross-validation fold had both classes")
        chosen_c = max(sorted(valid), key=lambda c: valid[c])  # ties go to the smallest C
        lr = _lr_pipeline(chosen_c).fit(x_train, y_train)
        tree = _tree_pipeline(seed).fit(x_train, y_train)

    coefficients = lr.named_steps["model"].coef_[0]
    importances = tree.named_steps["model"].feature_importances_
    return {
        "view": view,
        "feature_subset": None if feature_subset is None else sorted(str(f) for f in feature_subset),
        "n_train": n_train,
        "n_test": len(test_ids),
        "train_label_counts": {POSITIVE: int(y_train.sum()), CONTROL: int(n_train - y_train.sum())},
        "test_label_counts": {POSITIVE: int(y_test.sum()), CONTROL: int(y_test.size - y_test.sum())},
        "test_prevalence": float(y_test.mean()),
        "n_columns": len(names),
        "n_indicator_columns": sum(name.endswith(".missing") for name in names),
        "n_columns_dropped": n_dropped,
        "columns": names,
        "cv": {
            "splitter": splitter,
            "scoring": "roc_auc",
            "c_grid": [float(c) for c in c_grid],
            "mean_auroc_by_c": mean_auroc_by_c,
            "chosen_c": float(chosen_c),
        },
        "bootstrap": {"n_resamples": int(n_bootstrap), "unit": "episode" if groups is None else "group"},
        "logistic_regression": {
            **_test_metrics(y_test, lr.predict_proba(x_test)[:, 1], test_groups, n_bootstrap, seed),
            "converged": bool(np.all(lr.named_steps["model"].n_iter_ < 5000)),
            "top_coefficients": [
                {"column": names[j], "coef": float(coefficients[j])}
                for j in np.argsort(-np.abs(coefficients), kind="mergesort")[:10]
            ],
        },
        "decision_tree": {
            **_test_metrics(y_test, tree.predict_proba(x_test)[:, 1], test_groups, n_bootstrap, seed),
            "splits": _tree_splits(tree.named_steps["model"], names),
            "top_importances": [
                {"column": names[j], "importance": float(importances[j])}
                for j in np.argsort(-importances, kind="mergesort")[:10]
                if importances[j] > 0
            ],
        },
    }


# -------------------------------------------------------------------------------- loading


@dataclass(frozen=True)
class EpisodeArrays:
    """One episode as stored by v1: hourly values, missing mask (True = missing), metadata."""

    episode_id: str
    values: np.ndarray
    mask: np.ndarray
    meta: dict[str, Any]


def episode_stem(episode_id: str) -> str:
    """File stem of an episode: ``shock_episode_<episode_id>`` (prefix not doubled)."""
    episode_id = str(episode_id)
    return episode_id if episode_id.startswith(EPISODE_PREFIX) else EPISODE_PREFIX + episode_id


def load_episode(episode_dir: Path | str, episode_id: str) -> EpisodeArrays:
    """Open the three files of one episode by constructed path."""
    directory = Path(episode_dir)
    stem = episode_stem(episode_id)
    with open(directory / f"{stem}.json", encoding="utf-8") as handle:
        meta = json.load(handle)
    values = np.load(directory / f"{stem}.npy", allow_pickle=False)
    mask_name = Path(str(meta.get("missing_mask_file") or f"{stem}_missing_mask.npy")).name
    mask = np.load(directory / mask_name, allow_pickle=False)
    resolved_id = str(meta.get("episode_id") or stem[len(EPISODE_PREFIX):])
    return EpisodeArrays(episode_id=resolved_id, values=values, mask=mask.astype(bool), meta=meta)


def load_episodes(episode_dir: Path | str, episode_ids: Sequence[str]) -> list[EpisodeArrays]:
    """Load the given episodes sequentially (duplicates loaded once, order kept)."""
    return [load_episode(episode_dir, episode_id) for episode_id in dict.fromkeys(episode_ids)]


# ---------------------------------------------------------------------------------- audit


def _label_counts(metas: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    counts = Counter(episode_type(meta) for meta in metas)
    return {POSITIVE: int(counts.get(POSITIVE, 0)), CONTROL: int(counts.get(CONTROL, 0))}


def collect_summaries(
    episodes: Sequence[EpisodeArrays],
    spec: TaskSpec | None = None,
    *,
    apply_eligibility: bool = True,
    stay_anchors: Mapping[int, Sequence[pd.Timestamp]] | None = None,
) -> dict[str, Any]:
    """Eligibility (optional), transform and per-episode summaries, shared by both probes.

    ``spec=None`` keeps the v1 arrays as stored. With a spec, :func:`apply_task` runs first
    when ``apply_eligibility`` is set (stay anchors from the positives among ``episodes``
    unless given), then :func:`transform_episode` on every kept episode. Returns
    ``eligibility``, ``selected``, ``excluded``, ``summaries``, ``labels``,
    ``rows_by_label`` and ``transform``.
    """
    metas = [episode.meta for episode in episodes]
    selected = list(episodes)
    excluded: list[dict[str, Any]] = []
    eligibility = bool(spec is not None and apply_eligibility)
    if eligibility:
        kept_metas, excluded = apply_task(metas, spec, stay_anchors=stay_anchors)
        # apply_task returns the very meta objects it was given, so identity selects them.
        kept_ids = {id(meta) for meta in kept_metas}
        selected = [episode for episode in episodes if id(episode.meta) in kept_ids]

    summaries: dict[str, dict[str, dict[str, float]]] = {}
    labels: dict[str, int] = {}
    rows_by_label: dict[str, Counter] = {POSITIVE: Counter(), CONTROL: Counter()}
    dropped: set[str] = set()
    not_found: set[str] = set()
    trimmed = 0
    not_full = 0
    for episode in selected:
        values, mask = episode.values, episode.mask
        names = [str(name) for name in episode.meta["feature_names"]]
        if spec is not None:
            values, mask, names, info = transform_episode(values, mask, episode.meta, spec)
            dropped.update(info["dropped_features"])
            not_found.update(info["excluded_not_found"])
            trimmed += int(info["rows_trimmed_start"] > 0)
            not_full += int(not info["full_window"])
        if episode.episode_id in summaries:
            raise ValueError(f"duplicate episode id {episode.episode_id!r}")
        summaries[episode.episode_id] = summarize_episode(values, mask, names)
        labels[episode.episode_id] = episode_label(episode.meta)
        rows_by_label[episode_type(episode.meta)][int(values.shape[0])] += 1

    transform = None
    if spec is not None:
        transform = {
            "dropped_features": sorted(dropped),
            "excluded_not_found": sorted(not_found),
            "episodes_with_rows_trimmed_start": trimmed,
            "audited_not_full_window": not_full,
        }
    return {
        "eligibility": eligibility,
        "selected": selected,
        "excluded": excluded,
        "summaries": summaries,
        "labels": labels,
        "rows_by_label": rows_by_label,
        "transform": transform,
    }


def audit_loaded_episodes(
    episodes: Sequence[EpisodeArrays],
    spec: TaskSpec | None = None,
    *,
    apply_eligibility: bool = True,
    stay_anchors: Mapping[int, Sequence[pd.Timestamp]] | None = None,
    separability_threshold: float = DEFAULT_SEPARABILITY_THRESHOLD,
    include_excluded_rows: bool = True,
) -> dict[str, Any]:
    """Single-feature probes on already-loaded episodes, as v1 (``spec=None``) or under a
    v2 task spec (see :func:`collect_summaries` for the eligibility and transform steps)."""
    collected = collect_summaries(episodes, spec, apply_eligibility=apply_eligibility, stay_anchors=stay_anchors)
    eligibility = collected["eligibility"]
    selected = collected["selected"]
    excluded = collected["excluded"]
    metas = [episode.meta for episode in episodes]

    probes: list[dict[str, Any]] = []
    probe_error = None
    try:
        probes = single_feature_probes(collected["summaries"], collected["labels"])
    except ValueError as error:
        probe_error = str(error)

    result: dict[str, Any] = {
        "setting": "v1_raw" if spec is None else ("v2_task" if eligibility else "v2_transform_only"),
        "spec": None if spec is None else spec.to_dict(),
        "eligibility_applied": eligibility,
        "n_loaded": len(episodes),
        "label_counts_loaded": _label_counts(metas),
        "n_audited": len(selected),
        "label_counts": _label_counts([episode.meta for episode in selected]),
        "exclusions": summarize_exclusions(excluded) if eligibility else None,
        "rows_by_label": {
            kind: {str(n_rows): count for n_rows, count in sorted(counts.items())}
            for kind, counts in collected["rows_by_label"].items()
        },
        "transform": collected["transform"],
        "separability_threshold": separability_threshold,
        "probe_error": probe_error,
        "probes": probes,
        "flagged": flag_leaks(probes, separability_threshold),
    }
    if include_excluded_rows:
        result["excluded"] = excluded
    return result


def audit_episode_ids(
    episode_dir: Path | str,
    episode_ids: Sequence[str],
    spec: TaskSpec | None = None,
    *,
    apply_eligibility: bool = True,
    stay_anchors: Mapping[int, Sequence[pd.Timestamp]] | None = None,
    separability_threshold: float = DEFAULT_SEPARABILITY_THRESHOLD,
) -> dict[str, Any]:
    """Load only ``episode_ids`` from ``episode_dir`` and audit them.

    ``spec=None`` audits the v1 arrays as stored; a spec applies eligibility and
    :func:`transform_episode` first. Returns probe rows, flagged rows and label counts.
    """
    return audit_loaded_episodes(
        load_episodes(episode_dir, episode_ids),
        spec,
        apply_eligibility=apply_eligibility,
        stay_anchors=stay_anchors,
        separability_threshold=separability_threshold,
    )


def anchor_context_counts(
    metas: Sequence[Mapping[str, Any]],
    stay_anchors: Mapping[int, Sequence[pd.Timestamp]],
    spec: TaskSpec,
    lookback_hours: float = 24.0,
) -> dict[str, int]:
    """Residual-risk counts that the current rules do not exclude.

    ``positives_with_earlier_anchor_in_window``: another stay anchor in
    ``[anchor - window_hours, anchor)`` (clustered anchors). ``controls_after_recent_anchor``:
    a stay anchor in ``[window_start - lookback, window_start)``, i.e. the patient may
    already be on vasopressors during the control window. ``controls_after_any_anchor``: any
    stay anchor before ``window_start``.
    """
    window = pd.Timedelta(hours=float(spec.window_hours))
    lookback = pd.Timedelta(hours=float(lookback_hours))
    counts = Counter()
    for meta in metas:
        anchors = [pd.Timestamp(a) for a in stay_anchors.get(int(meta["stay_id"]), ())]
        if episode_type(meta) == POSITIVE:
            anchor = pd.Timestamp(meta["anchor_time"])
            counts["positives"] += 1
            if any(anchor - window <= other < anchor for other in anchors):
                counts["positives_with_earlier_anchor_in_window"] += 1
        else:
            start = pd.Timestamp(meta["window_start"])
            counts["controls"] += 1
            if any(start - lookback <= other < start for other in anchors):
                counts["controls_after_recent_anchor"] += 1
            if any(other < start for other in anchors):
                counts["controls_after_any_anchor"] += 1
    keys = (
        "positives",
        "positives_with_earlier_anchor_in_window",
        "controls",
        "controls_after_recent_anchor",
        "controls_after_any_anchor",
    )
    return {key: int(counts.get(key, 0)) for key in keys}


# --------------------------------------------------------------------------- split runner


def read_split_manifest(path: Path | str) -> dict[str, list[str]]:
    """Episode ids per split from a manifest like ``icde_comparator_split_manifest.json``."""
    with open(path, encoding="utf-8") as handle:
        data = json.load(handle)
    split = data.get("split", data)
    return {
        str(name): [str(item) for item in ids]
        for name, ids in split.items()
        if isinstance(ids, list) and all(isinstance(item, str) for item in ids)
    }


def find_probe(audit: Mapping[str, Any], feature: str, summary: str) -> dict[str, Any] | None:
    """The probe row for ``(feature, summary)`` with its 1-based rank, or None."""
    for rank, row in enumerate(audit.get("probes", ()), start=1):
        if row["feature"] == feature and row["summary"] == summary:
            return {**row, "rank": rank}
    return None


def _multivariate_suite(
    episodes: Sequence[EpisodeArrays],
    split_by_episode: Mapping[str, str],
    settings: Mapping[str, tuple[TaskSpec | None, bool]],
    stay_anchors: Mapping[int, Sequence[pd.Timestamp]],
    test_split: str,
    n_bootstrap: int,
) -> dict[str, Any]:
    """:func:`multivariate_probe` for every setting and view. Train = every split except
    ``test_split`` (val folded into train); CV folds and bootstrap grouped by subject."""
    groups = {
        episode.episode_id: episode.meta.get("subject_id") or episode.meta.get("stay_id") or episode.episode_id
        for episode in episodes
    }
    results: dict[str, Any] = {}
    for setting, (setting_spec, eligibility) in settings.items():
        collected = collect_summaries(episodes, setting_spec, apply_eligibility=eligibility, stay_anchors=stay_anchors)
        train = [eid for eid in collected["summaries"] if split_by_episode[eid] != test_split]
        test = [eid for eid in collected["summaries"] if split_by_episode[eid] == test_split]
        results[setting] = {}
        for view in MULTIVARIATE_VIEWS:
            try:
                results[setting][view] = multivariate_probe(
                    collected["summaries"], collected["labels"], train, test, view=view, groups=groups, n_bootstrap=n_bootstrap
                )
            except ValueError as error:
                results[setting][view] = {"view": view, "error": str(error)}
    return results


def _multivariate_headline(results: Mapping[str, Mapping[str, Mapping[str, Any]]]) -> dict[str, Any]:
    """Compact test metrics per ``setting/view``."""
    headline: dict[str, Any] = {}
    for setting, views in results.items():
        for view, result in views.items():
            key = f"{setting}/{view}"
            if "error" in result:
                headline[key] = {"error": result["error"]}
                continue
            headline[key] = {
                "n_train": result["n_train"],
                "n_test": result["n_test"],
                "test_prevalence": result["test_prevalence"],
                "chosen_c": result["cv"]["chosen_c"],
                **{
                    f"{short}_{metric}": result[model].get(metric)
                    for short, model in (("lr", "logistic_regression"), ("tree", "decision_tree"))
                    for metric in ("test_auroc", "test_auroc_ci95", "test_auprc", "test_auprc_ci95")
                },
            }
    return headline


def run_split_audit(
    manifest_path: Path | str,
    episode_dir: Path | str,
    spec: TaskSpec | None = None,
    *,
    separability_threshold: float = DEFAULT_SEPARABILITY_THRESHOLD,
    candidate_specs: Mapping[str, TaskSpec] | None = None,
    test_split: str = "test",
    multivariate_bootstrap: int = 1000,
) -> dict[str, Any]:
    """Load every manifest episode once and audit v1 against the v2 spec in memory.

    Single-feature variants: ``v1_raw``; ``v2_proposed`` (eligibility with stay anchors from
    the positives in the manifest, then transform); ``v2_transform_only`` (no eligibility);
    ``v2_without_refit`` (``reimpute_after_cutoff=False``); ``v2_with_control_paired_anchors``
    (index also holds each control's paired anchor); ``v1_raw_<split>`` per split. Each of
    ``candidate_specs`` (default :data:`CANDIDATE_SPECS`) gets a compact summary. The
    multivariate probe runs for ``v1_raw``, ``v2_proposed`` and ``v2_transform_only`` in
    every view, trained on all splits except ``test_split`` and scored on ``test_split``.
    """
    spec = spec or TaskSpec()
    splits = read_split_manifest(manifest_path)
    split_of: dict[str, str] = {}
    for name, ids in splits.items():
        for episode_id in ids:
            split_of.setdefault(episode_id, name)
    episodes = load_episodes(episode_dir, list(split_of))
    split_by_episode = {episode.episode_id: split_of[requested] for requested, episode in zip(split_of, episodes)}
    metas = [episode.meta for episode in episodes]
    positive_index = build_stay_anchor_index(metas)
    paired_index = build_stay_anchor_index(metas, include_control_anchors=True)
    common = {"separability_threshold": separability_threshold}

    audits: dict[str, Any] = {
        "v1_raw": audit_loaded_episodes(episodes, None, **common),
        "v2_proposed": audit_loaded_episodes(episodes, spec, stay_anchors=positive_index, **common),
        "v2_transform_only": audit_loaded_episodes(
            episodes, spec, apply_eligibility=False, include_excluded_rows=False, **common
        ),
        "v2_without_refit": audit_loaded_episodes(
            episodes,
            replace(spec, reimpute_after_cutoff=False),
            stay_anchors=positive_index,
            include_excluded_rows=False,
            **common,
        ),
        "v2_with_control_paired_anchors": audit_loaded_episodes(
            episodes, spec, stay_anchors=paired_index, include_excluded_rows=False, **common
        ),
    }
    for name in splits:
        subset = [episode for episode in episodes if split_by_episode[episode.episode_id] == name]
        audits[f"v1_raw_{name}"] = audit_loaded_episodes(subset, None, include_excluded_rows=False, **common)

    kept_metas, _ = apply_task(metas, spec, stay_anchors=positive_index)
    v1, v2 = audits["v1_raw"], audits["v2_proposed"]
    headline = {
        "v1_vasopressor_on_last": find_probe(v1, "vasopressor_on", "last"),
        "v1_window_n_rows": find_probe(v1, WINDOW_KEY, "n_rows"),
        "v1_top_probe": v1["probes"][0] if v1["probes"] else None,
        "v2_top_probe": v2["probes"][0] if v2["probes"] else None,
        "v1_n_flagged": len(v1["flagged"]),
        "v2_n_flagged": len(v2["flagged"]),
        "v2_label_counts": v2["label_counts"],
        "v2_exclusions": v2["exclusions"],
    }
    for name in splits:
        headline[f"v1_{name}_vasopressor_on_last"] = find_probe(audits[f"v1_raw_{name}"], "vasopressor_on", "last")

    candidates: dict[str, Any] = {}
    for name, candidate in (CANDIDATE_SPECS if candidate_specs is None else candidate_specs).items():
        audit = audit_loaded_episodes(
            episodes, candidate, stay_anchors=positive_index, include_excluded_rows=False, **common
        )
        candidates[name] = {
            "spec": candidate.to_dict(),
            "label_counts": audit["label_counts"],
            "exclusions": audit["exclusions"],
            "rows_by_label": audit["rows_by_label"],
            "n_flagged": len(audit["flagged"]),
            "probe_error": audit["probe_error"],
            "top_probes": audit["probes"][:5],
        }

    multivariate: dict[str, Any] = {"note": f"no {test_split!r} split in the manifest; skipped"}
    if test_split in splits and len(splits) > 1:
        settings = {"v1_raw": (None, False), "v2_proposed": (spec, True), "v2_transform_only": (spec, False)}
        results = _multivariate_suite(
            episodes, split_by_episode, settings, positive_index, test_split, multivariate_bootstrap
        )
        multivariate = {
            "train_splits": [name for name in splits if name != test_split],
            "test_split": test_split,
            "note": "val is folded into train; C is chosen by grouped CV on train only; groups = subject_id",
            "settings": results,
        }
        headline["multivariate"] = _multivariate_headline(results)
    return {
        "manifest": str(manifest_path),
        "episode_dir": str(episode_dir),
        "n_ids": len(split_of),
        "split_counts": {name: len(ids) for name, ids in splits.items()},
        "spec": spec.to_dict(),
        "headline": headline,
        "anchor_context_v2_kept": {
            "positive_anchor_index": anchor_context_counts(kept_metas, positive_index, spec),
            "with_control_paired_anchors": anchor_context_counts(kept_metas, paired_index, spec),
        },
        "audits": audits,
        "candidate_specs": candidates,
        "multivariate": multivariate,
    }


# ---------------------------------------------------------------------------------- output


def json_safe(value: Any) -> Any:
    """Convert numpy/pandas scalars to Python and non-finite floats to None for strict JSON."""
    if isinstance(value, Mapping):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return [json_safe(item) for item in value.tolist()]
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if math.isfinite(float(value)) else None
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return value


def format_probe_table(rows: Sequence[Mapping[str, Any]], k: int = 10) -> str:
    """Markdown table of the top-``k`` probe rows."""
    lines = [
        "| # | feature | summary | separability | AUROC | AUPRC (oriented) | direction | mean pos | mean ctrl |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for rank, row in enumerate(rows[:k], start=1):
        lines.append(
            "| {rank} | {feature} | {summary} | {sep:.3f} | {auroc:.3f} | {auprc:.3f} | {direction} | {mp} | {mn} |".format(
                rank=rank,
                feature=row["feature"],
                summary=row["summary"],
                sep=row["separability"],
                auroc=row["auroc"],
                auprc=row["auprc_oriented"],
                direction=row["direction"],
                mp="nan" if row["mean_pos"] is None or not math.isfinite(row["mean_pos"]) else f"{row['mean_pos']:.3f}",
                mn="nan" if row["mean_neg"] is None or not math.isfinite(row["mean_neg"]) else f"{row['mean_neg']:.3f}",
            )
        )
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    """Run :func:`run_split_audit` with the proposed default spec and write the JSON."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--manifest", default="docs/experiments/icde_comparator_split_manifest.json")
    parser.add_argument("--episode-dir", default="data/processed/mimic-iv-full-cw3-gap0-real/mimic-iv")
    parser.add_argument("--out-json", default="artifacts/v2/leakage_audit/locked500_audit.json")
    args = parser.parse_args(argv)

    result = run_split_audit(args.manifest, args.episode_dir)
    out = Path(args.out_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as handle:
        json.dump(json_safe(result), handle, indent=2, ensure_ascii=False)
    headline = result["headline"]
    print(f"wrote {out} ({result['n_ids']} episodes)")
    print(f"v1 flagged={headline['v1_n_flagged']} v2 flagged={headline['v2_n_flagged']} v2 kept={headline['v2_label_counts']}")
    for key, row in headline.get("multivariate", {}).items():
        if "error" in row:
            print(f"{key}: {row['error']}")
            continue
        print(
            f"{key}: n={row['n_train']}/{row['n_test']} prev={row['test_prevalence']:.3f} "
            f"LR AUROC={row['lr_test_auroc']} AUPRC={row['lr_test_auprc']} | "
            f"tree AUROC={row['tree_test_auroc']} AUPRC={row['tree_test_auprc']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
