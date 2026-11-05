from __future__ import annotations

from typing import Iterable
import warnings

import numpy as np

from .data_contract import EpisodeRecord


DEFAULT_INTERVENTION_FEATURES = ("ventilation_on", "vasopressor_on", "sedation_on")
DEFAULT_SUMMARY_STATS = ("min", "max", "mean", "std", "last", "slope", "missing_rate")


def _safe_stat(values: np.ndarray, func_name: str) -> float:
    if values.size == 0:
        return np.nan
    if func_name == "min":
        return float(np.min(values))
    if func_name == "max":
        return float(np.max(values))
    if func_name == "mean":
        return float(np.mean(values))
    if func_name == "std":
        return float(np.std(values))
    if func_name == "last":
        return float(values[-1])
    raise ValueError(f"unsupported stat: {func_name}")


def _compute_slope(series: np.ndarray, mask: np.ndarray) -> float:
    observed = np.where(~mask)[0]
    if observed.size < 2:
        return 0.0
    y = series[observed]
    x = observed.astype(float)
    x_center = x - x.mean()
    denom = float(np.sum(x_center**2))
    if denom <= 0.0:
        return 0.0
    slope = float(np.sum(x_center * (y - y.mean())) / denom)
    return slope


def _transition_count(series: np.ndarray, mask: np.ndarray) -> float:
    valid = series[~mask]
    if valid.size < 2:
        return 0.0
    clipped = (valid > 0.5).astype(int)
    return float(np.sum(np.abs(np.diff(clipped))))


def _compute_episode_feature_vector(
    matrix: np.ndarray,
    missing_mask: np.ndarray,
    feature_names: list[str],
    intervention_features: Iterable[str],
    summary_stats: Iterable[str],
) -> tuple[np.ndarray, list[str]]:
    values: list[float] = []
    names: list[str] = []

    intervention_set = set(intervention_features)
    for feature_idx, feature_name in enumerate(feature_names):
        series = matrix[:, feature_idx].astype(float)
        mask = missing_mask[:, feature_idx].astype(bool)
        observed_values = series[~mask]

        for stat_name in summary_stats:
            if stat_name == "slope":
                stat_value = _compute_slope(series, mask)
            elif stat_name == "missing_rate":
                stat_value = float(np.mean(mask.astype(float)))
            else:
                stat_value = _safe_stat(observed_values, stat_name)
            values.append(stat_value)
            names.append(f"{feature_name}__{stat_name}")

        if feature_name in intervention_set:
            values.append(_transition_count(series, mask))
            names.append(f"{feature_name}__transition_count")
            values.append(float(np.mean((series[~mask] > 0.5).astype(float))) if observed_values.size else 0.0)
            names.append(f"{feature_name}__on_ratio")

    return np.asarray(values, dtype=float), names


def build_handcrafted_feature_matrix(
    records: list[EpisodeRecord],
    intervention_features: Iterable[str] = DEFAULT_INTERVENTION_FEATURES,
    summary_stats: Iterable[str] = DEFAULT_SUMMARY_STATS,
) -> dict:
    if not records:
        return {
            "X": np.zeros((0, 0), dtype=float),
            "y": np.zeros((0,), dtype=int),
            "feature_names": [],
            "episode_ids": [],
            "stay_ids": [],
        }

    vectors: list[np.ndarray] = []
    labels: list[int] = []
    episode_ids: list[str] = []
    stay_ids: list[int] = []
    feature_name_order: list[str] | None = None

    for record in records:
        matrix = np.load(record.data_path)
        mask = np.load(record.mask_path)
        vector, names = _compute_episode_feature_vector(
            matrix=matrix,
            missing_mask=mask,
            feature_names=list(record.feature_names),
            intervention_features=intervention_features,
            summary_stats=summary_stats,
        )
        if feature_name_order is None:
            feature_name_order = names
        elif names != feature_name_order:
            raise ValueError("feature name order mismatch across episodes")
        vectors.append(vector)
        labels.append(1 if record.episode_type == "positive" else 0)
        episode_ids.append(record.episode_id)
        stay_ids.append(int(record.stay_id))

    X = np.vstack(vectors).astype(float)
    if np.isnan(X).any():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            col_median = np.nanmedian(X, axis=0)
        col_median = np.where(np.isnan(col_median), 0.0, col_median)
        row_idx, col_idx = np.where(np.isnan(X))
        X[row_idx, col_idx] = col_median[col_idx]

    return {
        "X": X,
        "y": np.asarray(labels, dtype=int),
        "feature_names": list(feature_name_order or []),
        "episode_ids": episode_ids,
        "stay_ids": stay_ids,
    }
