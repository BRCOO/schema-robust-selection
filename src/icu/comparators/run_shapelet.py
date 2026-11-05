"""Interpretable shapelet baseline.

Mines single-feature discriminative subsequences from positive episodes and
scores them by positive-versus-control distance separation.
"""

from __future__ import annotations

import numpy as np

from .mining_contract import MiningContract
from .subsequence_utils import impute_series, min_window_distance_raw, sliding_windows


def mine_shapelet_patterns(
    episodes: list[dict],
    support_grid: list[float],
    shapelet_length_grid: list[int],
    candidate_budget: int,
) -> list[dict]:
    """Mine simple discriminative shapelets from positive episodes."""
    positives = [ep for ep in episodes if int(ep.get("label", 0)) == 1]
    controls = [ep for ep in episodes if int(ep.get("label", 0)) == 0]
    if not positives or not controls:
        return []

    feature_names = list(positives[0].get("feature_names", []))
    lengths = [int(v) for v in shapelet_length_grid if int(v) > 1]
    if not feature_names or not lengths:
        return []

    candidates: list[dict] = []
    positive_subset = positives[: min(len(positives), 24)]
    for feature_idx, feature_name in enumerate(feature_names[: min(len(feature_names), 10)]):
        for length in lengths:
            for episode in positive_subset:
                matrix = np.asarray(episode.get("matrix"))
                mask = np.asarray(episode.get("mask"))
                if matrix.ndim != 2 or feature_idx >= matrix.shape[1]:
                    continue
                series = impute_series(matrix[:, feature_idx], mask[:, feature_idx] if mask.ndim == 2 else None)
                windows = sliding_windows(series, length, step=max(1, length // 2))
                for window in windows[:2]:
                    pos_dists = []
                    ctrl_dists = []
                    for pos in positives:
                        pos_matrix = np.asarray(pos.get("matrix"))
                        pos_mask = np.asarray(pos.get("mask"))
                        if pos_matrix.ndim != 2 or feature_idx >= pos_matrix.shape[1]:
                            continue
                        pos_series = impute_series(
                            pos_matrix[:, feature_idx],
                            pos_mask[:, feature_idx] if pos_mask.ndim == 2 else None,
                        )
                        pos_dists.append(min_window_distance_raw(pos_series, window))
                    for ctrl in controls:
                        ctrl_matrix = np.asarray(ctrl.get("matrix"))
                        ctrl_mask = np.asarray(ctrl.get("mask"))
                        if ctrl_matrix.ndim != 2 or feature_idx >= ctrl_matrix.shape[1]:
                            continue
                        ctrl_series = impute_series(
                            ctrl_matrix[:, feature_idx],
                            ctrl_mask[:, feature_idx] if ctrl_mask.ndim == 2 else None,
                        )
                        ctrl_dists.append(min_window_distance_raw(ctrl_series, window))
                    if len(pos_dists) < 2 or len(ctrl_dists) < 2:
                        continue

                    pos_mean = float(np.mean(pos_dists))
                    ctrl_mean = float(np.mean(ctrl_dists))
                    score_gap = ctrl_mean - pos_mean
                    if score_gap <= 0.0:
                        continue

                    accepted = False
                    for min_support in support_grid:
                        threshold = float(np.quantile(pos_dists, min(0.95, max(0.50, 1.0 - float(min_support)))))
                        support = float(np.mean(np.asarray(pos_dists, dtype=float) <= threshold))
                        if support + 1e-12 < float(min_support):
                            continue
                        control_support = float(np.mean(np.asarray(ctrl_dists, dtype=float) <= threshold))
                        score = score_gap * support * max(1e-6, 1.0 - control_support)
                        accepted = True
                        candidates.append(
                            {
                                "shapelet_id": f"shapelet_{len(candidates)}",
                                "feature": feature_name,
                                "feature_index": int(feature_idx),
                                "subsequence_length": int(length),
                                "threshold": float(threshold),
                                "support": float(support),
                                "control_support": float(control_support),
                                "pos_mean_distance": float(pos_mean),
                                "control_mean_distance": float(ctrl_mean),
                                "template": np.asarray(window, dtype=float).round(6).tolist(),
                                "score": float(score),
                                "min_support": float(min_support),
                            }
                        )
                    if not accepted:
                        threshold = float(np.quantile(pos_dists, 0.75))
                        support = float(np.mean(np.asarray(pos_dists, dtype=float) <= threshold))
                        control_support = float(np.mean(np.asarray(ctrl_dists, dtype=float) <= threshold))
                        score = score_gap * max(support, 1e-6) * max(1e-6, 1.0 - control_support)
                        candidates.append(
                            {
                                "shapelet_id": f"shapelet_{len(candidates)}",
                                "feature": feature_name,
                                "feature_index": int(feature_idx),
                                "subsequence_length": int(length),
                                "threshold": float(threshold),
                                "support": float(support),
                                "control_support": float(control_support),
                                "pos_mean_distance": float(pos_mean),
                                "control_mean_distance": float(ctrl_mean),
                                "template": np.asarray(window, dtype=float).round(6).tolist(),
                                "score": float(score),
                                "min_support": float(min(support_grid) if support_grid else 0.0),
                            }
                        )
                    if len(candidates) >= max(candidate_budget * 2, 240):
                        break
                if len(candidates) >= max(candidate_budget * 2, 240):
                    break
            if len(candidates) >= max(candidate_budget * 2, 240):
                break
        if len(candidates) >= max(candidate_budget * 2, 240):
            break

    if not candidates:
        for feature_idx, feature_name in enumerate(feature_names[: min(len(feature_names), 4)]):
            for length in lengths[: min(len(lengths), 2)]:
                for episode in positive_subset[:2]:
                    matrix = np.asarray(episode.get("matrix"))
                    mask = np.asarray(episode.get("mask"))
                    if matrix.ndim != 2 or feature_idx >= matrix.shape[1]:
                        continue
                    series = impute_series(matrix[:, feature_idx], mask[:, feature_idx] if mask.ndim == 2 else None)
                    windows = sliding_windows(series, length, step=max(1, length // 2))
                    if not windows:
                        continue
                    window = np.asarray(windows[0], dtype=float)
                    pos_dists = []
                    ctrl_dists = []
                    for pos in positives:
                        pos_matrix = np.asarray(pos.get("matrix"))
                        pos_mask = np.asarray(pos.get("mask"))
                        if pos_matrix.ndim != 2 or feature_idx >= pos_matrix.shape[1]:
                            continue
                        pos_series = impute_series(
                            pos_matrix[:, feature_idx],
                            pos_mask[:, feature_idx] if pos_mask.ndim == 2 else None,
                        )
                        pos_dists.append(min_window_distance_raw(pos_series, window))
                    for ctrl in controls:
                        ctrl_matrix = np.asarray(ctrl.get("matrix"))
                        ctrl_mask = np.asarray(ctrl.get("mask"))
                        if ctrl_matrix.ndim != 2 or feature_idx >= ctrl_matrix.shape[1]:
                            continue
                        ctrl_series = impute_series(
                            ctrl_matrix[:, feature_idx],
                            ctrl_mask[:, feature_idx] if ctrl_mask.ndim == 2 else None,
                        )
                        ctrl_dists.append(min_window_distance_raw(ctrl_series, window))
                    if not pos_dists or not ctrl_dists:
                        continue
                    threshold = float(np.quantile(pos_dists, 0.80))
                    support = float(np.mean(np.asarray(pos_dists, dtype=float) <= threshold))
                    control_support = float(np.mean(np.asarray(ctrl_dists, dtype=float) <= threshold))
                    score = max(1e-6, support - control_support + 0.1 * abs(float(np.mean(window))))
                    candidates.append(
                        {
                            "shapelet_id": f"shapelet_{len(candidates)}",
                            "feature": feature_name,
                            "feature_index": int(feature_idx),
                            "subsequence_length": int(length),
                            "threshold": float(max(threshold, 1e-6)),
                            "support": float(support),
                            "control_support": float(control_support),
                            "pos_mean_distance": float(np.mean(pos_dists)),
                            "control_mean_distance": float(np.mean(ctrl_dists)),
                            "template": np.asarray(window, dtype=float).round(6).tolist(),
                            "score": float(score),
                            "min_support": float(min(support_grid) if support_grid else 0.0),
                        }
                    )
                    if len(candidates) >= max(candidate_budget, 10):
                        break
                if len(candidates) >= max(candidate_budget, 10):
                    break
            if len(candidates) >= max(candidate_budget, 10):
                break

    candidates.sort(
        key=lambda row: (
            -float(row["score"]),
            -float(row["support"]),
            float(row["pos_mean_distance"]),
            row["feature"],
        )
    )
    return candidates[:candidate_budget]


def normalize_mining_output(
    mined: list[dict],
    top_k: int = 10,
) -> dict:
    """Normalize shapelet output into the shared discovery schema."""
    sorted_mined = sorted(mined, key=lambda x: -x.get("score", 0))[:top_k]
    return {
        "programs": {
            row["shapelet_id"]: {
                "score": float(row.get("score", 0)),
                "feature": str(row.get("feature", "")),
                "feature_index": int(row.get("feature_index", 0)),
                "subsequence_length": int(row.get("subsequence_length", 0)),
                "threshold": float(row.get("threshold", 0)),
                "support": float(row.get("support", 0)),
                "control_support": float(row.get("control_support", 0)),
                "pos_mean_distance": float(row.get("pos_mean_distance", 0)),
                "control_mean_distance": float(row.get("control_mean_distance", 0)),
                "template": list(row.get("template", [])),
                "min_support": float(row.get("min_support", 0)),
            }
            for row in sorted_mined
        },
        "program_count": len(sorted_mined),
        "top_k": top_k,
    }


def run_shapelet_baseline(
    episodes: list[dict],
    contract: MiningContract,
) -> dict:
    """Run shapelet mining and normalize output."""
    mined = mine_shapelet_patterns(
        episodes,
        support_grid=contract.support_grid,
        shapelet_length_grid=contract.shapelet_length_grid,
        candidate_budget=contract.candidate_budget,
    )
    return normalize_mining_output(mined=mined, top_k=contract.top_k)
