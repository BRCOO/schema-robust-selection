"""Matrix Profile-style motif baseline (ICDM 2016).

This implementation approximates cross-episode motif discovery over single
feature subsequences. It is label-agnostic during mining and therefore acts
as a classical motif primitive under the same frozen bridge.
"""

from __future__ import annotations

import numpy as np

from .mining_contract import MiningContract
from .subsequence_utils import impute_series, min_window_distance, sliding_windows, z_normalize


def compute_matrix_profile_motifs(
    episodes: list[dict],
    motif_distance_quantile: list[float],
    shapelet_length_grid: list[int],
    candidate_budget: int,
) -> list[dict]:
    """Mine cross-episode motifs from numeric subsequences."""
    if not episodes:
        return []

    feature_names = list(episodes[0].get("feature_names", []))
    lengths = [int(v) for v in shapelet_length_grid if int(v) > 1]
    if not feature_names or not lengths:
        return []

    candidates: list[dict] = []
    episode_subset = episodes[: min(len(episodes), 20)]
    for feature_idx, feature_name in enumerate(feature_names[: min(len(feature_names), 10)]):
        for length in lengths:
            for episode in episode_subset:
                matrix = np.asarray(episode.get("matrix"))
                mask = np.asarray(episode.get("mask"))
                if matrix.ndim != 2 or feature_idx >= matrix.shape[1]:
                    continue
                series = impute_series(matrix[:, feature_idx], mask[:, feature_idx] if mask.ndim == 2 else None)
                windows = sliding_windows(series, length, step=max(1, length // 2))
                for window in windows[:2]:
                    distances = []
                    for other in episodes:
                        other_matrix = np.asarray(other.get("matrix"))
                        other_mask = np.asarray(other.get("mask"))
                        if other_matrix.ndim != 2 or feature_idx >= other_matrix.shape[1]:
                            continue
                        other_series = impute_series(
                            other_matrix[:, feature_idx],
                            other_mask[:, feature_idx] if other_mask.ndim == 2 else None,
                        )
                        dist = min_window_distance(other_series, window)
                        if np.isfinite(dist):
                            distances.append(float(dist))
                    if len(distances) < 3:
                        continue
                    mean_dist = float(np.mean(distances))
                    for quantile in motif_distance_quantile:
                        threshold = float(np.quantile(distances, min(max(float(quantile), 0.0), 1.0)))
                        support = float(np.mean(np.asarray(distances, dtype=float) <= threshold))
                        score = support * (1.0 - float(quantile)) / max(mean_dist, 1e-6)
                        candidates.append(
                            {
                                "motif_id": f"matrix_profile_{len(candidates)}",
                                "feature": feature_name,
                                "feature_index": int(feature_idx),
                                "subsequence_length": int(length),
                                "distance_quantile": float(quantile),
                                "distance_threshold": float(threshold),
                                "support": float(support),
                                "mean_distance": float(mean_dist),
                                "template": z_normalize(window).round(6).tolist(),
                                "score": float(score),
                            }
                        )
                    if len(candidates) >= max(candidate_budget * 2, 200):
                        break
                if len(candidates) >= max(candidate_budget * 2, 200):
                    break
            if len(candidates) >= max(candidate_budget * 2, 200):
                break
        if len(candidates) >= max(candidate_budget * 2, 200):
            break

    candidates.sort(
        key=lambda row: (
            -float(row["score"]),
            -float(row["support"]),
            float(row["mean_distance"]),
            row["feature"],
        )
    )
    return candidates[:candidate_budget]


def normalize_mining_output(
    mined: list[dict],
    top_k: int = 10,
) -> dict:
    """Normalize raw mining output into top_k_programs schema."""
    sorted_mined = sorted(mined, key=lambda x: -x.get("score", 0))[:top_k]
    return {
        "programs": {
            m["motif_id"]: {
                "score": float(m.get("score", 0)),
                "feature": str(m.get("feature", "")),
                "feature_index": int(m.get("feature_index", 0)),
                "subsequence_length": int(m.get("subsequence_length", 0)),
                "distance_quantile": float(m.get("distance_quantile", 0)),
                "distance_threshold": float(m.get("distance_threshold", 0)),
                "support": float(m.get("support", 0)),
                "mean_distance": float(m.get("mean_distance", 0)),
                "template": list(m.get("template", [])),
            }
            for m in sorted_mined
        },
        "program_count": len(sorted_mined),
        "top_k": top_k,
    }


def run_matrix_profile_baseline(
    episodes: list[dict],
    contract: MiningContract,
) -> dict:
    """Run Matrix Profile motif discovery and normalize output."""
    mined = compute_matrix_profile_motifs(
        episodes,
        motif_distance_quantile=contract.motif_distance_quantile,
        shapelet_length_grid=contract.shapelet_length_grid,
        candidate_budget=contract.candidate_budget,
    )
    return normalize_mining_output(mined=mined, top_k=contract.top_k)
