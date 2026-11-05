"""Frozen bridge protocol for discovery-to-prediction mapping.

All discovery methods share one vectorization + validation-selected
logistic probe. This avoids method-specific bridge advantages.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


def build_bridge_features(
    programs: dict[str, Any],
    manifest: dict,
    feature_names: list[str] | None = None,
) -> tuple[np.ndarray, list[str]]:
    """Convert episode-level program firings into a binary feature matrix."""
    top_k = int(manifest.get("top_k", 10))
    encoding = manifest.get("bridge_encoding", "binary")

    all_program_ids: list[str] = []
    episode_rows: list[dict[str, float]] = []

    for episode_id, episode_programs in programs.items():
        row: dict[str, float] = {}
        sorted_programs = sorted(episode_programs.items(), key=lambda x: -x[1].get("score", 0))[:top_k]
        for prog_id, prog_data in sorted_programs:
            if encoding == "binary":
                row[prog_id] = 1.0
            else:
                row[prog_id] = float(prog_data.get("score", 1.0))
            if prog_id not in all_program_ids:
                all_program_ids.append(prog_id)
        episode_rows.append(row)

    if feature_names:
        ordered_names = feature_names
    else:
        ordered_names = all_program_ids

    matrix = np.zeros((len(episode_rows), len(ordered_names)), dtype=float)
    for i, row in enumerate(episode_rows):
        for j, name in enumerate(ordered_names):
            matrix[i, j] = row.get(name, 0.0)

    return matrix, ordered_names


def select_best_logistic_probe(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    c_grid: list[float],
    metric: str = "auprc",
) -> dict:
    """Select the best L2-regularized logistic probe by validation performance.

    When the validation set is empty, falls back to training-data metric
    so that a model is always selected and test evaluation can proceed.
    """
    from sklearn.metrics import average_precision_score, roc_auc_score

    if x_train.ndim != 2 or x_train.shape[0] == 0 or x_train.shape[1] == 0:
        return {
            "model": None,
            "selected_c": None,
            "best_validation_score": float("nan"),
        }
    if np.unique(y_train).size < 2:
        return {
            "model": None,
            "selected_c": None,
            "best_validation_score": float("nan"),
        }

    best_score = -1.0
    best_model = None
    best_c = None

    for c in c_grid:
        model = LogisticRegression(C=float(c), max_iter=1000, random_state=42, class_weight="balanced")
        model.fit(x_train, y_train)

        if x_val.shape[0] > 0:
            val_prob = model.predict_proba(x_val)[:, 1]
            if metric == "auprc":
                score = float(average_precision_score(y_val, val_prob))
            else:
                score = float(roc_auc_score(y_val, val_prob))
        elif x_train.shape[0] > 0:
            # Fallback: score on training data when no validation set
            train_prob = model.predict_proba(x_train)[:, 1]
            if metric == "auprc":
                score = float(average_precision_score(y_train, train_prob))
            else:
                score = float(roc_auc_score(y_train, train_prob))
        else:
            score = -1.0

        if score >= best_score:
            best_score = score
            best_model = model
            best_c = c

    return {
        "model": best_model,
        "selected_c": best_c,
        "best_validation_score": best_score,
    }


def export_bridge_result(
    best: dict,
    feature_names: list[str],
    test_programs: dict[str, Any],
    test_labels: np.ndarray,
    manifest: dict,
) -> dict:
    """Evaluate the selected bridge on the test set and export results."""
    x_test, _ = build_bridge_features(test_programs, manifest, feature_names=feature_names)

    if best["model"] is None or x_test.shape[0] == 0:
        return {
            "selected_c": best.get("selected_c"),
            "best_validation_score": best.get("best_validation_score"),
            "test_metrics": {},
            "rows": {},
        }

    from .metrics import compute_binary_metrics

    test_prob = best["model"].predict_proba(x_test)[:, 1]
    test_metrics = compute_binary_metrics(test_labels, test_prob)

    return {
        "selected_c": best.get("selected_c"),
        "best_validation_score": best.get("best_validation_score"),
        "test_metrics": test_metrics,
        "feature_count": len(feature_names),
    }


def fit_frozen_bridge(
    train_programs: dict[str, Any],
    val_programs: dict[str, Any],
    test_programs: dict[str, Any],
    labels: dict[str, list],
    manifest: dict,
) -> dict:
    """Run the full frozen bridge protocol.

    All discovery methods share exactly this bridge: same vectorization,
    same L2 regularization grid, same validation selection rule.
    """
    x_train, feature_names = build_bridge_features(train_programs, manifest)
    x_val, _ = build_bridge_features(val_programs, manifest, feature_names=feature_names)

    y_train = np.asarray(labels.get("train", []), dtype=int)
    y_val = np.asarray(labels.get("val", []), dtype=int)

    best = select_best_logistic_probe(
        x_train, y_train,
        x_val, y_val,
        c_grid=manifest.get("logistic_probe_grid", [0.01, 0.1, 1.0]),
        metric=manifest.get("bridge_selection_metric", "auprc"),
    )

    return export_bridge_result(
        best=best,
        feature_names=feature_names,
        test_programs=test_programs,
        test_labels=np.asarray(labels.get("test", []), dtype=int),
        manifest=manifest,
    )
