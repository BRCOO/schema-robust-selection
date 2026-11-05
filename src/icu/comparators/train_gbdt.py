from __future__ import annotations

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score


def _compute_binary_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> dict:
    metrics = {
        "count": int(y_true.shape[0]),
        "positive_ratio": float(np.mean(y_true)) if y_true.size else 0.0,
        "auroc": float("nan"),
        "auprc": float("nan"),
        "brier": float("nan"),
    }
    if y_true.size == 0:
        return metrics
    try:
        if np.unique(y_true).size > 1:
            metrics["auroc"] = float(roc_auc_score(y_true, y_prob))
            metrics["auprc"] = float(average_precision_score(y_true, y_prob))
    except Exception:
        pass
    try:
        metrics["brier"] = float(brier_score_loss(y_true, y_prob))
    except Exception:
        pass
    return metrics


def _indices_for_split(bundle: dict, split_ids: list[str]) -> np.ndarray:
    id_to_idx = {eid: idx for idx, eid in enumerate(bundle["episode_ids"])}
    indices = [id_to_idx[eid] for eid in split_ids if eid in id_to_idx]
    return np.asarray(indices, dtype=int)


def train_evaluate_gbdt(bundle: dict, split: dict, random_state: int = 42) -> dict:
    X = np.asarray(bundle["X"], dtype=float)
    y = np.asarray(bundle["y"], dtype=int)

    train_idx = _indices_for_split(bundle, list(split.get("train", [])))
    val_idx = _indices_for_split(bundle, list(split.get("val", [])))
    test_idx = _indices_for_split(bundle, list(split.get("test", [])))

    if train_idx.size == 0:
        raise ValueError("train split is empty")
    y_train = y[train_idx]
    if np.unique(y_train).size < 2:
        raise ValueError("train split must contain both classes")

    model = HistGradientBoostingClassifier(
        random_state=int(random_state),
        learning_rate=0.05,
        max_depth=4,
        max_iter=300,
    )
    model.fit(X[train_idx], y_train)

    def predict_split(indices: np.ndarray) -> tuple[dict, np.ndarray, np.ndarray]:
        if indices.size == 0:
            empty_y = np.asarray([], dtype=int)
            empty_prob = np.asarray([], dtype=float)
            return _compute_binary_metrics(empty_y, empty_prob), empty_y, empty_prob
        probs = model.predict_proba(X[indices])[:, 1]
        y_true = y[indices]
        return _compute_binary_metrics(y_true, probs), y_true, probs

    train_metrics, train_y, train_prob = predict_split(train_idx)
    val_metrics, val_y, val_prob = predict_split(val_idx)
    test_metrics, test_y, test_prob = predict_split(test_idx)

    return {
        "model_name": "gbdt_stats",
        "metrics": {
            "train": train_metrics,
            "val": val_metrics,
            "test": test_metrics,
        },
        "predictions": {
            "train": {"y_true": train_y.astype(int).tolist(), "y_prob": train_prob.astype(float).tolist()},
            "val": {"y_true": val_y.astype(int).tolist(), "y_prob": val_prob.astype(float).tolist()},
            "test": {"y_true": test_y.astype(int).tolist(), "y_prob": test_prob.astype(float).tolist()},
        },
        "split_counts": {
            "train": int(train_idx.size),
            "val": int(val_idx.size),
            "test": int(test_idx.size),
        },
    }
