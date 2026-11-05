"""Shared sequence bundle builder for deep learning baselines."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from .data_contract import EpisodeRecord


@dataclass
class SequenceBundle:
    """Padded sequence data bundle for training deep learning baselines."""
    feature_dim: int
    max_len: int
    train_x: np.ndarray
    train_m: np.ndarray
    train_d: np.ndarray
    train_y: np.ndarray
    train_len: np.ndarray
    val_x: np.ndarray
    val_m: np.ndarray
    val_d: np.ndarray
    val_y: np.ndarray
    val_len: np.ndarray
    test_x: np.ndarray
    test_m: np.ndarray
    test_d: np.ndarray
    test_y: np.ndarray
    test_len: np.ndarray


def _compute_delta(missing_mask: np.ndarray) -> np.ndarray:
    time_steps, feature_dim = missing_mask.shape
    delta = np.zeros((time_steps, feature_dim), dtype=float)
    for step in range(1, time_steps):
        prev = delta[step - 1]
        observed_prev = (~missing_mask[step - 1]).astype(float)
        delta[step] = (prev + 1.0) * (1.0 - observed_prev)
    return delta


def _records_by_id(records: list[EpisodeRecord]) -> dict[str, EpisodeRecord]:
    return {record.episode_id: record for record in records}


def _load_split_arrays(
    records_by_id: dict[str, EpisodeRecord],
    episode_ids: Iterable[str],
    max_len: int,
    feature_dim: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    values = []
    masks = []
    deltas = []
    labels = []
    lengths = []
    for episode_id in episode_ids:
        record = records_by_id.get(episode_id)
        if record is None:
            continue
        matrix = np.load(record.data_path).astype(float)
        missing_mask = np.load(record.mask_path).astype(bool)
        if matrix.ndim != 2 or matrix.shape[1] != feature_dim:
            continue
        steps = max(1, min(max_len, matrix.shape[0]))
        padded_x = np.zeros((max_len, feature_dim), dtype=float)
        padded_miss = np.ones((max_len, feature_dim), dtype=bool)
        padded_x[:steps] = matrix[:steps]
        padded_miss[:steps] = missing_mask[:steps]
        values.append(padded_x)
        masks.append((~padded_miss).astype(float))
        deltas.append(_compute_delta(padded_miss))
        labels.append(1 if record.episode_type == "positive" else 0)
        lengths.append(int(steps))
    if not values:
        return (
            np.zeros((0, max_len, feature_dim), dtype=float),
            np.zeros((0, max_len, feature_dim), dtype=float),
            np.zeros((0, max_len, feature_dim), dtype=float),
            np.zeros((0,), dtype=int),
            np.zeros((0,), dtype=int),
        )
    return (
        np.asarray(values, dtype=float),
        np.asarray(masks, dtype=float),
        np.asarray(deltas, dtype=float),
        np.asarray(labels, dtype=int),
        np.asarray(lengths, dtype=int),
    )


def _fit_normalization(train_x: np.ndarray, train_m: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if train_x.size == 0:
        return np.zeros((train_x.shape[-1],), dtype=float), np.ones((train_x.shape[-1],), dtype=float)
    observed = train_m > 0.5
    feature_dim = train_x.shape[-1]
    means = np.zeros((feature_dim,), dtype=float)
    stds = np.ones((feature_dim,), dtype=float)
    for feat in range(feature_dim):
        vals = train_x[:, :, feat][observed[:, :, feat]]
        if vals.size == 0:
            means[feat] = 0.0
            stds[feat] = 1.0
        else:
            means[feat] = float(vals.mean())
            std = float(vals.std())
            stds[feat] = std if std > 1e-6 else 1.0
    return means, stds


def _apply_normalization(x: np.ndarray, m: np.ndarray, means: np.ndarray, stds: np.ndarray) -> np.ndarray:
    normed = (x - means.reshape(1, 1, -1)) / stds.reshape(1, 1, -1)
    normed = np.where(m > 0.5, normed, 0.0)
    return normed.astype(float)


def build_sequence_bundle(
    records: list[EpisodeRecord],
    split: dict,
) -> SequenceBundle:
    """Build a padded sequence bundle from records and a split dict."""
    if not records:
        raise ValueError("records must not be empty")

    feature_dim = int(len(records[0].feature_names))
    max_len = max(int(record.shape[0]) for record in records)
    by_id = _records_by_id(records)

    train_ids = list(split.get("train", []))
    val_ids = list(split.get("val", []))
    test_ids = list(split.get("test", []))

    train_x, train_m, train_d, train_y, train_len = _load_split_arrays(by_id, train_ids, max_len=max_len, feature_dim=feature_dim)
    val_x, val_m, val_d, val_y, val_len = _load_split_arrays(by_id, val_ids, max_len=max_len, feature_dim=feature_dim)
    test_x, test_m, test_d, test_y, test_len = _load_split_arrays(by_id, test_ids, max_len=max_len, feature_dim=feature_dim)

    if train_x.shape[0] == 0:
        raise ValueError("train split is empty")
    if np.unique(train_y).size < 2:
        raise ValueError("train split must contain both classes")

    means, stds = _fit_normalization(train_x, train_m)
    train_x = _apply_normalization(train_x, train_m, means, stds)
    val_x = _apply_normalization(val_x, val_m, means, stds)
    test_x = _apply_normalization(test_x, test_m, means, stds)

    return SequenceBundle(
        feature_dim=feature_dim,
        max_len=max_len,
        train_x=train_x, train_m=train_m, train_d=train_d, train_y=train_y, train_len=train_len,
        val_x=val_x, val_m=val_m, val_d=val_d, val_y=val_y, val_len=val_len,
        test_x=test_x, test_m=test_m, test_d=test_d, test_y=test_y, test_len=test_len,
    )


def build_sparse_event_bundle(
    records: list[EpisodeRecord],
    split: dict,
    time_encoding: str = "relative_hour",
    include_observation_mask: bool = True,
) -> SequenceBundle:
    """Build a sequence bundle for sparse irregular series baselines.

    For STraTS-style models, the input is similar but with specific
    time encoding. Currently wraps the standard sequence bundle.
    """
    return build_sequence_bundle(records, split)


def fit_binary_sequence_model(
    model: torch.nn.Module,
    bundle: SequenceBundle,
    random_state: int = 42,
    epochs: int = 8,
    learning_rate: float = 1e-3,
    device: torch.device | None = None,
) -> dict:
    """Generic training loop for a binary sequence classifier.

    Returns the same metric/output dict format as other trainers.
    """
    import torch

    if device is None:
        device = torch.device("cpu")

    torch.manual_seed(int(random_state))
    np.random.seed(int(random_state))

    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(learning_rate), weight_decay=1e-3)

    pos_count = float(np.sum(bundle.train_y == 1))
    neg_count = float(np.sum(bundle.train_y == 0))
    pos_weight = torch.tensor([neg_count / max(pos_count, 1.0)], dtype=torch.float32, device=device)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    x_t = torch.tensor(bundle.train_x, dtype=torch.float32, device=device)
    m_t = torch.tensor(bundle.train_m, dtype=torch.float32, device=device)
    d_t = torch.tensor(bundle.train_d, dtype=torch.float32, device=device)
    y_t = torch.tensor(bundle.train_y, dtype=torch.float32, device=device)
    len_t = torch.tensor(bundle.train_len, dtype=torch.long, device=device)

    model.train()
    for _ in range(max(1, int(epochs))):
        optimizer.zero_grad(set_to_none=True)
        logits = model(x_t, m_t, d_t, len_t)
        loss = criterion(logits, y_t)
        if torch.isnan(loss):
            break
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
        optimizer.step()

    def predict_probs(x_np, m_np, d_np, len_np):
        if x_np.shape[0] == 0:
            return np.zeros((0,), dtype=float)
        model.eval()
        with torch.no_grad():
            logits = model(
                torch.tensor(x_np, dtype=torch.float32, device=device),
                torch.tensor(m_np, dtype=torch.float32, device=device),
                torch.tensor(d_np, dtype=torch.float32, device=device),
                torch.tensor(len_np, dtype=torch.long, device=device),
            )
            probs = torch.sigmoid(logits).cpu().numpy().astype(float)
        return probs

    train_prob = predict_probs(bundle.train_x, bundle.train_m, bundle.train_d, bundle.train_len)
    val_prob = predict_probs(bundle.val_x, bundle.val_m, bundle.val_d, bundle.val_len)
    test_prob = predict_probs(bundle.test_x, bundle.test_m, bundle.test_d, bundle.test_len)

    from .metrics import compute_binary_metrics

    return {
        "model_name": model.__class__.__name__,
        "metrics": {
            "train": compute_binary_metrics(bundle.train_y, train_prob),
            "val": compute_binary_metrics(bundle.val_y, val_prob),
            "test": compute_binary_metrics(bundle.test_y, test_prob),
        },
        "predictions": {
            "train": {"y_true": bundle.train_y.astype(int).tolist(), "y_prob": train_prob.astype(float).tolist()},
            "val": {"y_true": bundle.val_y.astype(int).tolist(), "y_prob": val_prob.astype(float).tolist()},
            "test": {"y_true": bundle.test_y.astype(int).tolist(), "y_prob": test_prob.astype(float).tolist()},
        },
        "split_counts": {
            "train": int(bundle.train_y.shape[0]),
            "val": int(bundle.val_y.shape[0]),
            "test": int(bundle.test_y.shape[0]),
        },
    }
