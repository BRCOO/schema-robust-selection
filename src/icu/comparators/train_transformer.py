from __future__ import annotations

from typing import Iterable

import numpy as np
import torch
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from .data_contract import EpisodeRecord


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


class _TransformerSeqClassifier(torch.nn.Module):
    def __init__(
        self,
        input_dim: int,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
        max_len: int = 256,
    ) -> None:
        super().__init__()
        self.input_proj = torch.nn.Linear(input_dim * 3, d_model)
        self.pos = torch.nn.Parameter(torch.zeros(1, max_len, d_model))
        layer = torch.nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = torch.nn.TransformerEncoder(layer, num_layers=num_layers)
        self.head = torch.nn.Sequential(
            torch.nn.LayerNorm(d_model),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(d_model, 1),
        )

    def forward(self, values: torch.Tensor, observed_mask: torch.Tensor, deltas: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        x = torch.cat([values, observed_mask, deltas], dim=-1)
        x = self.input_proj(x)
        seq_len = x.shape[1]
        x = x + self.pos[:, :seq_len, :]

        arange = torch.arange(seq_len, device=x.device).unsqueeze(0)
        key_padding_mask = arange >= lengths.unsqueeze(1)
        enc = self.encoder(x, src_key_padding_mask=key_padding_mask)

        valid = (~key_padding_mask).unsqueeze(-1).float()
        pooled = (enc * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1.0)
        return self.head(pooled).squeeze(-1)


def train_evaluate_transformer(
    records: list[EpisodeRecord],
    split: dict,
    random_state: int = 42,
    epochs: int = 10,
    d_model: int = 64,
    nhead: int = 4,
    num_layers: int = 2,
    learning_rate: float = 1e-3,
) -> dict:
    if not records:
        raise ValueError("records must not be empty")

    torch.manual_seed(int(random_state))
    np.random.seed(int(random_state))

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

    device = torch.device("cpu")
    model = _TransformerSeqClassifier(
        input_dim=feature_dim,
        d_model=int(d_model),
        nhead=int(nhead),
        num_layers=int(num_layers),
        dropout=0.1,
        max_len=max_len,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(learning_rate), weight_decay=1e-3)

    pos_count = float(np.sum(train_y == 1))
    neg_count = float(np.sum(train_y == 0))
    pos_weight = torch.tensor([neg_count / max(pos_count, 1.0)], dtype=torch.float32, device=device)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    x_t = torch.tensor(train_x, dtype=torch.float32, device=device)
    m_t = torch.tensor(train_m, dtype=torch.float32, device=device)
    d_t = torch.tensor(train_d, dtype=torch.float32, device=device)
    y_t = torch.tensor(train_y, dtype=torch.float32, device=device)
    len_t = torch.tensor(train_len, dtype=torch.long, device=device)

    val_x_t = torch.tensor(val_x, dtype=torch.float32, device=device)
    val_m_t = torch.tensor(val_m, dtype=torch.float32, device=device)
    val_d_t = torch.tensor(val_d, dtype=torch.float32, device=device)
    val_len_t = torch.tensor(val_len, dtype=torch.long, device=device)
    val_y_t = torch.tensor(val_y, dtype=torch.float32, device=device)

    best_state = None
    best_val_loss = float("inf")
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

        if val_x.shape[0] > 0:
            model.eval()
            with torch.no_grad():
                val_logits = model(val_x_t, val_m_t, val_d_t, val_len_t)
                val_loss = criterion(val_logits, val_y_t)
                if torch.isfinite(val_loss) and float(val_loss.item()) < best_val_loss:
                    best_val_loss = float(val_loss.item())
                    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            model.train()

    if best_state is not None:
        model.load_state_dict(best_state)

    def predict_probs(x_np: np.ndarray, m_np: np.ndarray, d_np: np.ndarray, len_np: np.ndarray) -> np.ndarray:
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

    train_prob = predict_probs(train_x, train_m, train_d, train_len)
    val_prob = predict_probs(val_x, val_m, val_d, val_len)
    test_prob = predict_probs(test_x, test_m, test_d, test_len)

    return {
        "model_name": "transformer_seq",
        "metrics": {
            "train": _compute_binary_metrics(train_y, train_prob),
            "val": _compute_binary_metrics(val_y, val_prob),
            "test": _compute_binary_metrics(test_y, test_prob),
        },
        "predictions": {
            "train": {"y_true": train_y.astype(int).tolist(), "y_prob": train_prob.astype(float).tolist()},
            "val": {"y_true": val_y.astype(int).tolist(), "y_prob": val_prob.astype(float).tolist()},
            "test": {"y_true": test_y.astype(int).tolist(), "y_prob": test_prob.astype(float).tolist()},
        },
        "split_counts": {
            "train": int(train_y.shape[0]),
            "val": int(val_y.shape[0]),
            "test": int(test_y.shape[0]),
        },
        "train_config": {
            "epochs": int(epochs),
            "d_model": int(d_model),
            "nhead": int(nhead),
            "num_layers": int(num_layers),
            "learning_rate": float(learning_rate),
            "random_state": int(random_state),
        },
    }
