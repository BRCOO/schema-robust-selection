"""Fit all 7 supervised comparator baselines and return callable predictors.

Each baseline returns:
    - model_name: str
    - family: str
    - predict: callable(records) -> np.ndarray of probabilities (same order)
    - internal_metrics: dict (test-split metrics on internal data)
    - internal_auprc: float

Usage:
    predictors = fit_all_supervised(records, split, seed=42, epochs=1)
    probs_lr = predictors["lr_stats"]["predict"](external_records)
"""

from __future__ import annotations

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler

from .data_contract import EpisodeRecord
from .feature_builder import build_handcrafted_feature_matrix
from .metrics import compute_binary_metrics
from .sequence_bundle import build_sequence_bundle
from .models.grud import GRUDLite
from .train_retain import RETAINClassifier
from .train_mtan import MTANClassifier
from .train_adacare import AdaCareClassifier
from .train_strats import STraTSClassifier


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _records_by_id(records: list[EpisodeRecord]) -> dict[str, EpisodeRecord]:
    return {r.episode_id: r for r in records}


def _compute_delta(missing_mask: np.ndarray) -> np.ndarray:
    T, D = missing_mask.shape
    delta = np.zeros((T, D), dtype=float)
    for t in range(1, T):
        prev = delta[t - 1]
        obs = (~missing_mask[t - 1]).astype(float)
        delta[t] = (prev + 1.0) * (1.0 - obs)
    return delta


def _load_padded_arrays(
    by_id: dict[str, EpisodeRecord],
    episode_ids: list[str],
    max_len: int,
    feature_dim: int,
):
    """Load episodes as padded arrays (x, mask, delta, lengths)."""
    values, masks, deltas, lengths = [], [], [], []
    for eid in episode_ids:
        rec = by_id.get(eid)
        if rec is None:
            continue
        mat = np.load(rec.data_path).astype(float)
        miss = np.load(rec.mask_path).astype(bool)
        if mat.ndim != 2 or mat.shape[1] != feature_dim:
            continue
        steps = max(1, min(max_len, mat.shape[0]))
        px = np.zeros((max_len, feature_dim), dtype=float)
        pm = np.ones((max_len, feature_dim), dtype=bool)
        px[:steps] = mat[:steps]
        pm[:steps] = miss[:steps]
        values.append(px)
        masks.append((~pm).astype(float))
        deltas.append(_compute_delta(pm))
        lengths.append(int(steps))

    n = len(values)
    if n == 0:
        return (np.zeros((0, max_len, feature_dim), dtype=float),
                np.zeros((0, max_len, feature_dim), dtype=float),
                np.zeros((0, max_len, feature_dim), dtype=float),
                np.zeros((0,), dtype=int))
    return (np.asarray(values, dtype=float),
            np.asarray(masks, dtype=float),
            np.asarray(deltas, dtype=float),
            np.asarray(lengths, dtype=int))


def _fit_normalization(train_x: np.ndarray, train_m: np.ndarray):
    if train_x.size == 0:
        return np.zeros(train_x.shape[-1]), np.ones(train_x.shape[-1])
    obs = train_m > 0.5
    D = train_x.shape[-1]
    means, stds = np.zeros(D), np.ones(D)
    for d in range(D):
        vals = train_x[:, :, d][obs[:, :, d]]
        if vals.size == 0:
            continue
        means[d] = float(vals.mean())
        s = float(vals.std())
        stds[d] = s if s > 1e-6 else 1.0
    return means, stds


def _apply_norm(x, m, means, stds):
    normed = (x - means.reshape(1, 1, -1)) / stds.reshape(1, 1, -1)
    return np.where(m > 0.5, normed, 0.0).astype(float)


# ---------------------------------------------------------------------------
# Sklearn predictors
# ---------------------------------------------------------------------------

def _fit_lr(records, split, seed):
    bundle = build_handcrafted_feature_matrix(records)
    X, y = np.asarray(bundle["X"], dtype=float), np.asarray(bundle["y"], dtype=int)
    train_idx = [i for i, eid in enumerate(bundle["episode_ids"]) if eid in split.get("train", [])]
    test_idx = [i for i, eid in enumerate(bundle["episode_ids"]) if eid in split.get("test", [])]
    if not train_idx or np.unique(y[train_idx]).size < 2:
        return {"predict": None, "internal_metrics": {}, "internal_auprc": float("nan")}

    scaler = StandardScaler()
    model = LogisticRegression(max_iter=1000, random_state=int(seed), class_weight="balanced")
    model.fit(scaler.fit_transform(X[train_idx]), y[train_idx])

    internal = {}
    if test_idx:
        internal = compute_binary_metrics(y[test_idx], model.predict_proba(scaler.transform(X[test_idx]))[:, 1])

    def predict(recs):
        if not recs:
            return np.zeros((0,), dtype=float)
        b = build_handcrafted_feature_matrix(recs)
        Xe = np.asarray(b["X"], dtype=float)
        return model.predict_proba(scaler.transform(Xe))[:, 1].astype(float) if Xe.shape[0] else np.zeros((0,), dtype=float)

    return {"predict": predict, "internal_metrics": internal, "internal_auprc": float(internal.get("auprc", float("nan")))}


def _fit_gbdt(records, split, seed):
    bundle = build_handcrafted_feature_matrix(records)
    X, y = np.asarray(bundle["X"], dtype=float), np.asarray(bundle["y"], dtype=int)
    train_idx = [i for i, eid in enumerate(bundle["episode_ids"]) if eid in split.get("train", [])]
    test_idx = [i for i, eid in enumerate(bundle["episode_ids"]) if eid in split.get("test", [])]
    if not train_idx or np.unique(y[train_idx]).size < 2:
        return {"predict": None, "internal_metrics": {}, "internal_auprc": float("nan")}

    model = HistGradientBoostingClassifier(random_state=int(seed), learning_rate=0.05, max_depth=4, max_iter=300)
    model.fit(X[train_idx], y[train_idx])

    internal = {}
    if test_idx:
        internal = compute_binary_metrics(y[test_idx], model.predict_proba(X[test_idx])[:, 1])

    def predict(recs):
        if not recs:
            return np.zeros((0,), dtype=float)
        b = build_handcrafted_feature_matrix(recs)
        Xe = np.asarray(b["X"], dtype=float)
        return model.predict_proba(Xe)[:, 1].astype(float) if Xe.shape[0] else np.zeros((0,), dtype=float)

    return {"predict": predict, "internal_metrics": internal, "internal_auprc": float(internal.get("auprc", float("nan")))}


# ---------------------------------------------------------------------------
# Torch predictors — generic factory
# ---------------------------------------------------------------------------

def _fit_torch_model(
    records, split, seed, epochs, model_cls, forward_needs_lengths=True,
    model_kwargs=None, trainer_kwargs=None,
):
    """Train a torch model and return a dict with predict callable."""
    model_kwargs = dict(model_kwargs or {})
    trainer_kwargs = dict(trainer_kwargs or {})

    bundle = build_sequence_bundle(records, split)
    D, Lmax = bundle.feature_dim, bundle.max_len
    means, stds = _fit_normalization(bundle.train_x, bundle.train_m)
    device = torch.device("cpu")

    torch.manual_seed(int(seed))
    np.random.seed(int(seed))

    model = model_cls(input_dim=D, **model_kwargs).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(trainer_kwargs.get("lr", 1e-3)), weight_decay=1e-3)

    x_t = torch.tensor(bundle.train_x, dtype=torch.float32, device=device)
    m_t = torch.tensor(bundle.train_m, dtype=torch.float32, device=device)
    d_t = torch.tensor(bundle.train_d, dtype=torch.float32, device=device)
    y_t = torch.tensor(bundle.train_y, dtype=torch.float32, device=device)
    len_t = torch.tensor(bundle.train_len, dtype=torch.long, device=device)

    pos_w = torch.tensor([float(np.sum(bundle.train_y == 0)) / max(float(np.sum(bundle.train_y == 1)), 1.0)],
                         dtype=torch.float32, device=device)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_w)

    model.train()
    for _ in range(max(1, int(epochs))):
        opt.zero_grad(set_to_none=True)
        if forward_needs_lengths:
            logits = model(x_t, m_t, d_t, len_t)
        else:
            logits = model(x_t, m_t, d_t)
        loss = criterion(logits, y_t)
        if torch.isnan(loss):
            break
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
        opt.step()

    # Internal test metrics (from bundle test split)
    model.eval()
    with torch.no_grad():
        if forward_needs_lengths:
            test_logits = model(
                torch.tensor(bundle.test_x, dtype=torch.float32, device=device),
                torch.tensor(bundle.test_m, dtype=torch.float32, device=device),
                torch.tensor(bundle.test_d, dtype=torch.float32, device=device),
                torch.tensor(bundle.test_len, dtype=torch.long, device=device),
            ) if bundle.test_x.shape[0] else torch.zeros(0)
        else:
            test_logits = model(
                torch.tensor(bundle.test_x, dtype=torch.float32, device=device),
                torch.tensor(bundle.test_m, dtype=torch.float32, device=device),
                torch.tensor(bundle.test_d, dtype=torch.float32, device=device),
            ) if bundle.test_x.shape[0] else torch.zeros(0)
        test_prob = torch.sigmoid(test_logits).cpu().numpy().astype(float) if test_logits.shape[0] else np.zeros(0)
    internal = compute_binary_metrics(bundle.test_y, test_prob)

    # Predict function for arbitrary records
    def predict(ext_records):
        if not ext_records:
            return np.zeros((0,), dtype=float)
        by_id = _records_by_id(ext_records)
        eids = [r.episode_id for r in ext_records]
        x, m, d, lens = _load_padded_arrays(by_id, eids, Lmax, D)
        if x.shape[0] == 0:
            return np.zeros((0,), dtype=float)
        x = _apply_norm(x, m, means, stds)

        model.eval()
        with torch.no_grad():
            x_t = torch.tensor(x, dtype=torch.float32, device=device)
            m_t = torch.tensor(m, dtype=torch.float32, device=device)
            d_t = torch.tensor(d, dtype=torch.float32, device=device)
            if forward_needs_lengths:
                logits = model(x_t, m_t, d_t, torch.tensor(lens, dtype=torch.long, device=device))
            else:
                logits = model(x_t, m_t, d_t)
            return torch.sigmoid(logits).cpu().numpy().astype(float)

    return {"predict": predict, "internal_metrics": internal, "internal_auprc": float(internal.get("auprc", float("nan")))}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def fit_all_supervised(
    records: list[EpisodeRecord],
    split: dict,
    seed: int = 42,
    epochs: int = 1,
) -> dict[str, dict]:
    """Fit all 7 supervised baselines and return {name: {predict, ...}}."""

    trainers = [
        ("lr_stats",        "statistical",       lambda: _fit_lr(records, split, seed)),
        ("gbdt_stats",      "statistical",       lambda: _fit_gbdt(records, split, seed)),
        ("grud_lite",       "missingness_aware", lambda: _fit_torch_model(
            records, split, seed, epochs, GRUDLite,
            forward_needs_lengths=False,
            model_kwargs={"hidden_size": 32, "dropout": 0.1},
            trainer_kwargs={"lr": 1e-3},
        )),
        ("retain_seq",      "classic_clinical",  lambda: _fit_torch_model(
            records, split, seed, epochs, RETAINClassifier,
            forward_needs_lengths=True,
            model_kwargs={"hidden_size": 32, "dropout": 0.1, "max_len": max(r.shape[0] for r in records)},
            trainer_kwargs={"lr": 1e-3},
        )),
        ("mtan_seq",        "modern_irregular",  lambda: _fit_torch_model(
            records, split, seed, epochs, MTANClassifier,
            forward_needs_lengths=True,
            model_kwargs={"latent_dim": 32, "dropout": 0.1, "max_len": max(r.shape[0] for r in records)},
            trainer_kwargs={"lr": 1e-3},
        )),
        ("adacare_seq",     "modern_clinical",   lambda: _fit_torch_model(
            records, split, seed, epochs, AdaCareClassifier,
            forward_needs_lengths=True,
            model_kwargs={"hidden_size": 32, "dropout": 0.1, "max_len": max(r.shape[0] for r in records)},
            trainer_kwargs={"lr": 1e-3},
        )),
        ("strats_seq",      "modern_dependency", lambda: _fit_torch_model(
            records, split, seed, epochs, STraTSClassifier,
            forward_needs_lengths=True,
            model_kwargs={"d_model": 32, "num_layers": 1, "dropout": 0.1, "max_len": max(r.shape[0] for r in records)},
            trainer_kwargs={"lr": 1e-3},
        )),
    ]

    predictors: dict[str, dict] = {}
    for name, family, fn in trainers:
        try:
            result = fn()
            result["model_name"] = name
            result["family"] = family
            predictors[name] = result
            auprc = result.get("internal_auprc", float("nan"))
            status = "ok" if result.get("predict") is not None else "no_predict"
            print(f"  {name} ({family}): {status}, AUPRC={auprc:.4f}")
        except Exception as e:
            print(f"  {name} ({family}): FAILED — {e}")
            predictors[name] = {"model_name": name, "family": family,
                                "predict": None, "internal_metrics": {},
                                "internal_auprc": float("nan")}

    return predictors


def evaluate_on_records(model_entry: dict, records: list[EpisodeRecord]) -> dict:
    """Evaluate a predictor (from fit_all_supervised) on a list of records."""
    fn = model_entry.get("predict")
    if fn is None or not records:
        return {"count": len(records), "auprc": float("nan")}
    probs = fn(records)
    if probs.shape[0] == 0:
        return {"count": 0, "auprc": float("nan")}
    labels = np.asarray([1 if r.episode_type == "positive" else 0 for r in records], dtype=int)
    return compute_binary_metrics(labels, probs)
