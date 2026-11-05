"""Per-candidate scores behind the proxy-fidelity analysis (Section 2 motivation figure): at steps 1 and 2 of forward
selection on a union pool, every remaining candidate's mRMR score, NMIFS score, SRS's cross-validated estimate (gain in
CV log-likelihood) and its held-out truth (gain in VALIDATION AUPRC). Same definitions as
scripts/v2/analyze_selection_proxies_heldout.py, which reports only the correlations.

  PYTHONDONTWRITEBYTECODE=1 /d/BRCoo/miniconda/python scripts/v2/paper_v3/dump_proxy_candidates.py gap1h
"""
import math
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from src.icu.v2.pipeline import episodes_by_split  # noqa: E402
from src.icu.v2.robust_selection import _Objective, grouped_folds  # noqa: E402
from src.icu.v2.selection_baselines import information_gain, mutual_information  # noqa: E402
from src.icu.v2.stats import metric_value  # noqa: E402
from src.icu.v2.union import union_pool  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data import generator_of  # noqa: E402


def entropy(rate: float) -> float:
    return 0.0 if rate <= 0 or rate >= 1 else -(rate * math.log2(rate) + (1 - rate) * math.log2(1 - rate))


def matrix(firings, ids):
    index = {e: i for i, e in enumerate(ids)}
    X = np.zeros((len(ids), len(firings)))
    for j, f in enumerate(firings):
        rows = [index[e] for e in f if e in index]
        X[rows, j] = 1.0
    return X


label = sys.argv[1] if len(sys.argv) > 1 else "gap1h"
pools = ROOT / f"artifacts/v2/comparison/f_mimic_iv_{label}/pools.pkl"
with open(pools, "rb") as h:
    state = pickle.load(h)
grouped = episodes_by_split(state["episodes"])
train, val = grouped["train"], grouped["val"]
ids = sorted(e.episode_id for e in train)
val_ids = [e.episode_id for e in val]
train_ids = frozenset(ids)
pos = frozenset(e.episode_id for e in train if e.label == 1)
subject_of = {e.episode_id: e.subject_id for e in train}
cands, firings, _ = union_pool(state)
tf = [firings[c.program_id] & train_ids for c in cands]
X = matrix(tf, ids)
Xv = matrix([firings[c.program_id] for c in cands], val_ids)
y = np.array([1.0 if e in pos else 0.0 for e in ids])
yv = np.array([float(e.label) for e in val])
obj = _Objective(X, y, grouped_folds(ids, subject_of, 3, 0), np.ones((1, len(cands))), np.array([1.0]), 1.0, "frozen")
relevance = np.array([information_gain(f, train_ids, pos) for f in tf])
h = np.array([entropy(len(f) / len(ids)) for f in tf])


def val_auprc(cols):
    if not cols:
        return float(yv.mean())
    model = LogisticRegression(C=1.0, class_weight="balanced", max_iter=1000).fit(X[:, cols], y)
    return metric_value(yv, model.predict_proba(Xv[:, cols])[:, 1], "auprc")


rows, selected, current_ll = [], [], 0.0
for step in range(3):
    rest = [j for j in range(len(cands)) if j not in selected]
    base = val_auprc(selected)
    ll_gain = np.array([obj(selected + [j]) - current_ll for j in rest])
    if step > 0:
        auprc_gain = np.array([val_auprc(selected + [j]) - base for j in rest])
        mi = np.array([[mutual_information(tf[j], tf[s], train_ids) for s in selected] for j in rest])
        scale = np.array([[min(h[j], h[s]) for s in selected] for j in rest])
        mrmr = relevance[rest] - mi.mean(axis=1)
        nmifs = relevance[rest] - np.where(scale > 0, mi / np.where(scale > 0, scale, 1), 0).mean(axis=1)
        for k, j in enumerate(rest):
            rows.append({"task": label, "step": step, "program_id": cands[j].program_id, "generator": generator_of(cands[j].program_id),
                         "train_rate": len(tf[j]) / len(ids), "mrmr": mrmr[k], "nmifs": nmifs[k], "srs_estimate": ll_gain[k],
                         "val_auprc_gain": auprc_gain[k]})
        print(f"step {step}: {len(rest)} candidates", flush=True)
    selected.append(rest[int(np.argmax(ll_gain))])
    current_ll = obj(selected)
out = ROOT / f"paper/v3_assets/source_data/proxy_candidates_{label}.csv"
pd.DataFrame(rows).to_csv(out, index=False)
print("wrote", out)
