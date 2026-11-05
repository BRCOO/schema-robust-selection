#!/usr/bin/env python3
"""Reviewer item J: the proxy-correlation analysis without circularity.

``analyze_selection_proxies.py`` measures a candidate's true marginal value by the gain in the
cross-validated log-likelihood that SRS itself maximizes. Here a second, independent truth is
added: the gain in VALIDATION AUPRC of the probe (L2 logistic, class-balanced, C = 1, fit on
TRAIN) when the candidate joins the current set. Two proxies, and SRS's own cross-validated estimate (``objective``, against the held-out truth only), are scored:
mRMR's MID score I(q; y) - mean_s I(q; s), and NMIFS's, whose redundancy is normalized by
min(H(q), H(s)), i.e. by firing rate. Spearman correlation over the remaining candidates at each
of the first ``--steps`` steps, within each generator and on the union; the set grows by the best
candidate by cross-validated log-likelihood, as in the original analysis.

  PYTHONDONTWRITEBYTECODE=1 /d/BRCoo/miniconda/python scripts/v2/analyze_selection_proxies_heldout.py \\
      --pools artifacts/v2/comparison/f_mimic_iv_gap1h/pools.pkl --out artifacts/v2/proxy_correlation/heldout_f_mimic_iv_gap1h.csv
"""

import argparse
import csv
import math
import pickle
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.icu.v2.pipeline import episodes_by_split  # noqa: E402
from src.icu.v2.robust_selection import _Objective, grouped_folds  # noqa: E402
from src.icu.v2.selection_baselines import information_gain, mutual_information  # noqa: E402
from src.icu.v2.stats import metric_value  # noqa: E402
from src.icu.v2.union import union_pool  # noqa: E402


def entropy(rate: float) -> float:
    return 0.0 if rate <= 0 or rate >= 1 else -(rate * math.log2(rate) + (1 - rate) * math.log2(1 - rate))


def matrix(firings, ids):
    row = {e: i for i, e in enumerate(ids)}
    X = np.zeros((len(ids), len(firings)))
    for j, f in enumerate(firings):
        for e in f:
            if e in row:
                X[row[e], j] = 1.0
    return X


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pools", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--steps", type=int, default=4)
    args = parser.parse_args(argv)
    with open(args.pools, "rb") as h:
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
    group = np.array([c.method for c in cands])

    def val_auprc(cols):
        if not cols:
            return float(yv.mean())
        model = LogisticRegression(C=1.0, class_weight="balanced", max_iter=1000).fit(X[:, cols], y)
        return metric_value(yv, model.predict_proba(Xv[:, cols])[:, 1], "auprc")

    selected, current_ll, records = [], 0.0, []
    for step in range(args.steps):
        rest = [j for j in range(len(cands)) if j not in selected]
        base_auprc = val_auprc(selected)
        truth = {"cv_loglik": np.array([obj(selected + [j]) - current_ll for j in rest]),
                 "val_auprc": np.array([val_auprc(selected + [j]) - base_auprc for j in rest])}
        if selected:
            mi = np.array([[mutual_information(tf[j], tf[s], train_ids) for s in selected] for j in rest])
            scale = np.array([[min(h[j], h[s]) for s in selected] for j in rest])
            proxies = {"mrmr": relevance[rest] - mi.mean(axis=1),
                       "nmifs": relevance[rest] - np.where(scale > 0, mi / np.where(scale > 0, scale, 1), 0).mean(axis=1)}
        else:
            proxies = {"mrmr": relevance[rest], "nmifs": relevance[rest]}
        proxies["objective"] = truth["cv_loglik"]  # SRS's own estimate, scored against the held-out truth
        for pname, score in proxies.items():
            for tname, gain in truth.items():
                if pname == "objective" and tname == "cv_loglik":
                    continue
                records.append({"step": step, "scope": "union", "proxy": pname, "truth": tname, "n": len(rest),
                                "spearman": spearmanr(score, gain).correlation})
                for m in sorted(set(group)):
                    mask = group[rest] == m
                    if mask.sum() >= 5:
                        records.append({"step": step, "scope": m, "proxy": pname, "truth": tname, "n": int(mask.sum()),
                                        "spearman": spearmanr(score[mask], gain[mask]).correlation})
        selected.append(rest[int(np.argmax(truth["cv_loglik"]))])
        current_ll = obj(selected)
        print(f"[proxy] step {step}: " + ", ".join(f"{r['proxy']}~{r['truth']} {r['spearman']:.2f}" for r in records if r['step'] == step and r['scope'] == 'union'), flush=True)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
