#!/usr/bin/env python3
"""Proxy fidelity, extended (review of the v3 experiments, round 2): the analysis of scripts/v2/analyze_selection_proxies_heldout.py
over the first ``--steps`` picks (default 20), plus shortlist recall: at each pick, whether the candidate with the largest gain
in VAL AUPRC is among the top 10 of each score, and the mean VAL gain of each score's top 10. Same definitions: the current set
grows along the schema-unaware likelihood-greedy path (the best candidate by cross-validated log-likelihood gain, source
schema); truth = gain in VAL AUPRC of an L2 logistic probe (class-balanced, C = 1) fit on TRAIN; proxies = mRMR's and NMIFS's
scores and SRS's cross-validated estimate (``objective``). Spearman correlations with a Fisher-z 95% interval.

  PYTHONDONTWRITEBYTECODE=1 /d/BRCoo/miniconda/python scripts/v2/paper_v3/proxy_fidelity_ext.py --pools artifacts/v2/comparison/f_mimic_iv_gap1h/pools.pkl --tag f_mimic_iv_gap1h
"""
import argparse
import importlib.util
import math
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
spec = importlib.util.spec_from_file_location("proxies", ROOT / "scripts/v2/analyze_selection_proxies_heldout.py")
px = importlib.util.module_from_spec(spec)
spec.loader.exec_module(px)

from sklearn.linear_model import LogisticRegression  # noqa: E402

from src.icu.v2.pipeline import episodes_by_split  # noqa: E402
from src.icu.v2.robust_selection import _Objective, grouped_folds  # noqa: E402
from src.icu.v2.selection_baselines import information_gain, mutual_information  # noqa: E402
from src.icu.v2.stats import metric_value  # noqa: E402
from src.icu.v2.union import union_pool  # noqa: E402


def fisher_ci(r: float, n: int) -> tuple[float, float]:
    if n <= 3 or not np.isfinite(r) or abs(r) >= 1:
        return float("nan"), float("nan")
    z, se = math.atanh(r), 1.0 / math.sqrt(n - 3)
    return math.tanh(z - 1.96 * se), math.tanh(z + 1.96 * se)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pools", required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--steps", type=int, default=20)
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
    X = px.matrix(tf, ids)
    Xv = px.matrix([firings[c.program_id] for c in cands], val_ids)
    y = np.array([1.0 if e in pos else 0.0 for e in ids])
    yv = np.array([float(e.label) for e in val])
    obj = _Objective(X, y, grouped_folds(ids, subject_of, 3, 0), np.ones((1, len(cands))), np.array([1.0]), 1.0, "frozen")
    relevance = np.array([information_gain(f, train_ids, pos) for f in tf])
    h = np.array([px.entropy(len(f) / len(ids)) for f in tf])
    group = np.array([c.method for c in cands])

    def val_auprc(cols):
        if not cols:
            return float(yv.mean())
        model = LogisticRegression(C=1.0, class_weight="balanced", max_iter=1000).fit(X[:, cols], y)
        return metric_value(yv, model.predict_proba(Xv[:, cols])[:, 1], "auprc")

    selected, current_ll, corr, recall = [], 0.0, [], []
    for step in range(args.steps):
        rest = [j for j in range(len(cands)) if j not in selected]
        base = val_auprc(selected)
        gain = np.array([val_auprc(selected + [j]) - base for j in rest])
        ll = np.array([obj(selected + [j]) - current_ll for j in rest])
        if selected:
            mi = np.array([[mutual_information(tf[j], tf[s], train_ids) for s in selected] for j in rest])
            scale = np.array([[min(h[j], h[s]) for s in selected] for j in rest])
            proxies = {"mrmr": relevance[rest] - mi.mean(axis=1),
                       "nmifs": relevance[rest] - np.where(scale > 0, mi / np.where(scale > 0, scale, 1), 0).mean(axis=1)}
        else:
            proxies = {"mrmr": relevance[rest], "nmifs": relevance[rest]}
        proxies["objective"] = ll
        best = int(np.argmax(gain))
        for pname, score in proxies.items():
            top = np.argsort(-score, kind="stable")[:10]
            rank_of_best = int((score > score[best]).sum()) + 1
            recall.append({"tag": args.tag, "step": step, "pick": step + 1, "proxy": pname, "best_in_top10": int(best in set(top.tolist())),
                           "rank_of_best": rank_of_best, "top10_mean_gain": float(gain[top].mean()), "best_gain": float(gain[best]),
                           "chosen_gain": float(gain[int(np.argmax(score))])})
            scopes = [("union", np.ones(len(rest), dtype=bool))] + [(m, group[rest] == m) for m in sorted(set(group))]
            for scope, mask in scopes:
                if mask.sum() < 5:
                    continue
                r = spearmanr(score[mask], gain[mask]).correlation
                lo, hi = fisher_ci(r, int(mask.sum()))
                corr.append({"tag": args.tag, "step": step, "pick": step + 1, "scope": scope, "proxy": pname, "n": int(mask.sum()),
                             "spearman": r, "low": lo, "high": hi})
        selected.append(rest[int(np.argmax(ll))])
        current_ll = obj(selected)
        u = {r["proxy"]: r["spearman"] for r in corr if r["step"] == step and r["scope"] == "union"}
        print(f"[fidelity] {args.tag} pick {step + 1}: " + ", ".join(f"{k} {v:.2f}" for k, v in u.items()), flush=True)
    out = ROOT / "artifacts/v2/proxy_correlation"
    pd.DataFrame(corr).to_csv(out / f"ext_corr_{args.tag}.csv", index=False)
    pd.DataFrame(recall).to_csv(out / f"ext_recall_{args.tag}.csv", index=False)
    print("wrote", out / f"ext_corr_{args.tag}.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
