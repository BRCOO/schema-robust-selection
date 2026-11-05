#!/usr/bin/env python3
"""Why proxy scores fail across generators: rank correlation of the mRMR score with the true
marginal value of each candidate, within each generator's candidates and over their union.

True value = increase of the cross-validated log-likelihood of the frozen probe (source schema
only, the fs_cv objective) when the candidate is added to the current set; proxy = mRMR's MID
score I(q; y) - mean_s I(q; s). Spearman correlation over the remaining candidates at each of the
first ``--steps`` steps of forward selection (the set grows by the best candidate by true value).

  PYTHONDONTWRITEBYTECODE=1 /d/BRCoo/miniconda/python scripts/v2/analyze_selection_proxies.py \
      --pools artifacts/v2/comparison/f_mimic_iv_gap1h/pools.pkl --out artifacts/v2/proxy_correlation/f_mimic_iv_gap1h.csv
"""

import argparse
import csv
import pickle
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.icu.v2.pipeline import episodes_by_split  # noqa: E402
from src.icu.v2.robust_selection import _Objective, grouped_folds  # noqa: E402
from src.icu.v2.selection_baselines import information_gain, mutual_information  # noqa: E402
from src.icu.v2.union import union_pool  # noqa: E402


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pools", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--steps", type=int, default=5)
    args = parser.parse_args(argv)
    with open(args.pools, "rb") as h:
        state = pickle.load(h)
    train = episodes_by_split(state["episodes"])["train"]
    ids = sorted(e.episode_id for e in train)
    train_ids = frozenset(ids)
    pos = frozenset(e.episode_id for e in train if e.label == 1)
    subject_of = {e.episode_id: e.subject_id for e in train}
    cands, firings, _ = union_pool(state)
    tf = [firings[c.program_id] & train_ids for c in cands]
    row = {e: i for i, e in enumerate(ids)}
    X = np.zeros((len(ids), len(cands)))
    for j, f in enumerate(tf):
        for e in f:
            X[row[e], j] = 1.0
    y = np.array([1.0 if e in pos else 0.0 for e in ids])
    obj = _Objective(X, y, grouped_folds(ids, subject_of, 3, 0), np.ones((1, len(cands))), np.array([1.0]), 1.0, "frozen")
    relevance = np.array([information_gain(f, train_ids, pos) for f in tf])
    group = np.array([c.method for c in cands])
    selected, current, records = [], 0.0, []
    for step in range(args.steps):
        rest = [j for j in range(len(cands)) if j not in selected]
        gains = np.array([obj(selected + [j]) - current for j in rest])
        red = np.array([np.mean([mutual_information(tf[j], tf[s], train_ids) for s in selected]) for j in rest]) if selected else np.zeros(len(rest))
        score = relevance[rest] - red
        pick = rest[int(np.argmax(score))]
        records.append({"step": step, "scope": "union", "n": len(rest), "spearman": spearmanr(score, gains).correlation,
                        "regret_of_proxy_pick": float(gains.max() - gains[rest.index(pick)])})
        for m in sorted(set(group)):
            mask = group[rest] == m
            if mask.sum() >= 5:
                records.append({"step": step, "scope": m, "n": int(mask.sum()), "spearman": spearmanr(score[mask], gains[mask]).correlation,
                                "regret_of_proxy_pick": float(gains[mask].max() - gains[mask][int(np.argmax(score[mask]))])})
        selected.append(rest[int(np.argmax(gains))])
        current = obj(selected)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
