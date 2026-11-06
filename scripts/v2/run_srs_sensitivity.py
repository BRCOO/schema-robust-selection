#!/usr/bin/env python3
"""Hyper-parameter sensitivity of SRS-C-V on the union pool (one run, K from the E3 manifest).

One hyper-parameter at a time is moved away from the SRS-C-V configuration used everywhere else
(surrogates, known targets eICU + HiRID, fast SRS with shortlist 10, 3 patient-grouped folds,
C = 1 in the selection objective, at most 5 surrogates, 20 random group-dropout scenarios, and
the dropout rate q picked on validation from {0, 0.25}):

* ``q`` fixed at 0, 0.1, 0.25 or 0.5 (no validation pick);
* ``max_surrogates`` 1, 3, 10; ``n_folds`` 5; ``C`` 0.1, 10; ``n_dropout`` 10, 40 -- each with the
  validation pick of q, as SRS-C-V.

Every registry is deployed with its surrogates. Reported: internal test AUPRC, mean target AUPRC,
the chosen q, and the overlap (|shared programs| / K) with the default registry, which is checked
to be the E3 ``srs_c_val`` registry.

  CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1 /d/BRCoo/miniconda/python scripts/v2/run_srs_sensitivity.py \\
      --e3-dir artifacts/v2/union/f_mimic_iv_gap1h --out-dir artifacts/v2/srs_sensitivity/f_mimic_iv_gap1h
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import pickle
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.icu.v2.bridge import fit_frozen_bridge_v2  # noqa: E402
from src.icu.v2.pipeline import episodes_by_split, restrict_firings  # noqa: E402
from src.icu.v2.robust_selection import scenario_set, schema_robust_select  # noqa: E402
from src.icu.v2.selection_baselines import SelectionInput, train_matrix  # noqa: E402
from src.icu.v2.stats import metric_value  # noqa: E402
from src.icu.v2.surrogates import CompositeRegistry, refire_composite, surrogate_chains  # noqa: E402
from src.icu.v2.union import union_pool  # noqa: E402

DEFAULT = {"max_surrogates": 5, "n_folds": 3, "C": 1.0, "n_dropout": 20, "seed": 0}
VARIANTS = [("q", q) for q in (0.0, 0.1, 0.25, 0.5)] + [("max_surrogates", m) for m in (1, 3, 10)] + [("n_folds", 5)] + \
           [("C", c) for c in (0.1, 10.0)] + [("n_dropout", d) for d in (10, 40)] + [("seed", s) for s in (1, 2, 3, 4)]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--e3-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--decision", default="loglik", choices=("loglik", "auprc"), help="auprc = SRS-AP")
    parser.add_argument("--reference", default="", help="registries.json:name the default run must reproduce (default: the E3 srs_c_val)")
    args = parser.parse_args(argv)
    started = time.perf_counter()
    spec = importlib.util.spec_from_file_location("sb", ROOT / "scripts/v2/run_selection_baselines.py")
    sb = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sb)
    e3 = Path(args.e3_dir)
    manifest = json.loads((e3 / "manifest.json").read_text(encoding="utf-8"))
    k = int(manifest["args"]["k"])
    pools_path = ROOT / manifest["args"]["pools"]
    with open(pools_path, "rb") as handle:
        state = pickle.load(handle)
    source = sb.source_schema(state, pools_path.parent / "config.json")
    grouped = episodes_by_split(state["episodes"])
    ids = {name: [e.episode_id for e in rows] for name, rows in grouped.items()}
    labels = {name: [e.label for e in rows] for name, rows in grouped.items()}
    train_ids = frozenset(ids["train"])
    train_pos = frozenset(e.episode_id for e in grouped["train"] if e.label == 1)
    subject_of = {e.episode_id: e.subject_id for e in grouped["train"]}
    known = {t.name: t.features for t in state["targets"] if t.name in ("eicu", "hirid")}
    candidates, firings, external = union_pool(state)
    inp = SelectionInput(tuple(candidates), {c.program_id: firings[c.program_id] & train_ids for c in candidates}, train_ids, train_pos)
    _, X, _ = train_matrix(inp)
    inp = SelectionInput(tuple(candidates), {}, train_ids, train_pos, matrix=X)
    chains_cache: dict[int, list] = {}
    test_y = np.array(labels["test"], dtype=int)

    def select(q: float, max_surrogates: int, n_folds: int, C: float, n_dropout: int, seed: int) -> CompositeRegistry:
        if max_surrogates not in chains_cache:
            chains_cache[max_surrogates] = surrogate_chains(candidates, X, max_surrogates)
        result = schema_robust_select(inp, k, scenario_set(source, known, dropout_q=q, n_dropout=n_dropout), subject_of, n_folds=n_folds, C=C,
                                      surrogates=True, max_surrogates=max_surrogates, shortlist=10, chains=chains_cache[max_surrogates],
                                      seed=seed, decision=args.decision)
        return CompositeRegistry(tuple(result.registry), result.surrogates)

    def evaluate(comp: CompositeRegistry) -> dict[str, float]:
        primaries = list(comp.primaries)
        f = {p.program_id: firings[p.program_id] for p in primaries}
        bridge = fit_frozen_bridge_v2(primaries, {s: restrict_firings(f, frozenset(ids[s])) for s in ids}, ids, labels)
        p_test = np.array([bridge["test_probabilities"][e] for e in ids["test"]])
        values = []
        for t in state["targets"]:
            eids = [e.episode_id for e in t.episodes]
            out = refire_composite(bridge["model_bundle"], comp, t.features, external[t.name], eids)
            values.append(metric_value([e.label for e in t.episodes], [out["probabilities"][e] for e in eids], "auprc"))
        return {"validation_score": bridge["validation_score"], "internal_auprc": metric_value(test_y, p_test, "auprc"), "mean_target_auprc": float(np.mean(values))}

    def with_pick(params: dict[str, Any]) -> tuple[CompositeRegistry, float, dict[str, float]]:
        pair = {q: select(q, **params) for q in (0.0, 0.25)}
        scores = {q: evaluate(comp) for q, comp in pair.items()}
        q = 0.25 if scores[0.25]["validation_score"] > scores[0.0]["validation_score"] + 1e-12 else 0.0
        return pair[q], q, scores[q]

    base, base_q, base_scores = with_pick(dict(DEFAULT))
    ref_path, ref_name = (args.reference.rsplit(":", 1) if args.reference else (str(e3 / "registries.json"), "srs_c_val"))
    saved = json.loads(Path(ref_path).read_text(encoding="utf-8"))[ref_name]
    reproduces = [p.program_id for p in base.primaries] == saved["primaries"]
    base_ids = {p.program_id for p in base.primaries}
    rows = [{"parameter": "default", "value": "", "chosen_q": base_q, "overlap_with_default": 1.0, "reproduces_e3": reproduces, **base_scores}]
    print(f"[sens] {e3}: default q = {base_q}, reproduces E3 srs_c_val: {reproduces} ({time.perf_counter() - started:.0f}s)", flush=True)
    for parameter, value in VARIANTS:
        params = dict(DEFAULT)
        if parameter == "q":
            comp, q = select(value, **params), value
            scores = evaluate(comp)
        else:
            params[parameter] = value
            comp, q, scores = with_pick(params)
        overlap = len(base_ids & {p.program_id for p in comp.primaries}) / max(1, len(base_ids))
        rows.append({"parameter": parameter, "value": value, "chosen_q": q, "overlap_with_default": overlap, **scores})
        print(f"[sens] {parameter}={value}: q {q}, internal {scores['internal_auprc']:.3f}, target {scores['mean_target_auprc']:.3f}, overlap {overlap:.1f} "
              f"({time.perf_counter() - started:.0f}s)", flush=True)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "sensitivity.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (out / "manifest.json").write_text(json.dumps({"e3_dir": str(e3), "k": k, "default": DEFAULT, "decision": args.decision, "reference": args.reference,
                                                   "elapsed_seconds": time.perf_counter() - started}, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
