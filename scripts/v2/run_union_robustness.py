#!/usr/bin/env python3
"""Decoder and subgroup robustness of the union-pool registries (E3 + add-on), for one run.

Decoder counterfactual: every registry is re-scored with a gradient-boosted tree decoder on the
same K binary program coordinates (interactions allowed; hyper-parameters chosen on VALIDATION
from ``GBDT_GRID``, fit on TRAIN only, as in ``run_decoder_counterfactual.py``). Targets use the
composite deployment for both decoders: each coordinate takes the firing of its first executable
branch (0 when none), nothing is refit at the target. Reported: internal test and mean target
AUPRC per decoder, Kendall tau between the two decoders' rankings of the registries, and the
paired bootstrap of SRS-C-V minus every other registry under the tree decoder.

Subgroups: sex (F / M) and age (< 65 / >= 65) on the internal test split and the eICU and HiRID
cohorts (demographics from the raw tables through ``run_subgroup_analysis.load_demographics``),
frozen logistic probe with surrogates: AUPRC per subgroup with patient-clustered bootstrap CIs, and
the paired bootstrap of SRS-C-V minus mRMR inside every subgroup.

  CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1 /d/BRCoo/miniconda/python scripts/v2/run_union_robustness.py \\
      --e3-dir artifacts/v2/union/f_mimic_iv_gap1h --extra-dir artifacts/v2/union_extra/union/f_mimic_iv_gap1h \\
      --out-dir artifacts/v2/union_robustness/f_mimic_iv_gap1h
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import itertools
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

from scipy.stats import kendalltau  # noqa: E402
from sklearn.ensemble import HistGradientBoostingClassifier  # noqa: E402

from src.icu.v2.bridge import compile_bridge, fit_frozen_bridge_v2  # noqa: E402
from src.icu.v2.pipeline import episodes_by_split, restrict_firings  # noqa: E402
from src.icu.v2.stats import metric_value, paired_cluster_bootstrap_diff  # noqa: E402
from src.icu.v2.surrogates import CompositeRegistry, refire_composite  # noqa: E402
from src.icu.v2.union import union_pool  # noqa: E402

GBDT_GRID = {"max_depth": (2, 3), "learning_rate": (0.05, 0.1)}
GBDT_FIXED = {"max_iter": 300, "min_samples_leaf": 20, "early_stopping": False, "random_state": 0}
REGISTRIES = "srs_c_val,srs_c_unknown_val,fs_cv,mrmr,mrmr_filter_known,mmrfs,l1_logistic,l1_dropout,jmi,cmim"
SUBGROUP_REGISTRIES = ("srs_c_val", "srs_c_unknown_val", "mrmr", "mrmr_filter_known", "fs_cv")


def load_module(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / "v2" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module  # dataclasses in the module look themselves up there
    spec.loader.exec_module(module)
    return module


def composite_matrix(comp: CompositeRegistry, schema, firings, episode_ids) -> np.ndarray:
    """Episodes x K coordinates under ``schema``: each coordinate fires as its first executable branch."""
    row = {e: i for i, e in enumerate(episode_ids)}
    X = np.zeros((len(episode_ids), len(comp.primaries)))
    for j, branch in enumerate(comp.resolve(schema)):
        if branch is None:
            continue
        for e in firings.get(branch.program_id, ()):
            i = row.get(e)
            if i is not None:
                X[i, j] = 1.0
    return X


def write_csv(path: Path, records: list[dict[str, Any]]) -> None:
    columns = sorted({k for r in records for k in r}, key=lambda c: (c not in ("registry", "decoder", "a", "b", "cohort", "subgroup_var", "subgroup", "target"), c))
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(records)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--e3-dir", required=True)
    parser.add_argument("--extra-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--registries", default=REGISTRIES)
    parser.add_argument("--reference", default="srs_c_val", help="the registry compared with every other one")
    parser.add_argument("--more-registries", default="", help="comma list of further registries.json files (same pool)")
    parser.add_argument("--extract-root", default="data/processed/v2/extract")
    parser.add_argument("--n-boot", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    started = time.perf_counter()
    sb, sg = load_module("run_selection_baselines"), load_module("run_subgroup_analysis")
    e3, extra = Path(args.e3_dir), Path(args.extra_dir)
    manifest = json.loads((e3 / "manifest.json").read_text(encoding="utf-8"))
    with open(ROOT / manifest["args"]["pools"], "rb") as handle:
        state = pickle.load(handle)
    grouped = episodes_by_split(state["episodes"])
    ids = {name: [e.episode_id for e in rows] for name, rows in grouped.items()}
    labels = {name: [e.label for e in rows] for name, rows in grouped.items()}
    _, firings, external = union_pool(state)
    by_id = {c.program_id: c for pool in state["pools"] for c in pool.candidates}
    saved = {**json.loads((e3 / "registries.json").read_text(encoding="utf-8")), **json.loads((extra / "registries.json").read_text(encoding="utf-8"))}
    for more in [x for x in args.more_registries.split(",") if x]:
        saved.update(json.loads(Path(more).read_text(encoding="utf-8")))
    ref = args.reference
    names = [n for n in args.registries.split(",") if n in saved]
    registries = {n: CompositeRegistry(tuple(by_id[i] for i in saved[n]["primaries"]),
                                       {pid: tuple(by_id[q] for q in qs) for pid, qs in saved[n]["surrogates"].items()}) for n in names}
    targets = list(state["targets"])
    test_y = np.array(labels["test"], dtype=int)
    test_g = np.array([e.subject_id for e in grouped["test"]])
    print(f"[robust] {e3}: {len(names)} registries", flush=True)

    decoder_rows, probs = [], {}
    for name, comp in registries.items():
        primaries = list(comp.primaries)
        f = {p.program_id: firings[p.program_id] for p in primaries}
        bridge = fit_frozen_bridge_v2(primaries, {s: restrict_firings(f, frozenset(ids[s])) for s in ids}, ids, labels)
        mats = {s: compile_bridge(primaries, restrict_firings(f, frozenset(ids[s])), ids[s], "binary")[0] for s in ids}
        best = None
        for depth, rate in itertools.product(GBDT_GRID["max_depth"], GBDT_GRID["learning_rate"]):
            model = HistGradientBoostingClassifier(max_depth=depth, learning_rate=rate, **GBDT_FIXED).fit(mats["train"], labels["train"])
            score = metric_value(labels["val"], model.predict_proba(mats["val"])[:, 1], "auprc")
            if best is None or score > best[0] + 1e-12:
                best = (score, depth, rate, model)
        gbdt = best[3]
        for decoder in ("logistic", "gbdt"):
            if decoder == "logistic":
                p_test = np.array([bridge["test_probabilities"][e] for e in ids["test"]])
            else:
                p_test = gbdt.predict_proba(mats["test"])[:, 1]
            row = {"registry": name, "decoder": decoder, "internal_auprc": metric_value(test_y, p_test, "auprc")}
            if decoder == "gbdt":
                row.update(gbdt_max_depth=best[1], gbdt_learning_rate=best[2], gbdt_validation_auprc=best[0])
            per_target = {}
            for t in targets:
                eids = [e.episode_id for e in t.episodes]
                if decoder == "logistic":
                    out = refire_composite(bridge["model_bundle"], comp, t.features, external[t.name], eids)
                    p = np.array([out["probabilities"][e] for e in eids])
                else:
                    p = gbdt.predict_proba(composite_matrix(comp, t.features, external[t.name], eids))[:, 1]
                per_target[t.name] = p
                row[f"{t.name}_auprc"] = metric_value([e.label for e in t.episodes], p, "auprc")
            row["mean_target_auprc"] = float(np.mean([row[f"{t.name}_auprc"] for t in targets]))
            decoder_rows.append(row)
            probs[(name, decoder)] = (p_test, per_target)
    print(f"[robust] decoders done ({time.perf_counter() - started:.0f}s)", flush=True)

    paired = []
    for decoder in ("logistic", "gbdt"):
        for b in names:
            if b == ref:
                continue
            diff = paired_cluster_bootstrap_diff(test_y, probs[(ref, decoder)][0], probs[(b, decoder)][0], test_g, metric="auprc", n_boot=args.n_boot, seed=args.seed)
            paired.append({"decoder": decoder, "a": ref, "b": b, "target": "internal", **diff})
            for record in sb.paired_target_bootstrap(targets, probs[(ref, decoder)][1], probs[(b, decoder)][1], n_boot=args.n_boot, seed=args.seed):
                paired.append({"decoder": decoder, "a": ref, "b": b, **record})
    rank = {}
    for metric in ("internal_auprc", "mean_target_auprc"):
        lr = [next(r[metric] for r in decoder_rows if r["registry"] == n and r["decoder"] == "logistic") for n in names]
        gb = [next(r[metric] for r in decoder_rows if r["registry"] == n and r["decoder"] == "gbdt") for n in names]
        rank[metric] = {"kendall_tau": float(kendalltau(lr, gb).statistic), "max_abs_change": float(np.max(np.abs(np.array(lr) - np.array(gb))))}
    print(f"[robust] paired decoder tests done ({time.perf_counter() - started:.0f}s); ranks {rank}", flush=True)

    ns = argparse.Namespace(demographics=None, extract_root=args.extract_root, gap=1)
    cohorts = [sg.attach_demographics("mimic_iv_test", grouped["test"], sg.load_demographics("mimic_iv", ns)[0])]
    for t in targets:
        if t.name in ("eicu", "hirid"):
            cohorts.append(sg.attach_demographics(t.name, t.episodes, sg.load_demographics(t.name, ns)[0]))
    subgroup_rows, subgroup_paired = [], []
    for cohort in cohorts:
        for name in [n for n in dict.fromkeys((ref, *SUBGROUP_REGISTRIES)) if n in names]:
            p_test, per_target = probs[(name, "logistic")]
            p = p_test if cohort.name == "mimic_iv_test" else per_target[cohort.name]
            for row in sg.subgroup_rows(cohort, p, n_boot=args.n_boot, seed=args.seed):
                subgroup_rows.append({"registry": name, **row})
        for b in [n for n in ("mrmr", "mrmr_filter_known", "nmifs") if n in names]:
            pa = probs[(ref, "logistic")][0] if cohort.name == "mimic_iv_test" else probs[(ref, "logistic")][1][cohort.name]
            pb = probs[(b, "logistic")][0] if cohort.name == "mimic_iv_test" else probs[(b, "logistic")][1][cohort.name]
            for (variable, level), mask in cohort.masks.items():
                if mask.sum() == 0 or np.unique(cohort.y[mask]).size < 2:
                    continue
                diff = paired_cluster_bootstrap_diff(cohort.y[mask], pa[mask], pb[mask], cohort.groups[mask], metric="auprc", n_boot=args.n_boot, seed=args.seed)
                subgroup_paired.append({"cohort": cohort.name, "subgroup_var": variable, "subgroup": level, "a": ref, "b": b, **diff})
    print(f"[robust] subgroups done ({time.perf_counter() - started:.0f}s)", flush=True)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for file_name, records in (("decoder.csv", decoder_rows), ("decoder_paired.csv", paired), ("subgroups.csv", subgroup_rows), ("subgroup_paired.csv", subgroup_paired)):
        write_csv(out / file_name, records)
    (out / "manifest.json").write_text(json.dumps({"e3_dir": str(e3), "extra_dir": str(extra), "args": vars(args), "rank_agreement": rank,
                                                   "cohort_audit": {c.name: c.audit for c in cohorts}, "elapsed_seconds": time.perf_counter() - started},
                                                  indent=2, default=str), encoding="utf-8")
    print(f"[robust] wrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
