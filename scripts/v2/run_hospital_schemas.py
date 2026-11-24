#!/usr/bin/env python3
"""Reviewer item A: real heterogeneous targets -- every eICU hospital with its own schema.

eICU pools many hospitals whose information systems record different feeds. A hospital's schema is
estimated from its own target episodes, without labels: a measured feature belongs to it when it
is observed in at least ``--coverage`` of the hospital's windows; a status feature (sedation) when
it is "on" in at least one of them. Ventilation is absent everywhere (harmonization decision).
No selector was told these schemas: SRS+T knows only the pooled eICU schema.

Every saved registry of a finished E3 run (not re-selected) is deployed at every hospital with
its frozen probe, each chain resolved against that hospital's schema (``surrogate``) or with
non-executable programs zeroed (``zero``); for reference also against the pooled eICU schema, as
in the paper (``pooled_schema``). AUPRC and log-loss over all eICU episodes and over the episodes
of hospitals whose schema lacks a feature of the pooled one (``affected``). Paired
patient-clustered bootstrap of SRS+T and SRS against the other registries.

  CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1 python scripts/v2/run_hospital_schemas.py [--runs union:f_mimic_iv_gap1h,...]
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _context():
    name = "e3_context"
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / "v2" / "e3_context.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


ctx = _context()

from src.icu.v2.stats import metric_value, paired_cluster_bootstrap_diff  # noqa: E402

REGISTRIES = ("srs_c_val", "srs_c_unknown_val", "fs_cv", "mrmr", "mrmr_filter_known", "mmrfs", "cmim", "jmi", "l1_logistic", "l1_dropout")
STATUS = ("sedation_on", "ventilation_on", "vasopressor_on")
EPS = 1e-6


def log_loss(y, p):
    p = np.clip(p, EPS, 1 - EPS)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def hospital_coverage(config_path: Path, cache: Path) -> dict:
    """Per hospital: episode count, windows with an observation / with an "on" value, per feature."""
    if cache.exists():
        return json.loads(cache.read_text(encoding="utf-8"))
    cfg = json.loads(config_path.read_text(encoding="utf-8"))["targets"]["eicu"]
    records = json.loads(Path(cfg["episodes"]).read_text(encoding="utf-8"))
    episode_dir = Path(cfg["episode_dir"])
    patient = pd.read_csv(ROOT / "data/raw/EICU 2.0数据/patient.csv.gz", usecols=["patientunitstayid", "hospitalid"])
    hospital_of = dict(zip(patient["patientunitstayid"].astype(int), patient["hospitalid"].astype(int)))
    count, observed, on = defaultdict(int), defaultdict(lambda: defaultdict(int)), defaultdict(lambda: defaultdict(int))
    episode_hospital = {}
    for rec in records:
        eid = rec["episode_id"]
        meta = json.loads((episode_dir / f"shock_episode_{eid}.json").read_text(encoding="utf-8"))
        values = np.load(episode_dir / f"shock_episode_{eid}.npy")
        h = str(hospital_of[int(meta["stay_id"])])
        episode_hospital[eid] = h
        count[h] += 1
        for j, f in enumerate(meta["feature_names"]):
            col = values[:, j]
            if np.isfinite(col).any():
                observed[h][f] += 1
            if np.nansum(col > 0) > 0:
                on[h][f] += 1
    out = {"count": dict(count), "observed": {h: dict(v) for h, v in observed.items()}, "on": {h: dict(v) for h, v in on.items()},
           "episode_hospital": episode_hospital}
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(out), encoding="utf-8")
    return out


def hospital_schemas(cov: dict, pooled: frozenset, threshold: float, min_episodes: int) -> dict[str, frozenset]:
    schemas = {}
    for h, n in cov["count"].items():
        if n < min_episodes:
            schemas[h] = pooled
            continue
        keep = set()
        for f in pooled:
            if f in STATUS:
                if cov["on"].get(h, {}).get(f, 0) > 0:
                    keep.add(f)
            elif cov["observed"].get(h, {}).get(f, 0) / n >= threshold:
                keep.add(f)
        schemas[h] = frozenset(keep)
    return schemas


def evaluate(run, schemas: dict[str, frozenset], episode_hospital: dict[str, str], registries, n_boot: int, seed: int, label: str,
             ref_t: str = "srs_c_val", ref: str = "srs_c_unknown_val"):
    target = next(t for t in run.targets if t.name == "eicu")
    pooled = frozenset(target.features)
    eids = [e.episode_id for e in target.episodes]
    y = np.array([e.label for e in target.episodes])
    groups = np.array([str(e.subject_id) for e in target.episodes])
    ext = run.external["eicu"]
    by_schema = defaultdict(list)
    for i, e in enumerate(eids):
        by_schema[schemas[episode_hospital[e]]].append(i)
    affected = np.array([schemas[episode_hospital[e]] != pooled for e in eids])
    rows, probs = [], {}
    for name in registries:
        if name not in run.registries:
            continue
        comp = run.registries[name]
        prim = list(comp.primaries)
        model = run.fit(prim)["model_bundle"]["model"]
        out = {"pooled_schema": model.predict_proba(run.matrix(comp.resolve(pooled), eids, ext))[:, 1]}
        for mode in ("surrogate", "zero"):
            p = np.zeros(len(eids))
            for schema, idx in by_schema.items():
                sub = [eids[i] for i in idx]
                cols = comp.resolve(schema) if mode == "surrogate" else [q if frozenset(q.features) <= schema else None for q in prim]
                p[idx] = model.predict_proba(run.matrix(cols, sub, ext))[:, 1]
            out[mode] = p
        for mode, p in out.items():
            probs[(name, mode)] = p
            for subset, mask in (("all", np.ones(len(eids), bool)), ("affected", affected)):
                if mask.sum() and len(set(y[mask])) == 2:
                    rows.append({"run": label, "registry": name, "mode": mode, "target": f"eicu_hospitals_{subset}",
                                 "auprc": metric_value(y[mask], p[mask], "auprc"), "logloss": log_loss(y[mask], p[mask]),
                                 "n_episodes": int(mask.sum())})
    paired = []
    for a, b in [(ref_t, b) for b in ("mrmr", "mrmr_filter_known", "mmrfs", "cmim", "l1_dropout", "fs_cv", "nmifs", ref)] + \
                [(ref, b) for b in ("mrmr", "mmrfs", "cmim", "l1_dropout", "fs_cv", "nmifs")]:
        for mode in ("surrogate", "zero", "pooled_schema"):
            if (a, mode) not in probs or (b, mode) not in probs:
                continue
            for subset, mask in (("all", np.ones(len(eids), bool)), ("affected", affected)):
                if not mask.sum():
                    continue
                d = paired_cluster_bootstrap_diff(y[mask], probs[(a, mode)][mask], probs[(b, mode)][mask], groups[mask], metric="auprc", n_boot=n_boot, seed=seed)
                paired.append({"run": label, "mode": mode, "subset": subset, "a": a, "b": b, **d})
    summary = {"n_episodes": len(eids), "n_affected": int(affected.sum()), "n_hospitals": len(set(episode_hospital[e] for e in eids)),
               "patterns": {",".join(sorted(pooled - s)) or "-": len(idx) for s, idx in by_schema.items()}}
    return rows, paired, summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", default="", help="comma list of root:label (default: the 16 ICU runs)")
    parser.add_argument("--registries", default=",".join(REGISTRIES))
    parser.add_argument("--coverage", type=float, default=0.05)
    parser.add_argument("--min-episodes", type=int, default=1)
    parser.add_argument("--n-boot", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-root", default=str(ROOT / "artifacts/v2/hospital_schemas"))
    parser.add_argument("--ref-t", default="srs_c_val", help="the method with known target schemas")
    parser.add_argument("--ref", default="srs_c_unknown_val", help="the source-only method")
    args = parser.parse_args(argv)
    runs = [tuple(x.split(":")) for x in args.runs.split(",") if x] or list(ctx.ICU_RUNS)
    registries = tuple(x for x in args.registries.split(",") if x)
    out_root = Path(args.out_root)
    tag = f"cov{args.coverage:g}_min{args.min_episodes}"
    for root, label in runs:
        out = out_root / tag / root / label
        if (out / "manifest.json").exists():
            print(f"[hospitals] {root}/{label}: done, skipped", flush=True)
            continue
        t0 = time.perf_counter()
        run = ctx.load_run(*ctx.run_dirs(root, label))
        task = label.replace("f_mimic_iv_", "")
        cov = hospital_coverage(run.pools_path.parent / "config.json", out_root / "coverage" / f"{task}.json")
        pooled = frozenset(next(t for t in run.targets if t.name == "eicu").features)
        schemas = hospital_schemas(cov, pooled, args.coverage, args.min_episodes)
        rows, paired, summary = evaluate(run, schemas, cov["episode_hospital"], registries, args.n_boot, args.seed, f"{root}/{label}", args.ref_t, args.ref)
        ctx.write_csv(out / "rows.csv", rows)
        ctx.write_csv(out / "paired.csv", paired)
        (out / "manifest.json").write_text(json.dumps({"args": vars(args), "summary": summary, "elapsed_seconds": time.perf_counter() - t0},
                                                      indent=2), encoding="utf-8")
        get = {(r["registry"], r["mode"], r["target"]): r["auprc"] for r in rows}
        print(f"[hospitals] {root}/{label} ({time.perf_counter() - t0:.0f}s) affected {summary['n_affected']}/{summary['n_episodes']}; "
              f"SRS+T / SRS / mRMR on affected: " + " / ".join(f"{get.get((n, 'surrogate', 'eicu_hospitals_affected'), float('nan')):.3f}"
                                                            for n in (args.ref_t, args.ref, "mrmr")), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
