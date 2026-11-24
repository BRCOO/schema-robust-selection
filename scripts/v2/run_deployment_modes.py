#!/usr/bin/env python3
"""Reviewer item D: is the edge of SRS a property of the selection or of the deployment?

The saved registries of finished E3 runs (not re-selected) are deployed at every target schema F
in six ways:

* ``zero``: the frozen probe, non-executable programs zeroed (``bridge.refire_external``);
* ``surrogate``: the frozen probe, each coordinate takes its first executable branch (the paper's
  deployment, ``surrogates.refire_composite``);
* ``soft``: the frozen probe; a non-executable coordinate takes a label-free prediction of its
  program's firing from the registry's programs that do execute under F (L2 logistic on TRAIN
  firings, fit at the source for F);
* ``robust``: a probe trained with schema dropout: TRAIN plus ``--copies`` copies in which every
  coarse group is dropped per episode with probability q (from ``--dropout-grid``, chosen on VAL),
  each coordinate resolved through its surrogate chain; deployed like ``surrogate``. It needs no
  target schema;
* ``reduced``: a probe re-fit at the source on the coordinates executable under F ("reduced
  models", Saar-Tsechansky and Provost 2007); needs F in advance and ships one probe per schema;
* ``reduced_surrogate``: the same on the coordinates resolved under F.

Every probe is the paper's: L2 logistic, class-balanced, C on VAL (``bridge.select_probe_on_validation``).
Per registry, mode and target: AUPRC and log-loss (nats per episode). Paired patient-clustered
bootstrap on the mean target AUPRC, per mode: SRS+T (``srs_c_val``) and SRS (``srs_c_unknown_val``)
against mRMR, SRS+T against its schema-unaware ablation (``fs_cv``) and against L1 + schema dropout.

  CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1 python scripts/v2/run_deployment_modes.py [--runs union:f_mimic_iv_gap1h,...] [--drive-k 5,10,20]
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression

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

from src.icu.v2.bridge import select_probe_on_validation  # noqa: E402
from src.icu.v2.stats import metric_value  # noqa: E402
from src.icu.v2.types import coarse_group  # noqa: E402

REGISTRIES = ("srs_c_val", "srs_c_unknown_val", "fs_cv", "mrmr", "mrmr_filter_known", "mmrfs", "cmim", "l1_dropout")
MODES = ("zero", "surrogate", "soft", "robust", "reduced", "reduced_surrogate")
EPS = 1e-6


def log_loss(y: np.ndarray, p: np.ndarray) -> float:
    p = np.clip(p, EPS, 1 - EPS)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def probe(run, X: dict[str, np.ndarray]):
    """The paper's probe on given TRAIN/VAL matrices; None when there is no coordinate."""
    if X["train"].shape[1] == 0:
        return None
    return select_probe_on_validation(X["train"], run.labels["train"], X["val"], run.labels["val"])["model"]


def predict(model, X: np.ndarray, base_rate: float) -> np.ndarray:
    return np.full(X.shape[0], base_rate) if model is None else model.predict_proba(X)[:, 1]


def robust_probe(run, comp, grid: list[float], copies: int, seed: int):
    """Schema-dropout training resolved through the chains; q on VAL (clean VAL, as for every probe)."""
    prim = list(comp.primaries)
    train = run.ids["train"]
    y = np.asarray(run.labels["train"])
    groups = sorted({coarse_group(f) for f in run.source})
    gid = {g: i for i, g in enumerate(groups)}
    chains = [comp.branches(p) for p in prim]
    pairs = [(i, b) for i, chain in enumerate(chains) for b in chain]
    B = run.matrix([b for _, b in pairs], train)
    need = np.zeros((len(pairs), len(groups)))
    for j, (_, b) in enumerate(pairs):
        for f in b.features:
            need[j, gid[coarse_group(f)]] = 1.0
    starts = np.cumsum([0] + [len(c) for c in chains])
    X0 = run.matrix(prim, train)
    Xva = run.matrix(prim, run.ids["val"])
    best = None
    for q in grid:
        rng = np.random.default_rng(seed)
        blocks = [X0]
        for _ in range(copies):
            keep = rng.random((len(train), len(groups))) >= q
            empty = np.flatnonzero(~keep.any(axis=1))
            keep[empty, rng.integers(len(groups), size=empty.size)] = True
            exe = ((~keep).astype(float) @ need.T) == 0
            Xc = np.zeros_like(X0)
            for i in range(len(prim)):
                block = exe[:, starts[i]:starts[i + 1]]
                has = block.any(axis=1)
                first = np.argmax(block, axis=1) + starts[i]
                Xc[has, i] = B[np.flatnonzero(has), first[has]]
            blocks.append(Xc)
        fit = select_probe_on_validation(np.vstack(blocks), np.tile(y, copies + 1), Xva, run.labels["val"])
        if best is None or fit["validation_score"] > best[0] + 1e-12:
            best = (fit["validation_score"], q, fit["model"])
    return best[2], best[1]


def soft_matrix(run, comp, schema: frozenset, eids: list[str], ext: dict, X_resolved: np.ndarray) -> np.ndarray:
    """Replace every non-executable coordinate by a label-free prediction of its program's firing."""
    prim = list(comp.primaries)
    pool = {}
    for p in prim:
        for b in comp.branches(p):
            if frozenset(b.features) <= schema:
                pool.setdefault(b.program_id, b)
    predictors = list(pool.values())
    X = X_resolved.copy()
    if not predictors:
        return X
    train = run.ids["train"]
    P_tr = run.matrix(predictors, train)
    P_t = run.matrix(predictors, eids, ext)
    for i, p in enumerate(prim):
        if frozenset(p.features) <= schema:
            continue
        target = run.matrix([p], train)[:, 0]
        if target.min() == target.max():
            X[:, i] = target[0]
            continue
        model = LogisticRegression(C=1.0, max_iter=1000).fit(P_tr, target)
        X[:, i] = model.predict_proba(P_t)[:, 1]
    return X


def evaluate_run(run, registries: tuple[str, ...], grid: list[float], copies: int, n_boot: int, seed: int, label: str, ref_t: str = "srs_c_val", ref: str = "srs_c_unknown_val"):
    sb = ctx.selection_baselines_module()
    rows, probs = [], {}
    split_ids = run.ids
    base_rate = float(np.mean(run.labels["train"]))
    y_test = np.asarray(run.labels["test"])
    for name in registries:
        if name not in run.registries:
            continue
        comp = run.registries[name]
        prim = list(comp.primaries)
        Xs = {s: run.matrix(prim, split_ids[s]) for s in ("train", "val", "test")}
        frozen = probe(run, Xs)
        rob, rob_q = robust_probe(run, comp, grid, copies, seed)
        internal = {"frozen": predict(frozen, Xs["test"], base_rate), "robust": predict(rob, Xs["test"], base_rate)}
        for which, p in internal.items():
            rows.append({"run": label, "registry": name, "mode": f"internal_{which}", "target": "internal_test",
                         "auprc": metric_value(y_test, p, "auprc"), "logloss": log_loss(y_test, p), "robust_q": rob_q})
        for target in run.targets:
            eids = [e.episode_id for e in target.episodes]
            y = np.array([e.label for e in target.episodes])
            ext = run.external[target.name]
            schema = frozenset(target.features)
            resolved = comp.resolve(schema)
            executable = [p for p in prim if frozenset(p.features) <= schema]
            X_zero = run.matrix([p if frozenset(p.features) <= schema else None for p in prim], eids, ext)
            X_sur = run.matrix(resolved, eids, ext)
            out = {
                "zero": predict(frozen, X_zero, base_rate),
                "surrogate": predict(frozen, X_sur, base_rate),
                "soft": predict(frozen, soft_matrix(run, comp, schema, eids, ext, X_sur), base_rate),
                "robust": predict(rob, X_sur, base_rate),
            }
            red = probe(run, {s: run.matrix(executable, split_ids[s]) for s in ("train", "val")})
            out["reduced"] = predict(red, run.matrix(executable, eids, ext), base_rate)
            kept = [b for b in resolved if b is not None]
            red_s = probe(run, {s: run.matrix(kept, split_ids[s]) for s in ("train", "val")})
            out["reduced_surrogate"] = predict(red_s, run.matrix(kept, eids, ext), base_rate)
            for mode, p in out.items():
                probs.setdefault((name, mode), {})[target.name] = p
                rows.append({"run": label, "registry": name, "mode": mode, "target": target.name,
                             "auprc": metric_value(y, p, "auprc"), "logloss": log_loss(y, p),
                             "n_executable": len(executable), "n_resolved": len(kept), "n_programs": len(prim), "robust_q": rob_q})
    paired = []
    for a, b in ((ref_t, "mrmr"), (ref, "mrmr"), (ref_t, "fs_cv"), (ref_t, "l1_dropout"), (ref_t, "nmifs"), (ref, "nmifs")):
        for mode in MODES:
            if (a, mode) not in probs or (b, mode) not in probs:
                continue
            for record in sb.paired_target_bootstrap(run.targets, probs[(a, mode)], probs[(b, mode)], n_boot=n_boot, seed=seed):
                if record["target"] == "mean_over_targets":
                    paired.append({"run": label, "mode": mode, "a": a, "b": b, **record})
    return rows, paired


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", default="", help="comma list of root:label (default: the 16 ICU runs)")
    parser.add_argument("--drive-k", default="", help="also the drive runs at these budgets, e.g. 5,10,20")
    parser.add_argument("--registries", default=",".join(REGISTRIES))
    parser.add_argument("--dropout-grid", default="0.25,0.5")
    parser.add_argument("--copies", type=int, default=4)
    parser.add_argument("--n-boot", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-root", default=str(ROOT / "artifacts/v2/deployment_modes"))
    parser.add_argument("--ref-t", default="srs_c_val", help="the method with known target schemas")
    parser.add_argument("--ref", default="srs_c_unknown_val", help="the source-only method")
    args = parser.parse_args(argv)
    # --runs none: no ICU run (only the drive budgets of --drive-k)
    runs = [] if args.runs == "none" else [tuple(x.split(":")) for x in args.runs.split(",") if x] or list(ctx.ICU_RUNS)
    jobs = [(f"{root}/{label}", *ctx.run_dirs(root, label)) for root, label in runs]
    jobs += [(f"drive_k{k}/f_backblaze_seagate_gap2d", *ctx.drive_dirs(int(k))) for k in args.drive_k.split(",") if k]
    registries = tuple(x for x in args.registries.split(",") if x)
    grid = [float(x) for x in args.dropout_grid.split(",") if x]
    out_root = Path(args.out_root)
    for label, e3_dir, extra_dir in jobs:
        out = out_root / label
        if (out / "manifest.json").exists():
            print(f"[deploy] {label}: done, skipped", flush=True)
            continue
        t0 = time.perf_counter()
        run = ctx.load_run(e3_dir, extra_dir)
        rows, paired = evaluate_run(run, registries, grid, args.copies, args.n_boot, args.seed, label, args.ref_t, args.ref)
        ctx.write_csv(out / "rows.csv", rows)
        ctx.write_csv(out / "paired.csv", paired)
        (out / "manifest.json").write_text(json.dumps({"e3_dir": str(e3_dir), "extra_dir": str(extra_dir), "args": vars(args),
                                                       "elapsed_seconds": time.perf_counter() - t0}, indent=2), encoding="utf-8")
        summary = {(r["registry"], r["mode"]): [] for r in rows}
        for r in rows:
            if not r["mode"].startswith("internal"):
                summary[(r["registry"], r["mode"])].append(r["auprc"])
        line = " ".join(f"{m}={np.mean(summary[(args.ref_t, m)]):.3f}/{np.mean(summary[('mrmr', m)]):.3f}" for m in MODES if (args.ref_t, m) in summary and ("mrmr", m) in summary)
        print(f"[deploy] {label} ({time.perf_counter() - t0:.0f}s) SRS+T/mRMR mean target: {line}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
