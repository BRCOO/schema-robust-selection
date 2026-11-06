#!/usr/bin/env python3
"""Joint patient bootstrap over the 16 ICU runs (v3 review, round 3, 2026-11-05, item H3).

The 16 runs share patients: the large pools reuse the vasopressor 1 h and AKI cohorts and splits, the vasopressor cohorts
overlap, and every run of a task deploys to the same eICU and HiRID patients. A pooled interval that treats runs or pools as
independent is too narrow. This driver resamples patients jointly across all runs instead: every MIMIC-IV patient (internal
TEST and the five controlled targets built from it), every eICU patient and every HiRID patient receives one bootstrap
multiplicity per replicate, shared by every run, dataset and registry in which the patient appears.

* ``--phase dump``: per run, the shipped (CV seed 0) registries of Table 2 deployed with surrogate chains exactly as in the
  paper (``run_union_ap.deploy``); labels, patient ids and probabilities per dataset to artifacts/v2/joint_bootstrap/<root>/<label>.npz.
* ``--phase boot``: B replicates of weighted average precision (sklearn's definition, ties grouped) per run, dataset and
  registry; artifacts/v2/joint_bootstrap/boot.npz.
* ``--phase summarize``: differences of means over the 16 runs (CV-based selectors averaged over their seeds, as in Table 2;
  every point value is the paper's, from q2_seeds_long.csv; the dump only supplies the bootstrap variation)
  with 95% intervals: point +- 1.96 sqrt(bootstrap variance of the seed-0 difference + between-seed variance of each
  seed mean); paper/v3_assets/source_data/q3_joint.csv.

  CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1 /d/BRCoo/miniconda/python scripts/v2/run_joint_bootstrap.py --phase dump
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
spec = importlib.util.spec_from_file_location("e3_context", ROOT / "scripts/v2/e3_context.py")
ctx = importlib.util.module_from_spec(spec)
sys.modules["e3_context"] = ctx
spec.loader.exec_module(ctx)

OUT = ROOT / "artifacts/v2/joint_bootstrap"
SRC = ROOT / "paper/v3_assets/source_data"
REAL = ("eicu", "hirid")
CONTROLLED = ("drop_labs", "drop_ventilation", "drop_bp", "drop_urine", "vitals_only")
METHODS = ("srs_ap_t", "srs_ap", "srs_c_val", "srs_c_unknown_val", "nmifs_ap_t", "srs_ap_unaware", "fs_cv", "nmifs", "mrmr",
           "mrmr_stratified", "jmi", "cmim", "mmrfs", "mmr_ig", "mms_xin_ig", "dpp_ig", "l1_logistic", "l1_dropout",
           "mrmr_filter_known", "filter_nmifs")


def _ap_module():
    spec = importlib.util.spec_from_file_location("run_union_ap", ROOT / "scripts/v2/run_union_ap.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def phase_dump(args) -> None:
    from src.icu.v2.selection_baselines import nmifs

    ap = _ap_module()
    for root, label in ctx.ICU_RUNS:
        out = OUT / root / f"{label}.npz"
        if out.exists():
            print(f"[dump] {root}/{label}: done, skipped", flush=True)
            continue
        t0 = time.perf_counter()
        e3_dir, extra_dir = ctx.run_dirs(root, label)
        run = ctx.load_run(e3_dir, extra_dir)
        regs = dict(run.registries)
        # the add-on registries: Filter+NMIFS (references), NMIFS+AP (nmifs_ap), Wrapper, AUPRC (union_ap ablations)
        for path, names in ((ROOT / "artifacts/v2/references" / root / label / "registries.json", ("filter_nmifs",)),
                            (ROOT / "artifacts/v2/nmifs_ap" / root / label / "registries.json", ("nmifs_ap_t",)),
                            (ROOT / "artifacts/v2/union_ap" / root / label / "ablations.json", ("srs_ap_unaware",))):
            for name, reg in json.loads(path.read_text(encoding="utf-8")).items():
                if name in names:
                    regs[name] = ctx._composite(reg, run.by_id)
        if "nmifs" not in regs:
            regs["nmifs"] = run.composite(nmifs(run.inp, run.k))
        missing = [m for m in METHODS if m not in regs]
        if missing:
            raise SystemExit(f"{root}/{label}: registries missing: {missing}")
        datasets = {"internal": run.grouped["test"], **{t.name: t.episodes for t in run.targets}}
        probs = {d: [] for d in datasets}
        for m in METHODS:
            p_test, per_target = ap.deploy(run, regs[m])
            probs["internal"].append(p_test)
            for t in run.targets:
                probs[t.name].append(per_target[t.name])
        arrays = {"methods": np.array(METHODS), "datasets": np.array(list(datasets))}
        for d, eps in datasets.items():
            arrays[f"y_{d}"] = np.array([e.label for e in eps], dtype=np.int8)
            arrays[f"g_{d}"] = np.array([str(e.subject_id) for e in eps])
            arrays[f"p_{d}"] = np.column_stack(probs[d]).astype(np.float64)
        out.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out, **arrays)
        print(f"[dump] {root}/{label} ({time.perf_counter() - t0:.0f}s): {len(METHODS)} registries, "
              + ", ".join(f"{d} {len(e)}" for d, e in datasets.items()), flush=True)


def database_of(dataset: str) -> str:
    return dataset if dataset in REAL else "mimic"


def ap_prepare(y: np.ndarray, p: np.ndarray):
    """Sort once (descending score); the ends of the groups of tied scores are the thresholds."""
    order = np.argsort(-p, kind="stable")
    ps = p[order]
    ends = np.r_[np.nonzero(np.diff(ps))[0], len(ps) - 1]
    return order, ends, y[order].astype(np.float64)


def weighted_ap(W: np.ndarray, order: np.ndarray, ends: np.ndarray, ys: np.ndarray) -> np.ndarray:
    """Average precision for each row of episode weights W (replicates x episodes), as sklearn with sample weights."""
    Ws = W[:, order]
    tp = np.cumsum(Ws * ys, axis=1)[:, ends]
    fp = np.cumsum(Ws * (1.0 - ys), axis=1)[:, ends]
    pos = tp[:, -1:]
    prec = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=(tp + fp) > 0)
    rec = np.divide(tp, pos, out=np.zeros_like(tp), where=pos > 0)
    drec = np.diff(np.concatenate([np.zeros((len(W), 1)), rec], axis=1), axis=1)
    return (drec * prec).sum(axis=1)


def phase_boot(args) -> None:
    from sklearn.metrics import average_precision_score

    runs = []
    for root, label in ctx.ICU_RUNS:
        z = np.load(OUT / root / f"{label}.npz", allow_pickle=False)
        runs.append((f"{root}/{label}", z))
    methods = [str(m) for m in runs[0][1]["methods"]]
    datasets = ["internal", *REAL, *CONTROLLED]
    # one index of patients per database, over every run
    index: dict[str, dict[str, int]] = {"mimic": {}, "eicu": {}, "hirid": {}}
    for _, z in runs:
        for d in datasets:
            ix = index[database_of(d)]
            for g in z[f"g_{d}"]:
                ix.setdefault(str(g), len(ix))
        test = set(map(str, z["g_internal"]))
        for d in CONTROLLED:
            assert set(map(str, z[f"g_{d}"])) <= test, "controlled targets must be built from internal TEST patients"
    sizes = {db: len(ix) for db, ix in index.items()}
    print(f"[boot] patients: {sizes}", flush=True)
    prep = []
    for run, z in runs:
        for d in datasets:
            y = z[f"y_{d}"].astype(int)
            g = np.array([index[database_of(d)][str(x)] for x in z[f"g_{d}"]])
            P = z[f"p_{d}"]
            per_m = []
            for j in range(len(methods)):
                order, ends, ys = ap_prepare(y, P[:, j])
                per_m.append((order, ends, ys))
                if args.check:
                    a = weighted_ap(np.ones((1, len(y))), order, ends, ys)[0]
                    b = average_precision_score(y, P[:, j])
                    assert abs(a - b) < 1e-10, (run, d, methods[j], a, b)
            prep.append((run, d, g, per_m))
    point = np.zeros((len(runs), len(methods), len(datasets)))
    for i, (run, d, g, per_m) in enumerate(prep):
        r, k = i // len(datasets), i % len(datasets)
        for j, (order, ends, ys) in enumerate(per_m):
            point[r, j, k] = weighted_ap(np.ones((1, len(g))), order, ends, ys)[0]
    rng = np.random.default_rng(args.seed)
    B, chunk = args.n_boot, args.chunk
    boot = np.zeros((B, len(runs), len(methods), len(datasets)))
    t0 = time.perf_counter()
    for start in range(0, B, chunk):
        nb = min(chunk, B - start)
        counts = {db: np.stack([np.bincount(rng.integers(0, n, n), minlength=n) for _ in range(nb)]).astype(np.float64)
                  for db, n in sizes.items()}
        for i, (run, d, g, per_m) in enumerate(prep):
            r, k = i // len(datasets), i % len(datasets)
            W = counts[database_of(d)][:, g]
            for j, (order, ends, ys) in enumerate(per_m):
                boot[start:start + nb, r, j, k] = weighted_ap(W, order, ends, ys)
        print(f"[boot] {start + nb}/{B} replicates ({time.perf_counter() - t0:.0f}s)", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT / "boot.npz", boot=boot, point=point, runs=np.array([r for r, _ in runs]), methods=np.array(methods),
                        datasets=np.array(datasets), seed=args.seed)
    print(f"[boot] wrote {OUT / 'boot.npz'}", flush=True)


def endpoint_values(arr: np.ndarray, datasets: list[str], endpoint: str) -> np.ndarray:
    """arr[..., dataset] -> endpoint value per leading index (internal, eicu, hirid, real, controlled, targets)."""
    idx = {d: i for i, d in enumerate(datasets)}
    if endpoint in idx:
        return arr[..., idx[endpoint]]
    groups = {"real": REAL, "controlled": CONTROLLED, "targets": REAL + CONTROLLED}[endpoint]
    return np.mean(np.stack([arr[..., idx[d]] for d in groups], axis=-1), axis=-1)


def pool_of(run: str) -> str:
    root, label = run.split("/")
    return ("large/" if root.startswith("union_large") else "standard/") + label


def seed_var_runs(seeds: dict, runs: list[str], a: str, b: str, e: str) -> np.ndarray:
    """Per run: between-seed variance of each seed mean (var / n), summed over the two registries."""
    out = np.zeros(len(runs))
    for r, run in enumerate(runs):
        for m in (a, b):
            grp = seeds.get((run, m))
            if grp is not None and e in grp and grp.seed.nunique() >= 2:
                x = grp[e].to_numpy(dtype=float)
                out[r] += float(x.var(ddof=1)) / len(x)
    return out


def random_effects(y: np.ndarray, draws: np.ndarray, seed_var: np.ndarray, runs: list[str]) -> dict:
    """Random effects per pool with correlated sampling errors. y: seed-aware difference per run; draws: bootstrap
    replicates x runs of the seed-0 difference (their covariance is the sampling covariance of the runs, shared patients
    included), plus the seed variance on the diagonal. The estimate is the mean over runs (the difference of the 16-run
    means); its variance is tau^2 sum_p (n_p / R)^2 + 1' Sigma 1 / R^2, tau^2 by moments from the pool means; t with P - 1 df."""
    from scipy.stats import t as student

    R = len(runs)
    sigma = np.cov(draws, rowvar=False, ddof=1) + np.diag(seed_var)
    pools = sorted({pool_of(r) for r in runs})
    A = np.zeros((len(pools), R))
    for r, run in enumerate(runs):
        p = pools.index(pool_of(run))
        A[p, r] = 1.0
    n_p = A.sum(axis=1)
    A = A / n_p[:, None]
    P = len(pools)
    ybar = A @ y
    C = A @ sigma @ A.T
    s2 = float(np.var(ybar, ddof=1))
    within = float(np.trace(C)) / P - (float(C.sum()) - float(np.trace(C))) / (P * (P - 1))
    tau2 = max(0.0, s2 - within)
    est = float(y.mean())
    var = tau2 * float(np.sum((n_p / R) ** 2)) + float(sigma.sum()) / R ** 2
    q = float(student.ppf(0.975, P - 1))
    return {"re_low": est - q * math.sqrt(var), "re_high": est + q * math.sqrt(var), "re_tau": math.sqrt(tau2),
            "re_se": math.sqrt(var), "pool_mean": float(ybar.mean())}


PAIRS_DEFAULT = (
    # SRS+T against every comparator of Table 2
    *[("srs_ap_t", m) for m in METHODS if m != "srs_ap_t"],
    # the pre-specified likelihood versions against NMIFS and mRMR
    ("srs_c_val", "nmifs"), ("srs_c_val", "mrmr"), ("srs_c_unknown_val", "nmifs"), ("srs_ap", "nmifs"), ("srs_ap", "mrmr"),
    # stepwise decomposition NMIFS -> SRS+T, and the AUPRC decision added last
    ("fs_cv", "nmifs"), ("srs_ap_unaware", "fs_cv"), ("srs_ap", "srs_ap_unaware"),
)


def phase_summarize(args) -> None:
    import pandas as pd

    z = np.load(OUT / "boot.npz", allow_pickle=False)
    boot, point = z["boot"], z["point"]
    runs, methods, datasets = [str(x) for x in z["runs"]], [str(x) for x in z["methods"]], [str(x) for x in z["datasets"]]
    mi = {m: i for i, m in enumerate(methods)}
    T = pd.read_csv(SRC / "q2_seeds_long.csv")  # every registry, run and CV seed (round2.py seeds_pooled)
    endpoints = ("internal", "eicu", "hirid", "real", "controlled", "targets")
    # check: seed-0 values of the dump equal the paper's seed-0 values
    worst = 0.0
    for e in ("internal", "real", "targets"):
        v = endpoint_values(point, datasets, e)
        for r, run in enumerate(runs):
            for m in methods:
                t = T[(T.run == run) & (T.registry == m) & (T.seed == 0)]
                if len(t):
                    worst = max(worst, abs(float(t.iloc[0][e]) - v[r, mi[m]]))
    print(f"[summarize] largest |seed-0 AUPRC (dump) - q2_seeds_long| = {worst:.2e}", flush=True)
    seeds = {}
    for (run, reg), grp in T.groupby(["run", "registry"]):
        seeds[(run, reg)] = grp
    rows, per_run = [], []
    for a, b in PAIRS_DEFAULT:
        if a not in mi or b not in mi:
            continue
        for e in endpoints:
            v_point = endpoint_values(point, datasets, e)
            v_boot = endpoint_values(boot, datasets, e)
            d0 = v_point[:, mi[a]] - v_point[:, mi[b]]
            db = (v_boot[:, :, mi[a]] - v_boot[:, :, mi[b]]).mean(axis=1)
            var_boot = float(np.var(db, ddof=1))
            means, var_seed, n_seeds = [], 0.0, {}
            for r, run in enumerate(runs):
                diff = 0.0
                for m, sign in ((a, 1.0), (b, -1.0)):
                    grp = seeds.get((run, m))
                    col = e if grp is not None and e in grp else None
                    if grp is None or col is None:
                        val = endpoint_values(point[r], datasets, e)[mi[m]]
                        n_seeds[m] = min(n_seeds.get(m, 1), 1)
                    elif grp.seed.nunique() < 2:
                        val = float(grp[col].iloc[0])  # the paper's (seed-0) value, as in phase_verdicts and the tables
                        n_seeds[m] = min(n_seeds.get(m, 1), 1)
                    else:
                        x = grp[col].to_numpy(dtype=float)
                        val = float(x.mean())
                        var_seed += float(x.var(ddof=1)) / len(x) / len(runs) ** 2
                        n_seeds[m] = min(n_seeds.get(m, 99), len(x))
                    diff += sign * val
                means.append(diff)
            est = float(np.mean(means))
            se = math.sqrt(var_boot + var_seed)
            lo_p, hi_p = np.percentile(db, [2.5, 97.5])
            re = random_effects(np.array(means), v_boot[:, :, mi[a]] - v_boot[:, :, mi[b]], seed_var_runs(seeds, runs, a, b, e), runs)
            # per run: the seed-aware difference with the run's own bootstrap variance plus the seed variance
            run_var = np.var(v_boot[:, :, mi[a]] - v_boot[:, :, mi[b]], axis=0, ddof=1) + seed_var_runs(seeds, runs, a, b, e)
            lo_r, hi_r = np.array(means) - 1.96 * np.sqrt(run_var), np.array(means) + 1.96 * np.sqrt(run_var)
            tally = f"{int((lo_r > 0).sum())}/{int(((lo_r <= 0) & (hi_r >= 0)).sum())}/{int((hi_r < 0).sum())}"
            for r, run in enumerate(runs):
                per_run.append({"a": a, "b": b, "endpoint": e, "run": run, "diff": float(means[r]), "se": float(math.sqrt(run_var[r]))})
            rows.append({"a": a, "b": b, "endpoint": e, "point": est, "low": est - 1.96 * se, "high": est + 1.96 * se,
                         "se_boot": math.sqrt(var_boot), "se_seed": math.sqrt(var_seed), "point_seed0": float(d0.mean()),
                         "boot_pct_low": float(lo_p), "boot_pct_high": float(hi_p), **re, "tally": tally, "n_runs": len(runs),
                         "n_seeds_a": n_seeds.get(a, 1), "n_seeds_b": n_seeds.get(b, 1), "n_boot": len(boot)})
    R = pd.DataFrame(rows)
    R.to_csv(SRC / "q3_joint.csv", index=False)
    pd.DataFrame(per_run).to_csv(SRC / "q3_joint_runs.csv", index=False)
    show = R[R.endpoint.isin(["internal", "real", "targets"])].copy()
    for c in ("point", "low", "high", "se_boot", "se_seed", "point_seed0", "re_low", "re_high", "re_tau"):
        show[c] = (100 * show[c]).round(2)
    pd.set_option("display.width", 250)
    print(show[["a", "b", "endpoint", "point", "low", "high", "re_low", "re_high", "re_tau", "se_boot", "se_seed", "n_seeds_a", "n_seeds_b"]].to_string(index=False))


VERDICT_PAIRS = (("srs_ap_t", "srs_c_val"), ("srs_ap_t", "srs_ap_unaware"), ("srs_ap_t", "nmifs_ap_t"), ("srs_ap_t", "srs_ap"),
                 ("srs_ap_t", "nmifs"), ("srs_ap", "nmifs"), ("srs_ap_t", "mrmr"))


def phase_verdicts(args) -> None:
    """Per run and dataset (internal and every target): the difference of the CV-seed means, with a 95% interval from the
    run's own joint-bootstrap variance of the seed-0 difference plus the between-seed variance of each mean."""
    import pandas as pd

    z = np.load(OUT / "boot.npz", allow_pickle=False)
    boot = z["boot"]
    runs, methods, datasets = [str(x) for x in z["runs"]], [str(x) for x in z["methods"]], [str(x) for x in z["datasets"]]
    mi, di = {m: i for i, m in enumerate(methods)}, {d: i for i, d in enumerate(datasets)}
    T = pd.read_csv(SRC / "q2_seeds_long.csv")
    rows = []
    for a, b in VERDICT_PAIRS:
        for d in datasets:
            for r, run in enumerate(runs):
                ga, gb = T[(T.run == run) & (T.registry == a)], T[(T.run == run) & (T.registry == b)]
                xa, xb = ga[d].to_numpy(dtype=float), gb[d].to_numpy(dtype=float)
                diff = float(xa.mean() - xb.mean())
                var = float(np.var(boot[:, r, mi[a], di[d]] - boot[:, r, mi[b], di[d]], ddof=1))
                for x in (xa, xb):
                    if len(x) > 1:
                        var += float(x.var(ddof=1)) / len(x)
                lo, hi = diff - 1.96 * math.sqrt(var), diff + 1.96 * math.sqrt(var)
                rows.append({"a": a, "b": b, "target": d, "run": run, "diff": diff, "low": lo, "high": hi,
                             "verdict": 1 if lo > 0 else (-1 if hi < 0 else 0), "n_seeds_a": len(xa), "n_seeds_b": len(xb)})
    R = pd.DataFrame(rows)
    R.to_csv(SRC / "q3_target_verdicts.csv", index=False)
    tab = R.groupby(["a", "b", "target"]).verdict.apply(lambda v: f"{(v == 1).sum()}/{(v == 0).sum()}/{(v == -1).sum()}").unstack()
    pd.set_option("display.width", 250)
    print(tab.to_string())


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--phase", choices=("dump", "boot", "summarize", "verdicts"), required=True)
    parser.add_argument("--n-boot", type=int, default=1000)
    parser.add_argument("--chunk", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--check", action="store_true", help="boot phase: check the weighted AP against sklearn at unit weights")
    args = parser.parse_args(argv)
    {"dump": phase_dump, "boot": phase_boot, "summarize": phase_summarize, "verdicts": phase_verdicts}[args.phase](args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
