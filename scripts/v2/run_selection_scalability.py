#!/usr/bin/env python3
"""Runtime and memory scalability of the SELECTION stage: SRS-C-V against the established selectors.

Complements ``run_scalability_benchmark.py`` (the discovery stage).

Protocol
--------
* Base data: the TRAIN split of the vasopressor 1 h large union pool (``--comparison``, 2274
  candidates), cached once as a binary firing matrix with the candidates' required features and
  kinds, the TRAIN labels and patient ids, the source schema and the known target schemas.
* Pool size |C| (``--sizes``): random subsets of the real pool below its size; above it, synthetic
  candidates -- a copy of a random real candidate with independent bit flips (rate ``--flip``) on
  every TRAIN episode, keeping the parent's required features and kind, so that executability,
  surrogates and redundancy behave like those of real programs. K = ``--k``.
* Episodes n (``--fractions``): nested patient-level subsamples of TRAIN, on the real pool.
* Budget (``--budgets``): K on the real pool.
* Every (method, setting) runs in a fresh process pinned to the performance cores
  (``--cpu-affinity``) with ``--blas-threads`` BLAS threads. Only the selection call is timed (data
  loading and pool synthesis are excluded). The driver samples the job's resident memory every 0.1 s;
  ``peak_mb`` is the job's peak and ``delta_mb`` the peak minus the job's memory just before the timed
  call. A job is stopped after ``--timeout`` seconds and reported as ``timeout``; methods with a
  quadratic Python reference implementation are not run above ``--reference-limit`` candidates.

Methods: ``srs_c_v`` -- SRS-C-V as deployed: the surrogate chains (computed once), then fast SRS-C
with q = 0 and with q = 0.25 (the validation pick between the two, two probe fits, is excluded);
``fs_cv`` -- fast SRS with the source schema only; ``srs_c_exact`` -- exact greedy SRS-C (q = 0);
``mrmr`` (vectorized, identical selections to the reference implementation), ``jmi``, ``cmim``,
``l1_logistic``; ``mmrfs`` (delta = 1) and ``dpp`` (alpha = 1) -- reference implementations on firing
sets, one value of their hyper-parameter (their validation tuning runs a grid of 4).

  CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1 python scripts/v2/run_selection_scalability.py
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import pickle
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

METHODS = ("srs_c_v", "fs_cv", "srs_c_exact", "mrmr", "jmi", "cmim", "l1_logistic", "mmrfs", "dpp")
#: SRS-AP (the AUPRC shortlist decision), timed like srs_c_v; run with --methods srs_ap_v
AP_METHODS = ("srs_ap_v",)
REFERENCE_ONLY = ("mmrfs", "dpp")  # Python reference implementations, quadratic in |C| (DPP also in memory)


# ------------------------------------------------------------------------------------------ data


def build_cache(comparison: Path, cache: Path) -> None:
    import importlib.util

    from src.icu.v2.pipeline import episodes_by_split
    from src.icu.v2.union import union_pool

    spec = importlib.util.spec_from_file_location("sb", ROOT / "scripts/v2/run_selection_baselines.py")
    sb = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sb)
    with open(comparison / "pools.pkl", "rb") as handle:
        state = pickle.load(handle)
    source = sb.source_schema(state, comparison / "config.json")
    train = episodes_by_split(state["episodes"])["train"]
    order = sorted(train, key=lambda e: e.episode_id)
    row = {e.episode_id: i for i, e in enumerate(order)}
    candidates, firings, _ = union_pool(state)
    X = np.zeros((len(order), len(candidates)), dtype=bool)
    for j, c in enumerate(candidates):
        for e in firings[c.program_id]:
            i = row.get(e)
            if i is not None:
                X[i, j] = True
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, X=X, y=np.array([e.label for e in order], dtype=np.int8),
                        subject=np.array([str(e.subject_id) for e in order]))
    meta = {"comparison": comparison.as_posix(), "source": sorted(source),
            "known": {t.name: sorted(t.features) for t in state["targets"] if t.name in ("eicu", "hirid")},
            "candidates": [{"id": c.program_id, "kind": c.kind, "features": sorted(c.features)} for c in candidates]}
    cache.with_suffix(".json").write_text(json.dumps(meta), encoding="utf-8")


def make_setting(cache: Path, n_candidates: int, fraction: float, flip: float, seed: int):
    """Candidates, float64 TRAIN matrix, labels and patient ids of one setting (deterministic)."""
    from src.icu.v2.types import ProgramRecord

    data = np.load(cache)
    meta = json.loads(cache.with_suffix(".json").read_text(encoding="utf-8"))
    X0, y, subject = data["X"], data["y"].astype(float), data["subject"]
    rng = np.random.default_rng(seed)
    if fraction < 1.0:  # nested patient subsample: the first round(f * N) patients of one seeded order
        patients = sorted(set(subject))
        keep = set(np.array(patients)[rng.permutation(len(patients))[: max(1, round(fraction * len(patients)))]])
        rows = np.flatnonzero([s in keep for s in subject])
        X0, y, subject = X0[rows], y[rows], subject[rows]
    base = [ProgramRecord(program_id=c["id"], method="real", kind=c["kind"], features=frozenset(c["features"])) for c in meta["candidates"]]
    m0 = len(base)
    if n_candidates <= m0:
        pick = np.sort(rng.permutation(m0)[:n_candidates])
        cands, X = [base[j] for j in pick], X0[:, pick].astype(np.float64)
    else:
        extra = n_candidates - m0
        parents = rng.integers(m0, size=extra)
        X = np.empty((X0.shape[0], n_candidates), dtype=np.float64)
        X[:, :m0] = X0
        for start in range(0, extra, 2048):
            stop = min(start + 2048, extra)
            noise = rng.random((X0.shape[0], stop - start)) < flip
            X[:, m0 + start:m0 + stop] = np.logical_xor(X0[:, parents[start:stop]], noise)
        cands = base + [ProgramRecord(program_id=f"syn_{i}", method="synthetic", kind=base[p].kind, features=base[p].features)
                        for i, p in enumerate(parents)]
    return cands, X, y, subject, meta


# ------------------------------------------------------------------------------------------- job


def run_job(spec: dict[str, Any], cache: Path) -> dict[str, Any]:
    import psutil

    from src.icu.v2.robust_selection import scenario_set, schema_robust_select
    from src.icu.v2.selection_baselines import SelectionInput, run_baseline
    from src.icu.v2.surrogates import surrogate_chains

    cands, X, y, subject, meta = make_setting(cache, spec["n_candidates"], spec["fraction"], spec["flip"], spec["seed"])
    ids = [f"e{i:06d}" for i in range(X.shape[0])]
    train_ids, positives = frozenset(ids), frozenset(e for e, label in zip(ids, y) if label == 1)
    subject_of = dict(zip(ids, subject))
    method, k = spec["method"], spec["k"]
    if method in REFERENCE_ONLY:  # the reference implementations read firing sets
        firings = {c.program_id: frozenset(ids[i] for i in np.flatnonzero(X[:, j])) for j, c in enumerate(cands)}
        inp = SelectionInput(tuple(cands), firings, train_ids, positives)
    else:
        inp = SelectionInput(tuple(cands), {}, train_ids, positives, matrix=X)
    source = frozenset(meta["source"])
    known = {name: frozenset(f) for name, f in meta["known"].items()}
    rss_before = psutil.Process().memory_info().rss
    phases: dict[str, float] = {}
    t0 = time.perf_counter()
    if method in ("srs_c_v", "srs_ap_v"):
        chains = surrogate_chains(cands, X)
        phases["chains"] = time.perf_counter() - t0
        for name, q in (("select_q0", 0.0), ("select_q25", 0.25)):
            t1 = time.perf_counter()
            result = schema_robust_select(inp, k, scenario_set(source, known, dropout_q=q), subject_of, surrogates=True, shortlist=10, chains=chains,
                                          decision="auprc" if method == "srs_ap_v" else "loglik")
            phases[name] = time.perf_counter() - t1
        selected = len(result.registry)
    elif method in ("fs_cv", "srs_c_exact"):
        surrogates = method == "srs_c_exact"
        scen = scenario_set(source, known if surrogates else {})
        result = schema_robust_select(inp, k, scen, subject_of, surrogates=surrogates, shortlist=None if surrogates else 10)
        selected = len(result.registry)
    else:
        value = {"mmrfs": 1, "dpp": 1.0}.get(method)  # one grid value (validation tuning runs the grid of 4)
        selected = len(run_baseline("mrmr_vectorized" if method == "mrmr" else method, inp, k, value))
    seconds = time.perf_counter() - t0
    return {"seconds": seconds, "phases": phases, "n_selected": selected, "rss_before_mb": rss_before / 2**20,
            "n_episodes": X.shape[0], "n_candidates": X.shape[1]}


# ---------------------------------------------------------------------------------------- driver


def parse_cpus(text: str) -> list[int]:
    """``a-b`` (inclusive) or ``a,b,c``."""
    if "-" in text:
        low, high = map(int, text.split("-"))
        return list(range(low, high + 1))
    return [int(c) for c in text.split(",") if c]


def launch(spec: dict[str, Any], args, log_dir: Path) -> dict[str, Any]:
    import psutil

    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONDONTWRITEBYTECODE="1",
               **{v: str(args.blas_threads) for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")})
    name = f"{spec['method']}_C{spec['n_candidates']}_f{spec['fraction']}_K{spec['k']}"
    log = open(log_dir / f"{name}.log", "w", encoding="utf-8")
    process = subprocess.Popen([sys.executable, __file__, "--job", json.dumps(spec), "--cache", str(args.cache)],
                               stdout=subprocess.PIPE, stderr=log, text=True, env=env, cwd=ROOT)
    handle = psutil.Process(process.pid)
    try:
        handle.cpu_affinity(parse_cpus(args.cpu_affinity))
    except (psutil.Error, ValueError):
        pass
    peak, stop = [0], threading.Event()

    def sample():
        while not stop.is_set():
            try:
                peak[0] = max(peak[0], handle.memory_info().rss)
            except psutil.Error:
                break
            stop.wait(0.1)

    sampler = threading.Thread(target=sample, daemon=True)
    sampler.start()
    started = time.perf_counter()
    try:
        out, _ = process.communicate(timeout=args.timeout)
        status = "ok" if process.returncode == 0 else f"exit {process.returncode}"
    except subprocess.TimeoutExpired:
        process.kill()
        out, _ = process.communicate()
        status = "timeout"
    stop.set()
    sampler.join()
    log.close()
    result = {}
    for line in reversed((out or "").strip().splitlines()):
        if line.startswith("{"):
            result = json.loads(line)
            break
    row = {**spec, "status": status, "wall_seconds": time.perf_counter() - started, "peak_mb": peak[0] / 2**20}
    if result:
        row.update(seconds=result["seconds"], n_selected=result["n_selected"], n_episodes=result["n_episodes"],
                   delta_mb=peak[0] / 2**20 - result["rss_before_mb"], **{f"phase_{k}": v for k, v in result["phases"].items()})
    return row


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--comparison", default="artifacts/v2/comparison/f_mimic_iv_gap1h_large")
    parser.add_argument("--cache", default="artifacts/v2/selection_scalability/base_gap1h_large.npz")
    parser.add_argument("--out-dir", default="artifacts/v2/selection_scalability")
    parser.add_argument("--methods", default=",".join(METHODS))
    parser.add_argument("--sizes", default="500,1000,2274,5000,10000,20000,50000,100000")
    parser.add_argument("--fractions", default="0.25,0.5,1.0")
    parser.add_argument("--budgets", default="5,10,20,40")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--flip", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--blas-threads", type=int, default=6)
    parser.add_argument("--cpu-affinity", default="0-11", help="CPU ids, a-b (inclusive) or a,b,c; default: the 12 logical CPUs of the 6 P-cores")
    parser.add_argument("--timeout", type=float, default=7200)
    parser.add_argument("--exact-limit", type=int, default=5000, help="largest |C| for srs_c_exact")
    parser.add_argument("--reference-limit", type=int, default=10000, help="largest |C| for the reference-only methods")
    parser.add_argument("--job", default="")
    args = parser.parse_args(argv)
    args.cache = Path(args.cache) if Path(args.cache).is_absolute() else ROOT / args.cache
    if args.job:
        print(json.dumps(run_job(json.loads(args.job), args.cache)), flush=True)
        return 0
    if not args.cache.exists():
        build_cache(ROOT / args.comparison, args.cache)
    meta = json.loads(args.cache.with_suffix(".json").read_text(encoding="utf-8"))
    real = len(meta["candidates"])
    out = ROOT / args.out_dir
    log_dir = out / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    methods = [m for m in args.methods.split(",") if m]
    settings = [("size", int(c), 1.0, args.k) for c in args.sizes.split(",") if c]
    settings += [("episodes", real, float(f), args.k) for f in args.fractions.split(",") if f and float(f) < 1.0]
    settings += [("budget", real, 1.0, int(b)) for b in args.budgets.split(",") if b and int(b) != args.k]
    path = out / "selection_scalability.csv"
    finished = ("ok", "timeout", "not run (limit)")  # failed jobs are run again
    rows = [r for r in csv.DictReader(open(path, encoding="utf-8")) if r["status"] in finished] if path.exists() else []
    done = {(r["method"], int(r["n_candidates"]), float(r["fraction"]), int(r["k"])) for r in rows}
    for axis, n_candidates, fraction, k in settings:
        for method in methods:
            if (method, n_candidates, fraction, k) in done:
                continue
            if (method == "srs_c_exact" and n_candidates > args.exact_limit) or (method in REFERENCE_ONLY and n_candidates > args.reference_limit):
                rows.append({"axis": axis, "method": method, "n_candidates": n_candidates, "fraction": fraction, "k": k, "status": "not run (limit)"})
            else:
                spec = {"method": method, "n_candidates": n_candidates, "fraction": fraction, "k": k, "flip": args.flip, "seed": args.seed}
                rows.append({"axis": axis, **launch(spec, args, log_dir)})
                r = rows[-1]
                print(f"[scal] {axis:8s} {method:12s} |C|={n_candidates:6d} f={fraction:.2f} K={k:2d}: {r['status']:8s} "
                      f"{float(r.get('seconds', 'nan')):9.1f}s peak {r['peak_mb']:8.0f} MB", flush=True)
            columns = sorted({c for r in rows for c in r}, key=lambda c: (c not in ("axis", "method", "n_candidates", "fraction", "k", "status"), c))
            with open(path, "w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=columns)
                writer.writeheader()
                writer.writerows(rows)
    (out / "README.md").write_text(__doc__ + f"\n\nArguments: {json.dumps({k: str(v) for k, v in vars(args).items() if k != 'job'})}\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
