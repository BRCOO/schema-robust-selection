#!/usr/bin/env python3
"""Do registries of programs add information to window statistics? (v3 review, round 4, substantive item 4.)

For the four standard ICU settings at K = 10 (the paper's union runs), the statistic models of run_stat_baselines.py (same
features, same fitting code, same targets) are refitted with the K programs of a registry added as binary columns: the
programs fire as at deployment, each coordinate resolved through its surrogate chain under the dataset's schema. Models:

* ``stats_lr`` / ``stats_gbdt``: all window statistics (L2 logistic regression / gradient-boosted trees), as in the paper;
  ``+srs`` / ``+nmifs``: the same with the SRS+T / NMIFS registry (CV seed 0) added.
* ``kstats``: the K statistics first chosen on an L1 path, refitted (the paper's K-statistics reference); ``kstats+srs`` and
  ``kstats+nmifs``: with the K programs added; ``kstats2``: the 2K statistics of the L1 path, the budget-matched control
  (is a program worth more than one more statistic?).
* ``prog_srs`` / ``prog_nmifs``: the programs alone with the same logistic fitting code.

AUPRC at internal TEST, eICU, HiRID and the five controlled targets; a joint patient bootstrap over the four settings
(MIMIC-IV, eICU and HiRID patients drawn once per replicate) for the pre-chosen contrasts. Outputs: artifacts/v2/hybrid/.

  CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 /d/BRCoo/miniconda/python scripts/v2/run_hybrid_programs_stats.py
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
OUT = ROOT / "artifacts/v2/hybrid"
TAGS = ("gap1h", "gap2h", "gap4h", "aki_gap6h")
K = 10
REAL = ("eicu", "hirid")
CONTROLLED = ("drop_labs", "drop_ventilation", "drop_bp", "drop_urine", "vitals_only")
CONTRASTS = (("stats_lr+srs", "stats_lr"), ("stats_gbdt+srs", "stats_gbdt"), ("kstats+srs", "kstats"), ("kstats+srs", "kstats2"),
             ("kstats+nmifs", "kstats2"), ("kstats+srs", "kstats+nmifs"), ("stats_lr+nmifs", "stats_lr"),
             # programs alone against K statistics (same fitting code; added with the predicate generator, 2026-11-08)
             ("prog_srs", "kstats"), ("prog_nmifs", "kstats"), ("prog_srs", "prog_nmifs"), ("stats_gbdt", "prog_srs"))


def _module(name: str, path: Path):
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


def phase_fit(args) -> None:
    from src.icu.v2.supervised import FileArrayLoader, build_external_cohort, build_supervised_inputs, load_episode_list, read_feature_names

    sb = _module("run_stat_baselines", ROOT / "scripts/v2/run_stat_baselines.py")
    ctx = _module("e3_context", ROOT / "scripts/v2/e3_context.py")
    OUT.mkdir(parents=True, exist_ok=True)
    for tag in TAGS:
        out = OUT / f"f_mimic_iv_{tag}.npz"
        if out.exists():
            print(f"[hybrid] {tag}: done, skipped", flush=True)
            continue
        t0 = time.perf_counter()
        task = "aki" if tag.startswith("aki") else "vaso"
        src_dir = ROOT / f"data/processed/v2/extract/mimic_iv_{tag}_full"
        episodes = load_episode_list(ROOT / f"artifacts/v2/task/f_mimic_iv_{tag}/episodes_split.json")
        names = read_feature_names(src_dir, episodes[0].episode_id)
        loader = FileArrayLoader(src_dir)
        inputs = build_supervised_inputs(episodes, names, loader, excluded_features=())
        tr, va, te = (inputs.splits[s] for s in ("train", "val", "test"))
        data = {"train": (tr.tabular, tr.y, tr.episode_ids, tr.subject_ids), "val": (va.tabular, va.y, va.episode_ids, va.subject_ids),
                "internal": (te.tabular, te.y, te.episode_ids, te.subject_ids)}
        for site in REAL:
            ext_dir = ROOT / f"data/processed/v2/extract/{site}_{tag}_full"
            ext_eps = load_episode_list(ROOT / f"artifacts/v2/task/f_{site}_{tag}/episodes_all.json")
            cohort = build_external_cohort(site, ext_eps, FileArrayLoader(ext_dir), inputs, feature_names=read_feature_names(ext_dir, ext_eps[0].episode_id))
            a = cohort.arrays
            data[site] = (a.tabular, a.y, a.episode_ids, a.subject_ids)
        test_eps = sorted([e for e in episodes if e.split == "test"], key=lambda e: {x: i for i, x in enumerate(te.episode_ids)}[e.episode_id])
        for name, drop in sb.controlled_targets(task).items():
            data[name] = (sb.masked_test(inputs, test_eps, loader, names, set(drop)), te.y, te.episode_ids, te.subject_ids)
        print(f"[hybrid] {tag}: statistics built ({time.perf_counter() - t0:.0f}s)", flush=True)
        # programs as deployed: internal splits fire on the source schema, targets resolve each chain under their schema
        run = ctx.load_run(*ctx.run_dirs("union", f"f_mimic_iv_{tag}"))
        targets = {t.name: t for t in run.targets}
        programs = {}
        for reg in ("srs_ap_t", "nmifs"):
            comp = run.registries[reg]
            cols = {}
            for d, (_, _, ids, _) in data.items():
                if d in ("train", "val", "internal"):
                    cols[d] = run.matrix(list(comp.primaries), list(ids))
                else:
                    t = targets[d]
                    cols[d] = run.matrix(comp.resolve(t.features), list(ids), run.external[d])
            programs[reg] = cols
        Xtr, ytr, Xva, yva = tr.tabular, tr.y, va.tabular, va.y
        k_cols = sb.l1_columns(Xtr, ytr, K)
        k2_cols = sb.l1_columns(Xtr, ytr, 2 * K)

        def design(name: str, d: str) -> np.ndarray:
            X = data[d][0]
            base, _, extra = name.partition("+")
            if base in ("stats_lr", "stats_gbdt"):
                parts = [X]
            elif base == "kstats":
                parts = [X[:, k_cols]]
            elif base == "kstats2":
                parts = [X[:, k2_cols]]
            else:  # prog_srs / prog_nmifs
                parts = []
                extra = base.split("_")[1]
            if extra:
                parts.append(programs["srs_ap_t" if extra == "srs" else "nmifs"][d])
            return np.hstack(parts)

        models = ("stats_lr", "stats_lr+srs", "stats_lr+nmifs", "stats_gbdt", "stats_gbdt+srs", "kstats", "kstats+srs", "kstats+nmifs", "kstats2",
                  "prog_srs", "prog_nmifs")
        datasets = ["internal", *REAL, *CONTROLLED]
        probs, info = {}, {}
        for m in models:
            if m.startswith("stats_gbdt"):
                predict, meta = sb.fit_gbdt(design(m, "train"), ytr, design(m, "val"), yva)
            else:
                model, meta = sb.fit_lr(design(m, "train"), ytr, design(m, "val"), yva)
                predict = (lambda mm: (lambda X: mm.predict_proba(X)[:, 1]))(model)
            info[m] = {k: v for k, v in meta.items() if k != "columns"}
            probs[m] = {d: predict(design(m, d)) for d in datasets}
            print(f"[hybrid] {tag} {m}: internal {sb.auprc(data['internal'][1], probs[m]['internal']):.3f}, real "
                  f"{np.mean([sb.auprc(data[d][1], probs[m][d]) for d in REAL]):.3f}", flush=True)
        arrays = {"models": np.array(models), "datasets": np.array(datasets)}
        for d in datasets:
            arrays[f"y_{d}"] = np.asarray(data[d][1], dtype=np.int8)
            arrays[f"g_{d}"] = np.array([str(s) for s in data[d][3]])
            arrays[f"p_{d}"] = np.column_stack([probs[m][d] for m in models])
        np.savez_compressed(out, **arrays)
        (OUT / f"f_mimic_iv_{tag}_info.json").write_text(json.dumps({"k_columns": [inputs.tabular_feature_names[c] for c in k_cols],
                                                                       "k2_columns": [inputs.tabular_feature_names[c] for c in k2_cols],
                                                                       "models": info}, indent=1, default=str), encoding="utf-8")
        print(f"[hybrid] {tag}: done ({time.perf_counter() - t0:.0f}s)", flush=True)


def database_of(d: str) -> str:
    return d if d in REAL else "mimic"


def phase_boot(args) -> None:
    import pandas as pd

    jb = _module("run_joint_bootstrap", ROOT / "scripts/v2/run_joint_bootstrap.py")
    loaded = [(tag, np.load(OUT / f"f_mimic_iv_{tag}.npz", allow_pickle=False)) for tag in TAGS]
    models, datasets = [str(m) for m in loaded[0][1]["models"]], [str(d) for d in loaded[0][1]["datasets"]]
    mi = {m: i for i, m in enumerate(models)}
    index = {"mimic": {}, "eicu": {}, "hirid": {}}
    for _, z in loaded:
        for d in datasets:
            for g in z[f"g_{d}"]:
                index[database_of(d)].setdefault(str(g), len(index[database_of(d)]))
    sizes = {db: len(ix) for db, ix in index.items()}
    prep = []
    for tag, z in loaded:
        for d in datasets:
            y = z[f"y_{d}"].astype(int)
            g = np.array([index[database_of(d)][str(x)] for x in z[f"g_{d}"]])
            prep.append((tag, d, g, [jb.ap_prepare(y, z[f"p_{d}"][:, j]) for j in range(len(models))]))
    R, M, D = len(loaded), len(models), len(datasets)
    point = np.zeros((R, M, D))
    for i, (_, _, g, per_m) in enumerate(prep):
        r, k = divmod(i, D)
        for j, (o, e, ys) in enumerate(per_m):
            point[r, j, k] = jb.weighted_ap(np.ones((1, len(g))), o, e, ys)[0]
    rng = np.random.default_rng(args.seed)
    boot = np.zeros((args.n_boot, R, M, D))
    for start in range(0, args.n_boot, 50):
        nb = min(50, args.n_boot - start)
        counts = {db: np.stack([np.bincount(rng.integers(0, n, n), minlength=n) for _ in range(nb)]).astype(float) for db, n in sizes.items()}
        for i, (_, d, g, per_m) in enumerate(prep):
            r, k = divmod(i, D)
            W = counts[database_of(d)][:, g]
            for j, (o, e, ys) in enumerate(per_m):
                boot[start:start + nb, r, j, k] = jb.weighted_ap(W, o, e, ys)
        print(f"[hybrid] boot {start + nb}/{args.n_boot}", flush=True)
    di = {d: i for i, d in enumerate(datasets)}

    def ep(arr, name):
        group = {"internal": ("internal",), "real": REAL, "controlled": CONTROLLED, "targets": REAL + CONTROLLED}[name]
        return np.mean(np.stack([arr[..., di[d]] for d in group], axis=-1), axis=-1)

    rows = []
    for m in models:
        for e in ("internal", "real", "controlled", "targets"):
            v = ep(point[:, mi[m]], e)
            rows.append({"model": m, "endpoint": e, "mean": float(v.mean()), **{tag: float(x) for (tag, _), x in zip(loaded, v)}})
    pd.DataFrame(rows).to_csv(OUT / "means.csv", index=False)
    out = []
    for a, b in CONTRASTS:
        for e in ("internal", "real", "controlled", "targets"):
            d = ep(point[:, mi[a]], e) - ep(point[:, mi[b]], e)
            db = ep(boot[:, :, mi[a]], e) - ep(boot[:, :, mi[b]], e)
            lo, hi = np.percentile(db.mean(axis=1), [2.5, 97.5])
            lo_r, hi_r = np.percentile(db, [2.5, 97.5], axis=0)
            out.append({"a": a, "b": b, "endpoint": e, "diff": float(d.mean()), "low": float(lo), "high": float(hi),
                        "better": int((lo_r > 0).sum()), "worse": int((hi_r < 0).sum()), **{tag: float(x) for (tag, _), x in zip(loaded, d)}})
    S = pd.DataFrame(out)
    S.to_csv(OUT / "contrasts.csv", index=False)
    S.to_csv(ROOT / "paper/v3_assets/source_data/q4_hybrid_contrasts.csv", index=False)
    pd.DataFrame(rows).to_csv(ROOT / "paper/v3_assets/source_data/q4_hybrid_means.csv", index=False)
    show = S.copy()
    for c in ("diff", "low", "high"):
        show[c] = (100 * show[c]).round(2)
    pd.set_option("display.width", 220)
    print(show[["a", "b", "endpoint", "diff", "low", "high", "better", "worse"]].to_string(index=False))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--phase", choices=("fit", "boot"), default="fit")
    parser.add_argument("--n-boot", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    {"fit": phase_fit, "boot": phase_boot}[args.phase](args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
