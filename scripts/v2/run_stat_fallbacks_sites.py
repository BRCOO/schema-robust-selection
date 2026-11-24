#!/usr/bin/env python3
"""Fallbacks for the K-statistics model at the eICU hospitals of the second confirmatory study (third review of 2026-11-11;
post hoc and exploratory, not part of the study's protocol).

For each of the study's six runs (vasopressor 1 h and AKI, K = 5, 10, 20) the study's K-statistics model (the K statistics
first chosen on an L1 path on the eICU source TRAIN, refitted; run_eicu_source_study.phase_stats) is deployed at every
dataset of the design with the study's defaults (``kstats``), with chains of statistics of other features (``kstats_chain``)
and with regression imputation fitted for each schema (``kstats_soft``), as in run_stat_fallbacks.py; every hospital group
keeps its own absent features. SRS+T (CV seed 0, chains) comes from the study's dump. The recomputed ``kstats`` must equal
the dump's. Joint patient bootstrap over the six runs. Outputs: artifacts/v2/stat_fallbacks/sites/.

  CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 python scripts/v2/run_stat_fallbacks_sites.py --phase fit
  ... --phase boot
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
ART = ROOT / "artifacts/v2"
OUT = ART / "stat_fallbacks/sites"
TAGS = ("gap1h", "aki_gap6h")
BUDGETS = (5, 10, 20)
MODELS = ("srs_ap_t", "kstats", "kstats_chain", "kstats_soft")
DATABASES = ("mimic_iv", "hirid")
CONTROLLED = ("drop_labs", "drop_bp", "drop_urine", "vitals_only")
CONTRASTS = (("srs_ap_t", "kstats"), ("srs_ap_t", "kstats_chain"), ("srs_ap_t", "kstats_soft"), ("kstats_chain", "kstats"),
             ("kstats_soft", "kstats"))


def _module(name: str, path: Path):
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


def phase_fit(args) -> None:
    from src.icu.v2.supervised import FileArrayLoader, build_supervised_inputs, load_episode_list, read_feature_names

    sb = _module("run_stat_baselines", ROOT / "scripts/v2/run_stat_baselines.py")
    fb = _module("run_stat_fallbacks", ROOT / "scripts/v2/run_stat_fallbacks.py")
    ctx = _module("e3_context", ROOT / "scripts/v2/e3_context.py")
    OUT.mkdir(parents=True, exist_ok=True)
    for tag in TAGS:
        label = f"e_eicu_{tag}"
        todo = [k for k in BUDGETS if not (OUT / f"{label}_k{k}.npz").exists()]
        if not todo:
            continue
        t0 = time.perf_counter()
        cfg = json.loads((ART / "comparison/configs" / f"{label}.json").read_text(encoding="utf-8"))
        schema = set(cfg["internal_features"])
        src_dir = Path(cfg["episode_dir"])
        episodes = load_episode_list(Path(cfg["cohort"]))
        feat = read_feature_names(src_dir, episodes[0].episode_id)
        loader = FileArrayLoader(src_dir)
        inputs = build_supervised_inputs(episodes, feat, loader, excluded_features=tuple(f for f in feat if f not in schema))
        names = list(inputs.tabular_feature_names)
        medians = np.asarray(inputs.preprocessing.tabular_medians, dtype=float)
        fill = lambda A: np.where(np.isfinite(A), A, medians)  # noqa: E731
        tr, va, te = (inputs.splits[s] for s in ("train", "val", "test"))
        order = {e: i for i, e in enumerate(te.episode_ids)}
        test_eps = sorted((e for e in episodes if e.split == "test"), key=lambda e: order[e.episode_id])
        by_id, absent = {}, {}  # dataset -> {episode_id: row of statistics}, absent features
        by_id["internal"] = dict(zip(te.episode_ids, fill(te.tabular)))
        absent["internal"] = set()
        for name, spec in cfg["targets"].items():
            eps = load_episode_list(Path(spec["episodes"]))
            t_dir = Path(spec["episode_dir"])
            ab = schema - set(spec["features"])
            X = fill(sb.masked_test(inputs, eps, FileArrayLoader(t_dir), read_feature_names(t_dir, eps[0].episode_id), ab))
            by_id[name], absent[name] = dict(zip((e.episode_id for e in eps), X)), ab
        for name, spec in cfg["schema_shift_targets"].items():
            X = fill(sb.masked_test(inputs, test_eps, loader, feat, set(spec["drop"])))
            by_id[name], absent[name] = dict(zip(te.episode_ids, X)), set(spec["drop"])
        Xtr, ytr, Xva, yva = fill(tr.tabular), tr.y, fill(va.tabular), va.y
        chains = fb.stat_chains(Xtr, names)
        print(f"[sites] {tag}: statistics built, {len(names)} columns, absent " + ", ".join(f"{d} {sorted(a)}" for d, a in absent.items() if a)
              + f" ({time.perf_counter() - t0:.0f}s)", flush=True)
        for k in todo:
            root = f"union_esrc2_k{k}"
            z = np.load(ART / "eicu_source2/dump" / root / f"{label}.npz", allow_pickle=False)
            methods = [str(m) for m in z["methods"]]
            run = ctx.load_run(*ctx.run_dirs(root, label))
            sites = [t for t in run.targets if t.name.startswith("site_")]
            assert [t.name for t in sites] == [str(s) for s in z["site_targets"]]
            groups = {"internal": [("internal", [e.episode_id for e in run.grouped["test"]])],
                      **{t.name: [(t.name, [e.episode_id for e in t.episodes])] for t in run.targets if not t.name.startswith("site_")},
                      "sites": [(t.name, [e.episode_id for e in t.episodes]) for t in sites]}
            cols = sb.l1_columns(Xtr, ytr, k)
            model, info = sb.fit_lr(Xtr[:, cols], ytr, Xva[:, cols], yva)
            arrays = {"models": np.array(MODELS), "datasets": z["datasets"]}
            check = 0.0
            used = {}
            for d in [str(x) for x in z["datasets"]]:
                parts = {m: [] for m in MODELS[1:]}
                for name, ids in groups[d]:
                    X = np.vstack([by_id[name][e] for e in ids])
                    parts["kstats"].append(model.predict_proba(X[:, cols])[:, 1])
                    Xc, used[name] = fb.deploy_chain(X, cols, names, chains, absent[name])
                    parts["kstats_chain"].append(model.predict_proba(Xc)[:, 1])
                    parts["kstats_soft"].append(model.predict_proba(fb.deploy_soft(Xtr, X, cols, names, absent[name]))[:, 1])
                p = {m: np.concatenate(v) for m, v in parts.items()}
                p["srs_ap_t"] = z[f"p_{d}"][:, methods.index("srs_ap_t")]
                check = max(check, float(np.max(np.abs(p["kstats"] - z[f"p_{d}"][:, methods.index("kstats")]))))
                arrays[f"y_{d}"], arrays[f"g_{d}"] = z[f"y_{d}"], z[f"g_{d}"]
                arrays[f"p_{d}"] = np.column_stack([p[m] for m in MODELS])
            assert check < 1e-9, f"{root}/{label}: recomputed kstats differ from the study's by {check}"
            np.savez_compressed(OUT / f"{label}_k{k}.npz", **arrays)
            (OUT / f"{label}_k{k}_info.json").write_text(json.dumps({"k_columns": [names[c] for c in cols], "kstats": info,
                                                                     "chains_used": used, "kstats_max_abs_diff_vs_study": check},
                                                                    indent=1, default=str), encoding="utf-8")
            ys, ps = arrays["y_sites"], arrays["p_sites"]
            print(f"[sites] {tag} K={k}: sites AUPRC " + ", ".join(f"{m} {sb.auprc(ys, ps[:, j]):.3f}" for j, m in enumerate(MODELS))
                  + f" (kstats check {check:.1e})", flush=True)


def database_of(d: str) -> str:
    return {"mimic_iv": "mimic", "hirid": "hirid"}.get(d, "eicu")


def phase_boot(args) -> None:
    import pandas as pd

    jb = _module("run_joint_bootstrap", ROOT / "scripts/v2/run_joint_bootstrap.py")
    keys = [(tag, k) for tag in TAGS for k in BUDGETS]
    loaded = [(f"{tag}_k{k}", np.load(OUT / f"e_eicu_{tag}_k{k}.npz", allow_pickle=False)) for tag, k in keys]
    models = [str(m) for m in loaded[0][1]["models"]]
    datasets = [str(d) for d in loaded[0][1]["datasets"]]
    mi, di = {m: i for i, m in enumerate(models)}, {d: i for i, d in enumerate(datasets)}
    index = {"mimic": {}, "eicu": {}, "hirid": {}}
    for _, z in loaded:
        for d in datasets:
            for g in z[f"g_{d}"]:
                index[database_of(d)].setdefault(str(g), len(index[database_of(d)]))
    sizes = {db: len(ix) for db, ix in index.items()}
    prep = []
    for _, z in loaded:
        for d in datasets:
            y = z[f"y_{d}"].astype(int)
            g = np.array([index[database_of(d)][str(x)] for x in z[f"g_{d}"]])
            prep.append((d, g, [jb.ap_prepare(y, z[f"p_{d}"][:, j]) for j in range(len(models))]))
    R, M, D = len(loaded), len(models), len(datasets)
    point = np.zeros((R, M, D))
    for i, (_, g, per_m) in enumerate(prep):
        r, k = divmod(i, D)
        for j, (o, e, ys) in enumerate(per_m):
            point[r, j, k] = jb.weighted_ap(np.ones((1, len(g))), o, e, ys)[0]
    rng = np.random.default_rng(args.seed)
    boot = np.zeros((args.n_boot, R, M, D))
    for start in range(0, args.n_boot, 50):
        nb = min(50, args.n_boot - start)
        counts = {db: np.stack([np.bincount(rng.integers(0, n, n), minlength=n) for _ in range(nb)]).astype(float) for db, n in sizes.items()}
        for i, (d, g, per_m) in enumerate(prep):
            r, k = divmod(i, D)
            W = counts[database_of(d)][:, g]
            for j, (o, e, ys) in enumerate(per_m):
                boot[start:start + nb, r, j, k] = jb.weighted_ap(W, o, e, ys)
    groups = {"sites": ("sites",), "databases": DATABASES, "controlled": CONTROLLED, "internal": ("internal",)}

    def ep(arr, name):
        return np.mean(np.stack([arr[..., di[d]] for d in groups[name]], axis=-1), axis=-1)

    names = [n for n, _ in loaded]
    rows = [{"model": m, "endpoint": e, "mean": float(ep(point[:, mi[m]], e).mean()), **dict(zip(names, map(float, ep(point[:, mi[m]], e))))}
            for m in models for e in groups]
    pd.DataFrame(rows).to_csv(OUT / "means.csv", index=False)
    out = []
    for a, b in CONTRASTS:
        for e in groups:
            d = ep(point[:, mi[a]], e) - ep(point[:, mi[b]], e)
            db = ep(boot[:, :, mi[a]], e) - ep(boot[:, :, mi[b]], e)
            lo, hi = np.percentile(db.mean(axis=1), [2.5, 97.5])
            lo_r, hi_r = np.percentile(db, [2.5, 97.5], axis=0)
            out.append({"a": a, "b": b, "endpoint": e, "diff": float(d.mean()), "low": float(lo), "high": float(hi),
                        "better": int((lo_r > 0).sum()), "worse": int((hi_r < 0).sum()), **dict(zip(names, map(float, d)))})
    S = pd.DataFrame(out)
    S.to_csv(OUT / "contrasts.csv", index=False)
    show = S.copy()
    for c in ["diff", "low", "high", *names]:
        show[c] = (100 * show[c]).round(2)
    pd.set_option("display.width", 250)
    print(show[["a", "b", "endpoint", "diff", "low", "high", "better", "worse", *names]].to_string(index=False))
    m = pd.DataFrame(rows).pivot(index="model", columns="endpoint", values="mean")
    print((100 * m).round(1).to_string())


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
