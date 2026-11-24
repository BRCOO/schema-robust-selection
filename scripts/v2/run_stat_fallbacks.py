#!/usr/bin/env python3
"""Fallback baselines for window statistics (third review of 2026-11-11, substance item 3; the owner asked for experiments
before the 2026-12-01 submission). Post hoc and exploratory.

Why not ship the K-statistics model with the same kind of fallback? For the four standard ICU settings and K = 5, 10, 20
(the 12 standard runs), the K statistics that enter first on an L1 path (run_stat_baselines.l1_columns) are refitted as in
the paper (StandardScaler + class-balanced L2 logistic regression, C on VAL) and deployed at every target in three ways; one
small tree and the registry complete the comparison:

* ``kstats``: a statistic of a feature the target lacks takes the paper's fixed default (missing rate 1, slope, transitions
  and on-ratio 0, other statistics the TRAIN median), the analogue of zeroing;
* ``kstats_chain``: such a statistic falls back to the first executable member of a chain of M = 5 statistics of other
  features, ranked by |Pearson r| on TRAIN (|r| >= 0.05) and mapped onto its scale by the TRAIN least-squares line: the
  analogue of a surrogate chain (catalog lookup, no target data, no refit);
* ``kstats_soft``: such a statistic is predicted by ridge regression from every statistic of the features the target has,
  fitted on TRAIN for that schema: the analogue of soft surrogates (needs the target schema at the source);
* ``tree_k``: one CART tree on the same K statistics (depth on VAL), whose splits on absent features fall back to CART
  surrogate splits on statistics of other features (M = 5, best agreement with the primary split among the TRAIN episodes
  at the node, kept when better than the majority rule), else to the larger child;
* ``prog_srs``: the run's SRS+T registry (CV seed 0) as deployed, each coordinate resolved through its chain under the
  dataset's schema, refitted with the same logistic code as the statistics (``prog_srs`` of run_hybrid_programs_stats.py).

AUPRC at internal TEST, eICU, HiRID and the five controlled targets; a joint patient bootstrap over the 12 runs (MIMIC-IV,
eICU and HiRID patients drawn once per replicate). Outputs: artifacts/v2/stat_fallbacks/.

  CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 python scripts/v2/run_stat_fallbacks.py --phase fit
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
OUT = ROOT / "artifacts/v2/stat_fallbacks"
TAGS = ("gap1h", "gap2h", "gap4h", "aki_gap6h")
BUDGETS = (5, 10, 20)
ROOT_OF_K = {5: "union_k5", 10: "union", 20: "union_k20"}
REAL = ("eicu", "hirid")
CONTROLLED = ("drop_labs", "drop_ventilation", "drop_bp", "drop_urine", "vitals_only")
MODELS = ("kstats", "kstats_chain", "kstats_soft", "tree_k", "prog_srs")
M_SURROGATES = 5
MIN_ABS_R = 0.05
TREE_DEPTHS = (2, 3, 4, 5, 6)
CONTRASTS = (("prog_srs", "kstats"), ("prog_srs", "kstats_chain"), ("prog_srs", "kstats_soft"), ("prog_srs", "tree_k"),
             ("kstats_chain", "kstats"), ("kstats_soft", "kstats"), ("kstats_soft", "kstats_chain"), ("tree_k", "kstats"))


def _module(name: str, path: Path):
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


def channel(name: str) -> str:
    return name.split("__")[0]


# --------------------------------------------------------------------------------------------- statistic fallbacks
def stat_chains(Xtr: np.ndarray, names: list[str]) -> dict[int, list[tuple[int, float, float, float]]]:
    """Chain of every statistic column: (column, slope, intercept, r) of the M most correlated columns of other features."""
    mu, sd = Xtr.mean(axis=0), Xtr.std(axis=0)
    ok = sd > 0
    Z = np.zeros_like(Xtr)
    Z[:, ok] = (Xtr[:, ok] - mu[ok]) / sd[ok]
    R = (Z.T @ Z) / len(Z)
    ch = np.array([channel(n) for n in names])
    chains = {}
    for j in range(len(names)):
        if not ok[j]:
            chains[j] = []
            continue
        order = np.argsort(-np.abs(R[j]), kind="stable")
        out = []
        for i in order:
            if len(out) == M_SURROGATES or abs(R[j, i]) < MIN_ABS_R:
                break
            if ch[i] == ch[j] or not ok[i]:
                continue
            slope = R[j, i] * sd[j] / sd[i]
            out.append((int(i), float(slope), float(mu[j] - slope * mu[i]), float(R[j, i])))
        chains[j] = out
    return chains


def deploy_chain(X: np.ndarray, cols: list[int], names: list[str], chains, absent: set[str]):
    Xk = X[:, cols].copy()
    used = {}
    for a, j in enumerate(cols):
        if channel(names[j]) not in absent:
            continue
        for i, slope, icpt, r in chains[j]:
            if channel(names[i]) not in absent:
                Xk[:, a] = icpt + slope * X[:, i]
                used[names[j]] = {"surrogate": names[i], "r": round(r, 3)}
                break
        else:
            used[names[j]] = None
    return Xk, used


def deploy_soft(Xtr: np.ndarray, X: np.ndarray, cols: list[int], names: list[str], absent: set[str]):
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler

    present = [i for i, n in enumerate(names) if channel(n) not in absent and Xtr[:, i].std() > 0]
    Xk = X[:, cols].copy()
    missing = [a for a, j in enumerate(cols) if channel(names[j]) in absent]
    if not missing or not present:
        return Xk
    scaler = StandardScaler().fit(Xtr[:, present])
    Ztr, Z = scaler.transform(Xtr[:, present]), scaler.transform(X[:, present])
    for a in missing:
        j = cols[a]
        Xk[:, a] = Ridge(alpha=1.0).fit(Ztr, Xtr[:, j]).predict(Z)
    return Xk


# --------------------------------------------------------------------------------------------- CART surrogate splits
def best_surrogate(v: np.ndarray, goes_left: np.ndarray):
    """Threshold and direction on column values v that agree best with the primary split; returns (agreement, s, dir)."""
    m = len(v)
    order = np.argsort(v, kind="stable")
    vs, gl = v[order], goes_left[order].astype(np.int64)
    cl = np.cumsum(gl)
    GL = int(cl[-1])
    r = np.nonzero(np.diff(vs))[0] + 1  # left = first r sorted samples, at every change of value
    if len(r) == 0:
        return None
    agree = m - r - GL + 2 * cl[r - 1]
    best_pos, best_neg = int(np.argmax(agree)), int(np.argmin(agree))
    a_pos, a_neg = agree[best_pos] / m, 1.0 - agree[best_neg] / m
    if a_pos >= a_neg:
        k = r[best_pos]
        return a_pos, float((vs[k - 1] + vs[k]) / 2), 1
    k = r[best_neg]
    return a_neg, float((vs[k - 1] + vs[k]) / 2), -1


def tree_surrogates(clf, Xtr_full: np.ndarray, cols: list[int], names: list[str]):
    t = clf.tree_
    path = clf.decision_path(Xtr_full[:, cols]).tocsc()
    sur, majority_left = {}, {}
    for node in range(t.node_count):
        if t.children_left[node] == -1:
            continue
        idx = path[:, node].nonzero()[0]
        f = cols[t.feature[node]]
        goes_left = Xtr_full[idx, f] <= t.threshold[node]
        p_left = goes_left.mean()
        majority_left[node] = bool(p_left >= 0.5)
        floor = min(p_left, 1 - p_left)
        cands = []
        for i, n in enumerate(names):
            if channel(n) == channel(names[f]):
                continue
            res = best_surrogate(Xtr_full[idx, i], goes_left)
            if res is None or floor <= 0:
                continue
            agreement, s, direction = res
            lam = (floor - (1 - agreement)) / floor
            if lam > 0:
                cands.append((lam, i, s, direction))
        cands.sort(key=lambda c: -c[0])
        sur[node] = [(i, s, d, round(lam, 3)) for lam, i, s, d in cands[:M_SURROGATES]]
    return sur, majority_left


def tree_predict(clf, X_full: np.ndarray, cols: list[int], names: list[str], sur, majority_left, absent: set[str]) -> np.ndarray:
    t = clf.tree_
    value = t.value[:, 0, :]
    p_leaf = value[:, 1] / value.sum(axis=1)
    out = np.empty(len(X_full))
    for r in range(len(X_full)):
        node = 0
        while t.children_left[node] != -1:
            f = cols[t.feature[node]]
            if channel(names[f]) not in absent:
                left = X_full[r, f] <= t.threshold[node]
            else:
                for i, s, d, _ in sur[node]:
                    if channel(names[i]) not in absent:
                        left = (X_full[r, i] <= s) if d > 0 else (X_full[r, i] > s)
                        break
                else:
                    left = majority_left[node]
            node = t.children_left[node] if left else t.children_right[node]
        out[r] = p_leaf[node]
    return out


def fit_tree(Xtr, ytr, Xva, yva, auprc):
    from sklearn.tree import DecisionTreeClassifier

    best = None
    for depth in TREE_DEPTHS:
        clf = DecisionTreeClassifier(max_depth=depth, min_samples_leaf=20, class_weight="balanced", random_state=0).fit(Xtr, ytr)
        score = auprc(yva, clf.predict_proba(Xva)[:, 1])
        if best is None or score > best[0]:
            best = (score, depth, clf)
    return best[2], {"max_depth": best[1], "val_auprc": best[0]}


# --------------------------------------------------------------------------------------------- phases
def phase_fit(args) -> None:
    from src.icu.v2.supervised import FileArrayLoader, build_external_cohort, build_supervised_inputs, load_episode_list, read_feature_names

    sb = _module("run_stat_baselines", ROOT / "scripts/v2/run_stat_baselines.py")
    ctx = _module("e3_context", ROOT / "scripts/v2/e3_context.py")
    OUT.mkdir(parents=True, exist_ok=True)
    for tag in [t for t in args.tags.split(",") if t]:
        todo = [k for k in BUDGETS if not (OUT / f"f_mimic_iv_{tag}_k{k}.npz").exists()]
        if not todo:
            print(f"[fallbacks] {tag}: done, skipped", flush=True)
            continue
        t0 = time.perf_counter()
        task = "aki" if tag.startswith("aki") else "vaso"
        src_dir = ROOT / f"data/processed/v2/extract/mimic_iv_{tag}_full"
        episodes = load_episode_list(ROOT / f"artifacts/v2/task/f_mimic_iv_{tag}/episodes_split.json")
        feat = read_feature_names(src_dir, episodes[0].episode_id)
        loader = FileArrayLoader(src_dir)
        inputs = build_supervised_inputs(episodes, feat, loader, excluded_features=())
        names = list(inputs.tabular_feature_names)
        medians = np.asarray(inputs.preprocessing.tabular_medians, dtype=float)
        fill = lambda A: np.where(np.isfinite(A), A, medians)  # noqa: E731
        tr, va, te = (inputs.splits[s] for s in ("train", "val", "test"))
        data = {"train": (fill(tr.tabular), tr.y, tr.episode_ids, tr.subject_ids), "val": (fill(va.tabular), va.y, va.episode_ids, va.subject_ids),
                "internal": (fill(te.tabular), te.y, te.episode_ids, te.subject_ids)}
        for site in REAL:
            ext_dir = ROOT / f"data/processed/v2/extract/{site}_{tag}_full"
            ext_eps = load_episode_list(ROOT / f"artifacts/v2/task/f_{site}_{tag}/episodes_all.json")
            cohort = build_external_cohort(site, ext_eps, FileArrayLoader(ext_dir), inputs, feature_names=read_feature_names(ext_dir, ext_eps[0].episode_id))
            a = cohort.arrays
            data[site] = (fill(a.tabular), a.y, a.episode_ids, a.subject_ids)
        test_eps = sorted([e for e in episodes if e.split == "test"], key=lambda e: {x: i for i, x in enumerate(te.episode_ids)}[e.episode_id])
        for name, drop in sb.controlled_targets(task).items():
            data[name] = (fill(sb.masked_test(inputs, test_eps, loader, feat, set(drop))), te.y, te.episode_ids, te.subject_ids)
        Xtr, ytr, Xva, yva = data["train"][0], data["train"][1], data["val"][0], data["val"][1]
        chains = stat_chains(Xtr, names)
        print(f"[fallbacks] {tag}: statistics built, {len(names)} columns ({time.perf_counter() - t0:.0f}s)", flush=True)
        datasets = ["internal", *REAL, *CONTROLLED]
        for k in todo:
            run = ctx.load_run(*ctx.run_dirs(ROOT_OF_K[k], f"f_mimic_iv_{tag}"))
            targets = {t.name: t for t in run.targets}
            absent = {d: (set(run.source) - set(targets[d].features)) if d in targets else set() for d in ["train", "val", *datasets]}
            assert all(d in targets for d in [*REAL, *CONTROLLED]), sorted(targets)
            comp = run.registries["srs_ap_t"]
            prog = {}
            for d, (_, _, ids, _) in data.items():
                if d in ("train", "val", "internal"):
                    prog[d] = run.matrix(list(comp.primaries), list(ids))
                else:
                    prog[d] = run.matrix(comp.resolve(targets[d].features), list(ids), run.external[d])
            cols = sb.l1_columns(Xtr, ytr, k)
            model_k, info_k = sb.fit_lr(Xtr[:, cols], ytr, Xva[:, cols], yva)
            model_p, info_p = sb.fit_lr(prog["train"], ytr, prog["val"], yva)
            tree, info_t = fit_tree(Xtr[:, cols], ytr, Xva[:, cols], yva, sb.auprc)
            sur, majority_left = tree_surrogates(tree, Xtr, cols, names)
            probs = {m: {} for m in MODELS}
            used_chain = {}
            for d in datasets:
                X, ab = data[d][0], absent[d]
                probs["kstats"][d] = model_k.predict_proba(X[:, cols])[:, 1]
                Xc, used_chain[d] = deploy_chain(X, cols, names, chains, ab)
                probs["kstats_chain"][d] = model_k.predict_proba(Xc)[:, 1]
                probs["kstats_soft"][d] = model_k.predict_proba(deploy_soft(Xtr, X, cols, names, ab))[:, 1]
                probs["tree_k"][d] = tree_predict(tree, X, cols, names, sur, majority_left, ab)
                probs["prog_srs"][d] = model_p.predict_proba(prog[d])[:, 1]
            for m in MODELS:
                print(f"[fallbacks] {tag} K={k} {m}: internal {sb.auprc(data['internal'][1], probs[m]['internal']):.3f}, real "
                      f"{np.mean([sb.auprc(data[d][1], probs[m][d]) for d in REAL]):.3f}, controlled "
                      f"{np.mean([sb.auprc(data[d][1], probs[m][d]) for d in CONTROLLED]):.3f}", flush=True)
            arrays = {"models": np.array(MODELS), "datasets": np.array(datasets)}
            for d in datasets:
                arrays[f"y_{d}"] = np.asarray(data[d][1], dtype=np.int8)
                arrays[f"g_{d}"] = np.array([str(s) for s in data[d][3]])
                arrays[f"p_{d}"] = np.column_stack([probs[m][d] for m in MODELS])
            np.savez_compressed(OUT / f"f_mimic_iv_{tag}_k{k}.npz", **arrays)
            info = {"k_columns": [names[c] for c in cols], "kstats": info_k, "prog_srs": info_p, "tree_k": info_t,
                    "chains_used": used_chain,
                    "chains_of_k_columns": {names[c]: [(names[i], round(r, 3)) for i, _, _, r in chains[c]] for c in cols},
                    "tree_nodes": int(tree.tree_.node_count), "tree_leaves": int(tree.get_n_leaves()),
                    "tree_surrogates": {str(n): [(names[i], lam) for i, _, _, lam in s] for n, s in sur.items()}}
            (OUT / f"f_mimic_iv_{tag}_k{k}_info.json").write_text(json.dumps(info, indent=1, default=str), encoding="utf-8")
        print(f"[fallbacks] {tag}: done ({time.perf_counter() - t0:.0f}s)", flush=True)


def database_of(d: str) -> str:
    return d if d in REAL else "mimic"


def phase_boot(args) -> None:
    import pandas as pd

    jb = _module("run_joint_bootstrap", ROOT / "scripts/v2/run_joint_bootstrap.py")
    keys = [(tag, k) for k in BUDGETS for tag in TAGS]
    loaded = [(f"{tag}_k{k}", np.load(OUT / f"f_mimic_iv_{tag}_k{k}.npz", allow_pickle=False)) for tag, k in keys]
    models, datasets = [str(m) for m in loaded[0][1]["models"]], [str(d) for d in loaded[0][1]["datasets"]]
    mi = {m: i for i, m in enumerate(models)}
    index = {"mimic": {}, "eicu": {}, "hirid": {}}
    for _, z in loaded:
        for d in datasets:
            for g in z[f"g_{d}"]:
                index[database_of(d)].setdefault(str(g), len(index[database_of(d)]))
    sizes = {db: len(ix) for db, ix in index.items()}
    prep = []
    for name, z in loaded:
        for d in datasets:
            y = z[f"y_{d}"].astype(int)
            g = np.array([index[database_of(d)][str(x)] for x in z[f"g_{d}"]])
            prep.append((name, d, g, [jb.ap_prepare(y, z[f"p_{d}"][:, j]) for j in range(len(models))]))
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
        print(f"[fallbacks] boot {start + nb}/{args.n_boot}", flush=True)
    di = {d: i for i, d in enumerate(datasets)}
    groups = {"internal": ("internal",), "real": REAL, "controlled": CONTROLLED, "targets": REAL + CONTROLLED, **{d: (d,) for d in REAL + CONTROLLED}}

    def ep(arr, name):
        return np.mean(np.stack([arr[..., di[d]] for d in groups[name]], axis=-1), axis=-1)

    names = [n for n, _ in loaded]
    rows = []
    for m in models:
        for e in groups:
            v = ep(point[:, mi[m]], e)
            rows.append({"model": m, "endpoint": e, "mean": float(v.mean()), **dict(zip(names, map(float, v)))})
    means = pd.DataFrame(rows)
    means.to_csv(OUT / "means.csv", index=False)
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
    show = S[S.endpoint.isin(["internal", "real", "controlled", "targets", "eicu"])].copy()
    for c in ("diff", "low", "high"):
        show[c] = (100 * show[c]).round(2)
    pd.set_option("display.width", 220)
    print(show[["a", "b", "endpoint", "diff", "low", "high", "better", "worse"]].to_string(index=False))
    m = means[means.endpoint.isin(["internal", "real", "controlled", "targets"])].pivot(index="model", columns="endpoint", values="mean")
    print((100 * m).round(1).to_string())


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--phase", choices=("fit", "boot"), default="fit")
    parser.add_argument("--tags", default=",".join(TAGS))
    parser.add_argument("--n-boot", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    {"fit": phase_fit, "boot": phase_boot}[args.phase](args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
