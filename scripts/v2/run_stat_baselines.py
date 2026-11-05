#!/usr/bin/env python3
"""Statistic baselines (review of the v3 experiments, round 2, item M5): the supervised references on window statistics
at every target, including the five controlled targets, and a budget-matched interpretable baseline that selects K window
statistics with an L1 path and refits an L2 logistic regression on them.

Inputs and protocol follow run_supervised_v2.py (--window stored, nothing excluded): the v1 handcrafted kernel per channel
(min, max, mean, std, last, slope, missing rate; transitions and on-ratio for interventions), training medians for
statistics of channels without observations, external cohorts and controlled targets scored with the internal (frozen)
preprocessing. A controlled target is MIMIC-IV TEST with the target's dropped features missing in every bin, exactly how eICU
encodes its absent ventilation channel.

Models (choices on VAL AUPRC only, one TEST pass):
* lr_stats: StandardScaler + class-balanced L2 logistic regression, C from (0.001, 0.01, 0.1, 1, 10).
* gbdt_stats: HistGradientBoosting (learning rate 0.05, no early stopping), max_depth from (3, 4, 6) and the number of
  boosting iterations (at most 500) on VAL.
* stats_l1_k{K}: the K statistics that enter first on an L1 path of class-balanced logistic regressions on standardized
  TRAIN statistics (largest C with at most K non-zero coefficients), then lr_stats on those K columns.

  CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1 /d/BRCoo/miniconda/python scripts/v2/run_stat_baselines.py [--labels ...]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.icu.v2.supervised import (  # noqa: E402
    FileArrayLoader,
    _episode_model_inputs,
    build_external_cohort,
    build_supervised_inputs,
    load_episode_list,
    read_feature_names,
)

LABS = ["Lactate", "pH", "Creatinine", "Glucose", "Bilirubin_Total", "WBC", "Platelets"]
C_GRID = (0.001, 0.01, 0.1, 1.0, 10.0)
DEPTHS = (3, 4, 6)
BUDGETS = (5, 10, 20)
OUT = ROOT / "artifacts/v2/stat_baselines"


def controlled_targets(task: str) -> dict[str, list[str]]:
    return {"drop_labs": LABS, "drop_ventilation": ["ventilation_on"], "drop_bp": ["SBP", "DBP", "MAP"], "drop_urine": ["urine_output"],
            "vitals_only": LABS + ["ventilation_on", "sedation_on", "urine_output"] + (["vasopressor_on"] if task == "aki" else [])}


def auprc(y, p) -> float:
    return float(average_precision_score(y, p))


def masked_test(inputs, test_eps, loader, feature_names, drop) -> np.ndarray:
    pre = inputs.preprocessing
    position = {f: i for i, f in enumerate(feature_names)}
    columns = np.asarray([-1 if f in drop else position.get(f, -1) for f in pre.input_features], dtype=np.int64)
    rows = []
    for ep in test_eps:
        values, missing = loader(ep)
        vector, _, _, _, _ = _episode_model_inputs(ep, values, missing, len(feature_names), columns, pre.input_features, pre.intervention_features)
        rows.append(vector)
    tab = np.vstack(rows)
    return np.where(np.isfinite(tab), tab, pre.tabular_medians)


def fit_lr(Xtr, ytr, Xva, yva):
    best = None
    for c in C_GRID:
        model = Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(C=c, max_iter=1000, class_weight="balanced"))])
        model.fit(Xtr, ytr)
        score = auprc(yva, model.predict_proba(Xva)[:, 1])
        if best is None or score > best[0]:
            best = (score, c, model)
    return best[2], {"C": best[1], "val_auprc": best[0]}


def fit_gbdt(Xtr, ytr, Xva, yva):
    best = None
    for depth in DEPTHS:
        model = HistGradientBoostingClassifier(learning_rate=0.05, max_depth=depth, max_iter=500, early_stopping=False, random_state=0)
        model.fit(Xtr, ytr)
        curve = np.asarray([auprc(yva, s[:, 1]) for s in model.staged_predict_proba(Xva)])
        n_iter = int(np.argmax(curve)) + 1
        if best is None or curve[n_iter - 1] > best[0]:
            best = (float(curve[n_iter - 1]), depth, n_iter, model)
    _, depth, n_iter, model = best

    def predict(X):
        for i, stage in enumerate(model.staged_predict_proba(X), start=1):
            if i == n_iter:
                return stage[:, 1]
    return predict, {"max_depth": depth, "max_iter": n_iter, "val_auprc": best[0]}


def l1_columns(Xtr, ytr, k: int) -> list[int]:
    scaler = StandardScaler().fit(Xtr)
    Z = scaler.transform(Xtr)
    chosen = []
    for c in np.logspace(-4, 1, 60):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            clf = LogisticRegression(penalty="l1", C=c, solver="liblinear", class_weight="balanced", max_iter=2000).fit(Z, ytr)
        nz = np.flatnonzero(np.abs(clf.coef_[0]) > 1e-10)
        if len(nz) > k:
            break
        chosen = nz.tolist()
    return chosen


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--labels", default="gap1h,gap2h,gap4h,aki_gap6h")
    args = parser.parse_args(argv)
    for tag in [x for x in args.labels.split(",") if x]:
        out = OUT / f"f_mimic_iv_{tag}"
        if (out / "rows.json").exists():
            print(f"[stats] {tag}: done, skipped", flush=True)
            continue
        t0 = time.perf_counter()
        task = "aki" if tag.startswith("aki") else "vaso"
        src_dir = ROOT / f"data/processed/v2/extract/mimic_iv_{tag}_full"
        episodes = load_episode_list(ROOT / f"artifacts/v2/task/f_mimic_iv_{tag}/episodes_split.json")
        names = read_feature_names(src_dir, episodes[0].episode_id)
        loader = FileArrayLoader(src_dir)
        inputs = build_supervised_inputs(episodes, names, loader, excluded_features=())
        tr, va, te = (inputs.splits[s] for s in ("train", "val", "test"))
        print(f"[stats] {tag}: inputs {tr.tabular.shape} / {va.tabular.shape} / {te.tabular.shape} ({time.perf_counter() - t0:.0f}s)", flush=True)
        targets = {}
        for site in ("eicu", "hirid"):
            ext_dir = ROOT / f"data/processed/v2/extract/{site}_{tag}_full"
            ext_eps = load_episode_list(ROOT / f"artifacts/v2/task/f_{site}_{tag}/episodes_all.json")
            ext_names = read_feature_names(ext_dir, ext_eps[0].episode_id)
            cohort = build_external_cohort(site, ext_eps, FileArrayLoader(ext_dir), inputs, feature_names=ext_names)
            targets[site] = (cohort.arrays.tabular, cohort.arrays.y)
        test_eps = [e for e in episodes if e.split == "test"]
        order = {e: i for i, e in enumerate(te.episode_ids)}
        test_eps = sorted(test_eps, key=lambda e: order[e.episode_id])
        for name, drop in controlled_targets(task).items():
            targets[name] = (masked_test(inputs, test_eps, loader, names, set(drop)), te.y)
        print(f"[stats] {tag}: targets built ({time.perf_counter() - t0:.0f}s)", flush=True)
        rows, details = [], {}

        def record(model_name, predict, info):
            row = {"model": model_name, "internal_auprc": auprc(te.y, predict(te.tabular))}
            for t, (X, y) in targets.items():
                row[f"{t}_auprc"] = auprc(y, predict(X))
            row["mean_target_auprc"] = float(np.mean([row[f"{t}_auprc"] for t in targets]))
            row["real_auprc"] = float(np.mean([row["eicu_auprc"], row["hirid_auprc"]]))
            rows.append(row)
            details[model_name] = info
            print(f"[stats] {tag} {model_name}: internal {row['internal_auprc']:.3f}, real {row['real_auprc']:.3f}, targets {row['mean_target_auprc']:.3f}",
                  flush=True)

        model, info = fit_lr(tr.tabular, tr.y, va.tabular, va.y)
        record("lr_stats", lambda X: model.predict_proba(X)[:, 1], info)
        predict, info = fit_gbdt(tr.tabular, tr.y, va.tabular, va.y)
        record("gbdt_stats", predict, info)
        for k in BUDGETS:
            cols = l1_columns(tr.tabular, tr.y, k)
            model_k, info = fit_lr(tr.tabular[:, cols], tr.y, va.tabular[:, cols], va.y)
            info["columns"] = [inputs.tabular_feature_names[c] for c in cols]
            record(f"stats_l1_k{k}", (lambda m, c: (lambda X: m.predict_proba(X[:, c])[:, 1]))(model_k, cols), info)
        out.mkdir(parents=True, exist_ok=True)
        (out / "rows.json").write_text(json.dumps({"rows": rows, "details": details}, indent=1), encoding="utf-8")
        print(f"[stats] {tag}: done ({time.perf_counter() - t0:.0f}s)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
