#!/usr/bin/env python3
"""Decoder counterfactual: does a more flexible decoder change the comparison between registries?

The frozen bridge is an L2 logistic regression on the K binary program indicators, with C
chosen on validation (``src.icu.v2.bridge``). This script re-scores the same registries with a
gradient-boosted tree decoder on the same K indicators (interactions allowed), with its
hyper-parameters chosen on VALIDATION from a small grid and the model fit on TRAIN only (no
train+val refit, as for the bridge). External targets are scored with the same survival mask
(non-survivable coordinates forced to 0, nothing refit or repaired).

Registries per method (K = 10): native top-K, the v2 selector with the a-priori weights
(``run_selector_analysis.base``), and the v2 selector with the validation-chosen weights from
``<selector-analysis-dir>/tuned.csv`` when that file exists.

  CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1 /d/BRCoo/miniconda/python scripts/v2/run_decoder_counterfactual.py \\
      --pools artifacts/v2/comparison/f_mimic_iv_gap1h/pools.pkl \\
      --selector-analysis-dir artifacts/v2/selector_analysis/f_mimic_iv_gap1h \\
      --out-dir artifacts/v2/decoder_counterfactual/f_mimic_iv_gap1h --selector-targets eicu,hirid
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
from typing import Any, Mapping, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sklearn.ensemble import HistGradientBoostingClassifier  # noqa: E402

from src.icu.v2.bridge import compile_bridge, fit_frozen_bridge_v2, refire_external  # noqa: E402
from src.icu.v2.pipeline import episodes_by_split, native_registry, restrict_firings, selector_registry  # noqa: E402
from src.icu.v2.stats import metric_value, patient_cluster_bootstrap  # noqa: E402

GBDT_GRID = {"max_depth": (2, 3), "learning_rate": (0.05, 0.1)}
GBDT_FIXED = {"max_iter": 300, "min_samples_leaf": 20, "early_stopping": False, "random_state": 0}


def _selector_analysis_module():
    spec = importlib.util.spec_from_file_location("run_selector_analysis", ROOT / "scripts" / "v2" / "run_selector_analysis.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def registries(state: Mapping[str, Any], targets: Sequence[str] | None, tuned_rows: Sequence[Mapping[str, str]]) -> list[tuple[str, str, list]]:
    """(method, selection, registry) for native, v2 a-priori weights and v2 validation-tuned weights."""
    sa = _selector_analysis_module()
    grouped = episodes_by_split(state["episodes"])
    train_ids = frozenset(e.episode_id for e in grouped["train"])
    schemas = {t.name: t.features for t in state["targets"] if not targets or t.name in targets}
    out = []
    for pool in state["pools"]:
        out.append((pool.method, "native", native_registry(pool, 10)))
        out.append((pool.method, "v2_selector", selector_registry(pool, sa.base(10), train_ids, schemas)[0]))
        chosen = next((r for r in tuned_rows if r["method"] == pool.method and r["selection"] == "v2_selector"), None)
        if chosen is not None:
            overrides = {key: float(chosen[key]) for key in sa.WEIGHT_GRID}
            out.append((pool.method, "v2_selector_tuned", selector_registry(pool, sa.base(10, **overrides), train_ids, schemas)[0]))
    return out


def _fit_gbdt(x_train: np.ndarray, y_train: np.ndarray, x_val: np.ndarray, y_val: np.ndarray) -> tuple[HistGradientBoostingClassifier, dict[str, Any], float]:
    best = None
    for values in itertools.product(*GBDT_GRID.values()):
        params = dict(zip(GBDT_GRID, values))
        model = HistGradientBoostingClassifier(**params, **GBDT_FIXED).fit(x_train, y_train)
        score = metric_value(y_val, model.predict_proba(x_val)[:, 1], "auprc")
        # ties: keep the earlier (simpler) grid point
        if best is None or score > best[2] + 1e-12:
            best = (model, params, score)
    return best


def _scores(episodes, probabilities: Mapping[str, float], n_boot: int, seed: int) -> dict[str, float]:
    y = np.array([e.label for e in episodes], dtype=int)
    p = np.array([probabilities[e.episode_id] for e in episodes], dtype=float)
    g = np.array([e.subject_id for e in episodes])
    auprc = patient_cluster_bootstrap(y, p, g, "auprc", n_boot=n_boot, seed=seed)
    return {"auprc": auprc["point"], "auprc_low": auprc["low"], "auprc_high": auprc["high"], "auroc": metric_value(y, p, "auroc"), "prevalence": float(y.mean())}


def evaluate(state: Mapping[str, Any], method: str, selection: str, registry: list, n_boot: int, seed: int) -> list[dict[str, Any]]:
    pool = next(p for p in state["pools"] if p.method == method)
    grouped = episodes_by_split(state["episodes"])
    ids = {name: [e.episode_id for e in rows] for name, rows in grouped.items()}
    labels = {name: [e.label for e in rows] for name, rows in grouped.items()}
    firings = {p.program_id: frozenset(pool.firings.get(p.program_id, frozenset())) for p in registry}
    split_firings = {name: restrict_firings(firings, frozenset(ids[name])) for name in ids}
    rows = []
    # 1. frozen L2 logistic bridge (exactly the main pipeline's bridge)
    bridge = fit_frozen_bridge_v2(registry, split_firings, ids, labels)
    lr_bundle = bridge["model_bundle"]
    # 2. gradient-boosted trees on the same K indicators
    x = {name: compile_bridge(registry, split_firings[name], ids[name])[0] for name in ids}
    y = {name: np.array(labels[name], dtype=int) for name in ids}
    gbdt, params, val_score = _fit_gbdt(x["train"], y["train"], x["val"], y["val"])
    gbdt_bundle = {"model": gbdt, "coordinate_names": [p.program_id for p in registry], "encoding": "binary"}
    gbdt_test = dict(zip(ids["test"], gbdt.predict_proba(x["test"])[:, 1]))
    for decoder, bundle, test_probs, hparams, vscore in (
        ("frozen_lr", lr_bundle, bridge["test_probabilities"], {"C": bridge["selected_c"]}, bridge["validation_score"]),
        ("gbdt", gbdt_bundle, gbdt_test, params, val_score),
    ):
        row: dict[str, Any] = {"method": method, "selection": selection, "decoder": decoder, "n_programs": len(registry), "hparams": json.dumps(hparams), "validation_score": vscore}
        row.update(_scores(grouped["test"], test_probs, n_boot, seed))
        for target in state["targets"]:
            refired = refire_external(bundle, registry, target.features, pool.external_firings.get(target.name, {}), [e.episode_id for e in target.episodes])
            scored = _scores(target.episodes, refired["probabilities"], n_boot, seed)
            row.update({f"{target.name}_{k}": v for k, v in scored.items()})
            row[f"{target.name}_schema_survival"] = refired["schema_survival"]
        rows.append(row)
    return rows


def kendall_tau(a: Sequence[float], b: Sequence[float]) -> float:
    pairs = list(itertools.combinations(range(len(a)), 2))
    if not pairs:
        return float("nan")
    s = sum(np.sign(a[i] - a[j]) * np.sign(b[i] - b[j]) for i, j in pairs)
    return float(s / len(pairs))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pools", required=True)
    parser.add_argument("--selector-analysis-dir", default=None, help="folder with tuned.csv (validation-chosen weights)")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--selector-targets", default="")
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    started = time.perf_counter()
    with open(args.pools, "rb") as handle:
        state = pickle.load(handle)
    targets = [t for t in args.selector_targets.split(",") if t] or None
    tuned_rows: list[dict[str, str]] = []
    if args.selector_analysis_dir and (Path(args.selector_analysis_dir) / "tuned.csv").exists():
        with open(Path(args.selector_analysis_dir) / "tuned.csv", encoding="utf-8") as handle:
            tuned_rows = list(csv.DictReader(handle))
    rows: list[dict[str, Any]] = []
    for method, selection, registry in registries(state, targets, tuned_rows):
        if not registry:
            continue
        rows += evaluate(state, method, selection, registry, args.n_boot, args.seed)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    columns = sorted({k for r in rows for k in r}, key=lambda c: (c not in ("method", "selection", "decoder", "n_programs"), c))
    with open(out / "decoder.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    keys = sorted({(r["method"], r["selection"]) for r in rows})
    lr = {(r["method"], r["selection"]): r for r in rows if r["decoder"] == "frozen_lr"}
    gb = {(r["method"], r["selection"]): r for r in rows if r["decoder"] == "gbdt"}
    summary = {
        "pools": args.pools,
        "args": vars(args),
        "gbdt_grid": GBDT_GRID,
        "gbdt_fixed": GBDT_FIXED,
        "n_registries": len(keys),
        "kendall_tau_internal_auprc": kendall_tau([lr[k]["auprc"] for k in keys], [gb[k]["auprc"] for k in keys]),
        "max_abs_internal_auprc_change": max(abs(gb[k]["auprc"] - lr[k]["auprc"]) for k in keys) if keys else None,
        "elapsed_seconds": time.perf_counter() - started,
    }
    for target in state["targets"]:
        summary[f"kendall_tau_{target.name}_auprc"] = kendall_tau([lr[k][f"{target.name}_auprc"] for k in keys], [gb[k][f"{target.name}_auprc"] for k in keys])
    (out / "decoder_summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "args"}, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
