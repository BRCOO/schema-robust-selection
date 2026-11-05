from __future__ import annotations

import csv
import importlib.util
import json
import pickle
from pathlib import Path

from src.icu.v2.pipeline import TargetCohort, evaluate_registry, native_registry
from src.icu.v2.splits import patient_level_split

from tests.v2.test_pipeline import FEATURES, _episodes, _pool

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "v2" / "run_decoder_counterfactual.py"


def _module():
    spec = importlib.util.spec_from_file_location("run_decoder_counterfactual", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _state():
    episodes, _ = patient_level_split(_episodes(80), fractions=(0.6, 0.2, 0.2), seed=3)
    target = TargetCohort(name="ext", features=frozenset(f for f in FEATURES if f != "Lactate"), episodes=tuple(_episodes(25, prefix="x", seed=9)))
    pools = [_pool("strong", episodes, signal=0.45, seed=1, external=[target]), _pool("weak", episodes, signal=0.1, seed=2, external=[target])]
    return {"pools": pools, "episodes": episodes, "targets": [target], "timings": {}, "n_candidates": {}}


def test_decoders_scored_and_frozen_lr_matches_the_pipeline(tmp_path):
    state = _state()
    pools = tmp_path / "pools.pkl"
    with open(pools, "wb") as handle:
        pickle.dump(state, handle)
    sa_dir = tmp_path / "sa"
    sa_dir.mkdir()
    with open(sa_dir / "tuned.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["method", "selection", "w_redundancy", "w_family", "w_anchor", "w_schema"])
        writer.writeheader()
        writer.writerow({"method": "strong", "selection": "v2_selector", "w_redundancy": 0.5, "w_family": 0.0, "w_anchor": 0.3, "w_schema": 0.0})
    out = tmp_path / "out"
    module = _module()
    assert module.main(["--pools", str(pools), "--selector-analysis-dir", str(sa_dir), "--out-dir", str(out), "--n-boot", "50", "--selector-targets", "ext"]) == 0
    with open(out / "decoder.csv", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    keys = {(r["method"], r["selection"], r["decoder"]) for r in rows}
    assert ("strong", "v2_selector_tuned", "gbdt") in keys and ("weak", "v2_selector_tuned", "gbdt") not in keys
    assert {r["decoder"] for r in rows} == {"frozen_lr", "gbdt"}
    for r in rows:
        assert float(r["auprc_low"]) <= float(r["auprc"]) <= float(r["auprc_high"])
        assert 0.0 <= float(r["ext_schema_survival"]) <= 1.0
    # the frozen LR decoder is exactly the main pipeline's bridge
    pool = state["pools"][0]
    reference = evaluate_registry(native_registry(pool, 10), pool.firings, state["episodes"], targets=state["targets"], external_firings=pool.external_firings, n_boot=50)
    lr = next(r for r in rows if r["method"] == "strong" and r["selection"] == "native" and r["decoder"] == "frozen_lr")
    assert abs(float(lr["auprc"]) - reference["internal"]["auprc"]) < 1e-12
    assert abs(float(lr["ext_auprc"]) - reference["external"]["ext"]["auprc"]) < 1e-12
    summary = json.loads((out / "decoder_summary.json").read_text(encoding="utf-8"))
    assert -1.0 <= summary["kendall_tau_internal_auprc"] <= 1.0


def test_kendall_tau():
    module = _module()
    assert module.kendall_tau([1, 2, 3], [10, 20, 30]) == 1.0
    assert module.kendall_tau([1, 2, 3], [30, 20, 10]) == -1.0
