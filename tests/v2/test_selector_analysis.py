from __future__ import annotations

import csv
import importlib.util
import pickle
from pathlib import Path

from src.icu.v2.pipeline import TargetCohort
from src.icu.v2.splits import patient_level_split

from tests.v2.test_pipeline import FEATURES, _episodes, _pool

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "v2" / "run_selector_analysis.py"


def _module():
    spec = importlib.util.spec_from_file_location("run_selector_analysis", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _state():
    episodes, _ = patient_level_split(_episodes(), fractions=(0.6, 0.2, 0.2), seed=1)
    target = TargetCohort(name="ext", features=frozenset(f for f in FEATURES if f != "Lactate"), episodes=tuple(_episodes(20, prefix="x", seed=7)))
    pools = [_pool("strong", episodes, signal=0.45, seed=1, external=[target]), _pool("noise", episodes, signal=0.0, seed=2, external=[target])]
    return {"pools": pools, "episodes": episodes, "targets": [target], "timings": {}, "n_candidates": {}}


def _rows(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def test_all_analyses_run_on_a_cached_state(tmp_path):
    pools = tmp_path / "pools.pkl"
    with open(pools, "wb") as handle:
        pickle.dump(_state(), handle)
    module = _module()
    out = tmp_path / "out"
    assert module.main(["--pools", str(pools), "--out-dir", str(out), "--n-boot", "20", "--tuned-n-boot", "20", "--stability-reps", "3", "--selector-targets", "ext"]) == 0
    k_rows = _rows(out / "k.csv")
    assert {r["k"] for r in k_rows} == {"5", "10", "20"} and len(k_rows) == 3 * 2 * (2 + 1)
    assert {(r["caps"], r["selection"]) for r in k_rows} == {("fixed", "native"), ("fixed", "v2_selector"), ("scaled", "v2_selector")}
    at_ten = {(r["method"], r["caps"]): r["auprc"] for r in k_rows if r["k"] == "10" and r["selection"] == "v2_selector"}
    assert all(at_ten[(m, "fixed")] == at_ten[(m, "scaled")] for m in ("strong", "noise"))  # scaled caps = defaults at K = 10
    ablation = _rows(out / "ablation.csv")
    assert {r["variant"] for r in ablation} == set(module.ABLATIONS) and all(r["selection"] == "v2_selector" for r in ablation)
    weights = _rows(out / "weights.csv")
    assert len(weights) == 36 * 2
    assert sum(r["is_default"] == "True" for r in weights) == 2
    for method in ("strong", "noise"):
        chosen = [r for r in weights if r["method"] == method and r["val_selected"] == "True"]
        assert len(chosen) == 1
        best = max(float(r["validation_score"]) for r in weights if r["method"] == method)
        assert abs(float(chosen[0]["validation_score"]) - best) < 1e-9
    tuned = _rows(out / "tuned.csv")
    assert {(r["method"], r["selection"]) for r in tuned} == {(m, s) for m in ("strong", "noise") for s in ("native", "v2_selector")}
    for r in tuned:
        if r["selection"] == "v2_selector":
            chosen = next(w for w in weights if w["method"] == r["method"] and w["val_selected"] == "True")
            assert all(float(r[key]) == float(chosen[key]) for key in module.WEIGHT_GRID)
            assert float(r["diff_low"]) <= float(r["diff_vs_native"]) <= float(r["diff_high"])
    # tuned can also run alone from a weights.csv written by an earlier run
    assert module.main(["--pools", str(pools), "--out-dir", str(out), "--analyses", "tuned", "--tuned-n-boot", "20", "--selector-targets", "ext"]) == 0
    assert _rows(out / "tuned.csv") == tuned
    stability = _rows(out / "stability.csv")
    assert {(r["method"], r["selection"]) for r in stability} == {(m, s) for m in ("strong", "noise") for s in ("topk_contrast", "v2_selector")}
    assert all(0.0 <= float(r["jaccard_to_full_mean"]) <= 1.0 for r in stability)
    registry = _rows(out / "registry.csv")
    assert registry and all(r["description"] for r in registry)


def test_describe_numeric_and_symbolic_programs():
    from src.icu.v2.types import ProgramRecord, make_symbolic_program

    module = _module()
    rule = make_symbolic_program("r", "erminer", "unordered_rule", ({"HR:q3", "MAP:q0"},), ({"Lactate:q3"},))
    assert module.describe(rule) == "{HR:q3, MAP:q0} => {Lactate:q3}"
    shapelet = ProgramRecord(
        program_id="s", method="shapelet", kind="shapelet", features=frozenset({"MAP"}), anchor_signature=frozenset({"hemodynamic"}),
        params={"feature": "MAP", "length": 3, "template": (1.0, 0.0, -1.0), "threshold": 0.5, "direction": "le"},
    )
    assert module.describe(shapelet).startswith("shapelet[MAP, L=3] near z-shape (+1.00, +0.00, -1.00)")
