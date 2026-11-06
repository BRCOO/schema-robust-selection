from __future__ import annotations

import csv
import json
import importlib.util
import pickle
from pathlib import Path

from src.icu.v2.pipeline import TargetCohort
from src.icu.v2.selection_baselines import BASELINES
from src.icu.v2.splits import patient_level_split

from tests.v2.test_pipeline import FEATURES, _episodes, _pool

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "v2" / "run_selection_baselines.py"


def _module():
    spec = importlib.util.spec_from_file_location("run_selection_baselines", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _rows(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def test_selection_baselines_end_to_end(tmp_path):
    episodes, _ = patient_level_split(_episodes(80), fractions=(0.6, 0.2, 0.2), seed=3)
    target = TargetCohort(name="ext", features=frozenset(f for f in FEATURES if f != "Lactate"), episodes=tuple(_episodes(25, prefix="x", seed=9)))
    pools = [_pool("strong", episodes, signal=0.45, seed=1, external=[target]), _pool("weak", episodes, signal=0.1, seed=2, external=[target])]
    with open(tmp_path / "pools.pkl", "wb") as handle:
        pickle.dump({"pools": pools, "episodes": episodes, "targets": [target], "timings": {}, "n_candidates": {}}, handle)
    sa = tmp_path / "sa"
    sa.mkdir()
    with open(sa / "tuned.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["method", "selection", "w_redundancy", "w_family", "w_anchor", "w_schema"])
        writer.writeheader()
        for method in ("strong", "weak"):
            writer.writerow({"method": method, "selection": "v2_selector", "w_redundancy": 0.5, "w_family": 0.8, "w_anchor": 0.3, "w_schema": 0.35})
    out = tmp_path / "out"
    assert _module().main(["--pools", str(tmp_path / "pools.pkl"), "--selector-analysis-dir", str(sa), "--out-dir", str(out), "--n-boot", "30", "--selector-targets", "ext"]) == 0

    rows = _rows(out / "rows.csv")
    expected = {"native", "v2_apriori", "v2_tuned", "v2_ig_apriori", "v2_ig_tuned", "v2_auto", "v2_ig_nocaps", "srs_known", "srs_robust25", "srs_robust50", "mrmr_filter_known", *BASELINES}
    for method in ("strong", "weak"):
        assert {r["selector"] for r in rows if r["method"] == method} == expected
    for r in rows:
        assert int(r["n_programs"]) <= 10
        if not r.get("error"):
            assert float(r["auprc_low"]) <= float(r["auprc"]) <= float(r["auprc_high"])
    # each tuned baseline keeps the grid value with the best validation score
    tuning = _rows(out / "tuning.csv")
    n_weight_grid = 36  # run_selector_analysis.WEIGHT_GRID, searched for the information-gain utility
    assert len(tuning) == 2 * (sum(len(grid) for _, _, grid in BASELINES.values()) + 2 * n_weight_grid)
    for method in ("strong", "weak"):
        for name, (_, param, _) in BASELINES.items():
            if param is None:
                continue
            grid_rows = [t for t in tuning if t["method"] == method and t["selector"] == name]
            best = max(float(t["validation_score"]) for t in grid_rows)
            chosen = next(r for r in rows if r["method"] == method and r["selector"] == name)
            assert abs(float(chosen["validation_score"]) - best) < 1e-9
    paired = _rows(out / "paired.csv")
    assert {p["reference"] for p in paired} == {"v2_tuned", "v2_auto", "native", "mrmr", "srs_known", "srs_robust25"}
    target_paired = _rows(out / "target_paired.csv")
    assert {r["target"] for r in target_paired} == {"ext", "mean_over_targets"}
    assert all(float(r["low"]) <= float(r["high"]) for r in target_paired)
    registries = json.loads((out / "registries.json").read_text(encoding="utf-8"))
    assert set(registries) == {"strong", "weak"} and set(registries["strong"]) == expected
    # v2_auto is one of the two tuned v2 registries (the one with the better validation score)
    for method in ("strong", "weak"):
        auto = next(r for r in rows if r["method"] == method and r["selector"] == "v2_auto")
        candidates = [r for r in rows if r["method"] == method and r["selector"] in ("v2_tuned", "v2_ig_tuned")]
        assert any(abs(float(c["auprc"]) - float(auto["auprc"])) < 1e-12 for c in candidates)
    assert all(float(p["low"]) <= float(p["point_diff"]) <= float(p["high"]) for p in paired)
