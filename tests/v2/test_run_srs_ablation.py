from __future__ import annotations

import csv
import importlib.util
import pickle
from pathlib import Path

from src.icu.v2.pipeline import TargetCohort
from src.icu.v2.splits import patient_level_split

from tests.v2.test_pipeline import FEATURES, _episodes, _pool

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "v2" / "run_srs_ablation.py"


def _module():
    spec = importlib.util.spec_from_file_location("run_srs_ablation", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_srs_ablation_end_to_end(tmp_path):
    episodes, _ = patient_level_split(_episodes(80), fractions=(0.6, 0.2, 0.2), seed=3)
    ext = TargetCohort(name="ext", features=frozenset(f for f in FEATURES if f != "Lactate"), episodes=tuple(_episodes(25, prefix="x", seed=9)))
    pools = [_pool("strong", episodes, signal=0.45, seed=1, external=[ext])]
    with open(tmp_path / "pools.pkl", "wb") as handle:
        pickle.dump({"pools": pools, "episodes": episodes, "targets": [ext], "timings": {}, "n_candidates": {}}, handle)
    out = tmp_path / "out"
    module = _module()
    assert module.main(["--pools", str(tmp_path / "pools.pkl"), "--out-dir", str(out), "--known-targets", "ext", "--n-boot", "30"]) == 0
    with open(out / "rows.csv", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert {r["variant"] for r in rows} == set(module.VARIANTS)
    for r in rows:
        assert int(r["n_programs"]) <= int(r["k"])
    with open(out / "target_paired.csv", encoding="utf-8") as handle:
        paired = list(csv.DictReader(handle))
    assert {p["reference"] for p in paired} <= set(module.VARIANTS) - {"srs_known"}
    assert {p["target"] for p in paired} == {"ext", "mean_over_targets"}
