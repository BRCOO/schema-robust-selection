from __future__ import annotations

import csv
import importlib.util
import json
import pickle
from pathlib import Path

from src.icu.v2.pipeline import TargetCohort
from src.icu.v2.splits import patient_level_split

from tests.v2.test_pipeline import FEATURES, _episodes, _pool

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "v2" / "run_registry_pairs.py"


def _module():
    spec = importlib.util.spec_from_file_location("run_registry_pairs", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_registry_pairs(tmp_path):
    episodes, _ = patient_level_split(_episodes(80), fractions=(0.6, 0.2, 0.2), seed=3)
    ext = TargetCohort(name="ext", features=frozenset(f for f in FEATURES if f != "Lactate"), episodes=tuple(_episodes(25, prefix="x", seed=9)))
    pool = _pool("strong", episodes, signal=0.45, seed=1, external=[ext])
    with open(tmp_path / "pools.pkl", "wb") as handle:
        pickle.dump({"pools": [pool], "episodes": episodes, "targets": [ext], "timings": {}, "n_candidates": {}}, handle)
    ids = [p.program_id for p in pool.candidates]
    (tmp_path / "a.json").write_text(json.dumps({"strong": {"x": ids[:3]}}), encoding="utf-8")
    (tmp_path / "b.json").write_text(json.dumps({"strong": {"y": ids[3:6], "z": ids[:3]}}), encoding="utf-8")
    out = tmp_path / "pairs.csv"
    assert _module().main(["--pools", str(tmp_path / "pools.pkl"), "--a", f"{tmp_path / 'a.json'}:x", "--b", f"{tmp_path / 'b.json'}:y", "--b", f"{tmp_path / 'b.json'}:z", "--out", str(out), "--n-boot", "30"]) == 0
    with open(out, encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert {r["b"] for r in rows} == {"y", "z"}
    assert {r["target"] for r in rows} == {"internal_test", "ext", "mean_over_targets"}
    # identical registries give a zero difference everywhere
    assert all(abs(float(r["point_diff"])) < 1e-12 for r in rows if r["b"] == "z")
