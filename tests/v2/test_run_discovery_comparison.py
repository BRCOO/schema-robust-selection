from __future__ import annotations

import csv
import importlib.util
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest

from src.icu.v2.materialize import episode_paths
from src.icu.v2.splits import patient_level_split
from src.icu.v2.types import EpisodeRecord

from tests.v2.test_pools import FEATURES, _arrays, _episodes, _java_ready, _write_fmg

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "v2" / "run_discovery_comparison.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("run_discovery_comparison", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _materialize(directory: Path, episodes, arrays) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    for e in episodes:
        values, mask, names = arrays[e.episode_id]
        paths = episode_paths(directory, e.episode_id)
        np.save(paths["values"], values)
        np.save(paths["mask"], mask)
        paths["meta"].write_text(json.dumps({"episode_id": e.episode_id, "feature_names": list(names)}), encoding="utf-8")
    return directory


def test_cli_end_to_end(tmp_path):
    episodes, _ = patient_level_split(_episodes(), fractions=(0.6, 0.2, 0.2), seed=4)
    target_episodes = _episodes(15, prefix="x")
    internal_dir = _materialize(tmp_path / "internal", episodes, _arrays(episodes))
    target_dir = _materialize(tmp_path / "target", target_episodes, _arrays(target_episodes, seed=9))
    (tmp_path / "cohort.json").write_text(json.dumps([asdict(e) for e in episodes]), encoding="utf-8")
    (tmp_path / "target.json").write_text(json.dumps([asdict(e) for e in target_episodes]), encoding="utf-8")
    fmg_dir, inputs = _write_fmg(tmp_path, episodes)
    (tmp_path / "ext").mkdir()
    _, ext_inputs = _write_fmg(tmp_path / "ext", target_episodes)

    methods = {
        "fmg_bridge": {"fmg_dir": str(fmg_dir), "contrastive_inputs": str(inputs)},
        "shapelet": {"lengths": [4], "top_k_per_feature": 2, "feature_names": ["HR", "Lactate"]},
    }
    if _java_ready():
        methods["prefixspan"] = {"minsup": 0.4, "max_pattern_length": 2, "pool_cap": 30}
    config = {
        "cohort": str(tmp_path / "cohort.json"),
        "episode_dir": str(internal_dir),
        "k": 3,
        "n_boot": 50,
        "methods": methods,
        "targets": {
            "ext": {
                "episodes": str(tmp_path / "target.json"),
                "episode_dir": str(target_dir),
                "features": list(FEATURES) + ["Creatinine"],
                "fmg_contrastive_inputs": str(ext_inputs),
            }
        },
        "reference": ["fmg_bridge", "native"],
        "schema_shift_targets": {"no_lactate": {"drop": ["Lactate"]}},
    }
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    module = _load_script()
    assert module.main(["--config", str(config_path), "--out-dir", str(tmp_path / "out")]) == 0
    with open(tmp_path / "out" / "comparison_rows.csv", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 2 * len(methods)
    assert all(not row.get("error") for row in rows), rows
    full = json.loads((tmp_path / "out" / "comparison_full.json").read_text(encoding="utf-8"))
    assert len(full["paired_vs_reference"]) == 2 * len(methods) - 1
    assert set(full["timings_seconds"]["pool_build"]) == set(methods)
    for row in rows:
        assert row.get("no_lactate_schema_survival") not in (None, "")
    lactate_rows = [r for r in rows if r["method"] == "shapelet"]
    assert any(float(r["no_lactate_schema_survival"]) < 1.0 for r in lactate_rows)  # Lactate shapelets lose survival
    assert len(json.loads((tmp_path / "out" / "provenance.json").read_text(encoding="utf-8"))["config_sha256"]) == 64


def test_cli_rejects_unknown_method(tmp_path):
    module = _load_script()
    episodes = [EpisodeRecord("a", 1, 1, 1, split="train")]
    (tmp_path / "cohort.json").write_text(json.dumps([asdict(e) for e in episodes]), encoding="utf-8")
    config = {"cohort": str(tmp_path / "cohort.json"), "episode_dir": str(tmp_path), "methods": {"dtw": {}}}
    with pytest.raises(ValueError, match="unknown methods"):
        module.run(config)


def test_saved_pools_reproduce_the_same_rows(tmp_path):
    episodes, _ = patient_level_split(_episodes(), fractions=(0.6, 0.2, 0.2), seed=4)
    internal_dir = _materialize(tmp_path / "internal", episodes, _arrays(episodes))
    (tmp_path / "cohort.json").write_text(json.dumps([asdict(e) for e in episodes]), encoding="utf-8")
    config = {
        "cohort": str(tmp_path / "cohort.json"),
        "episode_dir": str(internal_dir),
        "k": 3,
        "n_boot": 30,
        "methods": {"shapelet": {"lengths": [4], "top_k_per_feature": 2, "feature_names": ["HR", "Lactate"]}},
    }
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    module = _load_script()
    pools = tmp_path / "pools.pkl"
    assert module.main(["--config", str(config_path), "--out-dir", str(tmp_path / "a"), "--save-pools", str(pools)]) == 0
    assert module.main(["--config", str(config_path), "--out-dir", str(tmp_path / "b"), "--load-pools", str(pools)]) == 0
    a = (tmp_path / "a" / "comparison_rows.csv").read_text(encoding="utf-8")
    b = (tmp_path / "b" / "comparison_rows.csv").read_text(encoding="utf-8")
    assert a == b
