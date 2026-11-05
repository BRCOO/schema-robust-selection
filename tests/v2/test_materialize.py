from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.icu.v2.materialize import (
    core_observation_counts,
    episode_paths,
    materialize_cohort,
    materialize_episode,
    spec_sha256,
)
from src.icu.v2.task import TaskSpec

ROOT = Path(__file__).resolve().parents[2]
REAL_DIR = ROOT / "data" / "processed" / "mimic-iv-full-cw3-gap0-real" / "mimic-iv"
REAL_ID = "stay_30000484_21360115094000_positive"
FEATURES = [
    "HR", "RR", "SpO2", "SBP", "DBP", "MAP", "Temp", "ventilation_on", "Lactate", "pH",
    "Creatinine", "Glucose", "Bilirubin_Total", "WBC", "Platelets", "vasopressor_on", "sedation_on", "urine_output",
]


def _write_episode(directory: Path, episode_id: str, episode_type: str = "positive", rows: int = 13, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    anchor = pd.Timestamp("2136-01-15 09:40")
    if episode_type == "positive":
        window_end = anchor
    else:
        window_end = anchor - pd.Timedelta(hours=12)
    window_start = window_end - pd.Timedelta(hours=12)
    start = window_start.floor("60min")
    if rows < 13:
        start = window_end.floor("60min") - pd.Timedelta(hours=rows - 1)
        window_start = start
    values = rng.normal(size=(rows, len(FEATURES)))
    mask = rng.random((rows, len(FEATURES))) < 0.3
    mask[:, FEATURES.index("vasopressor_on")] = False
    meta = {
        "feature_names": FEATURES,
        "resample_minutes": 60,
        "start_time": start.isoformat(),
        "end_time": (start + pd.Timedelta(hours=rows - 1)).isoformat(),
        "shape": [rows, len(FEATURES)],
        "episode_id": episode_id,
        "episode_type": episode_type,
        "stay_id": 1,
        "subject_id": 2,
        "anchor_kind": "initiation",
        "anchor_time": anchor.isoformat(),
        "window_start": window_start.isoformat(),
        "window_end": window_end.isoformat(),
        "missing_mask_file": f"shock_episode_{episode_id}_missing_mask.npy",
        "intervention_features": ["ventilation_on", "vasopressor_on", "sedation_on"],
        "missingness_features": FEATURES[:7],
        "observation_density_features": FEATURES[:7],
    }
    paths = episode_paths(directory, episode_id)
    directory.mkdir(parents=True, exist_ok=True)
    np.save(paths["values"], values)
    np.save(paths["mask"], mask)
    paths["meta"].write_text(json.dumps(meta), encoding="utf-8")
    return meta


def test_materialize_episode_cuts_window_and_drops_anchor_feature(tmp_path):
    src, dst = tmp_path / "src", tmp_path / "dst"
    _write_episode(src, "stay_1_x_positive")
    meta = materialize_episode(src, dst, "stay_1_x_positive", TaskSpec())
    values = np.load(episode_paths(dst, "stay_1_x_positive")["values"])
    mask = np.load(episode_paths(dst, "stay_1_x_positive")["mask"])
    assert values.shape == mask.shape == (11, 17)
    assert "vasopressor_on" not in meta["feature_names"]
    assert meta["shape"] == [11, 17]
    assert meta["intervention_features"] == ["ventilation_on", "sedation_on"]
    # anchor 09:40, gap 1 h -> cutoff 08:40; last kept bin starts at 07:00
    assert pd.Timestamp(meta["end_time"]) == pd.Timestamp("2136-01-15 07:00")
    assert pd.Timestamp(meta["window_end"]) == pd.Timestamp("2136-01-15 08:40")
    assert meta["anchor_time"] == "2136-01-15T09:40:00"
    assert meta["v2"]["dropped_features"] == ["vasopressor_on"]
    assert meta["v2"]["task_spec_sha256"] == spec_sha256(TaskSpec())
    assert (meta["core_observed_channels"], meta["core_observed_points"]) == core_observation_counts(mask, meta["feature_names"])


def test_control_is_cut_relative_to_its_own_window_end(tmp_path):
    src, dst = tmp_path / "src", tmp_path / "dst"
    _write_episode(src, "stay_1_x_control", episode_type="control")
    meta = materialize_episode(src, dst, "stay_1_x_control", TaskSpec())
    assert meta["shape"][0] == 11
    assert pd.Timestamp(meta["window_end"]) == pd.Timestamp("2136-01-14 20:40")


def test_short_window_is_rejected(tmp_path):
    src, dst = tmp_path / "src", tmp_path / "dst"
    _write_episode(src, "stay_1_short_positive", rows=7)
    with pytest.raises(ValueError, match="full window"):
        materialize_episode(src, dst, "stay_1_short_positive", TaskSpec())


def test_refuses_source_directory_and_overwrite(tmp_path):
    src = tmp_path / "src"
    _write_episode(src, "stay_1_x_positive")
    with pytest.raises(ValueError, match="source directory"):
        materialize_episode(src, src, "stay_1_x_positive", TaskSpec())
    with pytest.raises(ValueError, match="source directory"):
        materialize_episode(src, src / "sub", "stay_1_x_positive", TaskSpec())
    dst = tmp_path / "dst"
    materialize_episode(src, dst, "stay_1_x_positive", TaskSpec())
    with pytest.raises(FileExistsError):
        materialize_episode(src, dst, "stay_1_x_positive", TaskSpec())
    materialize_episode(src, dst, "stay_1_x_positive", TaskSpec(), overwrite=True)


def test_materialize_cohort_writes_index_and_collects_failures(tmp_path):
    src, dst = tmp_path / "src", tmp_path / "dst"
    _write_episode(src, "stay_1_a_positive", seed=1)
    _write_episode(src, "stay_1_b_control", episode_type="control", seed=2)
    manifest = materialize_cohort(src, dst, ["stay_1_a_positive", "stay_1_b_control", "stay_missing_positive"], TaskSpec())
    assert (manifest["n_written"], manifest["n_failed"]) == (2, 1)
    assert manifest["failures"][0]["episode_id"] == "stay_missing_positive"
    index = json.loads((dst / "shock_episode_index.json").read_text(encoding="utf-8"))
    assert [row["episode_id"] for row in index] == ["stay_1_a_positive", "stay_1_b_control"]
    assert json.loads((dst / "materialization_manifest.json").read_text(encoding="utf-8"))["n_written"] == 2


def _import_v1_experiment():
    import importlib.util

    with warnings.catch_warnings():
        spec = importlib.util.spec_from_file_location("v1_run_mimic_experiment", ROOT / "scripts" / "run_mimic_experiment.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    return module


@pytest.mark.skipif(not (REAL_DIR / f"shock_episode_{REAL_ID}.json").exists(), reason="real v1 episode not available")
def test_real_episode_is_readable_by_v1_miner(tmp_path):
    dst = tmp_path / "dst"
    manifest = materialize_cohort(REAL_DIR, dst, [REAL_ID], TaskSpec())
    assert manifest["n_written"] == 1, manifest["failures"]
    rme = _import_v1_experiment()
    specs = rme._load_stay_specs(dst)
    assert [s["episode_id"] for s in specs] == [REAL_ID]
    meta = dict(specs[0])
    X = np.load(dst / f"shock_episode_{REAL_ID}.npy")
    meta["missing_mask"] = np.load(dst / meta["missing_mask_file"])
    assert X.shape == (11, 17)
    with warnings.catch_warnings():
        miner = rme.NonStationaryRuleMiner(**rme._default_miner_config())
        result = miner.fit((X, meta))
    assert "eventizer_name" in result
    labels = result["event_labels"]
    label_values = [str(v) for v in (labels.values() if isinstance(labels, dict) else labels)]
    assert label_values, "the v1 eventizer produced no events"
    assert not any(v.startswith("vasopressor_on") for v in label_values)
    assert result["all_rules"], "the v1 miner produced no rules on the materialized window"
