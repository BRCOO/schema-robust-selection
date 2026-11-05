from __future__ import annotations

import importlib.util
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from src.icu.v2.materialize import episode_paths
from src.icu.v2.splits import patient_level_split
from src.icu.v2.types import EpisodeRecord

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "v2" / "run_cohort_leakage_audit.py"
FEATURES = ["HR", "MAP", "Lactate", "RR"]


def _load_script():
    spec = importlib.util.spec_from_file_location("run_cohort_leakage_audit", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _cohort(tmp_path: Path, *, leak: bool) -> tuple[Path, Path]:
    rng = np.random.default_rng(0)
    raw = [EpisodeRecord(f"stay_{i}_{label}", 500 + i, i, label) for i in range(40) for label in (1, 0)]
    episodes, _ = patient_level_split(raw, fractions=(0.6, 0.2, 0.2), seed=2)
    episode_dir = tmp_path / ("leaky" if leak else "clean")
    episode_dir.mkdir()
    for e in episodes:
        values = rng.normal(size=(11, len(FEATURES)))
        if leak:
            values[-1, 2] = 5.0 if e.label else -5.0  # last Lactate encodes the label
        mask = rng.random(values.shape) < 0.15
        mask[-1, 2] = False
        paths = episode_paths(episode_dir, e.episode_id)
        np.save(paths["values"], values)
        np.save(paths["mask"], mask)
        meta = {
            "episode_id": e.episode_id,
            "episode_type": "positive" if e.label else "control",
            "feature_names": FEATURES,
            "shape": list(values.shape),
            "missing_mask_file": paths["mask"].name,
        }
        paths["meta"].write_text(json.dumps(meta), encoding="utf-8")
    episodes_path = tmp_path / f"episodes_{'leaky' if leak else 'clean'}.json"
    episodes_path.write_text(json.dumps([asdict(e) for e in episodes]), encoding="utf-8")
    return episodes_path, episode_dir


def test_gate_flags_planted_leak(tmp_path):
    module = _load_script()
    episodes_path, episode_dir = _cohort(tmp_path, leak=True)
    out = tmp_path / "leaky_audit.json"
    status = module.main(["--episodes", str(episodes_path), "--episode-dir", str(episode_dir), "--out-json", str(out), "--n-bootstrap", "50"])
    assert status == 2
    result = json.loads(out.read_text(encoding="utf-8"))
    assert result["flagged"][0]["feature"] == "Lactate"
    assert set(result["multivariate"]) == {"all", "observation_only", "values_only"}
    assert all("error" not in result["multivariate"][view] for view in result["multivariate"])


def test_gate_passes_clean_cohort(tmp_path):
    module = _load_script()
    episodes_path, episode_dir = _cohort(tmp_path, leak=False)
    out = tmp_path / "clean_audit.json"
    status = module.main(["--episodes", str(episodes_path), "--episode-dir", str(episode_dir), "--out-json", str(out), "--n-bootstrap", "50"])
    assert status == 0
    assert json.loads(out.read_text(encoding="utf-8"))["n_episodes"] == 80
