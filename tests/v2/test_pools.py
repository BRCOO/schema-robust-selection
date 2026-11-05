from __future__ import annotations

import json
import shutil
import subprocess

import numpy as np
import pytest

from src.icu.v2.baselines.spmf import DEFAULT_SPMF_JAR
from src.icu.v2.materialize import episode_paths
from src.icu.v2.pipeline import TargetCohort, compare_methods
from src.icu.v2.pools import (
    build_fmg_pool,
    build_numeric_pool,
    build_sequences,
    build_symbolic_pool,
    labels_and_groups,
    load_arrays,
)
from src.icu.v2.selector import SelectorConfig
from src.icu.v2.splits import patient_level_split
from src.icu.v2.types import EpisodeRecord

FEATURES = ("HR", "MAP", "Lactate", "RR")
ROWS = 10
RULE_A = "{Creatinine:sparse_measurement} => {RR:level_0}"
RULE_B = "{Lactate:up} => {HR:level_3}"


def _java_ready() -> bool:
    if shutil.which("java") is None or not DEFAULT_SPMF_JAR.exists():
        return False
    try:
        subprocess.run(["java", "-version"], capture_output=True, timeout=30, check=True)
    except (OSError, subprocess.SubprocessError):
        return False
    return True


def _arrays(episodes, seed=0):
    rng = np.random.default_rng(seed)
    arrays = {}
    for e in episodes:
        values = rng.normal(size=(ROWS, len(FEATURES)))
        if e.label == 1:
            values[-4:, 0] = [0.0, 1.0, 2.0, 3.0]  # rising HR shape before the cutoff
            values[:, 2] += 1.5  # higher Lactate
        mask = rng.random((ROWS, len(FEATURES))) < 0.15
        mask[:, 0] = False
        arrays[e.episode_id] = (values, mask, FEATURES)
    return arrays


def _episodes(n_patients=40, prefix=""):
    return [
        EpisodeRecord(f"{prefix}e{i}_{label}", 10_000 + i if not prefix else 90_000 + i, i, label)
        for i in range(n_patients)
        for label in (1, 0)
    ]


@pytest.fixture(scope="module")
def cohort():
    episodes, _ = patient_level_split(_episodes(), fractions=(0.6, 0.2, 0.2), seed=4)
    arrays = _arrays(episodes)
    labels, groups = labels_and_groups(episodes)
    train_ids = [e.episode_id for e in episodes if e.split == "train"]
    target_episodes = tuple(_episodes(15, prefix="x"))
    target_arrays = _arrays(target_episodes, seed=9)
    return episodes, arrays, labels, groups, train_ids, target_episodes, target_arrays


def test_load_arrays_reads_materialized_layout(tmp_path):
    names = list(FEATURES)
    paths = episode_paths(tmp_path, "stay_1_a_positive")
    np.save(paths["values"], np.zeros((3, 4)))
    np.save(paths["mask"], np.zeros((3, 4), dtype=bool))
    paths["meta"].write_text(json.dumps({"episode_id": "stay_1_a_positive", "feature_names": names}), encoding="utf-8")
    arrays = load_arrays(tmp_path, ["stay_1_a_positive", "stay_1_a_positive"])
    assert list(arrays) == ["stay_1_a_positive"] and arrays["stay_1_a_positive"][2] == tuple(names)


def test_build_sequences_fits_bins_on_train_only(cohort):
    episodes, arrays, _, _, train_ids, _, target_arrays = cohort
    seqs, external, bins = build_sequences(arrays, train_ids, extra={"ext": target_arrays})
    assert set(seqs) == set(arrays) and set(external["ext"]) == set(target_arrays)
    shifted = dict(arrays)
    test_id = next(e.episode_id for e in episodes if e.split == "test")
    values, mask, names = shifted[test_id]
    shifted[test_id] = (values + 100.0, mask, names)
    _, _, bins_shifted = build_sequences(shifted, train_ids)
    assert all(np.array_equal(bins[f], bins_shifted[f]) for f in bins)


def test_build_sequences_rejects_feature_order_mismatch(cohort):
    _, arrays, _, _, train_ids, _, _ = cohort
    broken = dict(arrays)
    first = next(iter(broken))
    values, mask, names = broken[first]
    broken[first] = (values, mask, tuple(reversed(names)))
    with pytest.raises(ValueError, match="feature order"):
        build_sequences(broken, train_ids)


@pytest.mark.skipif(not _java_ready(), reason="java or the SPMF jar is unavailable")
def test_symbolic_pool_with_prefixspan(cohort):
    _, arrays, labels, _, train_ids, _, target_arrays = cohort
    seqs, external, _ = build_sequences(arrays, train_ids, extra={"ext": target_arrays})
    pool = build_symbolic_pool(
        "prefixspan", seqs, labels, train_ids,
        mining_params={"minsup": 0.4, "max_pattern_length": 2}, pool_cap=30, external_sequences=external,
    )
    assert 0 < len(pool.candidates) <= 30
    assert all(set(ids) <= set(seqs) for ids in pool.firings.values())
    assert set(pool.external_firings["ext"]) == {p.program_id for p in pool.candidates}
    scores = [p.score for p in pool.candidates]
    assert scores == sorted(scores, reverse=True)


def test_numeric_pools_and_unknown_method(cohort):
    _, arrays, labels, groups, train_ids, _, target_arrays = cohort
    motif = build_numeric_pool(
        "matrix_profile", arrays, labels, train_ids, groups=groups,
        mining_params={"lengths": (3,), "top_k_per_feature": 2, "feature_names": ["HR"]},
        external_arrays={"ext": target_arrays},
    )
    shapelet = build_numeric_pool(
        "shapelet", arrays, labels, train_ids,
        mining_params={"lengths": (4,), "top_k_per_feature": 2, "feature_names": ["HR", "Lactate"]},
    )
    assert motif.candidates and shapelet.candidates
    assert all(p.kind == "motif" for p in motif.candidates)
    assert set(motif.external_firings["ext"]) == {p.program_id for p in motif.candidates}
    with pytest.raises(ValueError, match="unknown"):
        build_numeric_pool("dtw", arrays, labels, train_ids)


def _write_fmg(tmp_path, episodes):
    ranked = [
        {"program_id": "program_001", "representative_rule_id": RULE_A, "member_rule_ids": [RULE_A],
         "antecedent_features": ["Creatinine"], "consequent_features": ["RR"], "representative_score": 3.0},
        {"program_id": "program_002", "representative_rule_id": RULE_B, "member_rule_ids": [RULE_B],
         "antecedent_features": ["Lactate"], "consequent_features": ["HR"], "representative_score": 2.0},
    ]
    fmg_dir = tmp_path / "fmg"
    fmg_dir.mkdir()
    (fmg_dir / "fmg_programs_ranked.json").write_text(json.dumps(ranked), encoding="utf-8")
    (fmg_dir / "fmg_programs_selected.json").write_text(json.dumps(ranked[:1]), encoding="utf-8")
    rng = np.random.default_rng(3)
    inputs = []
    for e in episodes:
        rules = []
        if rng.random() < (0.7 if e.label else 0.2):
            rules.append({"rule_id": RULE_A})
        if rng.random() < 0.4:
            rules.append({"rule_id": RULE_B})
        inputs.append({"episode_id": e.episode_id, "episode_type": "positive" if e.label else "control", "rules": rules})
    path = tmp_path / "contrastive_inputs.json"
    path.write_text(json.dumps(inputs), encoding="utf-8")
    return fmg_dir, path


def test_fmg_pool_and_end_to_end_comparison(cohort, tmp_path):
    episodes, arrays, labels, groups, train_ids, target_episodes, target_arrays = cohort
    fmg_dir, inputs = _write_fmg(tmp_path, episodes)
    (tmp_path / "ext").mkdir()
    _, ext_inputs = _write_fmg(tmp_path / "ext", target_episodes)
    fmg = build_fmg_pool(fmg_dir, inputs, external_inputs={"ext": ext_inputs})
    assert [p.program_id for p in fmg.native_registry] == ["program_001"]
    assert len(fmg.candidates) == 2 and fmg.firings["program_001"]
    shapelet = build_numeric_pool(
        "shapelet", arrays, labels, train_ids,
        mining_params={"lengths": (4,), "top_k_per_feature": 2, "feature_names": ["HR", "Lactate"]},
        external_arrays={"ext": target_arrays},
    )
    target = TargetCohort("ext", frozenset({"HR", "Lactate", "RR", "Creatinine"}), target_episodes)
    out = compare_methods(
        [fmg, shapelet], episodes, targets=[target], selector_config=SelectorConfig(k=3), k=3,
        reference=("fmg_bridge", "native"), n_boot=100,
    )
    assert {(r["method"], r["selection"]) for r in out["rows"]} == {
        ("fmg_bridge", "native"), ("fmg_bridge", "v2_selector"), ("shapelet", "native"), ("shapelet", "v2_selector")
    }
    assert all("error" not in r for r in out["rows"]), out["rows"]
    assert len(out["paired_vs_reference"]) == 3
