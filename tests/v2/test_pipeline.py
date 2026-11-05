from __future__ import annotations

import math

import numpy as np
import pytest

from src.icu.v2.pipeline import (
    MethodPool,
    TargetCohort,
    compare_methods,
    episodes_by_split,
    evaluate_registry,
    native_registry,
)
from src.icu.v2.selector import SelectorConfig
from src.icu.v2.splits import patient_level_split
from src.icu.v2.types import EpisodeRecord, make_symbolic_program

FEATURES = ["HR", "MAP", "Lactate", "Creatinine", "RR", "SpO2", "Glucose", "WBC"]


def _episodes(n_patients: int = 60, prefix: str = "", seed: int = 0) -> list[EpisodeRecord]:
    rows = []
    for i in range(n_patients):
        for label in (1, 0):
            rows.append(EpisodeRecord(episode_id=f"{prefix}e{i}_{label}", subject_id=1000 * (seed + 1) + i, stay_id=i, label=label))
    return rows


def _pool(method: str, episodes, *, signal: float, n_programs: int = 12, seed: int = 0, external=None) -> MethodPool:
    rng = np.random.default_rng(seed)
    candidates, firings = [], {}
    external_firings: dict[str, dict[str, frozenset[str]]] = {}
    for j in range(n_programs):
        a, b = FEATURES[j % len(FEATURES)], FEATURES[(j + 3) % len(FEATURES)]
        program = make_symbolic_program(
            f"{method}_{j:03d}", method, "sequence", ({f"{a}:q1"}, {f"{b}:q3"}), score=float(n_programs - j)
        )
        candidates.append(program)
        p_pos, p_neg = 0.2 + signal, 0.2

        def fire(eps):
            return frozenset(e.episode_id for e in eps if rng.random() < (p_pos if e.label == 1 else p_neg))

        firings[program.program_id] = fire(episodes)
        for target in external or ():
            external_firings.setdefault(target.name, {})[program.program_id] = fire(target.episodes)
    return MethodPool(method=method, candidates=candidates, firings=firings, external_firings=external_firings)


@pytest.fixture(scope="module")
def setup():
    episodes, _ = patient_level_split(_episodes(), fractions=(0.6, 0.2, 0.2), seed=1)
    target = TargetCohort(
        name="ext",
        features=frozenset(f for f in FEATURES if f != "Lactate"),
        episodes=tuple(_episodes(25, prefix="x", seed=7)),
    )
    strong = _pool("strong", episodes, signal=0.45, seed=1, external=[target])
    noise = _pool("noise", episodes, signal=0.0, seed=2, external=[target])
    return episodes, target, strong, noise


def test_episodes_by_split_requires_all_splits():
    rows = [EpisodeRecord("a", 1, 1, 1, split="train"), EpisodeRecord("b", 2, 2, 0, split="test")]
    with pytest.raises(ValueError, match="empty splits"):
        episodes_by_split(rows)
    with pytest.raises(ValueError, match="split"):
        episodes_by_split([EpisodeRecord("a", 1, 1, 1, split="dev")])


def test_native_registry_dedups_and_truncates(setup):
    _, _, strong, _ = setup
    duplicate = make_symbolic_program("dup", "strong", "sequence", strong.candidates[0].elements, score=99.0)
    pool = MethodPool("strong", [duplicate, *strong.candidates], strong.firings)
    registry = native_registry(pool, 5)
    assert len(registry) == 5
    assert len({p.canonical_key() for p in registry}) == 5


def test_evaluate_registry_reports_internal_external_and_structure(setup):
    episodes, target, strong, _ = setup
    result = evaluate_registry(strong.candidates[:10], strong.firings, episodes, targets=[target], external_firings=strong.external_firings, n_boot=200)
    internal = result["internal"]
    assert internal["auprc_low"] <= internal["auprc"] <= internal["auprc_high"]
    assert internal["auroc"] > 0.6  # planted signal
    assert result["bridge"]["coordinate_manifest"]["n_coordinates"] == 10
    survival = result["structure"]["schema_survival"]["ext"]["value"]
    lactate = sum("Lactate" in p.features for p in strong.candidates[:10])
    assert math.isclose(survival, (10 - lactate) / 10)
    ext = result["external"]["ext"]
    assert ext["schema_survival"] == survival and ext["suppressed_firings"]  # firings reported for Lactate programs were zeroed
    assert ext["n_patients"] == 25


def test_evaluate_registry_returns_error_for_empty_registry(setup):
    episodes, _, strong, _ = setup
    assert evaluate_registry([], strong.firings, episodes)["error"] == "empty registry"


def test_compare_methods_rows_and_paired_differences(setup):
    episodes, target, strong, noise = setup
    out = compare_methods(
        [strong, noise],
        episodes,
        targets=[target],
        selector_config=SelectorConfig(k=6),
        k=6,
        reference=("strong", "native"),
        n_boot=200,
    )
    keys = {(row["method"], row["selection"]) for row in out["rows"]}
    assert keys == {("strong", "native"), ("strong", "v2_selector"), ("noise", "native"), ("noise", "v2_selector")}
    for row in out["rows"]:
        assert "error" not in row, row
        assert row["n_programs"] <= 6
        assert 0.0 <= row["redundancy"] <= 1.0
        assert "ext_auprc" in row and "ext_schema_survival" in row
    assert len(out["paired_vs_reference"]) == 3
    noise_vs_strong = next(d for d in out["paired_vs_reference"] if d["comparison"].startswith("noise|native"))
    assert noise_vs_strong["point_diff"] < 0
    assert out["details"]["strong|v2_selector"]["selector_trace"]


def test_compare_methods_rejects_mismatched_k_and_unknown_selection(setup):
    episodes, _, strong, _ = setup
    with pytest.raises(ValueError, match="differs"):
        compare_methods([strong], episodes, selector_config=SelectorConfig(k=5), k=10, n_boot=50)
    with pytest.raises(ValueError, match="unknown selections"):
        compare_methods([strong], episodes, selections=("oracle",), n_boot=50)
