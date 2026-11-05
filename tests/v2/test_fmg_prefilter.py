from __future__ import annotations

import json

import pytest

from src.icu.v2.fmg_prefilter import count_rule_support, iter_filtered_inputs, kept_rule_ids, prefilter_thresholds


def _write(tmp_path, items):
    path = tmp_path / "inputs.json"
    path.write_text(json.dumps(items), encoding="utf-8")
    return path


ITEMS = [
    {"episode_id": "p1", "episode_type": "positive", "rules": [{"rule_id": "a"}, {"rule_id": "b"}, {"rule_id": "a"}]},
    {"episode_id": "p2", "episode_type": "positive", "rules": [{"rule_id": "a"}, {"rule_id": "c"}]},
    {"episode_id": "c1", "episode_type": "control", "rules": [{"rule_id": "a"}, {"rule_id": "c"}]},
    {"episode_id": "c2", "episode_type": "control", "rules": [{"rule_id": "b"}]},
    {"episode_id": "t1", "episode_type": "positive", "rules": [{"rule_id": "b"}, {"rule_id": "c"}]},
]
SPLIT = {"p1": "train", "p2": "train", "c1": "train", "c2": "train", "t1": "test"}


def test_counts_unique_rules_per_episode_on_the_split(tmp_path):
    path = _write(tmp_path, ITEMS)
    positives, controls, episodes = count_rule_support(path, SPLIT, "train")
    assert positives == {"a": 2, "b": 1, "c": 1}  # duplicate "a" in p1 counted once; t1 (test) ignored
    assert controls == {"a": 1, "c": 1, "b": 1}
    assert episodes == {"positive": 2, "control": 2}
    all_pos, _, _ = count_rule_support(path)
    assert all_pos["b"] == 2  # t1 counted when no split filter is given


def test_kept_rules_and_filtered_inputs_keep_every_episode(tmp_path):
    path = _write(tmp_path, ITEMS)
    positives, controls, _ = count_rule_support(path, SPLIT, "train")
    keep = kept_rule_ids(positives, controls, 2, 1)
    assert keep == frozenset({"a"})
    out = list(iter_filtered_inputs(path, keep, SPLIT, "train"))
    assert [e["episode_id"] for e in out] == ["p1", "p2", "c1", "c2"]
    assert out[3]["rules"] == []  # c2 kept with no rules: totals count episodes
    assert all(r["rule_id"] == "a" for e in out for r in e["rules"])


def test_thresholds_refuse_non_exact_settings():
    assert prefilter_thresholds({"min_rule_positive_count": 2, "min_rule_control_count": 1}) == (2, 1)
    with pytest.raises(ValueError, match="zero_control"):
        prefilter_thresholds({"allow_zero_control_candidates": True})
    with pytest.raises(ValueError, match="recovery"):
        prefilter_thresholds({"rule_low_coverage_recovery_enabled": True})


def test_unknown_episode_type_raises(tmp_path):
    path = _write(tmp_path, [{"episode_id": "x", "episode_type": "other", "rules": []}])
    with pytest.raises(ValueError, match="episode_type"):
        count_rule_support(path)
