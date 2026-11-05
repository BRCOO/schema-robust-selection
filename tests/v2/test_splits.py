from __future__ import annotations

import json
import random
from dataclasses import replace

import pytest

from src.icu.v2.splits import SPLIT_NAMES, assert_patient_disjoint, patient_level_split, split_summary
from src.icu.v2.types import EpisodeRecord


def _patient(subject: int, n_pos: int, n_ctrl: int, stay: int | None = None, tag: str = "") -> list[EpisodeRecord]:
    stay_id = 50000 + subject if stay is None else stay
    labels = [1] * n_pos + [0] * n_ctrl
    return [
        EpisodeRecord(episode_id=f"s{subject}{tag}_e{j:03d}", subject_id=subject, stay_id=stay_id, label=label)
        for j, label in enumerate(labels)
    ]


def _v1_like_cohort() -> list[EpisodeRecord]:
    """31 patients and 474 episodes; patient 1000 has 144 episodes like the v1 validation patient."""
    episodes = _patient(1000, 40, 104)
    for i in range(30):
        n = 4 + (7 * i) % 15
        n_pos = 0 if i % 3 == 2 else min(n, 1 + i % 5)
        episodes += _patient(1001 + i, n_pos, n - n_pos)
    return episodes


def _cohort(n_patients: int = 120, seed: int = 0) -> list[EpisodeRecord]:
    """One third of the patients have positive episodes; sizes 1-12."""
    rng = random.Random(seed)
    episodes: list[EpisodeRecord] = []
    for i in range(n_patients):
        n = rng.randint(1, 12)
        n_pos = rng.randint(1, n) if i % 3 == 0 else 0
        episodes += _patient(2000 + i, n_pos, n - n_pos)
    return episodes


def _assignment(episodes: list[EpisodeRecord]) -> dict[str, str]:
    return {episode.episode_id: episode.split for episode in episodes}


def _has_both_classes(manifest: dict) -> bool:
    return all(manifest["splits"][name]["positives"] > 0 and manifest["splits"][name]["controls"] > 0 for name in SPLIT_NAMES)


def test_v1_failure_case_validation_is_not_one_patient_when_capped():
    episodes = _v1_like_cohort()
    assert len(episodes) == 474 and sum(e.subject_id == 1000 for e in episodes) == 144
    for seed in range(10):
        out, manifest = patient_level_split(episodes, seed=seed, max_episodes_per_patient=10)
        assert manifest["splits"]["val"]["patients"] >= 3
        assert abs(manifest["fractions_achieved"]["val"] - 0.1) < 0.05
        big = [e for e in out if e.subject_id == 1000]
        assert len(big) == 10 and {e.label for e in big} == {0, 1}
        assert _has_both_classes(manifest)
        assert manifest["n_patients"] == 31 and manifest["n_patients_capped"] > 0
        assert manifest["n_episodes"] + manifest["n_episodes_dropped_by_cap"] == 474
        assert_patient_disjoint(out)


def test_large_patient_does_not_swamp_validation_even_uncapped():
    for seed in range(10):
        _, manifest = patient_level_split(_v1_like_cohort(), seed=seed)
        assert manifest["splits"]["val"]["patients"] >= 3
        assert 1000 not in manifest["splits"]["val"]["patient_ids"]
        assert manifest["n_episodes_dropped_by_cap"] == 0


def test_split_groups_by_subject_not_stay():
    two_stays = _patient(9999, 2, 3, stay=1, tag="a") + _patient(9999, 1, 4, stay=2, tag="b")
    out, manifest = patient_level_split(_cohort() + two_stays, seed=3)
    assert len({e.split for e in out if e.subject_id == 9999}) == 1
    assert manifest["group_key"] == "subject_id"
    assert sum(9999 in manifest["splits"][name]["patient_ids"] for name in SPLIT_NAMES) == 1
    assert_patient_disjoint(out)


def test_split_is_deterministic_and_input_order_independent():
    episodes = _cohort()
    out, manifest = patient_level_split(episodes, seed=7, max_episodes_per_patient=6)
    again, manifest_again = patient_level_split(episodes, seed=7, max_episodes_per_patient=6)
    assert again == out and manifest_again == manifest

    shuffled = list(episodes)
    random.Random(1).shuffle(shuffled)
    out_shuffled, manifest_shuffled = patient_level_split(shuffled, seed=7, max_episodes_per_patient=6)
    assert _assignment(out_shuffled) == _assignment(out)
    assert manifest_shuffled == manifest

    other, _ = patient_level_split(episodes, seed=8, max_episodes_per_patient=6)
    assert _assignment(other) != _assignment(out)


def test_split_is_stratified_by_patient_positivity():
    out, manifest = patient_level_split(_cohort(n_patients=240), seed=0)
    positive_patients = {e.subject_id for e in out if e.label == 1}
    for has_positive in (True, False):
        stratum = [e for e in out if (e.subject_id in positive_patients) == has_positive]
        for name, fraction in manifest["fractions_requested"].items():
            achieved = sum(e.split == name for e in stratum) / len(stratum)
            assert abs(achieved - fraction) < 0.03, (has_positive, name, achieved)
    assert _has_both_classes(manifest)


def test_every_split_gets_both_classes_whenever_possible():
    feasible = infeasible = repaired = 0
    for k in range(150):
        rng = random.Random(k)
        episodes = [
            EpisodeRecord(episode_id=f"{p}_{j}", subject_id=p, stay_id=100 + p, label=int(rng.random() < 0.4))
            for p in range(rng.randint(3, 8))
            for j in range(rng.randint(1, 6))
        ]
        with_pos = len({e.subject_id for e in episodes if e.label == 1})
        with_ctrl = len({e.subject_id for e in episodes if e.label == 0})
        if with_pos >= 3 and with_ctrl >= 3:
            out, manifest = patient_level_split(episodes, seed=k)
            assert _has_both_classes(manifest)
            assert_patient_disjoint(out)
            feasible += 1
            repaired += manifest["n_class_repair_moves"] > 0
        else:
            with pytest.raises(ValueError, match="both classes"):
                patient_level_split(episodes, seed=k)
            infeasible += 1
    assert feasible > 0 and infeasible > 0 and repaired > 0  # every branch was exercised


def test_infeasible_split_raises_a_clear_error():
    episodes = _patient(1, 3, 3) + _patient(2, 2, 0) + _patient(3, 0, 5) + _patient(4, 0, 2)
    with pytest.raises(ValueError, match="need at least 3 patients with a positive episode"):
        patient_level_split(episodes)


def test_cap_keeps_one_of_each_class_and_is_deterministic():
    episodes = (
        _patient(1, 1, 30)  # both classes, capped to one positive and one control
        + _patient(2, 0, 20)  # controls only
        + _patient(3, 1, 1)  # under the cap: untouched
        + _patient(4, 2, 2)
        + _patient(5, 3, 0)
        + _patient(6, 0, 2)
    )
    out, manifest = patient_level_split(episodes, seed=0, max_episodes_per_patient=2)
    kept = {subject: [e for e in out if e.subject_id == subject] for subject in range(1, 7)}
    assert sorted(e.label for e in kept[1]) == [0, 1]
    assert sorted(e.label for e in kept[4]) == [0, 1]
    assert [e.label for e in kept[2]] == [0, 0]
    assert {e.episode_id for e in kept[3]} == {"s3_e000", "s3_e001"}
    assert manifest["max_episodes_per_patient"] == 2
    assert manifest["n_patients_capped"] == 4 and manifest["n_episodes_dropped_by_cap"] == 62 - 12

    shuffled = list(reversed(episodes))
    out_shuffled, _ = patient_level_split(shuffled, seed=0, max_episodes_per_patient=2)
    assert _assignment(out_shuffled) == _assignment(out)
    kept_controls = {
        next(e.episode_id for e in patient_level_split(episodes, seed=s, max_episodes_per_patient=2)[0] if e.subject_id == 1 and e.label == 0)
        for s in range(5)
    }
    assert len(kept_controls) > 1  # the seed, not the input order, picks which episodes stay


def test_returns_new_records_in_input_order():
    episodes = _cohort(n_patients=30)
    out, _ = patient_level_split(episodes, seed=2)
    assert [e.episode_id for e in out] == [e.episode_id for e in episodes]
    assert all(e.split == "" for e in episodes)
    assert all(new == replace(old, split=new.split) and new.split in SPLIT_NAMES for old, new in zip(episodes, out))


def test_argument_validation():
    episodes = _cohort(n_patients=30)
    for cap in (0, 1, 2.5, True):
        with pytest.raises(ValueError, match="max_episodes_per_patient"):
            patient_level_split(episodes, max_episodes_per_patient=cap)
    for fractions in ((0.8, 0.0, 0.2), (0.5, 0.2, 0.2), (0.7, 0.3), (1.2, -0.1, -0.1)):
        with pytest.raises(ValueError, match="fraction"):
            patient_level_split(episodes, fractions=fractions)
    with pytest.raises(ValueError, match="no episodes"):
        patient_level_split([])
    with pytest.raises(ValueError, match="duplicate"):
        patient_level_split(episodes + episodes[:1])
    with pytest.raises(ValueError, match="label"):
        patient_level_split(episodes + [EpisodeRecord("bad", 1, 1, label=2)])
    with pytest.raises(ValueError, match="stay_id"):
        patient_level_split(episodes + [replace(episodes[0], episode_id="moved", subject_id=123456)])


def test_manifest_reports_counts_fractions_and_sorted_patient_ids():
    episodes = _cohort()
    out, manifest = patient_level_split(episodes, fractions=(0.6, 0.2, 0.2), seed=4, max_episodes_per_patient=8)
    assert manifest["seed"] == 4
    assert manifest["fractions_requested"] == {"train": 0.6, "val": 0.2, "test": 0.2}
    assert sum(manifest["fractions_achieved"].values()) == pytest.approx(1.0)
    assert manifest["stratify_by"] == "patient_has_positive_episode"
    all_ids: list[int] = []
    for name in SPLIT_NAMES:
        row = manifest["splits"][name]
        assert row["patient_ids"] == sorted(row["patient_ids"])
        assert row["patients"] == len(row["patient_ids"])
        assert row["episodes"] == row["positives"] + row["controls"]
        assert row["episodes"] == sum(e.split == name for e in out)
        all_ids += row["patient_ids"]
    assert sorted(all_ids) == sorted({e.subject_id for e in episodes})
    assert sum(manifest["splits"][name]["episodes"] for name in SPLIT_NAMES) == manifest["n_episodes"] == len(out)
    json.dumps(manifest)


def test_assert_patient_disjoint_detects_leaks():
    out, _ = patient_level_split(_cohort(n_patients=30), seed=0)
    assert_patient_disjoint(out)
    first = out[0]
    other_split = next(name for name in SPLIT_NAMES if name != first.split)
    leaked = out + [replace(first, episode_id="leak", split=other_split)]
    with pytest.raises(ValueError, match="subject_id"):
        assert_patient_disjoint(leaked)
    with pytest.raises(ValueError, match="no split"):
        assert_patient_disjoint(out + [replace(first, episode_id="unassigned", split="")])
    stay_leak = [
        EpisodeRecord("a", subject_id=1, stay_id=7, label=1, split="train"),
        EpisodeRecord("b", subject_id=2, stay_id=7, label=0, split="test"),
    ]
    with pytest.raises(ValueError, match="stay_id"):
        assert_patient_disjoint(stay_leak)
    assert_patient_disjoint([])


def test_split_summary_counts_and_unassigned():
    episodes = [
        EpisodeRecord("a", subject_id=1, stay_id=11, label=1, split="train"),
        EpisodeRecord("b", subject_id=1, stay_id=12, label=0, split="train"),
        EpisodeRecord("c", subject_id=2, stay_id=21, label=0, split="test"),
        EpisodeRecord("d", subject_id=3, stay_id=31, label=1),
    ]
    summary = split_summary(episodes)
    assert summary["n_episodes"] == 4 and summary["n_patients"] == 3
    train = summary["splits"]["train"]
    assert (train["episodes"], train["positives"], train["controls"], train["patients"], train["stays"]) == (2, 1, 1, 1, 2)
    assert train["patients_with_positive"] == 1 and train["episode_fraction"] == 0.5
    assert summary["splits"]["val"] == {
        "episodes": 0, "positives": 0, "controls": 0, "patients": 0,
        "patients_with_positive": 0, "stays": 0, "episode_fraction": 0.0,
    }
    assert summary["splits"]["unassigned"]["positives"] == 1
    empty = split_summary([])
    assert empty["n_episodes"] == 0 and list(empty["splits"]) == list(SPLIT_NAMES)
