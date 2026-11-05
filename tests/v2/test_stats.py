from __future__ import annotations

import math

import numpy as np
import pytest

from src.icu.v2.stats import (
    METRICS,
    episode_bootstrap,
    metric_value,
    paired_cluster_bootstrap_diff,
    patient_cluster_bootstrap,
)


def _independent(n: int = 300, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < 0.3).astype(int)
    return y, y + rng.normal(size=n)


def _clustered(n_groups: int = 40, per_group: int = 15, seed: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Large patients whose episodes share a prevalence and a score shift."""
    rng = np.random.default_rng(seed)
    y, scores, groups = [], [], []
    for gid in range(n_groups):
        prevalence = 0.6 if rng.random() < 0.4 else 0.1
        shift = rng.normal()
        for _ in range(per_group):
            label = int(rng.random() < prevalence)
            y.append(label)
            scores.append(label + 1.5 * shift + 0.5 * rng.normal())
            groups.append(1000 + gid)
    return np.array(y), np.array(scores), np.array(groups)


def test_metric_value_matches_known_values():
    y, scores = [0, 0, 1, 1], [0.1, 0.4, 0.35, 0.8]
    assert metric_value(y, scores, "auroc") == pytest.approx(0.75)
    assert metric_value(y, scores, "auprc") == pytest.approx(0.8333333)
    assert metric_value(y, [0.0, 0.1, 0.8, 0.9]) == 1.0
    assert METRICS == ("auprc", "auroc")


def test_metric_value_rejects_bad_input():
    with pytest.raises(ValueError, match="both classes"):
        metric_value([1, 1, 1], [0.2, 0.5, 0.9])
    with pytest.raises(ValueError, match="metric"):
        metric_value([0, 1], [0.1, 0.9], "f1")
    with pytest.raises(ValueError):
        metric_value([0, 1, 1], [0.1, 0.9])
    with pytest.raises(ValueError):
        metric_value([0, 2], [0.1, 0.9])
    with pytest.raises(ValueError):
        metric_value([0, 1], [0.1, float("nan")])
    with pytest.raises(ValueError):
        metric_value([], [])


def test_cluster_bootstrap_reports_point_interval_and_counts():
    y, scores, groups = _clustered()
    out = patient_cluster_bootstrap(y, scores, groups, n_boot=300, seed=0)
    assert out["point"] == metric_value(y, scores, "auprc")
    assert out["low"] <= out["point"] <= out["high"]
    assert out["n_boot_used"] + out["n_skipped_single_class"] == 300
    assert out["n_groups"] == 40 and out["n_episodes"] == 600
    assert out["resample_unit"] == "cluster" and out["metric"] == "auprc"


def test_singleton_groups_agree_with_episode_bootstrap():
    y, scores = _independent()
    shuffled_ids = np.random.default_rng(5).permutation(len(y)) + 5000  # one patient per episode
    for metric in METRICS:
        cluster = patient_cluster_bootstrap(y, scores, shuffled_ids, metric=metric, n_boot=1000, seed=1)
        episode = episode_bootstrap(y, scores, metric=metric, n_boot=1000, seed=1)
        assert cluster["point"] == episode["point"]
        assert cluster["n_groups"] == episode["n_groups"] == len(y)
        assert abs(cluster["low"] - episode["low"]) < 0.02
        assert abs(cluster["high"] - episode["high"]) < 0.02
    # With groups in episode order the draws coincide exactly.
    in_order = patient_cluster_bootstrap(y, scores, np.arange(len(y)), n_boot=200, seed=3)
    v1_unit = episode_bootstrap(y, scores, n_boot=200, seed=3)
    assert (in_order["low"], in_order["high"]) == (v1_unit["low"], v1_unit["high"])
    assert v1_unit["resample_unit"] == "episode"


def test_correlated_groups_widen_the_interval():
    y, scores, groups = _clustered()
    for metric in METRICS:
        cluster = patient_cluster_bootstrap(y, scores, groups, metric=metric, n_boot=400, seed=0)
        episode = episode_bootstrap(y, scores, metric=metric, n_boot=400, seed=0)
        assert cluster["point"] == episode["point"]
        assert (cluster["high"] - cluster["low"]) > 1.3 * (episode["high"] - episode["low"])


def test_bootstrap_is_deterministic_for_a_fixed_seed():
    y, scores, groups = _clustered(n_groups=20, per_group=5)
    first = patient_cluster_bootstrap(y, scores, groups, n_boot=200, seed=11)
    assert patient_cluster_bootstrap(y, scores, groups, n_boot=200, seed=11) == first
    other = patient_cluster_bootstrap(y, scores, groups, n_boot=200, seed=12)
    assert (other["low"], other["high"]) != (first["low"], first["high"])
    # Row order does not matter: groups are coded by sorted id.
    perm = np.random.default_rng(0).permutation(len(y))
    shuffled = patient_cluster_bootstrap(y[perm], scores[perm], groups[perm], n_boot=200, seed=11)
    assert shuffled["low"] == pytest.approx(first["low"]) and shuffled["high"] == pytest.approx(first["high"])
    assert episode_bootstrap(y, scores, n_boot=100, seed=4) == episode_bootstrap(y, scores, n_boot=100, seed=4)


def test_single_group_gives_degenerate_interval():
    y, scores = _independent(n=50)
    out = patient_cluster_bootstrap(y, scores, [7] * 50, n_boot=50, seed=0)
    assert out["n_groups"] == 1 and out["n_skipped_single_class"] == 0
    assert out["low"] == pytest.approx(out["point"]) and out["high"] == pytest.approx(out["point"])


def test_single_class_resamples_are_skipped_and_counted():
    y = [1, 1, 1, 0, 0, 0]
    scores = [0.9, 0.8, 0.7, 0.3, 0.2, 0.1]
    groups = ["a", "a", "a", "b", "b", "b"]  # each patient has one class
    out = patient_cluster_bootstrap(y, scores, groups, n_boot=200, seed=0)
    assert 0 < out["n_skipped_single_class"] < 200
    assert out["n_boot_used"] + out["n_skipped_single_class"] == 200
    assert out["low"] == out["high"] == 1.0
    # When no resample is usable the bounds are NaN instead of a fake interval.
    runs = [patient_cluster_bootstrap(y, scores, groups, n_boot=1, seed=seed) for seed in range(20)]
    empty = [run for run in runs if run["n_boot_used"] == 0]
    assert empty
    assert all(math.isnan(run["low"]) and math.isnan(run["high"]) for run in empty)


def test_bootstrap_argument_validation():
    y, scores, groups = _clustered(n_groups=6, per_group=4)
    with pytest.raises(ValueError):
        patient_cluster_bootstrap(y, scores, groups, n_boot=0)
    with pytest.raises(ValueError):
        patient_cluster_bootstrap(y, scores, groups, ci=1.0)
    with pytest.raises(ValueError):
        patient_cluster_bootstrap(y, scores, groups[:-1])
    with pytest.raises(ValueError):
        patient_cluster_bootstrap(y, scores, groups, metric="brier")
    with pytest.raises(ValueError):
        episode_bootstrap(y, scores, ci=0.0)
    with pytest.raises(ValueError):
        paired_cluster_bootstrap_diff(y, scores, scores[:-1], groups)


def test_paired_diff_of_identical_scores_is_zero():
    y, scores, groups = _clustered(n_groups=20, per_group=5)
    out = paired_cluster_bootstrap_diff(y, scores, scores, groups, n_boot=200, seed=0)
    assert out["point_diff"] == 0.0
    assert out["low"] == out["high"] == 0.0
    assert out["frac_le_zero"] == 1.0
    assert out["n_boot_used"] + out["n_skipped_single_class"] == 200


def test_paired_diff_detects_a_better_scorer():
    rng = np.random.default_rng(2)
    groups = np.repeat(np.arange(60), 5)
    y = (rng.random(groups.size) < 0.3).astype(int)
    informative = y + 0.5 * rng.normal(size=y.size)
    noise = rng.normal(size=y.size)
    for metric in METRICS:
        out = paired_cluster_bootstrap_diff(y, informative, noise, groups, metric=metric, n_boot=300, seed=0)
        assert out["point_diff"] == pytest.approx(out["point_a"] - out["point_b"])
        assert out["point_diff"] > 0.0 and out["low"] > 0.0
        assert out["frac_le_zero"] < 0.05
        assert out["n_groups"] == 60


def test_paired_diff_is_antisymmetric_and_deterministic():
    y, scores, groups = _clustered(n_groups=30, per_group=6, seed=3)
    other = scores + np.random.default_rng(9).normal(size=scores.size)
    ab = paired_cluster_bootstrap_diff(y, scores, other, groups, metric="auroc", n_boot=300, seed=5)
    ba = paired_cluster_bootstrap_diff(y, other, scores, groups, metric="auroc", n_boot=300, seed=5)
    assert ab["point_diff"] == pytest.approx(-ba["point_diff"])
    assert ab["low"] == pytest.approx(-ba["high"]) and ab["high"] == pytest.approx(-ba["low"])
    assert paired_cluster_bootstrap_diff(y, scores, other, groups, metric="auroc", n_boot=300, seed=5) == ab
