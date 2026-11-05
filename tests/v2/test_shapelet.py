"""Tests for the exact information-gain shapelet baseline (v2) and shapelet firing."""

from __future__ import annotations

import math
from dataclasses import replace
from itertools import combinations

import numpy as np
import pytest

from src.icu.v2.baselines.motif import znorm
from src.icu.v2.baselines.shapelet import (
    best_split,
    discover_shapelets,
    mine_shapelet_programs,
    subsequence_distance,
)
from src.icu.v2.numeric_firing import fires_numeric, numeric_firing_sets, program_distance
from src.icu.v2.types import ProgramRecord, coarse_group, make_symbolic_program

SPIKE = np.array([0.0, 1.0, 0.0])
RAMP_UP = np.array([0.0, 1.0, 2.0])
RAMP_DOWN = np.array([5.0, 4.0, 3.0])


def _h(p: float) -> float:
    return 0.0 if p in (0.0, 1.0) else -p * math.log2(p) - (1.0 - p) * math.log2(1.0 - p)


IG_ONE_OF_FOUR = 1.0 - 0.75 * _h(1.0 / 3.0)  # one pure series split off a balanced set of four


def _episode(columns: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray, list[str]]:
    names = list(columns)
    values = np.column_stack([np.asarray(columns[name], dtype=float) for name in names])
    return values, np.zeros(values.shape, dtype=bool), names


def _hand_series() -> tuple[list[np.ndarray], list[int]]:
    """Two spikes (positives) and two ramps (controls), one window each.

    Length-normalized z-distances: spike-ramp = sqrt(2) (rho = 0), ramp-up-ramp-down = 2.
    """
    return [SPIKE, 5.0 * SPIKE + 10.0, RAMP_UP, RAMP_DOWN], [1, 1, 0, 0]


def _reference_split(d: list[float], y: list[int]) -> tuple[float, float, float]:
    """Plain-Python exhaustive split search with the documented rules."""
    n, n_pos = len(d), sum(y)
    order = sorted(range(n), key=lambda i: (d[i], i))
    ds, ys = [d[i] for i in order], [y[i] for i in order]
    n_finite = sum(math.isfinite(v) for v in ds)
    splits = []
    for k in range(1, n_finite):
        if not ds[k] - ds[k - 1] > 1e-8:
            continue
        near, far = ys[:k], ys[k:]
        gain = _h(n_pos / n) - k / n * _h(sum(near) / k) - (n - k) / n * _h(sum(far) / (n - k))
        far_finite = [v for v in ds[k:] if math.isfinite(v)]
        margin = sum(far_finite) / len(far_finite) - sum(ds[:k]) / k
        splits.append(((ds[k - 1] + ds[k]) / 2.0, max(gain, 0.0), margin))
    if not splits:
        return math.nan, 0.0, 0.0
    best_gain = max(s[1] for s in splits)
    tied = [s for s in splits if s[1] >= best_gain - 1e-9]
    best_margin = max(s[2] for s in tied)
    return min((s for s in tied if s[2] >= best_margin - 1e-9), key=lambda s: s[0])


def _shapelet_episodes(seed: int = 0) -> tuple[dict, dict]:
    """4 positives (noisy ramp + planted spike) and 4 controls (noisy ramps).

    Lactate is an all-zero channel and must yield no shapelet.
    """
    rng = np.random.default_rng(seed)
    episodes: dict[str, tuple] = {}
    labels: dict[str, int] = {}
    for k in range(4):
        for label, prefix in ((1, "pos"), (0, "ctl")):
            hr = np.linspace(0.0, 3.0, 10) + rng.normal(scale=0.05, size=10)
            if label:
                start = 2 + k
                hr[start : start + 3] = hr[start] + np.array([0.0, 4.0, 0.0])
            episodes[f"{prefix}{k}"] = _episode({"HR": hr, "Lactate": np.zeros(10)})
            labels[f"{prefix}{k}"] = label
    return episodes, labels


def _shapelet_program(direction: str, threshold: float = 0.7) -> ProgramRecord:
    return ProgramRecord(
        program_id=f"shapelet:HR:{direction}",
        method="shapelet",
        kind="shapelet",
        features=frozenset({"HR"}),
        anchor_signature=frozenset({coarse_group("HR")}),
        score=0.5,
        params={"feature": "HR", "template": tuple(znorm(RAMP_UP)), "threshold": threshold, "length": 3,
                "normalize": True, "direction": direction},
    )


# ---------------------------------------------------------------------------- distance


def test_subsequence_distance_is_the_min_length_normalized_window_distance():
    series = np.array([3.0, 0.0, 1.0, 0.0, 5.0, 6.0, 7.0])
    assert subsequence_distance(series, [10.0, 20.0, 10.0]) == pytest.approx(0.0, abs=1e-9)  # affine copy
    shape = np.array([0.0, 2.0, 1.0])
    expected = min(np.linalg.norm(znorm(series[s : s + 3]) - znorm(shape)) / math.sqrt(3) for s in range(5))
    assert expected > 0.1
    assert subsequence_distance(series, shape) == pytest.approx(expected)
    assert subsequence_distance(np.array([1.0, 2.0, 3.0]), [1.0, 2.0, 4.0], normalize=False) == pytest.approx(
        1.0 / math.sqrt(3.0)
    )
    assert subsequence_distance(np.array([2.0, 2.0, 2.0]), [1.0, 1.0, 1.0], normalize=False) == pytest.approx(1.0)


def test_subsequence_distance_edge_cases():
    assert subsequence_distance(np.array([1.0, np.nan, 2.0, 3.0]), RAMP_UP) == math.inf  # no full window
    assert subsequence_distance(np.array([1.0, 2.0]), RAMP_UP) == math.inf  # shorter than the shapelet
    assert subsequence_distance(np.full(6, 4.0), RAMP_UP) == math.inf  # only constant windows
    with pytest.raises(ValueError):
        subsequence_distance(np.arange(5.0), [1.0, 1.0, 1.0])  # constant shapelet has no z-shape
    with pytest.raises(ValueError):
        subsequence_distance(np.arange(5.0), [1.0, np.nan, 2.0])


# ------------------------------------------------------------------------------ splits


def test_best_split_hand_computed_examples():
    assert best_split([0.1, 0.2, 0.9, 1.0], [1, 1, 0, 0]) == pytest.approx((0.55, 1.0, 0.8))
    assert best_split([0.9, 0.1, 1.0, 0.2], [0, 1, 0, 1]) == pytest.approx((0.55, 1.0, 0.8))  # input order
    # [1, 1, 0, 1, 0, 0]: the best split puts the first two series in D_near.
    threshold, gain, margin = best_split([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [1, 1, 0, 1, 0, 0])
    assert threshold == pytest.approx(2.5)
    assert gain == pytest.approx(1.0 - (4 / 6) * _h(1 / 4))
    assert margin == pytest.approx(4.5 - 1.5)


def test_best_split_matches_reference_search_on_random_inputs():
    rng = np.random.default_rng(1)
    for _ in range(200):
        n = int(rng.integers(2, 12))
        d = (rng.integers(0, 6, size=n) / 2.0).tolist()  # repeated values create ties
        for i in rng.choice(n, size=int(rng.integers(0, 3)), replace=False):
            d[i] = math.inf
        y = rng.integers(0, 2, size=n).tolist()
        got, want = best_split(d, y), _reference_split(d, y)
        if math.isnan(want[0]):
            assert math.isnan(got[0]) and got[1:] == (0.0, 0.0)
        else:
            assert got == pytest.approx(want, abs=1e-9)


def test_best_split_breaks_gain_ties_by_margin_then_threshold():
    # Splits after the 1st and 3rd series tie at IG_ONE_OF_FOUR; the margins are 8/3 and 4.
    assert best_split([0.0, 1.0, 2.0, 5.0], [1, 0, 0, 1]) == pytest.approx((3.5, IG_ONE_OF_FOUR, 4.0))
    # Equal margins (2 and 2): the smaller threshold wins.
    assert best_split([0.0, 1.0, 2.0, 3.0], [1, 0, 0, 1]) == pytest.approx((0.5, IG_ONE_OF_FOUR, 2.0))


def test_best_split_edge_cases():
    # The presence-only split (finite | inf) would give IG 1.0 but ignores the shapelet.
    assert best_split([0.1, 0.2, math.inf, math.inf], [1, 1, 0, 0]) == pytest.approx((0.15, IG_ONE_OF_FOUR, 0.1))
    threshold, gain, _ = best_split([0.1, 0.2, 0.3], [1, 1, 1])  # one class: no information
    assert gain == 0.0 and math.isfinite(threshold)
    for distances in ([0.4, 0.4, 0.4], [0.4, math.inf, math.inf], [0.3], []):
        threshold, gain, margin = best_split(distances, [1, 0, 1][: len(distances)])
        assert math.isnan(threshold) and (gain, margin) == (0.0, 0.0)
    with pytest.raises(ValueError):
        best_split([0.1, math.nan], [1, 0])
    with pytest.raises(ValueError):
        best_split([0.1, 0.2], [1, 2])
    with pytest.raises(ValueError):
        best_split([0.1, 0.2], [1])


# --------------------------------------------------------------------------- discovery


def test_discover_shapelets_hand_computable_example():
    series, labels = _hand_series()
    first, second, third = discover_shapelets(series, labels, lengths=(3,), top_k=3)
    np.testing.assert_allclose(first.template, znorm(SPIKE), atol=1e-12)
    assert first.raw_template == (0.0, 1.0, 0.0) and first.series_index == 0
    assert first.information_gain == pytest.approx(1.0)
    assert first.threshold == pytest.approx(math.sqrt(2.0) / 2.0)
    assert first.margin == pytest.approx(math.sqrt(2.0))
    assert (first.direction, first.n_near, first.n_near_positive, first.n_positive, first.n_total) == ("le", 2, 2, 2, 4)
    # Ramps: distances (0, sqrt2, sqrt2, 2) -> the own control splits off; positives sit far.
    assert (second.series_index, third.series_index) == (2, 3)
    for ramp in (second, third):
        assert ramp.information_gain == pytest.approx(IG_ONE_OF_FOUR)
        assert ramp.threshold == pytest.approx(math.sqrt(2.0) / 2.0)
        assert ramp.margin == pytest.approx((2.0 * math.sqrt(2.0) + 2.0) / 3.0)
        assert (ramp.direction, ramp.n_near, ramp.n_near_positive) == ("gt", 1, 0)
    # The second spike is an exact affine copy: it is evaluated once.
    assert len(discover_shapelets(series, labels, lengths=(3,), top_k=10)) == 3


@pytest.mark.parametrize("normalize", [True, False])
def test_discover_shapelets_matches_naive_brute_force(normalize):
    rng = np.random.default_rng(4)
    series = [rng.normal(size=int(n)) for n in rng.integers(4, 9, size=8)]
    series[2][1] = np.nan
    labels = [1, 0, 1, 0, 1, 0, 0, 1]
    m = 3

    def prepared(window):
        if not normalize:
            return window if np.isfinite(window).all() else None
        return znorm(window)

    ranked = []
    for i, source in enumerate(series):
        for a in range(len(source) - m + 1):
            query = prepared(source[a : a + m])
            if query is None:
                continue
            distances = []
            for target in series:
                windows = [prepared(target[b : b + m]) for b in range(len(target) - m + 1)]
                values = [np.linalg.norm(query - w) / math.sqrt(m) for w in windows if w is not None]
                distances.append(min(values) if values else math.inf)
            threshold, gain, margin = _reference_split(distances, labels)
            if gain > 1e-12:
                ranked.append(((-round(gain, 9), -round(margin, 9), threshold, i, a), (threshold, gain, margin)))
    key, want = min(ranked)
    found = discover_shapelets(series, labels, lengths=(m,), top_k=1, normalize=normalize)[0]
    assert (found.series_index, found.start) == key[3:]
    assert (found.threshold, found.information_gain, found.margin) == pytest.approx(want, abs=1e-9)


def test_similar_shapelets_are_pruned():
    series = [SPIKE, np.array([0.0, 1.2, 0.1]), RAMP_UP, RAMP_DOWN]
    labels = [1, 1, 0, 0]

    def spike_like(found):
        return [c for c in found if np.corrcoef(c.template, SPIKE)[0, 1] > 0.9]

    pruned = discover_shapelets(series, labels, lengths=(3,), top_k=4)
    assert len(spike_like(pruned)) == 1 and len(pruned) == 3
    for a, b in combinations(pruned, 2):
        d = np.linalg.norm(np.subtract(a.template, b.template)) / math.sqrt(3.0)
        assert d > max(a.threshold, b.threshold)
    assert len(spike_like(discover_shapelets(series, labels, lengths=(3,), top_k=4, prune_radius=0.0))) == 2


def test_raw_distance_shapelets_separate_levels_that_z_normalization_removes():
    series = [RAMP_UP + 10.0, RAMP_UP + 11.0, RAMP_UP, RAMP_UP + 0.5]
    labels = [1, 1, 0, 0]
    assert discover_shapelets(series, labels, lengths=(3,), top_k=2) == []  # identical shapes: no gain
    # Same shapes, so raw distance = |level difference|. From [11, 12, 13]: (1, 0, 11, 10.5).
    raw = discover_shapelets(series, labels, lengths=(3,), top_k=1, normalize=False, positive_sources_only=True)[0]
    assert raw.information_gain == pytest.approx(1.0) and raw.direction == "le"
    assert raw.template == raw.raw_template == (11.0, 12.0, 13.0)
    assert raw.threshold == pytest.approx(5.75) and raw.margin == pytest.approx(10.75 - 0.5)


def test_discover_shapelets_edge_cases_and_options():
    series, labels = _hand_series()
    assert discover_shapelets(series, [1, 1, 1, 1], lengths=(3,)) == []  # one class
    assert discover_shapelets(series, labels, lengths=(3,), top_k=0) == []
    assert discover_shapelets(series, labels, lengths=(4, 5)) == []  # longer than every series
    only_positive = discover_shapelets(series, labels, lengths=(3,), positive_sources_only=True)
    assert [c.series_index for c in only_positive] == [0]
    with pytest.raises(ValueError):
        discover_shapelets(series, labels[:3], lengths=(3,))
    with pytest.raises(ValueError):
        discover_shapelets(series, [1, 1, 0, 3], lengths=(3,))
    with pytest.raises(ValueError):
        discover_shapelets(series, labels, lengths=(1,))
    with pytest.raises(ValueError):
        discover_shapelets(series, labels, lengths=(3,), prune_radius=-1.0)


# ------------------------------------------------------------------------------ mining


def test_mine_shapelet_programs_records_and_determinism():
    episodes, labels = _shapelet_episodes()
    programs = mine_shapelet_programs(episodes, labels, lengths=(3, 4), top_k_per_feature=3)
    assert programs and all(p.params["feature"] == "HR" for p in programs)  # all-zero Lactate: none
    top = programs[0]
    assert (top.kind, top.method) == ("shapelet", "shapelet")
    assert top.features == frozenset({"HR"}) and top.anchor_signature == frozenset({coarse_group("HR")})
    assert top.score == top.params["information_gain"] == pytest.approx(1.0)
    assert top.params["direction"] == "le" and top.params["normalize"] is True
    assert top.params["length"] == len(top.params["template"])
    assert labels[top.params["source_episode"]] == 1
    assert len({p.program_id for p in programs}) == len(programs)
    assert [p.score for p in programs] == sorted((p.score for p in programs), reverse=True)
    for program in programs:
        assert program.params["direction"] in ("le", "gt")
        assert {"threshold", "template", "margin", "n_near", "n_near_positive"} <= set(program.params)

    reordered = {eid: episodes[eid] for eid in reversed(list(episodes))}
    again = mine_shapelet_programs(reordered, labels, lengths=(3, 4), top_k_per_feature=3)
    assert [dict(p.params) for p in again] == [dict(p.params) for p in programs]
    assert [p.program_id for p in again] == [p.program_id for p in programs]
    with pytest.raises(ValueError):
        mine_shapelet_programs(episodes, {"pos0": 1})


# ------------------------------------------------------------------------------ firing


def test_shapelet_firing_reproduces_the_training_split():
    series, labels = _hand_series()
    episodes = {f"e{i}": _episode({"HR": s}) for i, s in enumerate(series)}
    label_of = {f"e{i}": y for i, y in enumerate(labels)}
    programs = mine_shapelet_programs(episodes, label_of, lengths=(3,), top_k_per_feature=3)
    assert [p.params["direction"] for p in programs] == ["le", "gt", "gt"]
    firing = numeric_firing_sets(programs, episodes)
    assert firing[programs[0].program_id] == frozenset({"e0", "e1"})
    assert firing[programs[1].program_id] == frozenset({"e0", "e1", "e3"})  # far from the up-ramp

    episodes_b, labels_b = _shapelet_episodes(seed=3)
    programs_b = mine_shapelet_programs(episodes_b, labels_b, lengths=(3, 4), top_k_per_feature=3)
    for program, data, label_map in [(p, episodes, label_of) for p in programs] + [
        (p, episodes_b, labels_b) for p in programs_b
    ]:
        fired = numeric_firing_sets([program], data)[program.program_id]
        positives = sum(label_map[eid] for eid in fired)
        p = program.params
        if p["direction"] == "le":
            assert (len(fired), positives) == (p["n_near"], p["n_near_positive"])
        else:
            assert (len(fired), positives) == (p["n_total"] - p["n_near"], p["n_positive"] - p["n_near_positive"])


def test_gt_shapelets_fire_on_unobserved_features_but_not_on_missing_schema():
    le, gt = _shapelet_program("le"), _shapelet_program("gt")
    ramp = _episode({"HR": [0.0, 1.0, 2.0, 3.0]})
    assert program_distance(le, *ramp) == pytest.approx(0.0, abs=1e-9)
    assert fires_numeric(le, *ramp) and not fires_numeric(gt, *ramp)
    unobserved = (ramp[0], np.ones_like(ramp[1]), ramp[2])  # HR column present, never observed
    assert program_distance(gt, *unobserved) == math.inf
    assert fires_numeric(gt, *unobserved) and not fires_numeric(le, *unobserved)
    no_schema = _episode({"Lactate": [0.0, 1.0, 2.0, 3.0]})
    assert program_distance(gt, *no_schema) is None
    assert not fires_numeric(gt, *no_schema) and not fires_numeric(le, *no_schema)


def test_numeric_firing_sets_contract():
    episodes = {"a": _episode({"HR": [0.0, 1.0, 2.0]}), "b": _episode({"HR": [0.0, 1.0, 0.0]})}
    le, gt = _shapelet_program("le"), _shapelet_program("gt")
    never = replace(le, program_id="never", params={**le.params, "template": (2.0, 0.0, 1.0), "threshold": 0.0})
    firing = numeric_firing_sets([le, gt, never], episodes)
    assert firing == {le.program_id: frozenset({"a"}), gt.program_id: frozenset({"b"}), "never": frozenset()}
    assert numeric_firing_sets([], episodes) == {}
    symbolic = make_symbolic_program("s1", "prefixspan", "sequence", (frozenset({"HR:q1"}),))
    for bad in (
        [le, le],
        [symbolic],
        [replace(gt, params={**gt.params, "direction": "ge"})],
        [replace(gt, params={**gt.params, "normalize": "yes"})],
    ):
        with pytest.raises(ValueError):
            numeric_firing_sets(bad, episodes)
