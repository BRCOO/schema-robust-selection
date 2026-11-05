"""Tests for the exact cross-episode matrix-profile motif baseline (v2) and motif firing."""

from __future__ import annotations

import math
from dataclasses import replace
from itertools import combinations

import numpy as np
import pytest

from src.icu.v2.baselines.motif import (
    build_window_bank,
    cross_episode_matrix_profile,
    discover_motifs,
    feature_series,
    mine_motif_programs,
    observed_series,
    sliding_windows,
    sq_distances,
    template_bank,
    znorm,
    znorm_rows,
)
from src.icu.v2.numeric_firing import fires_numeric, numeric_firing_sets, program_distance
from src.icu.v2.types import ProgramRecord, coarse_group, make_symbolic_program

MOTIF = np.array([0.0, 3.0, -1.0, 2.0])
REPEAT = np.array([4.0, -4.0, 4.0, -4.0])
CONTROL_MOTIF = np.array([1.0, -2.0, 2.5, 0.0])


def _episode(columns: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray, list[str]]:
    names = list(columns)
    values = np.column_stack([np.asarray(columns[name], dtype=float) for name in names])
    return values, np.zeros(values.shape, dtype=bool), names


def _planted_series(seed: int = 7) -> list[np.ndarray]:
    """MOTIF shared by series 0 and 3 (affine copy); REPEAT twice inside series 1 only."""
    rng = np.random.default_rng(seed)
    series = [rng.normal(size=14) for _ in range(6)]
    series[0][2:6] = MOTIF
    series[3][7:11] = 2.5 * MOTIF - 1.0 + rng.normal(scale=0.01, size=4)
    series[1][1:5] = REPEAT
    series[1][8:12] = REPEAT
    return series


def _naive_profile(series_list: list[np.ndarray], m: int, groups: list | None = None) -> list[np.ndarray]:
    groups = list(range(len(series_list))) if groups is None else groups
    profiles = []
    for i, series in enumerate(series_list):
        row = []
        for a in range(len(series) - m + 1):
            za = znorm(series[a : a + m])
            best = math.inf
            if za is not None:
                for j, other in enumerate(series_list):
                    if groups[j] == groups[i]:
                        continue
                    for b in range(len(other) - m + 1):
                        zb = znorm(other[b : b + m])
                        if zb is not None:
                            best = min(best, float(np.linalg.norm(za - zb)) / math.sqrt(m))
            row.append(best)
        profiles.append(np.array(row))
    return profiles


def _motif_episodes(seed: int = 0) -> tuple[dict, dict]:
    """3 positives + 3 controls. MOTIF is planted in pos0/pos1, CONTROL_MOTIF (exact
    affine copies, nn distance 0) in ctl0/ctl1. Lactate is an all-zero channel."""
    rng = np.random.default_rng(seed)
    episodes: dict[str, tuple] = {}
    labels: dict[str, int] = {}
    for k in range(3):
        for label, prefix in ((1, "pos"), (0, "ctl")):
            episodes[f"{prefix}{k}"] = _episode({"HR": rng.normal(size=12), "Lactate": np.zeros(12)})
            labels[f"{prefix}{k}"] = label
    episodes["pos0"][0][1:5, 0] = MOTIF
    episodes["pos1"][0][6:10, 0] = 2.0 * MOTIF + 3.0 + rng.normal(scale=0.01, size=4)
    episodes["ctl0"][0][2:6, 0] = CONTROL_MOTIF
    episodes["ctl1"][0][5:9, 0] = 3.0 * CONTROL_MOTIF - 2.0
    return episodes, labels


def _motif_program(threshold: float = 0.05, **params: object) -> ProgramRecord:
    return ProgramRecord(
        program_id="matrix_profile:HR:test",
        method="matrix_profile",
        kind="motif",
        features=frozenset({"HR"}),
        anchor_signature=frozenset({coarse_group("HR")}),
        params={"feature": "HR", "template": tuple(znorm(MOTIF)), "threshold": threshold, "length": 4,
                "normalize": True, **params},
    )


# ------------------------------------------------------------------ series and windows


def test_znorm_returns_shape_and_excludes_constant_or_missing_subsequences():
    np.testing.assert_allclose(znorm([1.0, 2.0, 3.0]), [-math.sqrt(1.5), 0.0, math.sqrt(1.5)])
    np.testing.assert_allclose(znorm(3.0 * MOTIF + 7.0), znorm(MOTIF))  # affine invariant
    assert np.std(znorm(MOTIF)) == pytest.approx(1.0)
    assert znorm([0.0, 0.0, 0.0]) is None  # v1 turned this into the template [0, 0, 0]
    assert znorm([1000.0, 1000.0, 1000.0]) is None  # constant up to rounding
    assert znorm([1.0, np.nan, 2.0]) is None
    assert znorm([]) is None
    z, valid = znorm_rows(np.array([[1.0, 2.0, 4.0], [5.0, 5.0, 5.0], [1.0, np.inf, 0.0]]))
    assert valid.tolist() == [True, False, False]
    assert np.isnan(z[1:]).all()
    with pytest.raises(ValueError):
        znorm_rows(np.zeros((2, 3)), eps=-1.0)


def test_observed_series_uses_only_observed_points_strict_or_forward_filled():
    column = np.array([7.0, 2.0, 3.0, np.nan, 5.0, 6.0])
    missing = np.array([True, False, True, False, False, False])  # True = missing
    strict = observed_series(column, missing)
    np.testing.assert_array_equal(strict, [np.nan, 2.0, np.nan, np.nan, 5.0, 6.0])  # imputed 7, 3 ignored
    np.testing.assert_array_equal(observed_series(column, missing, "ffill"), [np.nan, 2.0, 2.0, 2.0, 5.0, 6.0])
    np.testing.assert_array_equal(observed_series(column), [7.0, 2.0, 3.0, np.nan, 5.0, 6.0])  # mask None
    assert observed_series(np.array([]), None, "ffill").size == 0
    with pytest.raises(ValueError):
        observed_series(column, missing, "interpolate")
    with pytest.raises(ValueError):
        observed_series(column, missing[:3])


def test_feature_series_looks_features_up_by_name():
    values, mask, names = _episode({"HR": [1.0, 2.0, 3.0], "Lactate": [4.0, 5.0, 6.0]})
    mask[0, 1] = True
    np.testing.assert_array_equal(feature_series(values, mask, names, "Lactate"), [np.nan, 5.0, 6.0])
    np.testing.assert_array_equal(feature_series(values, None, names, "HR"), [1.0, 2.0, 3.0])
    assert feature_series(values, mask, names, "pH") is None  # schema lacks the feature
    with pytest.raises(ValueError):
        feature_series(values, mask[:, :1], names, "HR")
    with pytest.raises(ValueError):
        feature_series(values, mask, ["HR"], "HR")


def test_window_bank_keeps_only_fully_observed_nonconstant_windows():
    assert sliding_windows(np.arange(5.0), 3).shape == (3, 3)
    assert sliding_windows(np.arange(2.0), 3).shape == (0, 3)
    series = np.array([1.0, 2.0, 0.5, np.nan, 3.0, 3.0, 3.0, 4.0])
    bank = build_window_bank([series], 3)
    assert bank.start.tolist() == [0, 5]  # starts 1-3 touch the NaN, start 4 is constant
    np.testing.assert_allclose(bank.sq_norm, 1.0)  # z-normalized rows / sqrt(m) have unit norm
    raw_bank = build_window_bank([series], 3, normalize=False)
    assert raw_bank.start.tolist() == [0, 4, 5]  # without z-normalization constants are valid
    assert len(build_window_bank([], 3)) == 0
    with pytest.raises(ValueError):
        build_window_bank([series], 1)  # one point has no z-normalized shape
    d2 = sq_distances(template_bank(MOTIF), template_bank(np.vstack([2 * MOTIF + 1, -MOTIF])))
    np.testing.assert_allclose(d2, [[0.0, 4.0]])  # exact repeat -> 0, mirror image -> 2^2
    with pytest.raises(ValueError):
        template_bank([2.0, 2.0, 2.0])


# -------------------------------------------------------------------- matrix profile


@pytest.mark.parametrize("groups", [None, ["a", "b", "a", "c", "b"]])
def test_cross_episode_profile_equals_naive_brute_force(groups):
    rng = np.random.default_rng(3)
    series = [rng.normal(size=n) for n in (9, 7, 10, 3, 8)]
    series[0][4] = np.nan
    series[2][0:4] = 1.5  # constant run
    series[4] = rng.integers(0, 3, size=8).astype(float)  # discrete values repeat shapes exactly
    for m in (3, 4):
        profile, index = cross_episode_matrix_profile(series, m, groups)
        naive = _naive_profile(series, m, groups)
        group_of = list(range(len(series))) if groups is None else groups
        for i, (got, want) in enumerate(zip(profile, naive)):
            np.testing.assert_allclose(got, want, atol=1e-7)
            for start in np.flatnonzero(np.isfinite(got)):
                j, b = index[i][start]
                assert group_of[j] != group_of[i]
                d = np.linalg.norm(znorm(series[i][start : start + m]) - znorm(series[j][b : b + m])) / math.sqrt(m)
                assert d == pytest.approx(got[start], abs=1e-7)
            assert (index[i][~np.isfinite(got)] == -1).all()


def test_planted_cross_episode_motif_is_found_and_within_episode_repeat_is_not():
    series = _planted_series()
    profile, index = cross_episode_matrix_profile(series, 4)
    assert profile[0][2] < 0.01 and tuple(index[0][2]) == (3, 7)
    assert tuple(index[3][7]) == (0, 2)
    # A self-join would match the two copies of REPEAT in series 1 at distance 0.
    assert profile[1][1] > 0.1 and profile[1][8] > 0.1
    assert index[1][1][0] != 1 and index[1][8][0] != 1

    motifs = discover_motifs(series, lengths=(4,), top_k=3)
    first = motifs[0]
    assert (first.series_index, first.start, first.neighbor_series, first.neighbor_start) == (0, 2, 3, 7)
    np.testing.assert_allclose(first.template, znorm(MOTIF), atol=1e-12)
    assert first.raw_template == tuple(MOTIF)
    assert first.nn_distance == pytest.approx(profile[0][2])
    assert first.threshold == pytest.approx(2.0 * first.nn_distance)
    assert (first.support, first.n_other_series) == (1, 5)
    sources = {(motif.series_index, motif.start) for motif in motifs}
    assert (3, 7) not in sources  # the pair partner lies inside the first motif's radius
    assert not sources & {(1, 1), (1, 8)}


def test_groups_forbid_matches_within_the_same_patient():
    series = _planted_series()
    groups = ["p1", "p2", "p3", "p1", "p4", "p5"]  # series 0 and 3 are the same patient
    profile, index = cross_episode_matrix_profile(series, 4, groups)
    assert index[0][2][0] not in (0, 3) and profile[0][2] > 0.01
    top = discover_motifs(series, lengths=(4,), top_k=1, groups=groups)[0]
    assert {top.series_index, top.neighbor_series} != {0, 3}
    with pytest.raises(ValueError):
        cross_episode_matrix_profile(series, 4, groups[:2])


def test_constant_windows_are_never_motifs():
    rng = np.random.default_rng(11)
    flat_then_noise = [np.r_[np.zeros(6), rng.normal(size=6)] for _ in range(4)]
    profile, _ = cross_episode_matrix_profile(flat_then_noise, 3)
    assert all(np.isinf(p[:4]).all() for p in profile)  # windows inside the zero run
    motifs = discover_motifs(flat_then_noise, lengths=(3, 4), top_k=5)
    assert motifs
    for motif in motifs:
        assert np.std(motif.template) == pytest.approx(1.0)
        assert np.ptp(motif.raw_template) > 0.0
    # The v1 artifact: an all-zero channel gave the top-10 template [0, 0, 0].
    zeros = [np.zeros(12) for _ in range(5)]
    assert discover_motifs(zeros, top_k=3) == []
    assert all(np.isinf(p).all() for p in cross_episode_matrix_profile(zeros, 3)[0])


def test_selected_motifs_respect_rank_order_and_exclusion_radius():
    rng = np.random.default_rng(5)
    series = [rng.normal(size=12) for _ in range(8)]
    for min_radius in (0.0, 0.4):
        motifs = discover_motifs(series, lengths=(3, 4), top_k=6, min_radius=min_radius)
        assert len(motifs) == 6
        distances = [motif.nn_distance for motif in motifs]
        assert distances == sorted(distances)
        for motif in motifs:
            assert motif.threshold == pytest.approx(max(2.0 * motif.nn_distance, min_radius))
            assert 1 <= motif.support <= motif.n_other_series == 7
        for a, b in combinations(motifs, 2):
            if a.length == b.length:
                d = np.linalg.norm(np.subtract(a.template, b.template)) / math.sqrt(a.length)
                assert d > max(a.threshold, b.threshold)


def test_discover_motifs_edge_cases_and_validation():
    series = _planted_series()
    assert discover_motifs(series, (4,), top_k=0) == []
    assert discover_motifs(series[:1], (4,), top_k=3) == []  # no other episode to match
    assert discover_motifs([np.arange(3.0), np.arange(3.0)], (4,), top_k=3) == []  # too short
    with pytest.raises(ValueError):
        discover_motifs(series, (1,), top_k=3)
    with pytest.raises(ValueError):
        discover_motifs(series, (), top_k=3)
    with pytest.raises(ValueError):
        discover_motifs(series, (4,), top_k=3, radius_factor=0.0)
    with pytest.raises(ValueError):
        discover_motifs(series, (4,), top_k=-1)


# ---------------------------------------------------------------------------- mining


def test_mine_motif_programs_builds_records_from_positive_train_episodes():
    episodes, labels = _motif_episodes()
    programs = mine_motif_programs(episodes, labels, lengths=(4,), top_k_per_feature=2)
    assert programs and all(program.params["feature"] == "HR" for program in programs)  # Lactate is all zero
    top = programs[0]
    assert (top.kind, top.method) == ("motif", "matrix_profile")
    assert top.features == frozenset({"HR"}) and top.anchor_signature == frozenset({coarse_group("HR")})
    assert {top.params["source_episode"], top.params["neighbor_episode"]} == {"pos0", "pos1"}
    np.testing.assert_allclose(top.params["template"], znorm(MOTIF), atol=1e-12)
    assert top.params["length"] == len(top.params["template"]) == 4
    assert top.params["normalize"] is True and top.params["discovery"] == "positive_train"
    assert top.params["threshold"] == pytest.approx(2.0 * top.params["nn_distance"])
    assert top.score == pytest.approx(1.0 / (1.0 + top.params["nn_distance"]))
    assert len({program.program_id for program in programs}) == len(programs)
    assert [p.score for p in programs] == sorted((p.score for p in programs), reverse=True)

    everything = mine_motif_programs(episodes, labels, lengths=(4,), top_k_per_feature=2, positive_only=False)
    best = everything[0]
    assert {best.params["source_episode"], best.params["neighbor_episode"]} == {"ctl0", "ctl1"}
    assert best.params["nn_distance"] == 0.0 and best.score == 1.0 and best.params["threshold"] == 0.0
    assert best.params["discovery"] == "all_train"

    same_patient = {eid: "s1" if eid in ("pos0", "pos1") else eid for eid in episodes}
    grouped = mine_motif_programs(episodes, labels, groups=same_patient, lengths=(4,), top_k_per_feature=1)
    assert "pos2" in {grouped[0].params["source_episode"], grouped[0].params["neighbor_episode"]}

    assert mine_motif_programs(episodes, labels, feature_names=["Lactate", "pH"], lengths=(4,)) == []
    with pytest.raises(ValueError):
        mine_motif_programs(episodes, None)
    with pytest.raises(ValueError):
        mine_motif_programs(episodes, {"pos0": 1})


def test_motif_mining_is_deterministic_under_input_order():
    episodes, labels = _motif_episodes(seed=2)
    reordered = {eid: episodes[eid] for eid in reversed(list(episodes))}
    kwargs = dict(lengths=(3, 4, 5), top_k_per_feature=3, positive_only=False)
    first = mine_motif_programs(episodes, labels, **kwargs)
    second = mine_motif_programs(reordered, labels, **kwargs)
    assert [p.program_id for p in first] == [p.program_id for p in second]
    assert [dict(p.params) for p in first] == [dict(p.params) for p in second]
    assert [p.canonical_key() for p in first] == [p.canonical_key() for p in second]


# ---------------------------------------------------------------------------- firing


def test_motif_fires_iff_a_valid_window_is_within_threshold():
    program = _motif_program()
    background = np.linspace(0.0, 1.0, 10) ** 2
    with_motif = background.copy()
    with_motif[3:7] = 5.0 * MOTIF + 1.0
    values, mask, names = _episode({"Lactate": np.zeros(10), "HR": with_motif})
    assert fires_numeric(program, values, mask, names)
    assert program_distance(program, values, mask, names) == pytest.approx(0.0, abs=1e-9)

    mask[4, names.index("HR")] = True  # the stored (imputed) value at row 4 must not be used
    assert not fires_numeric(program, values, mask, names)
    nan_values = values.copy()
    nan_values[4, names.index("HR")] = np.nan
    assert not fires_numeric(program, nan_values, None, names)  # mask None: NaN = missing

    plain = _episode({"HR": background})
    assert program_distance(program, *plain) > 0.05 and not fires_numeric(program, *plain)
    no_hr = _episode({"Lactate": np.zeros(10)})
    assert program_distance(program, *no_hr) is None and not fires_numeric(program, *no_hr)
    unobserved = (plain[0], np.ones_like(plain[1]), plain[2])
    assert program_distance(program, *unobserved) == math.inf and not fires_numeric(program, *unobserved)
    flat = _episode({"HR": np.full(10, 3.0)})  # constant windows cannot match a z-normalized motif
    assert program_distance(program, *flat) == math.inf and not fires_numeric(program, *flat)


def test_motif_firing_honors_the_programs_missing_policy():
    series = np.array([0.0, 3.0, 3.0, -1.0, 2.0, 0.5])
    values, mask, names = _episode({"HR": series})
    mask[2, 0] = True  # LOCF turns rows 1-4 into [3, 3, -1, 2]
    template = tuple(znorm([3.0, 3.0, -1.0, 2.0]))
    strict = _motif_program(threshold=0.01, template=template)
    ffill = _motif_program(threshold=0.01, template=template, missing_policy="ffill")
    assert not fires_numeric(strict, values, mask, names)
    assert fires_numeric(ffill, values, mask, names)
    with pytest.raises(ValueError):
        fires_numeric(_motif_program(missing_policy="bfill"), values, mask, names)


def test_mined_motifs_fire_on_their_source_and_supporting_episodes():
    episodes, labels = _motif_episodes()
    programs = mine_motif_programs(episodes, labels, lengths=(3, 4), top_k_per_feature=3)
    firing = numeric_firing_sets(programs, episodes)
    assert set(firing) == {program.program_id for program in programs}
    positives = {eid for eid, label in labels.items() if label == 1}
    for program in programs:
        fired = firing[program.program_id]
        source = program.params["source_episode"]
        assert {source, program.params["neighbor_episode"]} <= fired
        assert len((fired & positives) - {source}) == program.params["support"]
        assert fired == frozenset(eid for eid, episode in episodes.items() if fires_numeric(program, *episode))


def test_firing_rejects_malformed_numeric_programs():
    episode = _episode({"HR": np.arange(6.0)})
    program = _motif_program()
    symbolic = make_symbolic_program("s1", "prefixspan", "sequence", (frozenset({"HR:q1"}),))
    for bad in (
        symbolic,
        replace(program, params={**program.params, "template": (2.0, 2.0, 2.0, 2.0)}),  # no shape
        replace(program, params={**program.params, "length": 3}),
        replace(program, params={**program.params, "threshold": -1.0}),
        replace(program, params={**program.params, "direction": "gt"}),  # motifs fire when close
        replace(program, features=frozenset({"RR"})),
    ):
        with pytest.raises(ValueError):
            fires_numeric(bad, *episode)
    with pytest.raises(ValueError):
        numeric_firing_sets([program, program], {"e1": episode})
