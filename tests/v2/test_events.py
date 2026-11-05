from __future__ import annotations

import numpy as np
import pytest

from src.icu.icu_eventizer import ICUEventizer
from src.icu.v2.events import (
    describe_vocabulary,
    eventize_episode,
    fit_feature_bins,
    translate_icu_eventizer_token,
)


def _states(seq, wanted: set[str]) -> list[set[str]]:
    return [{token for token in itemset if token.split(":", 1)[1] in wanted} for itemset in seq]


# ----------------------------------------------------------------- fit_feature_bins


def test_bins_are_global_train_quantiles_not_per_episode_ranges():
    low = np.arange(10, dtype=float).reshape(-1, 1)
    high = (100 + np.arange(10, dtype=float)).reshape(-1, 1)
    bins = fit_feature_bins([low, high], ["HR"], n_bins=4)

    pooled = np.concatenate([low[:, 0], high[:, 0]])
    expected = np.quantile(pooled, [0, 0.25, 0.5, 0.75, 1], method="inverted_cdf")
    np.testing.assert_allclose(bins["HR"], expected)
    np.testing.assert_allclose(bins["HR"], [0, 4, 9, 104, 109])  # cuts are observed values

    seq_low = eventize_episode(low, None, ["HR"], bins, observation_tokens=False)
    seq_high = eventize_episode(high, None, ["HR"], bins, observation_tokens=False)
    # A per-episode binning would spread each episode over q0..q3.
    assert {token for itemset in seq_low for token in itemset} == {"HR:q0", "HR:q1", "HR:q2"}
    assert {token for itemset in seq_high for token in itemset} == {"HR:q2", "HR:q3"}
    assert seq_low[9] == frozenset({"HR:q2"}) and seq_high[0] == frozenset({"HR:q2"})


def test_bins_ignore_masked_nonfinite_and_intervention_columns():
    values = np.array(
        [
            [1.0, 0.0, np.nan],
            [2.0, 1.0, np.nan],
            [3.0, 1.0, np.nan],
            [1e9, 0.0, np.nan],  # masked outlier
            [np.inf, 0.0, np.nan],
        ]
    )
    mask = np.zeros_like(values, dtype=bool)
    mask[3, 0] = True
    bins = fit_feature_bins([values], ["HR", "vasopressor_on", "pH"], n_bins=2, missing_masks=[mask])

    assert set(bins) == {"HR", "pH"}  # intervention features are not binned
    np.testing.assert_allclose(bins["HR"], [1.0, 2.0, 3.0])
    assert bins["pH"].size == 0  # never observed in train -> no value tokens
    seq = eventize_episode(values[:3], None, ["HR", "vasopressor_on", "pH"], bins, observation_tokens=False)
    assert [sorted(itemset) for itemset in seq] == [
        ["HR:q0"],
        ["HR:q1", "vasopressor_on:on"],
        ["HR:q1", "vasopressor_on:on"],
    ]


def test_bins_handle_tied_quantiles():
    zero_inflated = np.array([0.0] * 7 + [5.0, 10.0, 20.0]).reshape(-1, 1)
    binary = np.array([0.0] * 7 + [1.0] * 3).reshape(-1, 1)
    constant = np.full((10, 1), 7.0)
    bins = fit_feature_bins(
        [np.hstack([zero_inflated, binary, constant])],
        ["urine_output", "flag", "Temp"],
        n_bins=4,
    )
    edges = bins["urine_output"]
    np.testing.assert_allclose(edges, [0.0, 5.0, 20.0])  # tied cuts at the minimum dropped
    np.testing.assert_allclose(bins["flag"], [0.0, 1.0, 1.0])  # two bins, not collapsed
    np.testing.assert_allclose(bins["Temp"], [7.0, 7.0])  # one bin

    seq = eventize_episode(
        np.array([[0.0, 0.0, 7.0], [20.0, 1.0, 50.0]]),
        None,
        ["urine_output", "flag", "Temp"],
        bins,
        observation_tokens=False,
    )
    assert seq[0] == frozenset({"urine_output:q0", "flag:q0", "Temp:q0"})
    assert seq[1] == frozenset({f"urine_output:q{edges.size - 2}", "flag:q1", "Temp:q0"})
    with pytest.raises(ValueError):
        fit_feature_bins([zero_inflated], ["urine_output"], n_bins=0)


# ----------------------------------------------------------------- eventize_episode


def test_one_itemset_per_bin_and_no_length_cap():
    T = 200
    hr = np.linspace(60.0, 120.0, T)
    lactate = np.full(T, np.nan)
    lactate[-1] = 4.0  # only observed in the last bin, the one nearest the anchor
    values = np.column_stack([hr, lactate])
    bins = fit_feature_bins([values], ["HR", "Lactate"])
    seq = eventize_episode(values, None, ["HR", "Lactate"], bins, observation_tokens=False)

    assert len(seq) == T
    assert all(isinstance(itemset, frozenset) for itemset in seq)
    assert seq[-1] == frozenset({"HR:q3", "Lactate:q0"})  # same-bin events stay together
    assert all(len(itemset) == 1 for itemset in seq[:-1])
    assert eventize_episode(np.empty((0, 2)), None, ["HR", "Lactate"], bins) == ()


def test_empty_bins_are_kept_as_empty_itemsets():
    values = np.array([[1.0], [np.nan], [2.0]])
    bins = fit_feature_bins([values], ["Glucose"])
    seq = eventize_episode(values, None, ["Glucose"], bins, observation_tokens=False)
    assert len(seq) == 3 and seq[1] == frozenset()


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_observation_and_intervention_rules_match_icu_eventizer(seed):
    rng = np.random.default_rng(seed)
    T = 30
    names = ["HR", "Lactate", "pH", "vasopressor_on", "urine_output"]
    values = np.column_stack(
        [
            rng.normal(80, 10, T),
            rng.gamma(2.0, 1.0, T),
            np.full(T, np.nan),  # never observed
            (np.cumsum(rng.random(T) < 0.25) % 2).astype(float),
            rng.gamma(1.0, 50.0, T),
        ]
    )
    mask = np.zeros_like(values, dtype=bool)
    mask[:, 0] = rng.random(T) < 0.3
    mask[:, 1] = rng.random(T) < 0.7
    mask[:, 2] = True
    values[mask] = np.nan

    eventizer = ICUEventizer(
        feature_names=names,
        n_bins=4,
        window_size=6,
        quantile_stride=1,
        change_threshold=0.25,
        support_min=None,
        support_max=None,
    )
    v1 = eventizer.transform(values, feature_meta={"feature_names": names, "missing_mask": mask})
    v1_labels = [{eventizer.describe_event(event_id) for event_id in step} for step in v1]

    bins = fit_feature_bins([values], names, missing_masks=[mask])
    v2 = eventize_episode(values, mask, names, bins, observation_tokens=True, trend_tokens=True)

    wanted = {"missing", "sparse_measurement", "onset", "offset"}
    assert _states(v2, wanted) == _states(v1_labels, wanted)
    assert any(_states(v2, {"missing"})) and any(_states(v2, {"sparse_measurement"}))


def test_trend_tokens_compare_with_previous_observation_across_gaps():
    bins = {"HR": np.array([0.0, 10.0, 20.0, 30.0, 40.0])}
    values = np.array([[5.0], [np.nan], [np.nan], [35.0], [15.0], [16.0]])
    seq = eventize_episode(values, None, ["HR"], bins, observation_tokens=False, trend_tokens=True)
    assert [sorted(itemset) for itemset in seq] == [
        ["HR:q0"],
        [],
        [],
        ["HR:q3", "HR:up"],
        ["HR:down", "HR:q1"],
        ["HR:q1"],
    ]
    vent = np.array([[0.0], [1.0], [1.0], [0.0]])
    seq = eventize_episode(vent, None, ["ventilation_on"], {}, trend_tokens=True)
    assert [sorted(itemset) for itemset in seq] == [
        [],
        ["ventilation_on:on", "ventilation_on:onset"],
        ["ventilation_on:on"],
        ["ventilation_on:offset"],
    ]


def test_observation_tokens_flag_and_feature_scope():
    values = np.array([[np.nan, 0.0], [np.nan, 0.0], [1.0, 0.0], [np.nan, 0.0]])
    names = ["Creatinine", "urine_output"]
    bins = fit_feature_bins([np.array([[1.0, 0.0], [2.0, 10.0]])], names)
    with_obs = eventize_episode(values, None, names, bins)
    without = eventize_episode(values, None, names, bins, observation_tokens=False)
    assert [sorted(s) for s in with_obs] == [
        ["Creatinine:missing", "urine_output:q0"],
        ["urine_output:q0"],
        ["Creatinine:q0", "Creatinine:sparse_measurement", "urine_output:q0"],
        ["Creatinine:missing", "Creatinine:sparse_measurement", "urine_output:q0"],
    ]
    assert all(not any(":missing" in t or ":sparse" in t for t in s) for s in without)


def test_eventize_rejects_bad_inputs():
    bins = {"HR": np.array([0.0, 1.0])}
    with pytest.raises(ValueError):
        eventize_episode(np.zeros((3, 2)), None, ["HR"], bins)
    with pytest.raises(ValueError):
        eventize_episode(np.zeros((3, 1)), np.zeros((2, 1), dtype=bool), ["HR"], bins)
    with pytest.raises(KeyError):
        eventize_episode(np.zeros((3, 1)), None, ["Lactate"], bins)
    with pytest.raises(ValueError):
        eventize_episode(np.zeros((3, 2)), None, ["HR", "HR"], bins)


# ------------------------------------------------------------- vocabulary helpers


def test_describe_vocabulary_covers_every_emitted_token():
    rng = np.random.default_rng(7)
    names = ["HR", "Lactate", "ventilation_on", "urine_output"]
    values = np.column_stack(
        [rng.normal(80, 10, 40), rng.gamma(2, 1, 40), rng.random(40) < 0.5, rng.gamma(1, 50, 40)]
    ).astype(float)
    mask = np.zeros_like(values, dtype=bool)
    mask[:, 1] = rng.random(40) < 0.6
    bins = fit_feature_bins([values], names, missing_masks=[mask])
    seq = eventize_episode(values, mask, names, bins, trend_tokens=True)
    rows = describe_vocabulary(names, bins, trend_tokens=True)

    vocabulary = {row["token"] for row in rows}
    assert {token for itemset in seq for token in itemset} <= vocabulary
    assert len(vocabulary) == len(rows)
    assert {"token", "feature", "family", "definition", "icu_eventizer"} <= set(rows[0])
    by_token = {row["token"]: row for row in rows}
    assert by_token["HR:q0"]["definition"].endswith(f"value < {bins['HR'][1]:.4g}")
    assert by_token["ventilation_on:on"]["family"] == "intervention"
    assert "ventilation_on:q0" not in vocabulary
    assert "urine_output:missing" not in vocabulary
    plain = {row["token"] for row in describe_vocabulary(names, bins, observation_tokens=False)}
    assert "Lactate:missing" not in plain and "HR:up" not in plain


def test_translate_icu_eventizer_token():
    assert translate_icu_eventizer_token("HR:level_2") == "HR:q2"
    assert translate_icu_eventizer_token("HR:deterioration") == "HR:up"
    assert translate_icu_eventizer_token("SpO2:deterioration") == "SpO2:down"
    assert translate_icu_eventizer_token("SpO2:recovery") == "SpO2:up"
    assert translate_icu_eventizer_token("Temp:missing") == "Temp:missing"
    assert translate_icu_eventizer_token("Temp:sparse_measurement") == "Temp:sparse_measurement"
    assert translate_icu_eventizer_token("vasopressor_on:up") == "vasopressor_on:onset"
    assert translate_icu_eventizer_token("vasopressor_on:down") == "vasopressor_on:offset"
    assert translate_icu_eventizer_token("vasopressor_on:onset") == "vasopressor_on:onset"
    assert translate_icu_eventizer_token("vasopressor_on:level_0") is None
    assert translate_icu_eventizer_token("urine_output:deterioration") is None
    assert translate_icu_eventizer_token("HR:unknown_state") is None
