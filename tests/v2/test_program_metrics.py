from __future__ import annotations

import json

import pytest

from src.icu.shock_contrastive_miner import MECHANISM_GROUP_BY_FEATURE
from src.icu.v2.program_metrics import (
    DEFINITIONS,
    N_CANONICAL_FEATURES,
    N_COARSE_GROUPS,
    anchor_group_coverage,
    coverage_upper_bounds,
    exact_duplicate_ratio,
    feature_coverage,
    firing_survival,
    jaccard,
    pairwise_redundancy,
    schema_survival,
    set_redundancy,
    summarize_program_set,
)
from src.icu.v2.types import ProgramRecord, coarse_group, make_symbolic_program

ALL_FEATURES = frozenset(MECHANISM_GROUP_BY_FEATURE)


def _rule(pid: str, antecedent: list[set[str]], consequent: list[set[str]], method: str = "m", score: float = 0.0):
    return make_symbolic_program(
        pid,
        method,
        "ordered_rule",
        tuple(frozenset(x) for x in antecedent),
        tuple(frozenset(x) for x in consequent),
        score=score,
    )


def _motif(pid: str, feature: str, template: list[float], threshold: float, method: str = "mp") -> ProgramRecord:
    return ProgramRecord(
        program_id=pid,
        method=method,
        kind="motif",
        features=frozenset({feature}),
        anchor_signature=frozenset({coarse_group(feature)}),
        params={"feature": feature, "template": template, "threshold": threshold, "length": len(template)},
    )


CR_RR = _rule("cr_rr", [{"Creatinine:sparse_measurement"}], [{"RR:level_0"}])
LAC_RR = _rule("lac_rr", [{"Lactate:q3"}], [{"RR:level_1"}])
HR_MAP = _rule("hr_map", [{"HR:q1"}], [{"MAP:q0"}])
FIRINGS = {
    "cr_rr": frozenset({"e1", "e2"}),
    "lac_rr": frozenset({"e2", "e3"}),
    "hr_map": frozenset({"e1", "e2"}),
}


def test_jaccard_values_and_empty_convention():
    assert jaccard(frozenset({"a", "b"}), frozenset({"b", "c"})) == pytest.approx(1 / 3)
    assert jaccard(frozenset({"a"}), {"a"}) == 1.0
    assert jaccard(frozenset({"a"}), frozenset()) == 0.0
    # Two never-firing programs are not evidence of redundancy.
    assert jaccard(frozenset(), frozenset()) == 0.0
    assert jaccard(set(), set()) == 0.0


def test_pairwise_redundancy_is_max_of_three_jaccards():
    same_anchor = pairwise_redundancy(CR_RR, LAC_RR, FIRINGS)
    assert same_anchor["feature_jaccard"] == pytest.approx(1 / 3)
    assert same_anchor["anchor_jaccard"] == 1.0  # both terminals are respiratory
    assert same_anchor["firing_jaccard"] == pytest.approx(1 / 3)
    assert same_anchor["value"] == 1.0

    same_firing = pairwise_redundancy(CR_RR, HR_MAP, FIRINGS)
    assert same_firing["feature_jaccard"] == 0.0
    assert same_firing["anchor_jaccard"] == 0.0
    assert same_firing["firing_jaccard"] == 1.0
    assert same_firing["value"] == 1.0


def test_pairwise_redundancy_treats_missing_firing_ids_as_empty():
    out = pairwise_redundancy(CR_RR, HR_MAP, {})
    assert out == {"feature_jaccard": 0.0, "anchor_jaccard": 0.0, "firing_jaccard": 0.0, "value": 0.0}
    # Only one side fires: still no firing overlap.
    assert pairwise_redundancy(CR_RR, HR_MAP, {"cr_rr": frozenset({"e1"})})["firing_jaccard"] == 0.0


def test_set_redundancy_is_mean_over_unordered_pairs():
    # pairs: (cr_rr, lac_rr) = 1, (cr_rr, hr_map) = 1, (lac_rr, hr_map) = max(0, 0, 1/3)
    assert set_redundancy([CR_RR, LAC_RR, HR_MAP], FIRINGS) == pytest.approx((1 + 1 + 1 / 3) / 3)
    assert set_redundancy([HR_MAP, CR_RR, LAC_RR], FIRINGS) == pytest.approx((1 + 1 + 1 / 3) / 3)
    assert set_redundancy([], FIRINGS) == 0.0
    assert set_redundancy([CR_RR], FIRINGS) == 0.0


def test_set_redundancy_rejects_one_id_for_two_programs():
    clash = _rule("cr_rr", [{"HR:q1"}], [{"SpO2:q0"}])
    with pytest.raises(ValueError, match="ambiguous"):
        set_redundancy([CR_RR, clash], FIRINGS)
    # The same program listed twice is a duplicate, not an ambiguity.
    assert set_redundancy([CR_RR, CR_RR], FIRINGS) == 1.0


def test_dedup_by_key_does_not_imply_zero_redundancy():
    # v1 FMG reported redundancy 0 because it deduplicated on the key it then counted.
    a = _rule("a", [{"Creatinine:sparse_measurement"}], [{"RR:level_0"}])
    b = _rule("b", [{"Creatinine:q3"}], [{"RR:level_1"}])
    assert exact_duplicate_ratio([a, b]) == 0.0
    assert set_redundancy([a, b], {}) == 1.0


def test_exact_duplicate_ratio_ignores_ids_methods_and_scores():
    assert exact_duplicate_ratio([]) == 0.0
    assert exact_duplicate_ratio([CR_RR]) == 0.0
    same_as_cr_rr = _rule("other_id", [{"Creatinine:sparse_measurement"}], [{"RR:level_0"}], method="x", score=9.0)
    hr_map_again = _rule("hr_map_2", [{"HR:q1"}], [{"MAP:q0"}], method="y")
    assert exact_duplicate_ratio([CR_RR, same_as_cr_rr, HR_MAP, hr_map_again]) == 0.5


def test_exact_duplicate_ratio_for_numeric_programs():
    m1 = _motif("m1", "Lactate", [0.1, 0.2, 0.3], 1.5)
    m2 = _motif("m2", "Lactate", [0.1 + 1e-9, 0.2, 0.3], 1.5)  # same template after rounding
    m3 = _motif("m3", "Lactate", [0.1, 0.2, 0.3], 2.0)  # different threshold
    assert exact_duplicate_ratio([m1, m2, m3]) == pytest.approx(1 / 3)


def test_coverage_counts_unions_of_signatures():
    assert anchor_group_coverage([]) == 0
    assert feature_coverage([]) == 0
    assert anchor_group_coverage([CR_RR, LAC_RR, HR_MAP]) == 2  # respiratory, hemodynamic
    assert feature_coverage([CR_RR, LAC_RR, HR_MAP]) == 5
    # A two-feature terminal from one group adds one group; from two groups adds two.
    seq = make_symbolic_program("s", "prefixspan", "sequence", (frozenset({"HR:q3"}), frozenset({"Lactate:q3", "pH:q0"})))
    assert seq.anchor_signature == frozenset({"perfusion_acidbase"})
    assert anchor_group_coverage([seq]) == 1
    multi = _rule("multi", [{"pH:q0"}], [{"HR:q3", "RR:level_0"}])
    assert anchor_group_coverage([multi]) == 2


def test_coverage_upper_bounds_single_group_terminals_cannot_exceed_nine():
    assert (N_COARSE_GROUPS, N_CANONICAL_FEATURES) == (9, 18)
    consequents = ["HR", "RR", "Lactate", "Creatinine", "Bilirubin_Total", "WBC", "Glucose", "Temp", "vasopressor_on", "SpO2"]
    programs = [_rule(f"p{i}", [{"pH:q0"}], [{f"{feature}:q1"}]) for i, feature in enumerate(consequents)]
    bounds = coverage_upper_bounds(programs)
    assert bounds["k"] == 10
    assert bounds["anchor_group_coverage"] == 9
    assert bounds["anchor_group_coverage_max"] == 9
    assert bounds["anchor_group_coverage_max_single_group_terminals"] == 9
    assert bounds["anchor_group_coverage_can_exceed_k"] is False
    assert bounds["feature_coverage"] == 11
    assert bounds["feature_coverage_max"] == 18
    assert bounds["out_of_taxonomy_groups"] == [] and bounds["out_of_taxonomy_features"] == []


def test_coverage_upper_bounds_multi_group_terminals_can_exceed_k():
    programs = [
        _rule("a", [{"pH:q0"}], [{"HR:q1", "RR:q1"}]),
        _rule("b", [{"pH:q0"}], [{"Lactate:q1", "Creatinine:q1"}]),
        _rule("c", [{"pH:q0"}], [{"WBC:q1", "Glucose:q1"}]),
    ]
    bounds = coverage_upper_bounds(programs)
    assert bounds["anchor_group_coverage"] == 6
    assert bounds["anchor_group_coverage_max"] == 6
    assert bounds["anchor_group_coverage_max_single_group_terminals"] == 3
    assert bounds["anchor_group_coverage_can_exceed_k"] is True
    assert bounds["n_multi_group_terminals"] == 3


def test_coverage_upper_bounds_edge_cases():
    empty = coverage_upper_bounds([])
    assert empty["k"] == 0
    assert empty["anchor_group_coverage_max"] == 0
    assert empty["feature_coverage_max"] == 0
    assert empty["anchor_group_coverage_can_exceed_k"] is False
    # Features outside the shared taxonomy are listed and widen the universe.
    gcs = _rule("gcs", [{"HR:q1"}], [{"GCS:q1"}])
    bounds = coverage_upper_bounds([gcs], n_groups=1, n_features=1)
    assert bounds["out_of_taxonomy_groups"] == ["other:GCS"]
    assert bounds["out_of_taxonomy_features"] == ["GCS"]
    assert bounds["anchor_group_coverage"] <= bounds["anchor_group_coverage_max"]
    assert bounds["feature_coverage"] <= bounds["feature_coverage_max"]


def test_schema_survival_uses_all_selected_programs_as_denominator():
    motif = _motif("m1", "Lactate", [0.0, 1.0], 1.0)
    out = schema_survival([CR_RR, LAC_RR, HR_MAP, motif], ALL_FEATURES - {"Lactate"})
    assert out == {"value": 0.5, "surviving_ids": ["cr_rr", "hr_map"], "denominator": 4}
    assert schema_survival([CR_RR], ALL_FEATURES)["value"] == 1.0
    assert schema_survival([], ALL_FEATURES) == {"value": 0.0, "surviving_ids": [], "denominator": 0}


def test_firing_survival_counts_programs_firing_on_any_target_episode():
    motif = _motif("m1", "Lactate", [0.0, 1.0], 1.0)
    external = {"cr_rr": frozenset({"t1"}), "lac_rr": frozenset(), "m1": {"t2", "t3"}}  # hr_map missing
    out = firing_survival([CR_RR, LAC_RR, HR_MAP, motif], external)
    assert out == {"value": 0.5, "firing_ids": ["cr_rr", "m1"], "denominator": 4}
    assert firing_survival([], external) == {"value": 0.0, "firing_ids": [], "denominator": 0}


def test_summarize_program_set_uses_top_k_in_given_order():
    dup = _rule("cr_rr_b", [{"Creatinine:sparse_measurement"}], [{"RR:level_0"}])
    programs = [CR_RR, LAC_RR, HR_MAP, dup]
    firings = dict(FIRINGS, cr_rr_b=frozenset({"e1", "e2"}))

    top2 = summarize_program_set(programs, firings, k=2)
    assert top2["program_ids"] == ["cr_rr", "lac_rr"]
    assert top2["n_programs"] == 2 and top2["n_pairs"] == 1
    assert top2["redundancy"] == 1.0
    assert top2["redundancy_components"]["feature_jaccard_mean"] == pytest.approx(1 / 3)
    assert top2["redundancy_components"]["anchor_jaccard_mean"] == 1.0
    assert top2["exact_duplicate_ratio"] == 0.0
    assert top2["anchor_group_coverage"] == 1
    assert top2["feature_coverage"] == 3

    full = summarize_program_set(programs, firings)
    assert full["k"] is None and full["n_programs"] == 4
    assert full["redundancy"] == pytest.approx(set_redundancy(programs, firings))
    assert full["exact_duplicate_ratio"] == 0.25 and full["n_distinct_programs"] == 3
    assert full["coverage_upper_bounds"] == coverage_upper_bounds(programs)

    short = summarize_program_set(programs, firings, k=10)
    assert short["k"] == 10 and short["n_programs"] == 4
    assert summarize_program_set(programs, firings, k=0)["n_programs"] == 0
    with pytest.raises(ValueError):
        summarize_program_set(programs, firings, k=-1)


def test_summarize_program_set_reports_targets_and_never_firing_programs():
    programs = [CR_RR, LAC_RR, HR_MAP]
    summary = summarize_program_set(
        programs,
        {"cr_rr": frozenset({"e1"})},
        target_schemas={"full": ALL_FEATURES, "no_lactate": ALL_FEATURES - {"Lactate"}},
        external_firings={"no_lactate": {"cr_rr": frozenset({"t1"}), "lac_rr": frozenset({"t2"})}},
    )
    assert summary["n_never_fire_train"] == 2
    assert summary["never_fire_train_ids"] == ["lac_rr", "hr_map"]
    assert summary["schema_survival"]["full"]["value"] == 1.0
    assert summary["schema_survival"]["no_lactate"]["surviving_ids"] == ["cr_rr", "hr_map"]
    assert summary["firing_survival"]["no_lactate"]["value"] == pytest.approx(2 / 3)
    # lac_rr fires on a target that has no Lactate: flagged as target-specific repair.
    assert summary["firing_without_schema_survival"] == {"no_lactate": ["lac_rr"]}
    assert set(summary["definitions"]) == set(DEFINITIONS)
    assert "max{J(sigma_f(p), sigma_f(q))" in summary["definitions"]["pairwise_redundancy"]
    json.dumps(summary)  # must stay serializable for manifests


def test_metrics_are_identical_across_methods_and_ids():
    def build(method: str, prefix: str) -> tuple[list[ProgramRecord], dict]:
        programs = [
            _rule(f"{prefix}1", [{"Creatinine:sparse_measurement"}], [{"RR:level_0"}], method=method),
            _rule(f"{prefix}2", [{"Lactate:q3"}], [{"RR:level_1"}], method=method),
            _rule(f"{prefix}3", [{"HR:q1"}], [{"MAP:q0"}], method=method),
        ]
        firings = {f"{prefix}1": frozenset({"e1", "e2"}), f"{prefix}2": frozenset({"e2"})}
        return programs, firings

    keys = ["redundancy", "redundancy_components", "exact_duplicate_ratio", "anchor_group_coverage", "feature_coverage", "coverage_upper_bounds"]
    fmg = summarize_program_set(*build("fmg_bridge", "program_00"), target_schemas={"t": ALL_FEATURES})
    base = summarize_program_set(*build("prefixspan", "prefixspan_support_0.2_pattern_"), target_schemas={"t": ALL_FEATURES})
    assert {key: fmg[key] for key in keys} == {key: base[key] for key in keys}
    assert fmg["schema_survival"]["t"]["value"] == base["schema_survival"]["t"]["value"]
