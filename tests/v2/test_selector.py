from __future__ import annotations

import math
import random
from itertools import combinations

import pytest

from src.icu.shock_contrastive_miner import MECHANISM_GROUP_BY_FEATURE
from src.icu.v2.selector import (
    CARE_PROCESS_FEATURES,
    TERMS,
    SelectorConfig,
    consolidate,
    consolidation_key,
    program_family,
    select,
    terminal_features,
)
from src.icu.v2.types import ProgramRecord, coarse_group, make_symbolic_program

ALL_FEATURES = frozenset(MECHANISM_GROUP_BY_FEATURE)
CARE = frozenset(CARE_PROCESS_FEATURES)
# Isolates one behavior: no repair gate and no positional care cap unless a test sets them.
PLAIN = {"min_anchor_coverage": None, "care_process_cap": None}
UTILITY_ONLY = {**PLAIN, "w_family": 0.0, "w_anchor": 0.0, "w_schema": 0.0, "w_care": 0.0}


def _rule(pid: str, antecedent: list[str], consequent: list[str], score: float, method: str = "fmg", lead=None):
    params = {} if lead is None else {"lead_time_hours_mean": lead}
    return make_symbolic_program(
        pid,
        method,
        "ordered_rule",
        tuple(frozenset({token}) for token in antecedent),
        tuple(frozenset({token}) for token in consequent),
        score=score,
        params=params,
    )


def _feature_jaccard(p: ProgramRecord, q: ProgramRecord) -> float:
    """Stub redundancy so the tests do not depend on program_metrics."""
    union = p.features | q.features
    return len(p.features & q.features) / len(union) if union else 0.0


POOL_SPEC = [
    ("HR:q3", "vasopressor_on:up", 3.5),  # care-process terminal, highest score
    ("Lactate:q3", "MAP:q0", 3.0),
    ("Creatinine:sparse_measurement", "RR:level_0", 2.9),
    ("HR:q2", "SpO2:q0", 2.8),
    ("pH:q0", "MAP:q1", 2.7),
    ("WBC:q3", "Temp:q2", 2.6),
    ("Glucose:q3", "Lactate:q2", 2.5),
    ("Platelets:q0", "Bilirubin_Total:q3", 2.4),
    ("urine_output:q0", "Creatinine:q3", 2.3),
    ("SBP:q0", "HR:q3", 2.2),
    ("DBP:q0", "RR:q3", 2.1),
    ("Temp:q3", "WBC:q3", 2.0),
    ("MAP:q0", "urine_output:q0", 1.9),
    ("RR:q3", "pH:q0", 1.8),
    ("Bilirubin_Total:q3", "Platelets:q0", 1.7),
    ("SpO2:q0", "Glucose:q3", 1.6),
    ("Lactate:q2", "ventilation_on:up", 1.5),  # care-process terminal
    ("sedation_on:on", "MAP:q2", 1.4),  # care feature only in the antecedent
]


def _pool() -> list[ProgramRecord]:
    return [_rule(f"p{i:02d}", [a], [c], s, lead=float(i % 7)) for i, (a, c, s) in enumerate(POOL_SPEC)]


def _ids(programs) -> list[str]:
    return [program.program_id for program in programs]


def _phase(trace, phase):
    return [record for record in trace if record["phase"] == phase]


def test_default_config_mirrors_the_documented_v1_locked_values():
    config = SelectorConfig()
    assert (config.k, config.w_utility, config.w_family, config.w_anchor) == (10, 1.0, 0.8, 0.3)
    assert (config.w_schema, config.w_lead, config.w_redundancy, config.w_care) == (0.35, 0.0, 0.0, 0.1)
    assert (config.max_per_anchor_group, config.max_per_family) == (2, None)
    assert (config.care_process_cap, config.care_process_cap_window) == (0, 5)
    assert (config.pool_size, config.pool_per_anchor_group, config.pool_per_family) == (20, 3, 3)
    assert (config.min_anchor_coverage, config.min_family_coverage, config.max_redundancy) == (6, None, None)
    assert (config.max_repair_swaps, config.lead_gap_hours) == (25, 1.0)
    assert set(config.care_process_features) == {f for f, g in MECHANISM_GROUP_BY_FEATURE.items() if g == "care_process"}


@pytest.mark.parametrize(
    "kwargs",
    [
        {"k": 0},
        {"k": True},
        {"w_family": -0.1},
        {"w_utility": float("nan")},
        {"max_per_anchor_group": 0},
        {"pool_size": 0},
        {"pool_per_family": -1},
        {"care_process_cap": -1},
        {"care_process_cap_window": 0},
        {"lead_gap_hours": 0.0},
        {"max_redundancy": -1.0},
        {"care_process_features": "vasopressor_on"},
    ],
)
def test_selector_config_rejects_invalid_values(kwargs):
    with pytest.raises(ValueError):
        SelectorConfig(**kwargs)


def test_family_and_terminal_features_are_generator_agnostic():
    rule = _rule("r", ["Lactate:q3"], ["MAP:q0"], 1.0)
    sequence = make_symbolic_program(
        "s", "prefixspan", "sequence", (frozenset({"HR:q1", "MAP:q0"}), frozenset({"vasopressor_on:1"}))
    )
    motif = ProgramRecord(
        program_id="m",
        method="matrix_profile",
        kind="motif",
        features=frozenset({"Lactate"}),
        anchor_signature=frozenset({coarse_group("Lactate")}),
        params={"feature": "Lactate", "template": [0.0, 1.0], "threshold": 0.5, "length": 2},
    )
    assert program_family(rule) == {"perfusion_acidbase", "hemodynamic"}
    assert terminal_features(rule) == {"MAP"}
    assert terminal_features(sequence) == {"vasopressor_on"}
    assert program_family(sequence) == {"hemodynamic", "care_process"}
    assert terminal_features(motif) == {"Lactate"} and program_family(motif) == {"perfusion_acidbase"}
    assert consolidation_key(rule) == (("Lactate", "MAP"), ("hemodynamic",), "fmg")


def test_consolidate_keeps_highest_score_per_signature_and_ignores_input_order():
    a1 = _rule("a1", ["Lactate:q3"], ["MAP:q0"], 1.0)
    a2 = _rule("a2", ["Lactate:q1"], ["MAP:q2"], 2.0)  # same sigma_f, sigma_a, method as a1
    a3 = _rule("a3", ["Lactate:q2", "MAP:q1"], ["MAP:q0"], 2.0)  # same group, ties a2 -> smaller id wins
    reverse = _rule("b", ["MAP:q0"], ["Lactate:q3"], 1.5)  # same sigma_f, other sigma_a -> kept
    other_method = _rule("a0", ["Lactate:q3"], ["MAP:q0"], 1.0, method="prefixspan")  # kept
    rows = [a1, a2, a3, reverse, other_method]
    out = consolidate(rows)
    assert _ids(out) == ["a2", "b", "a0"]
    assert _ids(consolidate(list(reversed(rows)))) == _ids(out)
    assert _ids(consolidate(out)) == _ids(out)
    assert consolidate([]) == []
    with pytest.raises(ValueError, match="non-finite"):
        consolidate([_rule("x", ["HR:q1"], ["RR:q1"], float("nan"))])
    with pytest.raises(ValueError, match="two different programs"):
        consolidate([_rule("x", ["HR:q1"], ["RR:q1"], 1.0), _rule("x", ["HR:q1"], ["SpO2:q1"], 1.0)])
    with pytest.raises(TypeError):
        consolidate([{"program_id": "x"}])


def test_select_respects_k_and_default_caps():
    for k in (1, 4, 10):
        registry, trace = select(_pool(), SelectorConfig(k=k), {}, {"eicu": ALL_FEATURES})
        assert len(registry) == k
        assert len(set(_ids(registry))) == k
        assert trace[-1]["n_selected"] == k
        anchors = [program.anchor_signature for program in registry]
        assert max(anchors.count(anchor) for anchor in anchors) <= 2
        assert not any(terminal_features(program) & CARE for program in registry[:5])
    with pytest.raises(TypeError):
        select(_pool(), {"k": 3}, {}, {})


def test_anchor_group_cap_is_hard_and_reported_when_binding():
    respiratory = [_rule(f"r{i}", [f"{f}:q1"], ["RR:level_0"], 5.0 - i * 0.1) for i, f in enumerate(["HR", "Lactate", "pH"])]
    others = [_rule("o1", ["HR:q2"], ["MAP:q0"], 1.0), _rule("o2", ["Glucose:q1"], ["Temp:q0"], 0.5)]
    config = SelectorConfig(k=4, max_per_anchor_group=2, **UTILITY_ONLY)
    registry, trace = select(respiratory + others, config, {}, {})
    assert _ids(registry) == ["r0", "r1", "o1", "o2"]
    third = _phase(trace, "greedy")[2]
    assert third["active_caps"] == ["anchor_group"]
    assert third["blocked_better"][0]["program_id"] == "r2"

    only_respiratory, trace = select(respiratory, config, {}, {})
    assert _ids(only_respiratory) == ["r0", "r1"]
    assert trace[-1]["stop_reason"] == "hard_caps_block_remaining"


def test_family_cap_is_hard():
    same_family = [
        _rule("lac_map", ["Lactate:q3"], ["MAP:q0"], 3.0),
        _rule("ph_map", ["pH:q0"], ["MAP:q1"], 2.9),
        _rule("map_lac", ["MAP:q0"], ["Lactate:q3"], 2.8),
    ]
    novel = _rule("cr_rr", ["Creatinine:q3"], ["RR:level_0"], 1.0)
    registry, trace = select(same_family + [novel], SelectorConfig(k=3, max_per_family=1, **UTILITY_ONLY), {}, {})
    assert _ids(registry) == ["lac_map", "cr_rr"]
    assert trace[-1]["stop_reason"] == "hard_caps_block_remaining"


def test_care_process_cap_blocks_only_inside_its_window():
    care = _rule("care", ["HR:q3"], ["vasopressor_on:up"], 5.0)
    antecedent_only = _rule("sed", ["sedation_on:on"], ["MAP:q2"], 4.0)  # not care-terminal
    rest = [_rule(f"x{i}", [f"{f}:q1"], [f"{c}:q0"], 3.0 - i * 0.1) for i, (f, c) in enumerate(
        [("Lactate", "RR"), ("Creatinine", "Temp"), ("WBC", "Glucose"), ("Platelets", "Bilirubin_Total")]
    )]
    config = SelectorConfig(k=4, care_process_cap_window=2, **{**UTILITY_ONLY, "care_process_cap": 0})
    registry, trace = select([care, antecedent_only] + rest, config, {}, {})
    assert _ids(registry) == ["sed", "x0", "care", "x1"]
    assert _phase(trace, "greedy")[0]["active_caps"] == ["care_process_window"]

    wide = SelectorConfig(k=4, care_process_cap_window=5, **{**UTILITY_ONLY, "care_process_cap": 0})
    registry, _ = select([care, antecedent_only] + rest, wide, {}, {})
    assert "care" not in _ids(registry)

    penalized = SelectorConfig(k=1, **{**PLAIN, "w_care": 0.1})
    _, trace = select([antecedent_only], penalized, {}, {})
    assert _phase(trace, "greedy")[0]["components"]["care_process"] == pytest.approx(0.5)


def test_select_is_deterministic_and_independent_of_input_order():
    config = SelectorConfig(k=8, w_redundancy=0.2, w_lead=0.1, min_anchor_coverage=7)
    targets = {"eicu": ALL_FEATURES - {"Bilirubin_Total"}}
    first, first_trace = select(_pool(), config, {}, targets, redundancy_fn=_feature_jaccard)
    second, second_trace = select(_pool(), config, {}, targets, redundancy_fn=_feature_jaccard)
    assert _ids(first) == _ids(second)
    assert first_trace == second_trace
    shuffled = _pool()
    random.Random(7).shuffle(shuffled)
    third, third_trace = select(shuffled, config, {}, targets, redundancy_fn=_feature_jaccard)
    assert _ids(third) == _ids(first)
    assert [r["gain"] for r in _phase(third_trace, "greedy")] == [r["gain"] for r in _phase(first_trace, "greedy")]


def test_prefers_novel_family_over_redundant_higher_utility_candidate_when_weighted():
    top = _rule("top", ["Lactate:q3"], ["MAP:q0"], 3.0)
    duplicate = _rule("dup", ["pH:q0"], ["MAP:q1"], 2.5)  # same family and sigma_a as `top`
    novel = _rule("novel", ["Creatinine:sparse_measurement"], ["RR:level_0"], 1.0)
    pool = [top, duplicate, novel]
    diverse = SelectorConfig(k=2, w_family=0.8, w_anchor=0.3, w_schema=0.0, w_redundancy=1.0, w_care=0.0, **PLAIN)
    registry, trace = select(pool, diverse, {}, {}, redundancy_fn=_feature_jaccard)
    assert _ids(registry) == ["top", "novel"]
    second = _phase(trace, "greedy")[1]
    assert second["new_family_groups"] == ["renal", "respiratory"]
    assert second["components"]["utility"] == pytest.approx(0.0)
    assert second["runner_up"]["program_id"] == "dup"
    assert second["runner_up"]["gain"] == pytest.approx(0.75 - 1.0 / 3.0)
    assert second["gain"] == pytest.approx(2 * 0.8 + 0.3)

    utility_only = SelectorConfig(k=2, **UTILITY_ONLY)
    registry, _ = select(pool, utility_only, {}, {}, redundancy_fn=_feature_jaccard)
    assert _ids(registry) == ["top", "dup"]


def test_trace_components_sum_to_reported_gains_and_gains_telescope_to_objective():
    config = SelectorConfig(
        k=6, w_utility=1.0, w_family=0.8, w_anchor=0.3, w_schema=0.35, w_lead=0.2, w_redundancy=0.25, w_care=0.1,
        min_anchor_coverage=None,
    )
    targets = {"eicu": ALL_FEATURES - {"Bilirubin_Total"}, "hirid": ALL_FEATURES - {"urine_output", "Platelets"}}
    train_firings = {"p01": frozenset({"t1"}), "p02": frozenset({"t2", "t3"})}

    def mapping_redundancy(p, q):
        return {"value": _feature_jaccard(p, q)}

    registry, trace = select(_pool(), config, train_firings, targets, redundancy_fn=mapping_redundancy)
    signed = {
        "utility": config.w_utility, "family_coverage": config.w_family, "anchor_coverage": config.w_anchor,
        "schema_survival": config.w_schema, "lead_time": config.w_lead,
        "redundancy": -config.w_redundancy, "care_process": -config.w_care,
    }
    greedy = _phase(trace, "greedy")
    assert [record["rank"] for record in greedy] == list(range(1, len(registry) + 1))
    for record in greedy:
        assert set(record["components"]) == set(TERMS) == set(record["weighted"])
        for term, value in record["components"].items():
            assert record["weighted"][term] == pytest.approx(signed[term] * value)
        assert record["gain"] == pytest.approx(math.fsum(record["weighted"].values()), abs=1e-12)
        assert record["objective_after"] - record["objective_before"] == pytest.approx(record["gain"], abs=1e-9)
    final = trace[-1]
    assert final["phase"] == "final" and final["n_repair_swaps"] == 0
    assert final["objective"] == pytest.approx(math.fsum(record["gain"] for record in greedy), abs=1e-9)
    assert final["objective"] == pytest.approx(math.fsum(final["weighted_terms"].values()), abs=1e-12)
    pairs = list(combinations(registry, 2))
    assert final["redundancy_mean"] == pytest.approx(sum(_feature_jaccard(p, q) for p, q in pairs) / len(pairs))
    assert final["terms"]["redundancy"] == pytest.approx(sum(_feature_jaccard(p, q) for p, q in pairs))
    assert set(final["selected_without_train_firing"]) == set(_ids(registry)) - {"p01", "p02"}
    assert set(final["nonsurvivable_by_target"]) == {"eicu", "hirid"}
    assert trace[0]["phase"] == "setup" and trace[0]["redundancy"] == "caller-supplied redundancy_fn"


def test_repair_swaps_restore_a_declared_anchor_coverage_gate():
    pool = [
        _rule("r1", ["HR:q1"], ["RR:level_0"], 5.0),
        _rule("r2", ["Lactate:q1"], ["SpO2:q0"], 4.9),
        _rule("r3", ["Creatinine:q1"], ["MAP:q0"], 4.8),
        _rule("r4", ["Glucose:q1"], ["HR:q0"], 4.7),
        _rule("r5", ["WBC:q1"], ["Temp:q0"], 1.0),
    ]
    config = SelectorConfig(k=4, max_per_anchor_group=2, **{**UTILITY_ONLY, "min_anchor_coverage": 3})
    registry, trace = select(pool, config, {}, {})
    assert _ids(registry) == ["r1", "r2", "r3", "r5"]
    (swap,) = _phase(trace, "repair")
    assert (swap["swap_out"], swap["swap_in"], swap["rank"]) == ("r4", "r5", 4)
    assert swap["delta"] == pytest.approx(-(4.7 - 1.0) / 4.0)
    final = trace[-1]
    assert final["gates"]["anchor_coverage"]["pass"] and final["repair_stop_reason"] == "gates_satisfied"
    assert final["objective"] == pytest.approx(final["sum_of_greedy_gains"] + final["sum_of_repair_deltas"])

    no_repair = SelectorConfig(k=4, max_per_anchor_group=2, max_repair_swaps=0, **{**UTILITY_ONLY, "min_anchor_coverage": 3})
    registry, trace = select(pool, no_repair, {}, {})
    assert _ids(registry) == ["r1", "r2", "r3", "r4"]
    assert trace[-1]["repair_stop_reason"] == "repair_disabled"
    assert not trace[-1]["gates"]["anchor_coverage"]["pass"]

    unreachable = SelectorConfig(k=4, **{**UTILITY_ONLY, "min_anchor_coverage": 9})
    _, trace = select(pool, unreachable, {}, {})
    assert trace[-1]["repair_stop_reason"] == "no_valid_improving_swap"


def test_schema_survival_term_prefers_survivable_programs():
    bili = _rule("bili", ["Bilirubin_Total:q3"], ["MAP:q0"], 2.0)
    lac = _rule("lac", ["Lactate:q3"], ["MAP:q1"], 2.0)
    target = {"eicu": ALL_FEATURES - {"Bilirubin_Total"}}
    with_schema = SelectorConfig(k=1, **{**PLAIN, "w_schema": 0.35})
    registry, trace = select([bili, lac], with_schema, {}, target)
    assert _ids(registry) == ["lac"]
    assert _phase(trace, "greedy")[0]["components"]["schema_survival"] == 1.0
    without_schema = SelectorConfig(k=1, **{**PLAIN, "w_schema": 0.0})
    registry, _ = select([bili, lac], without_schema, {}, target)
    assert _ids(registry) == ["bili"]  # exact tie: equal score, then the smaller program_id


def test_bounded_pool_views_admit_low_score_units_of_uncovered_groups():
    high = [
        _rule("h1", ["HR:q1"], ["RR:level_0"], 3.0),
        _rule("h2", ["HR:q2"], ["SpO2:q0"], 2.9),
        _rule("h3", ["SBP:q1"], ["RR:level_1"], 2.8),
    ]
    low = _rule("low", ["Creatinine:q3"], ["Temp:q0"], 0.1)
    config = SelectorConfig(k=3, pool_size=2, pool_per_anchor_group=1, pool_per_family=0, **PLAIN)
    registry, trace = select(high + [low], config, {}, {})
    setup = trace[0]
    assert setup["pool_views"]["quality"] == ["h1", "h2"]
    assert setup["pool_ids"] == ["h1", "h2", "low"]
    assert setup["utility_normalization"]["min"] == 0.1 and setup["utility_normalization"]["max"] == 3.0
    assert _ids(registry) == ["h1", "low", "h2"]

    narrow = SelectorConfig(k=3, pool_size=2, pool_per_anchor_group=0, pool_per_family=0, **PLAIN)
    registry, trace = select(high + [low], narrow, {}, {})
    assert _ids(registry) == ["h1", "h2"]
    assert trace[-1]["stop_reason"] == "pool_exhausted"


def test_redundancy_is_not_evaluated_when_its_weight_and_gate_are_off():
    def must_not_run(p, q):
        raise AssertionError("redundancy_fn must not be called")

    registry, trace = select(_pool(), SelectorConfig(k=4, **PLAIN), {}, {}, redundancy_fn=must_not_run)
    assert len(registry) == 4
    assert trace[0]["redundancy"].startswith("not evaluated")
    assert all(record["components"]["redundancy"] is None for record in _phase(trace, "greedy"))
    assert trace[-1]["redundancy_mean"] is None


def test_select_on_empty_candidates_returns_empty_registry():
    registry, trace = select([], SelectorConfig(), {}, {})
    assert registry == []
    assert [record["phase"] for record in trace] == ["setup", "final"]
    assert trace[-1]["stop_reason"] == "empty_pool" and trace[-1]["objective"] == 0.0


def test_schema_view_lets_the_schema_term_reach_low_scoring_executable_units():
    # four high-scoring lab programs fill the quality view; the executable vitals program scores low
    labs = [_rule(f"lab{i}", ["Lactate:q3"], [f"{f}:q0"], 3.0 - 0.1 * i) for i, f in enumerate(["pH", "Creatinine", "Glucose", "WBC"])]
    vital = _rule("vital", ["HR:q3"], ["MAP:q0"], 0.1)
    no_labs = {"no_labs": ALL_FEATURES - {"Lactate", "pH", "Creatinine", "Glucose", "WBC", "Platelets", "Bilirubin_Total"}}
    base = {**PLAIN, "pool_size": 2, "pool_per_anchor_group": 0, "pool_per_family": 0, "w_schema": 10.0}
    registry, trace = select(labs + [vital], SelectorConfig(k=1, **base), {}, no_labs)
    assert trace[0]["pool_views"]["schema"] == ["vital"]
    assert _ids(registry) == ["vital"]
    # without the schema view the executable program never enters the pool, whatever w_schema is
    registry, trace = select(labs + [vital], SelectorConfig(k=1, **{**base, "pool_per_schema": None}), {}, no_labs)
    assert trace[0]["pool_views"]["schema"] == [] and "vital" not in trace[0]["pool_ids"]
    assert _ids(registry) == ["lab0"]
    # no schema term: no schema view, so the pool is the one without schema information
    _, trace = select(labs + [vital], SelectorConfig(k=1, **{**base, "w_schema": 0.0}), {}, no_labs)
    assert trace[0]["pool_views"]["schema"] == []
    # no target schema: the view is empty
    _, trace = select(labs + [vital], SelectorConfig(k=1, **base), {}, {})
    assert trace[0]["pool_views"]["schema"] == []


def test_pool_per_schema_validation():
    with pytest.raises(ValueError):
        SelectorConfig(pool_per_schema=-1)
    assert SelectorConfig(pool_per_schema=None).pool_per_schema is None
