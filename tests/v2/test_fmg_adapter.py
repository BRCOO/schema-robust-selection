from __future__ import annotations

import hashlib
import importlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path

import pytest

from src.icu.shock_contrastive_miner import extract_rule_records
from src.icu.v2 import fmg_adapter as fa
from src.icu.v2.types import ProgramRecord

ROOT = Path(__file__).resolve().parents[2]

RULE_A = "{Lactate:up} => {MAP:level_0}"
RULE_A2 = "{Lactate:up} -> {Lactate:up} => {MAP:level_1}"  # same program key as RULE_A
RULE_B = "{Creatinine:sparse_measurement} => {RR:level_0}"
RULE_C = "{HR:up, SpO2:down} -> {Temp:level_2} => {Glucose:level_1}"  # multi-token braces
RULE_D = "{pH:down} => {WBC:up}"
RULE_E = "{Glucose:up} => {Bilirubin_Total:level_3}"
RULE_F = "{SBP:down} => {vasopressor_on:up}"  # care-process consequent
RULE_TEST_ONLY = "{Platelets:down} => {urine_output:level_0}"
RULE_MOSTLY_TEST = "{DBP:up} => {Temp:up}"  # like program_005: no train support
RULE_MISSING = "{HR:missing} => {RR:missing}"
RULE_BACKGROUND = "{Temp:level_1} => {HR:level_1}"
RULE_ZERO_CONTROL = "{RR:up} => {SpO2:level_0}"  # removed by min_rule_control_count=1
RULE_SPARSE = "{Creatinine:sparse_measurement} -> {Lactate:sparse_measurement} => {Glucose:missing}"  # ratio 2/3
LEAKY_RULES = {RULE_TEST_ONLY, RULE_MOSTLY_TEST}

ALL6 = (1, 2, 3, 4, 5, 6)
# rule -> (train positives, train controls, test positives, test controls), 1-based indices
MEMBERSHIP = {
    RULE_A: ((1, 2, 3, 4), (1,), (1,), (1, 2)),
    RULE_A2: ((1, 2), (2,), (), ()),
    RULE_B: ((1, 2, 5), (2,), (2,), (3,)),
    RULE_C: ((2, 3, 6), (3,), (), (1,)),
    RULE_D: ((3, 4, 5, 6), (4, 5), (3,), (4,)),
    RULE_E: ((1, 5), (6,), (4,), (5,)),
    RULE_F: ((1, 2, 3, 4, 5), (1,), (5,), (6,)),
    RULE_TEST_ONLY: ((), (), ALL6, (1,)),
    RULE_MOSTLY_TEST: ((1,), (), (1, 2, 3, 4, 5), (2,)),
    RULE_MISSING: (ALL6, ALL6, ALL6, ALL6),
    RULE_BACKGROUND: (ALL6, ALL6, ALL6, ALL6),
    RULE_ZERO_CONTROL: ((1, 2, 3), (), (1,), ()),
    RULE_SPARSE: ((2, 4, 6), (5,), (2,), (3,)),
}
GROUPS = (("train", "positive"), ("train", "control"), ("test", "positive"), ("test", "control"))

# Forces the v1 autotune loop (rme:5373-5500) on the synthetic cohort.
STRICT_GATES = {
    "algorithm_ready_top5_unique_groups_min": 9,
    "algorithm_ready_top10_unique_groups_min": 12,
    "algorithm_ready_top5_score_min_threshold": 1.5,
}


def _labels(side: str) -> list[str]:
    return [token.strip() for group in re.findall(r"\{([^}]*)\}", side) for token in group.split(",")]


def _rule_record(rule_id: str, lead_time_steps: float = 3.0) -> dict:
    """A rule record with the fields of extract_rule_records (scm:396-428)."""
    antecedent_side, consequent_side = rule_id.split(" => ")
    antecedent, consequent = _labels(antecedent_side), _labels(consequent_side)
    labels = antecedent + consequent
    sparse = sum(":sparse_measurement" in label for label in labels)
    return {
        "rule_id": rule_id,
        "antecedent_features": sorted({label.split(":", 1)[0] for label in antecedent}),
        "consequent_features": sorted({label.split(":", 1)[0] for label in consequent}),
        "lead_time_steps": float(lead_time_steps),
        "resample_minutes": 60,
        "is_missing_only": all(":missing" in label for label in labels),
        "is_sparse_measurement_dominant": sparse * 2 >= len(labels),
        "sparse_measurement_ratio": sparse / len(labels),
    }


def _cohort() -> tuple[list[dict], list[str], list[str]]:
    episodes: list[dict] = []
    for slot, (split, kind) in enumerate(GROUPS):
        for index in ALL6:
            rules = [_rule_record(rule_id) for rule_id, members in MEMBERSHIP.items() if index in members[slot]]
            episodes.append({"episode_id": f"{split}_{kind}_{index}", "episode_type": kind, "rules": rules})
    train_ids = [episode["episode_id"] for episode in episodes if episode["episode_id"].startswith("train_")]
    test_ids = [episode["episode_id"] for episode in episodes if episode["episode_id"].startswith("test_")]
    return episodes, train_ids, test_ids


def _train_inputs() -> list[dict]:
    episodes, train_ids, _ = _cohort()
    return fa.filter_contrastive_inputs(episodes, train_ids)


def _rule_ids(program: dict) -> set[str]:
    return set(program.get("member_rule_ids", [])) | {program.get("representative_rule_id")}


def _canonical(programs: list[dict]) -> str:
    return json.dumps(programs, sort_keys=True, default=str)


def _assert_identical_to_v1(ours: fa.FmgMiningResult, reference: list[dict], work_dir: Path) -> None:
    """Programs and every selection diagnostic must equal v1's own summary (rme:5541-5559)."""
    assert _canonical(ours.programs) == _canonical(reference)
    summary = json.loads((work_dir / "v1_output" / "mimic_iv_summary.json").read_text(encoding="utf-8"))
    pairs = {
        "program_constrained_selection": ours.selector,
        "contrastive_qc": ours.qc,
        "algorithm_ready_gates": ours.gates,
        "program_score_calibration": ours.stages["program_score_calibration"],
        "program_support_confidence_adjustment": ours.stages["program_support_confidence_adjustment"],
        "program_preselect_coverage": ours.stages["program_preselect_coverage"],
        "rule_low_coverage_recovery": ours.stages["rule_low_coverage_recovery"],
        "contrastive_rule_candidate_pool": ours.stages["rule_candidate_pool"],
    }
    for key, value in pairs.items():
        assert _canonical(summary[key]) == _canonical(json.loads(json.dumps(value))), key
    assert len(summary["contrastive_rules"]) == ours.rule_count


def test_load_split_manifest_reads_v1_layout_rows_and_rejects_conflicts(tmp_path: Path) -> None:
    v1 = tmp_path / "v1.json"
    v1.write_text(json.dumps({"name": "x", "split": {"seed": 11, "train": ["a", "b"], "val": [], "test": ["c"]}}))
    assert fa.load_split_manifest(v1) == {"a": "train", "b": "train", "c": "test"}

    rows = tmp_path / "rows.json"
    rows.write_text(json.dumps([{"episode_id": "a", "split": "train"}, {"episode_id": "c", "split": "test"}]))
    assert fa.load_split_manifest(rows) == {"a": "train", "c": "test"}

    conflict = tmp_path / "conflict.json"
    conflict.write_text(json.dumps({"split": {"train": ["a"], "val": [], "test": ["a"]}}))
    with pytest.raises(ValueError, match="both"):
        fa.load_split_manifest(conflict)
    no_train = tmp_path / "no_train.json"
    no_train.write_text(json.dumps({"split": {"test": ["a"]}}))
    with pytest.raises(ValueError, match="split.train"):
        fa.load_split_manifest(no_train)

    real = fa.load_split_manifest(ROOT / "docs/experiments/icde_comparator_split_manifest.json")
    assert sum(name == "train" for name in real.values()) == 256
    assert sum(name == "test" for name in real.values()) == 100


def test_filter_contrastive_inputs_keeps_order_and_rejects_duplicates() -> None:
    episodes, train_ids, _ = _cohort()
    kept = fa.filter_contrastive_inputs(episodes, set(train_ids))
    assert [episode["episode_id"] for episode in kept] == train_ids
    assert kept[0] is episodes[0]
    assert fa.filter_contrastive_inputs(episodes, []) == []
    with pytest.raises(ValueError, match="duplicate"):
        fa.filter_contrastive_inputs(episodes + episodes[:1], train_ids)
    with pytest.raises(TypeError):
        fa.filter_contrastive_inputs(episodes, "train_positive_1")


def test_iter_json_array_streams_hashes_and_validates(tmp_path: Path) -> None:
    payload = [{"episode_id": "é1", "rules": [{"rule_id": RULE_C}]}, 12345, [1.5, None], "x, y", {}]
    raw = b"\xef\xbb\xbf \n" + json.dumps(payload, ensure_ascii=False, indent=1).encode("utf-8") + b"\n "
    path = tmp_path / "inputs.json"
    path.write_bytes(raw)
    for chunk_size in (1, 2, 7, 1 << 20):
        hasher = hashlib.sha256()
        assert list(fa.iter_json_array(path, chunk_size=chunk_size, hasher=hasher)) == payload
        assert hasher.hexdigest() == hashlib.sha256(raw).hexdigest()

    numbers = tmp_path / "numbers.json"
    numbers.write_text("[12345,6789]")
    assert list(fa.iter_json_array(numbers, chunk_size=2)) == [12345, 6789]
    empty = tmp_path / "empty.json"
    empty.write_text(" [ ] ")
    assert list(fa.iter_json_array(empty)) == []
    for bad in ("", "{}", "[1 2]", "[1,", "[1] x"):
        path.write_text(bad)
        with pytest.raises(ValueError):
            list(fa.iter_json_array(path, chunk_size=3))


@dataclass(frozen=True)
class _FakeMotif:
    itemsets: tuple[tuple[int, ...], ...]


@dataclass(frozen=True)
class _FakeRule:
    antecedent: _FakeMotif
    consequent: _FakeMotif
    delta_min: int = 1
    delta_max: int = 3


def _v1_rule_id(antecedent: tuple[tuple[int, ...], ...], consequent: tuple[tuple[int, ...], ...]) -> str:
    labels = {1: "HR:up", 2: "SpO2:down", 3: "Temp:level_2", 4: "Glucose:level_1", 5: "WBC:up"}
    rule = _FakeRule(_FakeMotif(antecedent), _FakeMotif(consequent))
    records = extract_rule_records([rule], event_labels=labels, feature_names=["HR", "SpO2", "Temp"], n_bins=4)
    return records[0]["rule_id"]


def test_parse_fmg_rule_id_round_trips_multi_token_braces() -> None:
    rule_id = _v1_rule_id(((1, 2), (3,)), ((4, 5),))  # produced by v1 extract_rule_records
    assert rule_id == "{HR:up, SpO2:down} -> {Temp:level_2} => {Glucose:level_1, WBC:up}"
    antecedent, consequent = fa.parse_fmg_rule_id(rule_id)
    assert antecedent == (frozenset({"HR:up", "SpO2:down"}), frozenset({"Temp:level_2"}))
    assert consequent == (frozenset({"Glucose:level_1", "WBC:up"}),)
    assert fa.format_fmg_rule_id(antecedent, consequent) == rule_id

    unsorted = _v1_rule_id(((2, 1), (3,)), ((5, 4),))  # v1 keeps event-id order inside braces
    assert unsorted == "{SpO2:down, HR:up} -> {Temp:level_2} => {WBC:up, Glucose:level_1}"
    assert fa.parse_fmg_rule_id(unsorted) == (antecedent, consequent)
    assert fa.parse_fmg_rule_id(fa.format_fmg_rule_id(*fa.parse_fmg_rule_id(unsorted))) == (antecedent, consequent)

    locked = "{Lactate:sparse_measurement} -> {Lactate:sparse_measurement} -> {Creatinine:sparse_measurement} => {Glucose:missing}"
    parsed = fa.parse_fmg_rule_id(locked)
    assert len(parsed[0]) == 3 and parsed[0][0] == parsed[0][1]
    assert fa.format_fmg_rule_id(*parsed) == locked

    for bad in (
        "{A:x} -> {B:y}",
        "{A:x} => {B:y} => {C:z}",
        "{A:x, } => {B:y}",
        "{A:x} => ",
        "A:x => {B:y}",
        "{A:x, A:x} => {B:y}",
        "{A:x} =>{B:y}",
        "{A:x {B:y}} => {C:z}",
    ):
        with pytest.raises(ValueError):
            fa.parse_fmg_rule_id(bad)
    with pytest.raises(ValueError):
        fa.format_fmg_rule_id((), (frozenset({"A:x"}),))


def _locked_like_program() -> dict:
    # Shape of the locked program_001 (contract_quality_complete_v1 raw.fmg_bridge.programs).
    return {
        "program_id": "program_001",
        "program_label": "Creatinine -> RR",
        "antecedent_features": ["Creatinine"],
        "consequent_features": ["RR"],
        "mechanism_groups": ["respiratory", "respiratory::RR"],
        "representative_rule_id": "{Creatinine:sparse_measurement} => {RR:level_0}",
        "member_rule_ids": [
            "{Creatinine:sparse_measurement} => {RR:level_0}",
            "{Creatinine:sparse_measurement} => {RR:level_3}",
            "{Creatinine:sparse_measurement} => {RR:up}",
        ],
        "representative_score": 3.297604088288461,
        "representative_score_support": {
            "positive_count": 10,
            "control_count": 1,
            "positive_total": 60,
            "control_total": 59,
            "support_ratio": 0.09243697478991597,
        },
        "lead_time_hours_mean": 5.5,
        "lead_time_steps_mean": 5.5,
        "rule_count": 28,
    }


def test_fmg_programs_to_records_keeps_v1_fields_and_validates() -> None:
    (record,) = fa.fmg_programs_to_records([_locked_like_program()])
    assert isinstance(record, ProgramRecord)
    assert (record.program_id, record.method, record.kind) == ("program_001", "fmg_bridge", "ordered_rule")
    assert record.elements == (frozenset({"Creatinine:sparse_measurement"}),)
    assert record.consequent == (frozenset({"RR:level_0"}),)
    assert record.features == frozenset({"Creatinine", "RR"})
    assert record.anchor_signature == frozenset({"respiratory"})
    assert record.score == pytest.approx(3.297604088288461)
    assert record.params["member_rule_ids"][1] == "{Creatinine:sparse_measurement} => {RR:level_3}"
    assert record.params["supports"]["positive_total"] == 60
    assert record.params["lead_time_hours_mean"] == 5.5
    assert record.params["v1_mechanism_groups"] == ("respiratory", "respiratory::RR")
    as_json = fa.program_record_to_json(record)
    assert json.loads(json.dumps(as_json))["consequent"] == [["RR:level_0"]]

    (unnamed,) = fa.fmg_programs_to_records([{k: v for k, v in _locked_like_program().items() if k != "program_id"}])
    assert unnamed.program_id == "fmg_program_001"  # v1 fallback naming (run_contract_quality_complete:259)
    bad_member = dict(_locked_like_program(), member_rule_ids=["{Creatinine:up} => {HR:up}"])
    bad_features = dict(_locked_like_program(), consequent_features=["HR"])
    no_representative = dict(_locked_like_program(), representative_rule_id="")
    for program in (bad_member, bad_features, no_representative):
        with pytest.raises(ValueError):
            fa.fmg_programs_to_records([program])
    with pytest.raises(ValueError, match="duplicate"):
        fa.fmg_programs_to_records([_locked_like_program(), _locked_like_program()])


def test_firing_sets_match_v1_contract_quality_semantics() -> None:
    rule_c_variant = "{HR:up, SpO2:down} -> {Temp:level_3} => {Glucose:level_0}"
    programs = [
        {"program_id": "p1", "representative_rule_id": RULE_A, "member_rule_ids": [RULE_A, RULE_A2]},
        {"program_id": "p2", "representative_rule_id": RULE_B, "member_rule_ids": []},
        {"program_id": "p3", "representative_rule_id": RULE_C, "member_rule_ids": [rule_c_variant]},
        {"program_id": "p4", "representative_rule_id": RULE_E, "member_rule_ids": [RULE_E]},
    ]
    missing_flagged = dict(_rule_record(RULE_A2), is_missing_only=True)  # v1 firing ignores rule flags
    inputs = [
        {"episode_id": "e1", "episode_type": "positive", "rules": [missing_flagged]},
        {"episode_id": "e2", "episode_type": "control", "rules": [_rule_record(RULE_B), _rule_record(RULE_D)]},
        {"episode_id": "e3", "episode_type": "positive", "rules": [_rule_record(RULE_C)]},
        {"episode_id": "e4", "episode_type": "positive", "rules": [_rule_record(RULE_D)]},
        {"episode_id": "e5", "episode_type": "control", "rules": []},
        {"episode_id": "e6", "episode_type": "control", "rules": [_rule_record(rule_c_variant)]},
    ]
    records = fa.fmg_programs_to_records(programs)
    ours = fa.fmg_firing_sets(records, {episode["episode_id"]: episode for episode in inputs})
    assert ours == {
        "p1": frozenset({"e1"}),
        "p2": frozenset({"e2"}),
        "p3": frozenset({"e3", "e6"}),
        "p4": frozenset(),
    }

    v1 = importlib.import_module("scripts.run_contract_quality_complete")
    by_program: dict[str, set[str]] = {program["program_id"]: set() for program in programs}
    for episode_id, row in v1._fmg_firings_for_inputs(inputs, programs).items():
        for program_id in row:
            by_program[program_id].add(episode_id)
    assert ours == {program_id: frozenset(ids) for program_id, ids in by_program.items()}

    bare = ProgramRecord("x", "fmg_bridge", "ordered_rule", (frozenset({"A:x"}),), (frozenset({"B:y"}),))
    with pytest.raises(ValueError, match="no v1 rule ids"):
        fa.fmg_firing_sets([bare], {})


def test_train_only_mining_never_selects_test_only_rules() -> None:
    episodes, train_ids, _ = _cohort()
    profile = fa.v1_locked_reference_profile()

    on_all = fa.mine_fmg_programs(episodes, profile)
    selected_on_all = set().union(*(_rule_ids(program) for program in on_all[:10]))
    assert LEAKY_RULES <= selected_on_all  # the v1 setting does pick them

    on_train = fa.mine_fmg_programs(fa.filter_contrastive_inputs(episodes, train_ids), profile)
    assert on_train
    for program in on_train:
        assert not _rule_ids(program) & (LEAKY_RULES | {RULE_ZERO_CONTROL, RULE_MISSING})
        assert program["program_label"] not in {"Platelets -> urine_output", "DBP -> Temp"}

    train_inputs = fa.filter_contrastive_inputs(episodes, train_ids)
    firings = fa.fmg_firing_sets(
        fa.fmg_programs_to_records(on_train), {episode["episode_id"]: episode for episode in train_inputs}
    )
    for program_id, episode_ids in firings.items():  # unlike program_005, every program fires on train
        assert sum(episode_id.startswith("train_positive") for episode_id in episode_ids) >= 2, program_id


def test_support_totals_equal_train_counts() -> None:
    episodes, train_ids, _ = _cohort()
    profile = fa.v1_locked_reference_profile()
    train_inputs = fa.filter_contrastive_inputs(episodes, train_ids)
    for program in fa.mine_fmg_programs(train_inputs, profile):
        support = program["representative_score_support"]
        assert (support["positive_total"], support["control_total"]) == (6, 6)
        representative = program["representative_rule_id"]
        holders = [ep for ep in train_inputs if representative in {rule["rule_id"] for rule in ep["rules"]}]
        assert support["positive_count"] == sum(episode["episode_type"] == "positive" for episode in holders)
        assert support["control_count"] == sum(episode["episode_type"] == "control" for episode in holders)
    for program in fa.mine_fmg_programs(episodes, profile):
        support = program["representative_score_support"]
        assert (support["positive_total"], support["control_total"]) == (12, 12)


def test_guard_raises_on_contaminated_supports() -> None:
    episodes, train_ids, _ = _cohort()
    profile = fa.v1_locked_reference_profile()
    train_inputs = fa.filter_contrastive_inputs(episodes, train_ids)
    on_train = fa.mine_fmg_programs(train_inputs, profile)
    fa.assert_train_only_supports(on_train, train_ids, episodes)  # passes on any superset of inputs
    fa.assert_train_only_supports(fa.fmg_programs_to_records(on_train), train_ids, train_inputs)

    with pytest.raises(fa.TrainSupportViolation, match="exceed train totals"):
        fa.assert_train_only_supports(fa.mine_fmg_programs(episodes, profile), train_ids, episodes)

    inflated = json.loads(json.dumps(on_train[0]))
    inflated["representative_score_support"]["positive_count"] += 1  # totals still look like train
    with pytest.raises(fa.TrainSupportViolation, match="exceeds the train support"):
        fa.assert_train_only_supports([inflated], train_ids, episodes)
    leaky = dict(inflated, representative_rule_id=RULE_MOSTLY_TEST)
    leaky["representative_score_support"] = dict(inflated["representative_score_support"], positive_count=6, control_count=1)
    with pytest.raises(fa.TrainSupportViolation, match=re.escape(RULE_MOSTLY_TEST)):
        fa.assert_train_only_supports([leaky], train_ids, episodes)
    with pytest.raises(fa.TrainSupportViolation, match="no representative rule or support"):
        fa.assert_train_only_supports([{"program_id": "p", "representative_rule_id": RULE_A}], train_ids, episodes)
    with pytest.raises(TypeError):
        fa.assert_train_only_supports(on_train, "train_positive_1", episodes)


def test_mining_matches_v1_entrypoint_including_autotune_and_transfer_target(tmp_path: Path) -> None:
    train_inputs = _train_inputs()
    locked = fa.v1_locked_reference_profile()
    strict = dict(locked, **STRICT_GATES)
    # The plain locked JSON leaves program_topk_care_process_max_count at -1, so the
    # locked-gate fallback of rme:4645-4652 applies (the formal profile sets it to 0).
    locked_json = json.loads((ROOT / "docs/experiments/strict_profile_icde_locked.json").read_text(encoding="utf-8-sig"))
    assert "program_topk_care_process_max_count" not in locked_json

    first = fa.run_fmg_contrastive_stage(train_inputs, strict)
    second_config = fa.transfer_target_config(strict, first)
    second = fa.run_fmg_contrastive_stage(train_inputs, second_config)
    cases = {
        "locked": (locked, fa.run_fmg_contrastive_stage(train_inputs, locked)),
        "locked_json": (locked_json, fa.run_fmg_contrastive_stage(train_inputs, locked_json)),
        "strict_gates": (strict, first),
        "transfer_target": (second_config, second),
    }
    for name, (config, ours) in cases.items():
        reference = fa.mine_fmg_programs_via_v1_entrypoint(train_inputs, config, tmp_path / name)
        _assert_identical_to_v1(ours, reference, tmp_path / name)
        assert _canonical(fa.mine_fmg_programs(train_inputs, config)) == _canonical(reference), name

    assert first.selector["autotune"]["attempts"], "strict gates must exercise the autotune port"
    assert second.selector["autotune"]["attempts"]
    assert second.selector["autotune"]["final_transfer_target"]["enabled"] is True
    tolerances = fa.v1_transfer_tolerances()
    assert second_config["transfer_baseline_top5_unique_groups"] == first.qc["mechanism_diversity"]["top5_unique_groups"]
    assert second_config["transfer_max_drop_top10_unique_groups"] == tolerances["transfer_max_drop_top10_unique_groups"]
    disabled = fa.without_transfer_target(second_config)
    assert disabled["transfer_baseline_top5_unique_groups"] == -1 and math.isnan(disabled["transfer_baseline_top5_score_min"])


def _wide_train_cohort() -> list[dict]:
    """10 high-score hemodynamic programs (2 per anchor, the locked same-anchor cap) and 5
    diverse programs scored just below the 0.85 constrained-selection quality floor."""
    membership: dict[str, tuple[tuple[int, ...], tuple[int, ...]]] = {}
    for k, (anchor, consequent) in enumerate(
        ((a, c) for a in ("Lactate", "pH", "Creatinine", "Glucose", "SpO2") for c in ("HR", "SBP")), start=1
    ):
        membership[f"{{{anchor}:up}} => {{{consequent}:level_0}}"] = (ALL6 if k % 2 else ALL6[:5], ((k % 6) + 1,))
    diverse = (
        ("WBC", "RR"),
        ("Platelets", "Temp"),
        ("Bilirubin_Total", "urine_output"),
        ("urine_output", "Bilirubin_Total"),
        ("Temp", "WBC"),
    )
    for j, (anchor, consequent) in enumerate(diverse):
        positives = (1, 2, 3, 4) if j % 2 else (3, 4, 5, 6)
        membership[f"{{{anchor}:down}} => {{{consequent}:level_3}}"] = (positives, ((j % 6) + 1,))
    return [
        {
            "episode_id": f"train_{kind}_{index}",
            "episode_type": kind,
            "rules": [_rule_record(rule_id) for rule_id, members in membership.items() if index in members[slot]],
        }
        for slot, kind in enumerate(("positive", "control"))
        for index in ALL6
    ]


def test_autotune_accept_branch_matches_v1_entrypoint(tmp_path: Path) -> None:
    # An unreachable transfer baseline makes v1 try relax_quality_floor_for_top10 and accept it.
    forced = dict(
        fa.v1_locked_reference_profile(),
        transfer_baseline_top5_unique_groups=99,
        transfer_baseline_top10_unique_groups=99,
        transfer_baseline_top5_score_min=0.0,
        transfer_max_drop_top5_unique_groups=0,
        transfer_max_drop_top10_unique_groups=0,
        transfer_max_drop_top5_score_min=0.0,
    )
    inputs = _wide_train_cohort()
    ours = fa.run_fmg_contrastive_stage(inputs, forced)
    rounds = {attempt["round"] for attempt in ours.selector["autotune"]["attempts"]}
    assert rounds == {1, 2}, "round 2 only runs after a round-1 trial was accepted"
    reference = fa.mine_fmg_programs_via_v1_entrypoint(inputs, forced, tmp_path / "v1")
    _assert_identical_to_v1(ours, reference, tmp_path / "v1")


def test_resolve_v1_profile_rejects_unknown_and_non_train_keys() -> None:
    resolved = fa.resolve_v1_profile({"min_rule_positive_count": 3, "min_episode_timesteps": 1})
    assert resolved["min_rule_positive_count"] == 3
    assert resolved["sparse_measurement_penalty"] == 0.35  # v1 argparse default
    assert resolved["upstream_only_keys_without_effect"] == ["min_episode_timesteps"]
    assert fa.resolve_v1_profile({"hard_exclude_sparse_measurement_dominant": True})["exclude_sparse_measurement_dominant"]
    assert math.isnan(fa.resolve_v1_profile({"transfer_baseline_top5_score_min": None})["transfer_baseline_top5_score_min"])
    for bad in (
        {"no_such_key": 1},
        {"exclude_sparse_measurement_dominant": True},  # not an argparse key: v1 would silently ignore it
        {"limit": 120},
        {"stability_cache_glob": "artifacts/*/mimic_iv_summary.json"},
        {"shadow_predictive_blend_weight": 0.2},
        {"input": "data/processed/mimic-iv"},
    ):
        with pytest.raises(ValueError):
            fa.resolve_v1_profile(bad)
    with pytest.raises(TypeError):
        fa.resolve_v1_profile([("min_rule_positive_count", 2)])
    with pytest.raises(ValueError, match="no training inputs"):
        fa.mine_fmg_programs([], {})


def test_cli_help_and_synthetic_end_to_end(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    cli = importlib.import_module("scripts.v2.run_fmg_train_only")
    with pytest.raises(SystemExit) as exit_info:
        cli.main(["--help"])
    assert exit_info.value.code == 0
    assert "--contrastive-inputs" in capsys.readouterr().out

    episodes, train_ids, test_ids = _cohort()
    inputs_path = tmp_path / "contrastive_inputs.json"
    inputs_path.write_text(json.dumps(episodes), encoding="utf-8")
    manifest_path = tmp_path / "split.json"
    manifest_path.write_text(json.dumps({"split": {"seed": 1, "train": train_ids, "val": [], "test": test_ids}}))
    out = tmp_path / "out"
    argv = [
        "--contrastive-inputs", str(inputs_path),
        "--split-manifest", str(manifest_path),
        "--output-dir", str(out),
        "--top-k", "4",
        "--verify-v1-parity",
    ]  # fmt: skip
    with pytest.raises(SystemExit):
        cli.main(argv[:-1] + ["--top-k", "0"])
    provenance = cli.main(argv)

    assert provenance["inputs"]["contrastive_inputs"]["sha256"] == hashlib.sha256(inputs_path.read_bytes()).hexdigest()
    assert provenance["mining_episodes"]["count"] == 12
    assert (provenance["mining_episodes"]["positive"], provenance["mining_episodes"]["control"]) == (6, 6)
    assert provenance["episode_counts_by_split"]["test"] == {"positive": 6, "control": 6}
    assert provenance["guards"]["train_only_supports"] == "passed"
    assert provenance["v1_parity"]["identical"] is True
    assert provenance["config"]["transfer_target"]["mode"] == "two-pass"

    ranked = json.loads((out / "fmg_programs_ranked.json").read_text(encoding="utf-8"))
    selected = json.loads((out / "fmg_programs_selected.json").read_text(encoding="utf-8"))
    records = json.loads((out / "fmg_program_records.json").read_text(encoding="utf-8"))
    firings = json.loads((out / "fmg_train_firings.json").read_text(encoding="utf-8"))
    assert selected == ranked[:4] and [row["program_id"] for row in records] == provenance["result"]["selected_program_ids"]
    assert not set().union(*(_rule_ids(program) for program in ranked)) & LEAKY_RULES
    assert all(episode_id in train_ids for ids in firings.values() for episode_id in ids)

    effective = json.loads((out / "fmg_profile_effective.json").read_text(encoding="utf-8"))
    in_memory = fa.mine_fmg_programs(fa.filter_contrastive_inputs(episodes, train_ids), effective)
    assert cli._json_ready(in_memory) == ranked
