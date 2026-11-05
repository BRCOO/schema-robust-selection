from __future__ import annotations

import math
import random

import pytest

from src.icu.v2.patterns import (
    contrast_score,
    dedup_programs,
    firing_sets,
    fires,
    rank_by_contrast,
)
from src.icu.v2.types import ProgramRecord, make_symbolic_program


def S(*bins: str) -> tuple[frozenset[str], ...]:
    """Event sequence from space-separated tokens per bin ("" is an empty bin)."""
    return tuple(frozenset(b.split()) for b in bins)


def seq_program(*elements: str, pid: str = "p", score: float = 0.0) -> ProgramRecord:
    return make_symbolic_program(pid, "test", "sequence", tuple(frozenset(e.split()) for e in elements), score=score)


def rule(kind: str, antecedent: tuple[str, ...], consequent: tuple[str, ...], pid: str = "r") -> ProgramRecord:
    return make_symbolic_program(
        pid,
        "test",
        kind,
        tuple(frozenset(e.split()) for e in antecedent),
        tuple(frozenset(e.split()) for e in consequent),
    )


# -------------------------------------------------------------------------- fires


def test_sequence_itemset_must_occur_within_one_bin():
    assert fires(seq_program("a b"), S("a b"))
    assert fires(seq_program("a b"), S("x", "a b c"))
    assert not fires(seq_program("a b"), S("a", "b"))


def test_sequence_allows_gaps_but_needs_strictly_increasing_bins():
    assert fires(seq_program("a", "c"), S("a", "b", "", "c"))
    assert not fires(seq_program("c", "a"), S("a", "b", "c"))
    assert not fires(seq_program("a", "a"), S("a b"))
    assert fires(seq_program("a", "a"), S("a", "x", "a"))
    assert not fires(seq_program("a", "a b"), S("a b"))
    assert fires(seq_program("a", "a b"), S("a", "a", "a b"))
    assert not fires(seq_program("a"), ())


def test_ordered_rule_puts_consequent_strictly_after_antecedent():
    r = rule("ordered_rule", ("a", "b"), ("c",))
    assert fires(r, S("a", "b", "c"))
    assert fires(r, S("a", "x", "b", "", "c"))
    assert not fires(r, S("b", "a", "c"))  # antecedent order matters
    assert not fires(r, S("a", "b c"))  # consequent in the same bin as the antecedent end


def test_unordered_rule_follows_rulegrowth_semantics():
    r = rule("unordered_rule", ("a b",), ("c",))
    assert fires(r, S("b", "a", "c"))  # antecedent items in any order
    assert fires(r, S("a b", "c"))
    assert fires(r, S("c", "a", "b", "c"))
    assert not fires(r, S("a b c"))  # needs a split between X and Y
    assert not fires(r, S("a", "c", "b"))  # b only after the last c
    assert not fires(r, S("a", "c"))  # b missing
    multi = rule("unordered_rule", ("a",), ("c d",))
    assert fires(multi, S("a", "c", "d"))  # consequent items may span bins
    assert fires(multi, S("a c", "c d"))
    assert not fires(multi, S("c", "a", "d"))
    ordered = rule("ordered_rule", ("a", "b"), ("c",))
    assert not fires(ordered, S("b", "a", "c")) and fires(rule("unordered_rule", ("a b",), ("c",)), S("b", "a", "c"))


def test_fires_rejects_numeric_and_malformed_programs():
    with pytest.raises(ValueError):
        fires(ProgramRecord(program_id="m", method="mp", kind="motif"), S("a"))
    with pytest.raises(ValueError):
        fires(ProgramRecord(program_id="s", method="sh", kind="shapelet"), S("a"))
    with pytest.raises(ValueError):
        fires(ProgramRecord(program_id="e", method="t", kind="sequence"), S("a"))
    with pytest.raises(ValueError):
        fires(rule("ordered_rule", ("a",), ()), S("a"))
    with pytest.raises(ValueError):
        fires(make_symbolic_program("x", "t", "sequence", (frozenset({"a"}), frozenset())), S("a", "b"))
    with pytest.raises(ValueError):
        fires(rule("sequence", ("a",), ("b",)), S("a", "b"))


# -------------------------------------------------------------------- firing_sets


def _random_db(rng: random.Random, n: int, alphabet: str = "abcde") -> dict[str, tuple[frozenset[str], ...]]:
    return {
        f"ep{i}": tuple(
            frozenset(rng.sample(alphabet, rng.randint(0, 3))) for _ in range(rng.randint(0, 6))
        )
        for i in range(n)
    }


def _random_programs(rng: random.Random, n: int, alphabet: str = "abcdef") -> list[ProgramRecord]:
    programs = []
    for i in range(n):
        def itemsets(k: int) -> tuple[frozenset[str], ...]:
            return tuple(frozenset(rng.sample(alphabet, rng.randint(1, 2))) for _ in range(k))

        kind = rng.choice(["sequence", "ordered_rule", "unordered_rule"])
        if kind == "sequence":
            programs.append(make_symbolic_program(f"p{i}", "t", kind, itemsets(rng.randint(1, 3))))
        else:
            programs.append(make_symbolic_program(f"p{i}", "t", kind, itemsets(rng.randint(1, 2)), itemsets(1)))
    return programs


def test_firing_sets_match_brute_force():
    rng = random.Random(3)
    sequences = _random_db(rng, 60)
    programs = _random_programs(rng, 80)
    result = firing_sets(programs, sequences)
    assert set(result) == {p.program_id for p in programs}
    for program in programs:
        expected = frozenset(eid for eid, seq in sequences.items() if fires(program, seq))
        assert result[program.program_id] == expected
    assert any(result.values()) and not all(result.values())


def test_firing_sets_edge_cases():
    sequences = {"e1": S("a", "b"), "e2": ()}
    assert firing_sets([seq_program("z")], sequences) == {"p": frozenset()}
    assert firing_sets([], sequences) == {}
    assert firing_sets([seq_program("a")], {}) == {"p": frozenset()}
    with pytest.raises(ValueError):
        firing_sets([seq_program("a"), seq_program("b")], sequences)  # duplicate ids
    with pytest.raises(ValueError):
        firing_sets([ProgramRecord(program_id="m", method="mp", kind="motif")], sequences)


# ------------------------------------------------------------------ dedup_programs


def test_dedup_keeps_highest_score_in_input_order():
    a = seq_program("x", "y", pid="A", score=0.1)
    b = seq_program("y", pid="B", score=0.5)
    c = seq_program("x", "y", pid="C", score=0.9)
    d = seq_program("x", "y", pid="D", score=0.9)
    e = rule("ordered_rule", ("x",), ("y",), pid="E")  # same itemsets, different kind
    assert [p.program_id for p in dedup_programs([a, b, c, d, e])] == ["B", "C", "E"]
    assert dedup_programs([]) == []
    same_items_other_order = make_symbolic_program("F", "t", "sequence", (frozenset({"b", "a"}),), score=1.0)
    assert [p.program_id for p in dedup_programs([seq_program("a b", pid="G"), same_items_other_order])] == ["F"]


# ------------------------------------------------------------------ contrast_score


def test_contrast_score_formula_and_edges():
    assert contrast_score(10, 10, 0, 10) == pytest.approx(math.log2(21.0))
    assert contrast_score(3, 10, 1, 20) == pytest.approx(math.log2((3.5 / 11) / (1.5 / 21)))
    assert contrast_score(3, 10, 1, 20, eps=1.0) == pytest.approx(math.log2((4 / 12) / (2 / 22)))
    assert contrast_score(5, 10, 10, 20) == 0.0  # same rate
    assert contrast_score(0, 10, 10, 10) == 0.0  # depleted in positives -> clamped
    assert contrast_score(0, 0, 0, 0) == 0.0
    assert contrast_score(4, 4, 0, 0) == pytest.approx(math.log2((4.5 / 5) / 0.5))
    for bad in [(-1, 10, 0, 10), (11, 10, 0, 10), (0, 10, 3, 2), (0, -1, 0, 0)]:
        with pytest.raises(ValueError):
            contrast_score(*bad)
    with pytest.raises(ValueError):
        contrast_score(1, 2, 0, 2, eps=0.0)


# ---------------------------------------------------------------- rank_by_contrast


def test_rank_by_contrast_rescores_sorts_and_breaks_ties():
    sequences = {
        "p1": S("a", "b"),
        "p2": S("a", "b"),
        "p3": S("a"),
        "c1": S("a"),
        "c2": S("b", "a"),
        "c3": S(""),
    }
    labels = {"p1": 1, "p2": 1, "p3": 1, "c1": 0, "c2": 0, "c3": 0, "unused": 1}
    a_then_b = seq_program("a", "b", pid="ab", score=0.7)
    only_a = seq_program("a", pid="a", score=0.9)
    rule_ab = rule("unordered_rule", ("a",), ("b",), pid="rule")  # same firing as a_then_b
    never = seq_program("z", pid="z")
    ranked = rank_by_contrast([never, only_a, rule_ab, a_then_b], sequences, labels)

    assert [p.program_id for p in ranked] == ["ab", "rule", "a", "z"]
    top = ranked[0]
    assert top.score == pytest.approx(contrast_score(2, 3, 0, 3))
    assert dict(top.params) == {
        "pos_count": 2,
        "pos_total": 3,
        "neg_count": 0,
        "neg_total": 3,
        "mining_score": 0.7,
    }
    assert ranked[2].score == pytest.approx(contrast_score(3, 3, 2, 3))
    assert ranked[3].score == 0.0 and ranked[3].params["pos_count"] == 0
    assert a_then_b.score == 0.7 and "pos_count" not in a_then_b.params  # originals untouched
    assert top.features == a_then_b.features and top.kind == "sequence"


def test_rank_by_contrast_validates_labels():
    program = seq_program("a")
    with pytest.raises(KeyError):
        rank_by_contrast([program], {"e1": S("a")}, {})
    with pytest.raises(ValueError):
        rank_by_contrast([program], {"e1": S("a")}, {"e1": 2})
    assert rank_by_contrast([], {"e1": S("a")}, {"e1": 1}) == []


def test_firing_counts_match_firing_sets_on_random_data():
    import random

    from src.icu.v2.patterns import firing_counts, firing_sets
    from src.icu.v2.types import make_symbolic_program

    rng = random.Random(7)
    tokens = [f"F{i}:q{j}" for i in range(4) for j in range(3)]
    sequences = {
        f"e{n}": tuple(frozenset(rng.sample(tokens, rng.randint(1, 4))) for _ in range(rng.randint(2, 6)))
        for n in range(60)
    }
    positives = {episode_id for episode_id in sequences if rng.random() < 0.4}
    programs = [
        make_symbolic_program(f"p{k}", "prefixspan", "sequence", tuple({rng.choice(tokens)} for _ in range(rng.randint(1, 3))))
        for k in range(40)
    ]
    sets = firing_sets(programs, sequences)
    counts = firing_counts(programs, sequences, positives)
    for program in programs:
        hits = sets[program.program_id]
        assert counts[program.program_id] == (len(hits & positives), len(hits - positives))
