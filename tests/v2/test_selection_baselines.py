from __future__ import annotations

import math

import pytest

from src.icu.v2.selection_baselines import (
    BASELINES,
    SelectionInput,
    dpp_greedy,
    information_gain,
    l1_logistic,
    mmr,
    mmrfs,
    mms_greedy,
    mrmr,
    mutual_information,
    normalized_scores,
    run_baseline,
)
from src.icu.v2.types import make_symbolic_program

POS = frozenset(f"p{i}" for i in range(10))
NEG = frozenset(f"n{i}" for i in range(10))
IDS = POS | NEG


def _program(pid: str, feature: str, score: float, anchor_feature: str | None = None):
    elements = ({f"{feature}:q3"},) if anchor_feature is None else ({f"{feature}:q3"}, {f"{anchor_feature}:q0"})
    return make_symbolic_program(pid, "toy", "sequence", elements, score=score)


def _input(programs, firings):
    return SelectionInput(candidates=tuple(programs), train_firings=firings, train_ids=IDS, train_positives=POS)


def test_information_measures():
    assert information_gain(POS, IDS, POS) == pytest.approx(1.0)  # perfect predictor: IG = H(Y) = 1 bit
    assert information_gain(IDS, IDS, POS) == pytest.approx(0.0)  # fires everywhere: no information
    assert mutual_information(POS, POS, IDS) == pytest.approx(1.0)
    half = frozenset(list(POS)[:5]) | frozenset(list(NEG)[:5])
    assert mutual_information(half, POS, IDS) == pytest.approx(0.0, abs=1e-12)
    assert normalized_scores([_program("a", "HR", 3.0), _program("b", "MAP", 1.0)]) == [1.0, 0.0]
    assert normalized_scores([_program("a", "HR", 2.0), _program("b", "MAP", 2.0)]) == [1.0, 1.0]


def _duplicate_pool():
    # a and a_dup fire on the same episodes (a_dup slightly lower score); b is distinct but weaker
    a = _program("a", "HR", 10.0)
    a_dup = _program("a_dup", "HR", 9.5, anchor_feature="SpO2")
    b = _program("b", "Lactate", 6.0)
    c = _program("c", "Creatinine", 0.0)
    fa = frozenset(list(POS)[:8]) | frozenset(list(NEG)[:1])
    firings = {"a": fa, "a_dup": fa, "b": frozenset(list(POS)[5:]) | frozenset(list(NEG)[5:7]), "c": frozenset(list(NEG)[:6])}
    return [a, a_dup, b, c], firings


@pytest.mark.parametrize("name", ["mms_xin", "dpp", "mmr"])
def test_redundancy_aware_methods_skip_the_duplicate(name):
    programs, firings = _duplicate_pool()
    value = {"mmr": 0.5, "dpp": 1.0}.get(name)
    chosen = [p.program_id for p in run_baseline(name, _input(programs, firings), 2, value)]
    assert chosen[0] == "a" and chosen[1] == "b"


def test_mms_gain_formula():
    programs, firings = _duplicate_pool()
    # S = (1.0, 0.95, 0.6, 0.0); after a: gain(a_dup) = 0.95 - 1*0.95 = 0 < gain(b) = 0.6 - J(a,b)*0.6
    assert [p.program_id for p in mms_greedy(_input(programs, firings), 3)] == ["a", "b", "a_dup"]


def test_mmrfs_coverage_stops_and_skips():
    programs, firings = _duplicate_pool()
    # delta = 1: a covers p0..p7, b then covers p8, p9 -> every positive covered once; c (negative
    # class) covers n0..n5 -> still eligible; a_dup covers nothing new -> discarded
    chosen = [p.program_id for p in mmrfs(_input(programs, firings), 10, delta=1)]
    assert "a_dup" not in chosen and chosen[0] == "a"
    assert set(chosen) <= {"a", "b", "c"}
    unlimited = [p.program_id for p in mmrfs(_input(programs, firings), 10, delta=math.inf)]
    assert len(unlimited) >= len(chosen)


def test_mrmr_prefers_relevant_then_non_redundant():
    programs, firings = _duplicate_pool()
    chosen = [p.program_id for p in mrmr(_input(programs, firings), 2)]
    assert chosen[0] == "a" and chosen[1] != "a_dup"


def test_dpp_stops_when_determinant_vanishes():
    same = frozenset(list(POS)[:4])
    programs = [_program(f"x{i}", f, 1.0) for i, f in enumerate(["HR", "MAP", "RR"])]
    out = dpp_greedy(_input(programs, {p.program_id: same for p in programs}), 3)
    assert len(out) == 1


def test_l1_logistic_returns_at_most_k_supervised_programs():
    programs, firings = _duplicate_pool()
    out = l1_logistic(_input(programs, firings), 2)
    assert 1 <= len(out) <= 2 and {p.program_id for p in out} <= {p.program_id for p in programs}


def test_all_baselines_are_deterministic_and_bounded():
    programs, firings = _duplicate_pool()
    for name, (_, _, grid) in BASELINES.items():
        for value in grid:
            first = [p.program_id for p in run_baseline(name, _input(programs, firings), 3, value)]
            again = [p.program_id for p in run_baseline(name, _input(programs, dict(reversed(list(firings.items())))), 3, value)]
            assert first == again and len(first) <= 3 and len(set(first)) == len(first)
