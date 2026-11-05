"""Firing semantics and the contrast score shared by all symbolic programs (v2).

Every symbolic program (FMG rules and the SPMF baselines alike) is executed on the
same ``EventSequence`` objects (see ``events.py``) by the functions below, so firing,
support, and ranking do not depend on which miner produced a program. Positions are
time-bin indices.

* ``sequence``: elements ``E1..Em`` fire iff positions ``p1 < ... < pm`` exist with
  ``Ei <= seq[pi]``. Gaps are allowed, and the items of one element must co-occur in
  a single bin. This is PrefixSpan / SPADE containment.
* ``ordered_rule``: antecedent elements followed by consequent elements, all at
  strictly increasing positions (the ``sequence`` test on ``elements + consequent``).
* ``unordered_rule``: RuleGrowth containment. With ``X`` the union of the antecedent
  itemsets and ``Y`` the union of the consequent itemsets, the rule fires iff a split
  ``k`` exists such that every item of ``X`` occurs in some bin ``<= k`` and every item
  of ``Y`` in some bin ``> k``, i.e. ``max_x first(x) < min_y last(y)``.
* ``motif`` / ``shapelet`` are numeric and are matched by distance in their own
  modules; :func:`fires` raises for them.

Programs with no elements, an empty itemset, a consequent on a ``sequence``, or a
rule without a consequent are rejected as malformed.

Contrast score (one rule for every symbolic miner)
--------------------------------------------------
With ``a`` = positives where the program fires out of ``n+`` positives and ``b`` =
controls where it fires out of ``n-`` controls::

    p+ = (a + eps) / (n+ + 2 eps),   p- = (b + eps) / (n- + 2 eps)
    score = max(0, log2(p+ / p-))

That is the log2 growth rate (relative risk) of additively smoothed firing rates.
``eps = 0.5`` is the add-one-half (Jeffreys / Krichevsky-Trofimov) estimate, and
``eps = 1`` would be Laplace's add-one rule. The score is 0 for programs that are not
enriched in positives and stays finite when a program never fires on controls.
"""

from __future__ import annotations

import math
from dataclasses import replace
from typing import AbstractSet, Iterable, Mapping, Sequence

from src.icu.v2.types import EventSequence, Itemset, ProgramRecord

SYMBOLIC_KINDS = ("sequence", "ordered_rule", "unordered_rule")


def _union(itemsets: tuple[Itemset, ...]) -> frozenset[str]:
    return frozenset(token for itemset in itemsets for token in itemset)


def _check_program(program: ProgramRecord) -> None:
    if program.kind not in SYMBOLIC_KINDS:
        raise ValueError(
            f"fires() handles symbolic kinds only; {program.kind!r} programs are matched "
            "numerically by their own baseline module"
        )
    if not program.elements:
        raise ValueError(f"program {program.program_id!r} has no elements")
    if any(not itemset for itemset in program.elements + program.consequent):
        raise ValueError(f"program {program.program_id!r} contains an empty itemset")
    if program.kind == "sequence" and program.consequent:
        raise ValueError(f"sequence program {program.program_id!r} must not have a consequent")
    if program.kind != "sequence" and not program.consequent:
        raise ValueError(f"rule program {program.program_id!r} has no consequent")


def _fires_ordered(elements: tuple[Itemset, ...], seq: EventSequence) -> bool:
    # Greedy earliest matching is optimal for subsequence containment.
    cursor = 0
    for itemset in seq:
        if elements[cursor] <= itemset:
            cursor += 1
            if cursor == len(elements):
                return True
    return False


def _fires_unordered(antecedent: frozenset[str], consequent: frozenset[str], seq: EventSequence) -> bool:
    first: dict[str, int] = {}
    last: dict[str, int] = {}
    for position, itemset in enumerate(seq):
        for token in itemset:
            if token in antecedent and token not in first:
                first[token] = position
            if token in consequent:
                last[token] = position
    if len(first) < len(antecedent) or len(last) < len(consequent):
        return False
    return max(first.values()) < min(last.values())


def _fires_checked(program: ProgramRecord, seq: EventSequence) -> bool:
    if program.kind == "unordered_rule":
        return _fires_unordered(_union(program.elements), _union(program.consequent), seq)
    return _fires_ordered(program.elements + program.consequent, seq)


def fires(program: ProgramRecord, seq: EventSequence) -> bool:
    """Return whether a symbolic program fires on one event sequence (see module doc)."""
    _check_program(program)
    return _fires_checked(program, seq)


def _postings(episode_seqs: Sequence[EventSequence]) -> dict[str, int]:
    postings: dict[str, int] = {}
    for index, seq in enumerate(episode_seqs):
        bit = 1 << index
        for token in _union(tuple(seq)):
            postings[token] = postings.get(token, 0) | bit
    return postings


def _candidate_bits(program: ProgramRecord, postings: Mapping[str, int], everyone: int) -> int:
    candidates = everyone
    for token in _union(program.elements + program.consequent):
        candidates &= postings.get(token, 0)
        if not candidates:
            break
    return candidates


def firing_sets(
    programs: Iterable[ProgramRecord],
    sequences: Mapping[str, EventSequence],
) -> dict[str, frozenset[str]]:
    """Map each program id to the ids of the sequences on which it fires.

    An inverted token index (Python-int bitsets) pre-filters candidate sequences; the
    exact :func:`fires` semantics decide the rest. Program ids must be unique.
    """
    episode_ids = list(sequences)
    episode_seqs = [sequences[episode_id] for episode_id in episode_ids]
    postings = _postings(episode_seqs)
    everyone = (1 << len(episode_ids)) - 1

    result: dict[str, frozenset[str]] = {}
    for program in programs:
        if program.program_id in result:
            raise ValueError(f"duplicate program_id {program.program_id!r}")
        _check_program(program)
        candidates = _candidate_bits(program, postings, everyone)
        fired: list[str] = []
        while candidates:
            lowest = candidates & -candidates
            index = lowest.bit_length() - 1
            if _fires_checked(program, episode_seqs[index]):
                fired.append(episode_ids[index])
            candidates ^= lowest
        result[program.program_id] = frozenset(fired)
    return result


def firing_counts(
    programs: Iterable[ProgramRecord],
    sequences: Mapping[str, EventSequence],
    positive_ids: AbstractSet[str],
) -> dict[str, tuple[int, int]]:
    """``program_id -> (positive hits, other hits)`` without materializing firing sets.

    Same semantics and pre-filter as :func:`firing_sets`; used for ranking very large
    candidate pools (hundreds of thousands of SPMF patterns), where storing every firing
    set costs gigabytes.
    """
    episode_ids = list(sequences)
    episode_seqs = [sequences[episode_id] for episode_id in episode_ids]
    postings = _postings(episode_seqs)
    everyone = (1 << len(episode_ids)) - 1
    positive_bits = 0
    for index, episode_id in enumerate(episode_ids):
        if episode_id in positive_ids:
            positive_bits |= 1 << index

    result: dict[str, tuple[int, int]] = {}
    for program in programs:
        if program.program_id in result:
            raise ValueError(f"duplicate program_id {program.program_id!r}")
        _check_program(program)
        candidates = _candidate_bits(program, postings, everyone)
        pos = neg = 0
        while candidates:
            lowest = candidates & -candidates
            index = lowest.bit_length() - 1
            if _fires_checked(program, episode_seqs[index]):
                if lowest & positive_bits:
                    pos += 1
                else:
                    neg += 1
            candidates ^= lowest
        result[program.program_id] = (pos, neg)
    return result


def dedup_programs(programs: Iterable[ProgramRecord]) -> list[ProgramRecord]:
    """Drop exact duplicates by ``canonical_key()``, keeping the highest score.

    The result is the input order restricted to the kept programs; on equal scores the
    first occurrence is kept.
    """
    programs = list(programs)
    best: dict[tuple, int] = {}
    for index, program in enumerate(programs):
        key = program.canonical_key()
        if key not in best or program.score > programs[best[key]].score:
            best[key] = index
    kept = set(best.values())
    return [program for index, program in enumerate(programs) if index in kept]


def contrast_score(
    pos_count: int,
    pos_total: int,
    neg_count: int,
    neg_total: int,
    eps: float = 0.5,
) -> float:
    """Smoothed log2 growth rate of firing in positives vs controls (see module doc)."""
    if eps <= 0:
        raise ValueError("eps must be > 0")
    for count, total in ((pos_count, pos_total), (neg_count, neg_total)):
        if total < 0 or count < 0 or count > total:
            raise ValueError(f"invalid count/total: {count}/{total}")
    pos_rate = (pos_count + eps) / (pos_total + 2.0 * eps)
    neg_rate = (neg_count + eps) / (neg_total + 2.0 * eps)
    return max(0.0, math.log2(pos_rate / neg_rate))


def rank_by_contrast(
    programs: Iterable[ProgramRecord],
    sequences: Mapping[str, EventSequence],
    labels: Mapping[str, int],
    *,
    eps: float = 0.5,
) -> list[ProgramRecord]:
    """Rescore programs with :func:`contrast_score` and sort them, best first.

    The episode universe is ``sequences`` (use TRAIN episodes in the v2 protocol); every
    key needs a 0/1 label. Returns copies whose ``score`` is the contrast score and whose
    ``params`` gain ``pos_count``, ``pos_total``, ``neg_count``, ``neg_total``, and
    ``mining_score`` (the score before rescoring). Ties are broken by
    ``canonical_key()`` and then ``program_id``.
    """
    programs = list(programs)
    positives: set[str] = set()
    for episode_id in sequences:
        if episode_id not in labels:
            raise KeyError(f"no label for episode {episode_id!r}")
        label = labels[episode_id]
        if label not in (0, 1):
            raise ValueError(f"label of {episode_id!r} must be 0 or 1, got {label!r}")
        if label == 1:
            positives.add(episode_id)
    pos_total = len(positives)
    neg_total = len(sequences) - pos_total
    counts = firing_counts(programs, sequences, positives)
    rescored = []
    for program in programs:
        pos_count, neg_count = counts[program.program_id]
        params = dict(program.params)
        params.update(
            pos_count=pos_count,
            pos_total=pos_total,
            neg_count=neg_count,
            neg_total=neg_total,
            mining_score=program.score,
        )
        score = contrast_score(pos_count, pos_total, neg_count, neg_total, eps)
        rescored.append(replace(program, score=score, params=params))
    rescored.sort(key=lambda program: (-program.score, program.canonical_key(), program.program_id))
    return rescored
