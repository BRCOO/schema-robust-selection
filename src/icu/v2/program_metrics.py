"""Method-independent program-set metrics for the v2 protocol.

Every function reads only ``ProgramRecord`` fields and firing sets keyed by
``program_id``, so FMG-Bridge and every baseline are scored by the same code. This
replaces the two v1 formulas: FMG used an exact duplicate-signature ratio and a union of
``mechanism_groups`` that mixed coarse and feature-level labels, while baselines used
``1 - unique tokens / total tokens`` with program-id strings counted as tokens.

Notation (paper Definition 3):

- sigma_f(p) = ``p.features``: base features the program needs to execute.
- sigma_a(p) = ``p.anchor_signature``: coarse mechanism groups of the terminal element.
- z_p: TRAIN episode ids where p fires. Callers build ``train_firings`` on the training
  split only; a program missing from the mapping is treated as never firing.

All functions are pure and do no I/O.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from collections.abc import Set as AbstractSet
from itertools import combinations

from src.icu.shock_contrastive_miner import MECHANISM_GROUP_BY_FEATURE
from src.icu.v2.types import OTHER_GROUP_PREFIX, ProgramRecord

FiringSets = Mapping[str, AbstractSet[str]]

N_COARSE_GROUPS = len(set(MECHANISM_GROUP_BY_FEATURE.values()))  # 9
N_CANONICAL_FEATURES = len(MECHANISM_GROUP_BY_FEATURE)  # 18

DEFINITIONS: dict[str, str] = {
    "jaccard": "J(A, B) = |A intersect B| / |A union B|; J(empty, empty) = 0",
    "pairwise_redundancy": (
        "r(p, q) = max{J(sigma_f(p), sigma_f(q)), J(sigma_a(p), sigma_a(q)), J(z_p, z_q)}; "
        "z_p = TRAIN episode ids where p fires (paper Definition 3)"
    ),
    "redundancy": "R(P) = mean of r(p, q) over unordered pairs {p, q} of P; 0 if |P| < 2",
    "redundancy_components": "mean over unordered pairs of each Jaccard term of r(p, q) separately",
    "exact_duplicate_ratio": "(|P| - #distinct canonical_key(p)) / |P|; 0 if |P| = 0",
    "anchor_group_coverage": (
        "|union_p sigma_a(p)|; sigma_a = coarse mechanism groups of the terminal element "
        "(consequent for rules, last itemset for sequences); at most 9 groups"
    ),
    "feature_coverage": "|union_p sigma_f(p)|; sigma_f = base features the program needs; at most 18",
    "schema_survival": (
        "|{p in P : sigma_f(p) subset of F_target}| / |P|; denominator = all selected programs"
    ),
    "firing_survival": "|{p in P : p fires on >= 1 target episode}| / |P|; denominator = all selected programs",
    "firing_without_schema_survival": (
        "programs that fire on a target whose schema lacks one of their features "
        "(must be empty; non-empty means target-specific repair)"
    ),
    "top_k": "every metric uses the first k programs in the given ranked order",
}

_EMPTY: frozenset[str] = frozenset()


def _as_frozenset(values: AbstractSet[str] | Iterable[str]) -> frozenset[str]:
    return values if isinstance(values, frozenset) else frozenset(values)


def _firing_set(firings: FiringSets, program_id: str) -> frozenset[str]:
    return _as_frozenset(firings.get(program_id, _EMPTY))


def _mean(values: Iterable[float]) -> float:
    items = list(values)
    return math.fsum(items) / len(items) if items else 0.0


def _check_unambiguous_ids(programs: Sequence[ProgramRecord]) -> None:
    """Firing sets are keyed by id, so one id must not name two different programs."""
    seen: dict[str, tuple] = {}
    for program in programs:
        key = program.canonical_key()
        if seen.setdefault(program.program_id, key) != key:
            raise ValueError(
                f"program_id {program.program_id!r} names two different programs; "
                "firing sets keyed by program_id would be ambiguous"
            )


def jaccard(a: AbstractSet[str], b: AbstractSet[str]) -> float:
    """Jaccard overlap ``|a & b| / |a | b|`` with ``J(empty, empty) = 0.0``.

    The empty-empty case is 0 on purpose: two programs that never fire on the training
    split (or carry an empty signature) share no observed evidence, so they are not
    evidence of redundancy. Scoring them as identical would let dead programs make a
    method look redundant, and would hide them instead of reporting them as unsupported.
    """
    a = _as_frozenset(a)
    b = _as_frozenset(b)
    union = len(a | b)
    if union == 0:
        return 0.0
    return len(a & b) / union


def pairwise_redundancy(p: ProgramRecord, q: ProgramRecord, train_firings: FiringSets) -> dict[str, float]:
    """Paper Definition 3 for one pair, with the three Jaccard terms kept for audit.

    ``train_firings`` maps program_id to the TRAIN episode ids where it fires; a missing
    id is an empty set.
    """
    feature_j = jaccard(p.features, q.features)
    anchor_j = jaccard(p.anchor_signature, q.anchor_signature)
    firing_j = jaccard(_firing_set(train_firings, p.program_id), _firing_set(train_firings, q.program_id))
    return {
        "feature_jaccard": feature_j,
        "anchor_jaccard": anchor_j,
        "firing_jaccard": firing_j,
        "value": max(feature_j, anchor_j, firing_j),
    }


def _pairwise_rows(programs: Sequence[ProgramRecord], train_firings: FiringSets) -> list[dict[str, float]]:
    return [pairwise_redundancy(p, q, train_firings) for p, q in combinations(programs, 2)]


def set_redundancy(programs: Sequence[ProgramRecord], train_firings: FiringSets) -> float:
    """Mean of ``pairwise_redundancy(...)["value"]`` over unordered pairs; 0.0 below 2 programs."""
    programs = list(programs)
    _check_unambiguous_ids(programs)
    return _mean(row["value"] for row in _pairwise_rows(programs, train_firings))


def exact_duplicate_ratio(programs: Sequence[ProgramRecord]) -> float:
    """Share of programs that repeat an earlier ``canonical_key()``; 0.0 for an empty set.

    Ids, methods, and scores are ignored, so one pattern emitted under several ids (the
    v1 baseline failure) counts as duplicates.
    """
    programs = list(programs)
    if not programs:
        return 0.0
    distinct = len({program.canonical_key() for program in programs})
    return (len(programs) - distinct) / len(programs)


def anchor_group_coverage(programs: Sequence[ProgramRecord]) -> int:
    """Number of distinct coarse groups in the union of the programs' sigma_a."""
    return len(frozenset().union(*(program.anchor_signature for program in programs)))


def feature_coverage(programs: Sequence[ProgramRecord]) -> int:
    """Number of distinct base features in the union of the programs' sigma_f."""
    return len(frozenset().union(*(program.features for program in programs)))


def coverage_upper_bounds(
    programs: Sequence[ProgramRecord],
    n_groups: int = N_COARSE_GROUPS,
    n_features: int = N_CANONICAL_FEATURES,
) -> dict:
    """Ceilings for the two coverage metrics, reported next to the observed values.

    Anchor-group coverage counts the coarse groups of terminal elements. A terminal
    element whose features all sit in one group (a single-feature consequent, the last
    itemset of a single-feature sequence, any motif or shapelet) adds at most one group,
    so such a set of K programs has coverage <= min(n_groups, K). Coverage can exceed K
    only when some terminal elements contain several features from different groups; in
    general coverage <= min(n_groups, sum_p |sigma_a(p)|). With 9 groups, K = 10 programs
    therefore reach at most 9.

    Feature coverage counts every feature a program needs (antecedent and consequent), so
    it is bounded by min(n_features, sum_p |sigma_f(p)|) and routinely exceeds K for
    multi-feature rules.

    Labels outside the shared taxonomy (``other:<feature>`` groups and features without a
    group) widen the universe; they are listed so they stay visible. Both lists are empty
    for the canonical 18-feature schema.
    """
    programs = list(programs)
    k = len(programs)
    groups = frozenset().union(*(program.anchor_signature for program in programs))
    features = frozenset().union(*(program.features for program in programs))
    other_groups = sorted(group for group in groups if group.startswith(OTHER_GROUP_PREFIX))
    other_features = sorted(feature for feature in features if feature not in MECHANISM_GROUP_BY_FEATURE)
    group_universe = int(n_groups) + len(other_groups)
    feature_universe = int(n_features) + len(other_features)
    anchor_max = min(group_universe, sum(len(program.anchor_signature) for program in programs))
    return {
        "k": k,
        "n_groups": int(n_groups),
        "n_features": int(n_features),
        "anchor_group_coverage": len(groups),
        "anchor_group_coverage_max": anchor_max,
        "anchor_group_coverage_max_single_group_terminals": min(group_universe, k),
        "anchor_group_coverage_can_exceed_k": anchor_max > k,
        "n_multi_group_terminals": sum(1 for program in programs if len(program.anchor_signature) > 1),
        "feature_coverage": len(features),
        "feature_coverage_max": min(feature_universe, sum(len(program.features) for program in programs)),
        "out_of_taxonomy_groups": other_groups,
        "out_of_taxonomy_features": other_features,
    }


def schema_survival(programs: Sequence[ProgramRecord], target_features: AbstractSet[str]) -> dict:
    """Share of ALL selected programs whose sigma_f is a subset of the target schema.

    The denominator is ``len(programs)``, never only the programs that compiled into
    bridge coordinates. A program with an empty sigma_f survives vacuously, so adapters
    must fill ``features``. An empty program set has value 0.0.
    """
    programs = list(programs)
    target = _as_frozenset(target_features)
    surviving = [program.program_id for program in programs if _as_frozenset(program.features) <= target]
    denominator = len(programs)
    return {
        "value": len(surviving) / denominator if denominator else 0.0,
        "surviving_ids": surviving,
        "denominator": denominator,
    }


def firing_survival(programs: Sequence[ProgramRecord], external_firings: FiringSets) -> dict:
    """Share of ALL selected programs that fire on at least one target episode.

    ``external_firings`` maps program_id to target episode ids; a missing id never fires.
    An empty program set has value 0.0.
    """
    programs = list(programs)
    _check_unambiguous_ids(programs)
    firing = [program.program_id for program in programs if _firing_set(external_firings, program.program_id)]
    denominator = len(programs)
    return {
        "value": len(firing) / denominator if denominator else 0.0,
        "firing_ids": firing,
        "denominator": denominator,
    }


def summarize_program_set(
    programs: Sequence[ProgramRecord],
    train_firings: FiringSets,
    target_schemas: Mapping[str, AbstractSet[str]] | None = None,
    external_firings: Mapping[str, FiringSets] | None = None,
    k: int | None = None,
) -> dict:
    """All v2 program-set metrics on the top-k programs (by the given order).

    ``k=None`` uses every program; a ``k`` above the list length uses every program and
    ``n_programs < k`` shows the shortfall. ``target_schemas`` maps a target name to its
    available features; ``external_firings`` maps a target name to program_id -> target
    episode ids. The result is JSON-serializable and carries a ``definitions`` dict.
    """
    programs = list(programs)
    if k is not None:
        if int(k) != k or k < 0:
            raise ValueError(f"k must be None or a non-negative integer, got {k!r}")
        programs = programs[: int(k)]
    _check_unambiguous_ids(programs)

    rows = _pairwise_rows(programs, train_firings)
    never_fire = [program.program_id for program in programs if not _firing_set(train_firings, program.program_id)]
    schemas = dict(target_schemas or {})
    firings = dict(external_firings or {})
    schema_rows = {name: schema_survival(programs, schemas[name]) for name in sorted(schemas)}
    firing_rows = {name: firing_survival(programs, firings[name]) for name in sorted(firings)}
    inconsistent: dict[str, list[str]] = {}
    for name in sorted(set(schemas) & set(firings)):
        surviving = set(schema_rows[name]["surviving_ids"])
        inconsistent[name] = [pid for pid in firing_rows[name]["firing_ids"] if pid not in surviving]
    return {
        "k": None if k is None else int(k),
        "n_programs": len(programs),
        "program_ids": [program.program_id for program in programs],
        "methods": sorted({program.method for program in programs}),
        "redundancy": _mean(row["value"] for row in rows),
        "redundancy_components": {
            "feature_jaccard_mean": _mean(row["feature_jaccard"] for row in rows),
            "anchor_jaccard_mean": _mean(row["anchor_jaccard"] for row in rows),
            "firing_jaccard_mean": _mean(row["firing_jaccard"] for row in rows),
        },
        "n_pairs": len(rows),
        "exact_duplicate_ratio": exact_duplicate_ratio(programs),
        "n_distinct_programs": len({program.canonical_key() for program in programs}),
        "anchor_group_coverage": anchor_group_coverage(programs),
        "feature_coverage": feature_coverage(programs),
        "coverage_upper_bounds": coverage_upper_bounds(programs),
        "n_never_fire_train": len(never_fire),
        "never_fire_train_ids": never_fire,
        "schema_survival": schema_rows,
        "firing_survival": firing_rows,
        "firing_without_schema_survival": inconsistent,
        "definitions": dict(DEFINITIONS),
    }
