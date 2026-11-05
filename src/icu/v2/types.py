"""Shared data types for the v2 protocol.

Every discovery method (FMG-Bridge and all baselines) is converted into
``ProgramRecord`` objects so that redundancy, coverage, survival, and bridge
compilation are computed by one implementation for all methods.

Event tokens have the form ``"<feature>:<state>"`` (for example ``"Lactate:q3"`` or
``"Creatinine:sparse_measurement"``). An itemset is the set of tokens observed in one
time bin; an episode is a time-ordered tuple of itemsets.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

from src.icu.shock_contrastive_miner import MECHANISM_GROUP_BY_FEATURE

Itemset = frozenset[str]
EventSequence = tuple[Itemset, ...]

PROGRAM_KINDS = (
    "sequence",  # ordered itemsets, gaps allowed (PrefixSpan, CM-SPADE)
    "ordered_rule",  # ordered antecedent itemsets, then ordered consequent itemsets (FMG rules)
    "unordered_rule",  # all antecedent items (any order) before all consequent items (RuleGrowth)
    "motif",  # numeric subsequence matched by z-normalized distance (matrix profile)
    "shapelet",  # numeric subsequence matched by distance threshold (shapelets)
)

OTHER_GROUP_PREFIX = "other:"

#: Coarse groups of the drive (Backblaze SMART) features of the second domain
#: (``extract/backblaze.py``). The ICU features keep the v1 taxonomy.
DRIVE_GROUP_BY_FEATURE = {
    "smart_1": "drive:error_rates", "smart_7": "drive:error_rates",
    "smart_5": "drive:media", "smart_197": "drive:media", "smart_198": "drive:media",
    "smart_187": "drive:reported_errors",
    "smart_188": "drive:interface", "smart_199": "drive:interface",
    "smart_3": "drive:mechanical", "smart_4": "drive:mechanical", "smart_10": "drive:mechanical",
    "smart_192": "drive:mechanical", "smart_193": "drive:mechanical",
    "smart_9": "drive:usage", "smart_12": "drive:usage", "smart_240": "drive:usage",
    "smart_241": "drive:usage", "smart_242": "drive:usage",
    "smart_190": "drive:thermal", "smart_194": "drive:thermal",
}


def feature_of(token: str) -> str:
    """Return the base feature of an event token (``"Lactate:q3"`` -> ``"Lactate"``)."""
    return token.split(":", 1)[0]


def coarse_group(feature: str) -> str:
    """Map a base feature to its coarse mechanism group (shared taxonomy for all methods)."""
    return MECHANISM_GROUP_BY_FEATURE.get(feature) or DRIVE_GROUP_BY_FEATURE.get(feature, OTHER_GROUP_PREFIX + feature)


def features_of_itemsets(itemsets: tuple[Itemset, ...]) -> frozenset[str]:
    return frozenset(feature_of(token) for itemset in itemsets for token in itemset)


@dataclass(frozen=True)
class ProgramRecord:
    """One discovered program in a method-independent form.

    ``features`` is sigma_f (base features required to execute the program) and
    ``anchor_signature`` is sigma_a (coarse groups of the terminal element's features).
    Both are filled by :func:`make_symbolic_program` for symbolic kinds; numeric kinds
    (motif, shapelet) set them explicitly from the matched feature.
    """

    program_id: str
    method: str
    kind: str
    elements: tuple[Itemset, ...] = ()
    consequent: tuple[Itemset, ...] = ()
    features: frozenset[str] = frozenset()
    anchor_signature: frozenset[str] = frozenset()
    score: float = 0.0
    params: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.kind not in PROGRAM_KINDS:
            raise ValueError(f"unknown program kind: {self.kind!r}")

    def canonical_key(self) -> tuple:
        """Identity used for exact-duplicate detection. Ignores id, score, and params
        except the numeric payload that defines a motif or shapelet."""
        numeric = ()
        if self.kind in ("motif", "shapelet"):
            numeric = (
                self.params.get("feature"),
                tuple(round(float(v), 6) for v in self.params.get("template", ())),
                round(float(self.params.get("threshold", 0.0)), 6),
            )
        return (
            self.kind,
            tuple(tuple(sorted(itemset)) for itemset in self.elements),
            tuple(tuple(sorted(itemset)) for itemset in self.consequent),
            numeric,
        )


def make_symbolic_program(
    program_id: str,
    method: str,
    kind: str,
    elements: tuple[Itemset, ...],
    consequent: tuple[Itemset, ...] = (),
    score: float = 0.0,
    params: Mapping[str, object] | None = None,
) -> ProgramRecord:
    """Build a symbolic program and derive sigma_f and sigma_a with the shared rules.

    sigma_a uses the consequent for rule kinds and the last element for sequences.
    """
    if kind not in ("sequence", "ordered_rule", "unordered_rule"):
        raise ValueError(f"{kind!r} is not a symbolic kind")
    elements = tuple(frozenset(itemset) for itemset in elements)
    consequent = tuple(frozenset(itemset) for itemset in consequent)
    terminal = consequent if consequent else elements[-1:]
    return ProgramRecord(
        program_id=program_id,
        method=method,
        kind=kind,
        elements=elements,
        consequent=consequent,
        features=features_of_itemsets(elements + consequent),
        anchor_signature=frozenset(coarse_group(f) for f in features_of_itemsets(terminal)),
        score=float(score),
        params=dict(params or {}),
    )


@dataclass(frozen=True)
class EpisodeRecord:
    """One labeled prediction window. ``label`` is 1 for positive, 0 for control."""

    episode_id: str
    subject_id: int
    stay_id: int
    label: int
    anchor_kind: str = ""
    anchor_time: str = ""
    window_start: str = ""
    window_end: str = ""
    split: str = ""
