"""The union of every generator's candidate pool (cross-generator selection, E3).

Program ids are generator-prefixed, so the union of the de-duplicated pools is disjoint;
firings and target firings are each generator's own.
"""

from __future__ import annotations

from typing import Any, Mapping

from src.icu.v2.patterns import dedup_programs
from src.icu.v2.types import ProgramRecord


def union_pool(state: Mapping[str, Any]) -> tuple[list[ProgramRecord], dict[str, frozenset[str]], dict[str, dict[str, frozenset[str]]]]:
    """(candidates, internal firings by program id, target name -> program id -> firings)."""
    candidates: list[ProgramRecord] = []
    firings: dict[str, frozenset[str]] = {}
    external: dict[str, dict[str, frozenset[str]]] = {}
    for pool in state["pools"]:
        for c in dedup_programs(pool.candidates):
            candidates.append(c)
            firings[c.program_id] = frozenset(pool.firings.get(c.program_id, frozenset()))
            for name, fired in pool.external_firings.items():
                external.setdefault(name, {})[c.program_id] = frozenset(fired.get(c.program_id, frozenset()))
    return candidates, firings, external
