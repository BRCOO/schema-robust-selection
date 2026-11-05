"""Exact support pre-filter for FMG contrastive inputs (memory reduction only).

v1's contrastive stage (``src/icu/shock_contrastive_miner.py::score_contrastive_rule_records``)
drops every rule with fewer than ``min_rule_positive_count`` positive episodes or fewer than
``min_rule_control_count`` control episodes before scoring, and the episode totals it uses
come from the episode list, not from the rules. Rules below those thresholds can therefore
never be scored, consolidated, or selected. Removing them from the inputs only shrinks
memory (18.3M rule records for 3.9k train episodes in the 2026-10-21 interim cohort), as
long as every episode is kept and the counts are taken over exactly the episodes the stage
will see.

Counts here ignore the ``is_missing_only`` / sparse flags, so they are >= v1's counts and
the kept set is a superset of the rules v1 can rank. Configurations that use low-support
rules elsewhere (zero-control candidates, low-coverage recovery) are refused. Exactness is
checked against v1's own outputs by ``scripts/v2/prefilter_fmg_inputs.py --verify-on``.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any, Iterator, Mapping

from src.icu.v2.fmg_adapter import iter_json_array


def prefilter_thresholds(resolved_profile: Mapping[str, Any]) -> tuple[int, int]:
    """``(min_positive, min_control)`` from a resolved v1 profile; refuse non-exact settings."""
    if bool(resolved_profile.get("allow_zero_control_candidates", False)):
        raise ValueError("allow_zero_control_candidates uses rules without controls; the pre-filter would not be exact")
    if bool(resolved_profile.get("rule_low_coverage_recovery_enabled", False)):
        raise ValueError("low-coverage recovery may re-admit low-support rules; the pre-filter would not be exact")
    return int(resolved_profile.get("min_rule_positive_count", 1)), int(resolved_profile.get("min_rule_control_count", 0))


def count_rule_support(
    path: str | Path,
    split_of: Mapping[str, str] | None = None,
    split: str = "train",
) -> tuple[Counter, Counter, dict[str, int]]:
    """Per rule id, the number of positive and control episodes containing it.

    Only episodes whose ``split_of`` value equals ``split`` are counted; ``split_of=None``
    counts every episode (v1 mined on all of them).
    """
    positives: Counter = Counter()
    controls: Counter = Counter()
    episodes: dict[str, int] = {"positive": 0, "control": 0}
    for item in iter_json_array(path):
        if split_of is not None and split_of.get(str(item.get("episode_id"))) != split:
            continue
        kind = str(item.get("episode_type"))
        ids = {str(rule["rule_id"]) for rule in item.get("rules", ())}
        if kind == "positive":
            positives.update(ids)
        elif kind == "control":
            controls.update(ids)
        else:
            raise ValueError(f"unknown episode_type {kind!r}")
        episodes[kind] += 1
    return positives, controls, episodes


def kept_rule_ids(positives: Counter, controls: Counter, min_positive: int, min_control: int) -> frozenset[str]:
    return frozenset(rule_id for rule_id, count in positives.items() if count >= min_positive and controls.get(rule_id, 0) >= min_control)


def iter_filtered_inputs(
    path: str | Path,
    keep: frozenset[str],
    split_of: Mapping[str, str] | None = None,
    split: str = "train",
) -> Iterator[dict]:
    """Yield every episode of the split with its rules restricted to ``keep``.

    Episodes whose rules all drop out are still yielded (with an empty rule list), because
    v1's positive/control totals count episodes.
    """
    for item in iter_json_array(path):
        if split_of is not None and split_of.get(str(item.get("episode_id"))) != split:
            continue
        rules = [rule for rule in item.get("rules", ()) if str(rule["rule_id"]) in keep]
        yield {**item, "rules": rules}


def compact_episode(item: Mapping[str, Any], cache: dict[tuple[str, ...], tuple[str, ...]]) -> dict:
    """Same episode with shared strings: interned rule ids, cached feature tuples.

    Values are unchanged; only object identity is shared, which cuts memory when millions
    of rule records repeat the same ids and feature lists. Feature lists become tuples
    (v1 copies them with ``list(...)`` before use); exactness is checked against v1 by
    ``tests``-style parity runs (see scripts/v2/run_fmg_train_only.py --compact-memory).
    """
    import sys

    rules = []
    for rule in item.get("rules", ()):
        compact = dict(rule)
        compact["rule_id"] = sys.intern(str(compact["rule_id"]))
        for key in ("antecedent_features", "consequent_features"):
            if key in compact and compact[key] is not None:
                features = tuple(sys.intern(str(f)) for f in compact[key])
                compact[key] = cache.setdefault(features, features)
        rules.append(compact)
    return {**item, "rules": rules}
