"""Build :class:`~src.icu.v2.pipeline.MethodPool` objects from v2 data.

All mining and ranking uses TRAIN episodes only; firing sets are then computed on every
internal episode and on each external target, so the pipeline can compile the bridge and
refire without repair. Episode arrays come from a materialized v2 directory
(``scripts/v2/materialize_task_episodes.py``), FMG candidates from
``scripts/v2/run_fmg_train_only.py``.

Representation note: v1 episode arrays are robust-scaled per episode, so the shared
quantile bins of :mod:`src.icu.v2.events` describe per-episode-relative levels unless the
arrays were re-materialized from raw values (audit section 7.3).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np

from src.icu.v2.baselines.motif import mine_motif_programs
from src.icu.v2.baselines.shapelet import mine_shapelet_programs
from src.icu.v2.baselines.spmf import mine_cmspade, mine_erminer, mine_prefixspan, mine_rulegrowth
from src.icu.v2.events import eventize_episode, fit_feature_bins
from src.icu.v2.fmg_adapter import fmg_firing_sets, fmg_programs_to_records, iter_json_array
from src.icu.v2.leakage_audit import load_episode
from src.icu.v2.numeric_firing import numeric_firing_sets
from src.icu.v2.patterns import dedup_programs, firing_sets, rank_by_contrast
from src.icu.v2.pipeline import MethodPool
from src.icu.v2.types import EpisodeRecord, EventSequence, ProgramRecord

ArrayTriple = tuple[np.ndarray, np.ndarray, Sequence[str]]

SPMF_MINERS: dict[str, Callable[..., list[ProgramRecord]]] = {
    "prefixspan": mine_prefixspan,
    "cmspade": mine_cmspade,
    "rulegrowth": mine_rulegrowth,
    "erminer": mine_erminer,
}
NUMERIC_MINERS: dict[str, Callable[..., list[ProgramRecord]]] = {
    "matrix_profile": mine_motif_programs,
    "shapelet": mine_shapelet_programs,
}


def load_arrays(episode_dir: str | Path, episode_ids: Iterable[str]) -> dict[str, ArrayTriple]:
    """``episode_id -> (values, mask, feature_names)`` read by constructed path (mask True = missing)."""
    arrays: dict[str, ArrayTriple] = {}
    for episode_id in dict.fromkeys(episode_ids):
        loaded = load_episode(episode_dir, episode_id)
        arrays[episode_id] = (loaded.values, loaded.mask, tuple(loaded.meta["feature_names"]))
    return arrays


def build_sequences(
    arrays: Mapping[str, ArrayTriple],
    train_ids: Iterable[str],
    *,
    n_bins: int = 4,
    observation_tokens: bool = True,
    trend_tokens: bool = False,
    extra: Mapping[str, Mapping[str, ArrayTriple]] | None = None,
) -> tuple[dict[str, EventSequence], dict[str, dict[str, EventSequence]], dict[str, np.ndarray]]:
    """Fit global bins on TRAIN arrays and eventize internal and external episodes.

    Every episode must share the train feature order; a target with a different schema
    should be materialized onto the canonical feature list first.
    """
    train_ids = list(train_ids)
    names = arrays[train_ids[0]][2]
    for episode_id, (_, _, other) in arrays.items():
        if tuple(other) != tuple(names):
            raise ValueError(f"episode {episode_id!r} has feature order {list(other)}, expected {list(names)}")
    bins = fit_feature_bins(
        [arrays[i][0] for i in train_ids],
        names,
        n_bins=n_bins,
        missing_masks=[arrays[i][1] for i in train_ids],
    )

    def _eventize(source: Mapping[str, ArrayTriple]) -> dict[str, EventSequence]:
        out = {}
        for episode_id, (values, mask, feature_names) in source.items():
            if tuple(feature_names) != tuple(names):
                raise ValueError(f"episode {episode_id!r} feature order differs from train")
            out[episode_id] = eventize_episode(
                values, mask, feature_names, bins, observation_tokens=observation_tokens, trend_tokens=trend_tokens
            )
        return out

    return _eventize(arrays), {name: _eventize(source) for name, source in (extra or {}).items()}, bins


def build_symbolic_pool(
    method: str,
    sequences: Mapping[str, EventSequence],
    labels: Mapping[str, int],
    train_ids: Iterable[str],
    *,
    mining_params: Mapping[str, Any],
    pool_cap: int = 200,
    external_sequences: Mapping[str, Mapping[str, EventSequence]] | None = None,
) -> MethodPool:
    """SPMF miner on TRAIN positives, contrast-ranked on TRAIN, de-duplicated, capped.

    ``mining_params`` are passed to the miner (e.g. ``minsup``, ``max_pattern_length``,
    ``minconf``, ``max_antecedent``, ``max_consequent``).
    """
    if method not in SPMF_MINERS:
        raise ValueError(f"unknown symbolic method {method!r}; choose from {sorted(SPMF_MINERS)}")
    train_ids = list(train_ids)
    train_seqs = {i: sequences[i] for i in train_ids}
    positives = [train_seqs[i] for i in train_ids if labels[i] == 1]
    mined = SPMF_MINERS[method](positives, **dict(mining_params))
    ranked = rank_by_contrast(mined, train_seqs, {i: labels[i] for i in train_ids})
    candidates = dedup_programs(ranked)[: int(pool_cap)]
    return MethodPool(
        method=method,
        candidates=candidates,
        firings=firing_sets(candidates, sequences),
        external_firings={name: firing_sets(candidates, seqs) for name, seqs in (external_sequences or {}).items()},
    )


def build_numeric_pool(
    method: str,
    arrays: Mapping[str, ArrayTriple],
    labels: Mapping[str, int],
    train_ids: Iterable[str],
    *,
    groups: Mapping[str, Any] | None = None,
    mining_params: Mapping[str, Any] | None = None,
    external_arrays: Mapping[str, Mapping[str, ArrayTriple]] | None = None,
) -> MethodPool:
    """Exact motif or shapelet discovery on TRAIN arrays; firing on all episodes.

    ``groups`` (episode id -> subject id) keeps motif neighbours on other patients; it is
    ignored by the shapelet miner.
    """
    if method not in NUMERIC_MINERS:
        raise ValueError(f"unknown numeric method {method!r}; choose from {sorted(NUMERIC_MINERS)}")
    train_ids = list(train_ids)
    train_arrays = {i: arrays[i] for i in train_ids}
    train_labels = {i: labels[i] for i in train_ids}
    params = dict(mining_params or {})
    if method == "matrix_profile":
        candidates = mine_motif_programs(train_arrays, train_labels, groups=groups, **params)
    else:
        candidates = mine_shapelet_programs(train_arrays, train_labels, **params)
    return MethodPool(
        method=method,
        candidates=candidates,
        firings=numeric_firing_sets(candidates, arrays),
        external_firings={name: numeric_firing_sets(candidates, source) for name, source in (external_arrays or {}).items()},
    )


def _program_rule_ids(records: Sequence[ProgramRecord]) -> frozenset[str]:
    ids: set[str] = set()
    for record in records:
        ids.update(str(rule_id) for rule_id in record.params.get("member_rule_ids", ()) or ())
        representative = str(record.params.get("representative_rule_id", "") or "")
        if representative:
            ids.add(representative)
    return frozenset(ids)


def _inputs_by_episode(path: str | Path, keep: set[str] | None = None, rule_ids: frozenset[str] | None = None) -> dict[str, dict]:
    """Stream a contrastive_inputs.json and keep, per episode, only the rule ids needed.

    Firing only intersects an episode's rule ids with the candidates' member and
    representative rule ids, so every other rule record is dropped while streaming (a
    full cache holds millions of records).
    """
    out: dict[str, dict] = {}
    for item in iter_json_array(path):
        episode_id = str(item.get("episode_id"))
        if keep is not None and episode_id not in keep:
            continue
        rules = item.get("rules", [])
        if rule_ids is not None:
            rules = [{"rule_id": str(rule["rule_id"])} for rule in rules if str(rule["rule_id"]) in rule_ids]
        out[episode_id] = {"episode_id": episode_id, "rules": rules}
    return out


def build_fmg_pool(
    fmg_dir: str | Path,
    contrastive_inputs: str | Path,
    *,
    episode_ids: Iterable[str] | None = None,
    external_inputs: Mapping[str, str | Path] | None = None,
    method: str = "fmg_bridge",
) -> MethodPool:
    """FMG-Bridge candidates (v1 ranked list, mined on TRAIN by run_fmg_train_only.py).

    ``native_registry`` is FMG's own v1 selection (``fmg_programs_selected.json``); the
    candidate pool for the v2 selector is the full ranked list. Firing uses v1 semantics
    on each episode's own mined rules (label-free, so test/external firing is legitimate).
    """
    fmg_dir = Path(fmg_dir)
    ranked = json.loads((fmg_dir / "fmg_programs_ranked.json").read_text(encoding="utf-8"))
    selected = json.loads((fmg_dir / "fmg_programs_selected.json").read_text(encoding="utf-8"))
    candidates = fmg_programs_to_records(ranked, method=method)
    by_id = {record.program_id: record for record in candidates}
    native = [by_id[str(program["program_id"])] for program in selected]
    keep = set(episode_ids) if episode_ids is not None else None
    needed = _program_rule_ids(candidates)
    inputs = _inputs_by_episode(contrastive_inputs, keep, needed)
    return MethodPool(
        method=method,
        candidates=candidates,
        firings=fmg_firing_sets(candidates, inputs),
        external_firings={
            name: fmg_firing_sets(candidates, _inputs_by_episode(path, None, needed))
            for name, path in (external_inputs or {}).items()
        },
        native_registry=native,
    )


def labels_and_groups(episodes: Sequence[EpisodeRecord]) -> tuple[dict[str, int], dict[str, int]]:
    return {e.episode_id: e.label for e in episodes}, {e.episode_id: e.subject_id for e in episodes}
