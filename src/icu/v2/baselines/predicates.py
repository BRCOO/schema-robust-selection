"""Window-statistic predicates: the sixth generator (v3 review, round 5, 2026-11-08).

A predicate compares one statistic of one feature over the episode window with a TRAIN quantile, e.g. "the maximum
lactate in the window is at least its TRAIN 75th percentile". Predicates are Boolean programs like the others: they
fire on an episode or not, require exactly one feature (sigma_f = {feature}), and compile to one SQL aggregate over the
long observation table (``sql_registry.program_sql``).

Statistics are computed on the observed bins of the window (finite and not masked), in the canonical units of the
harmonized features, with definitions that SQL reproduces exactly:

* measured features: ``min``, ``max``, ``mean``, ``std`` (population), ``last`` (value at the latest observed bin),
  ``slope`` (least-squares slope of the value on the bin index, at least two observations) and ``missing`` (share of
  unobserved bins, always defined);
* status features (``events.INTERVENTION_FEATURES``): ``onratio`` (share of observed bins that are on, value > 0.5),
  ``transitions`` (changes between consecutive observed bins) and ``last`` (whether the latest observed bin is on).

Candidates, fixed before any result was seen: for every statistic, ``stat >= q`` at the TRAIN quantiles 0.5, 0.75 and
0.9 and ``stat <= q`` at 0.1, 0.25 and 0.5 (quantiles over the TRAIN episodes where the statistic is defined). A
predicate fires when its statistic is defined and satisfies the comparison within a relative tolerance of 1e-9 (the
same in SQL). Predicates whose TRAIN firing rate is below 2% or above 98% are dropped, duplicates of an earlier
predicate's TRAIN firings are dropped, and the rest are ranked by TRAIN information gain; ``pool_cap`` keeps the top.
"""
from __future__ import annotations

import math
from typing import Iterable, Mapping, Sequence

import numpy as np

from src.icu.v2.events import INTERVENTION_FEATURES, INTERVENTION_ON_THRESHOLD
from src.icu.v2.types import ProgramRecord, coarse_group

MEASURED_STATS = ("min", "max", "mean", "std", "last", "slope", "missing")
STATUS_STATS = ("onratio", "transitions", "last")
GE_QUANTILES = (0.5, 0.75, 0.9)
LE_QUANTILES = (0.1, 0.25, 0.5)
MIN_RATE, MAX_RATE = 0.02, 0.98
REL_TOL = 1e-9
METHOD = "statistic"


def tolerance(threshold: float) -> float:
    return REL_TOL * max(1.0, abs(float(threshold)))


def window_statistics(values: np.ndarray, mask: np.ndarray | None, names: Sequence[str], features: Iterable[str]) -> dict[tuple[str, str], float]:
    """``(feature, stat) -> value`` for one episode (NaN where the statistic is undefined)."""
    matrix = np.asarray(values, dtype=float)
    observed_all = np.isfinite(matrix)
    if mask is not None:
        observed_all &= ~np.asarray(mask, dtype=bool)
    keep = set(features)
    out: dict[tuple[str, str], float] = {}
    n = matrix.shape[0]
    for j, f in enumerate(names):
        if f not in keep:
            continue
        obs = observed_all[:, j]
        x = matrix[obs, j]
        t = np.flatnonzero(obs).astype(float)
        if f in INTERVENTION_FEATURES:
            on = (x > INTERVENTION_ON_THRESHOLD).astype(float)
            out[(f, "onratio")] = float(on.mean()) if on.size else math.nan
            out[(f, "transitions")] = float(np.sum(on[1:] != on[:-1])) if on.size else math.nan
            out[(f, "last")] = float(on[-1]) if on.size else math.nan
            continue
        if x.size:
            out[(f, "min")], out[(f, "max")], out[(f, "mean")] = float(x.min()), float(x.max()), float(x.mean())
            out[(f, "std")] = float(x.std())
            out[(f, "last")] = float(x[-1])
        else:
            for s in ("min", "max", "mean", "std", "last"):
                out[(f, s)] = math.nan
        if x.size >= 2:
            tc = t - t.mean()
            out[(f, "slope")] = float(np.sum(tc * (x - x.mean())) / np.sum(tc * tc))
        else:
            out[(f, "slope")] = math.nan
        out[(f, "missing")] = 1.0 - x.size / n if n else math.nan
    return out


def statistic_table(arrays: Mapping[str, tuple], features: Sequence[str]) -> tuple[list[str], list[tuple[str, str]], np.ndarray]:
    """Episodes x (feature, stat) matrix of window statistics."""
    episode_ids = list(arrays)
    columns = [(f, s) for f in features for s in (STATUS_STATS if f in INTERVENTION_FEATURES else MEASURED_STATS)]
    table = np.full((len(episode_ids), len(columns)), math.nan)
    for i, e in enumerate(episode_ids):
        values, mask, names = arrays[e]
        stats = window_statistics(values, mask, names, features)
        for k, key in enumerate(columns):
            table[i, k] = stats.get(key, math.nan)
    return episode_ids, columns, table


def fires(column: np.ndarray, op: str, threshold: float) -> np.ndarray:
    tol = tolerance(threshold)
    defined = np.isfinite(column)
    if op == "ge":
        return defined & (column >= threshold - tol)
    return defined & (column <= threshold + tol)


def _information_gain(fired: np.ndarray, y: np.ndarray) -> float:
    def h(p: float) -> float:
        return 0.0 if p <= 0.0 or p >= 1.0 else -(p * math.log2(p) + (1 - p) * math.log2(1 - p))

    n = len(y)
    base = h(y.mean())
    rest = 0.0
    for part in (fired, ~fired):
        if part.any():
            rest += part.sum() / n * h(y[part].mean())
    return base - rest


def mine_statistic_predicates(train_ids: Sequence[str], episode_ids: Sequence[str], columns: Sequence[tuple[str, str]], table: np.ndarray,
                              labels: Mapping[str, int], pool_cap: int) -> list[ProgramRecord]:
    """Candidate predicates from TRAIN quantiles, filtered, deduplicated and ranked by TRAIN information gain."""
    row = {e: i for i, e in enumerate(episode_ids)}
    tr = np.array([row[e] for e in train_ids])
    y = np.array([labels[e] for e in train_ids], dtype=float)
    candidates, seen = [], {}
    for k, (f, stat) in enumerate(columns):
        col = table[tr, k]
        defined = col[np.isfinite(col)]
        if defined.size == 0:
            continue
        specs = [("ge", q) for q in GE_QUANTILES] + [("le", q) for q in LE_QUANTILES]
        for op, q in specs:
            threshold = float(np.quantile(defined, q))
            fired = fires(col, op, threshold)
            rate = float(fired.mean())
            if rate < MIN_RATE or rate > MAX_RATE:
                continue
            key = fired.tobytes()
            gain = _information_gain(fired, y)
            pid = f"stat:{f}:{stat}:{op}{int(round(100 * q))}"
            record = ProgramRecord(program_id=pid, method=METHOD, kind="predicate", features=frozenset({f}),
                                   anchor_signature=frozenset({coarse_group(f)}), score=gain,
                                   params={"feature": f, "stat": stat, "op": op, "quantile": q, "threshold": threshold, "train_rate": rate})
            if key in seen and seen[key].score >= gain:
                continue
            seen[key] = record
    candidates = sorted(seen.values(), key=lambda r: (-r.score, r.program_id))
    return candidates[:pool_cap]


def predicate_firing_sets(programs: Sequence[ProgramRecord], episode_ids: Sequence[str], columns: Sequence[tuple[str, str]],
                          table: np.ndarray) -> dict[str, frozenset[str]]:
    index = {c: k for k, c in enumerate(columns)}
    ids = np.array(list(episode_ids))
    out = {}
    for p in programs:
        col = table[:, index[(p.params["feature"], p.params["stat"])]]
        out[p.program_id] = frozenset(ids[fires(col, p.params["op"], float(p.params["threshold"]))].tolist())
    return out
