"""Temporal programs as SQL: run a registry inside the target's database (reviewer item G, v2).

A registry ships as SQL text. The target needs nothing but its own observations in the long table

    obs(episode_id VARCHAR, t INTEGER, feature VARCHAR, value DOUBLE)    -- value NULL = not observed

with one row per episode, hourly bin ``t`` and feature of its schema (a dense grid: every bin of
every schema feature has a row), plus the frozen source bin edges ``edges(feature, c1, c2, c3)``.

* :func:`token_sql` derives the shared event vocabulary of ``events.py`` in SQL: value tokens
  ``f:q<k>`` (k = number of inner edges <= value), intervention tokens ``f:on`` (value > 0.5),
  and the observation tokens ``f:missing`` (unobserved run lengths 1, 3, 6, ...) and
  ``f:sparse_measurement`` (trailing 6-bin window of >= 3 bins, partly observed, observed
  fraction <= 1/2), with window functions.
* :func:`program_sql` compiles one program into a query that returns the episodes where it fires,
  with the semantics of ``patterns.py`` (``sequence`` and ``ordered_rule``: elements at strictly
  increasing bins, greedy earliest matching; ``unordered_rule``: max first(x) < min last(y)) and
  of ``numeric_firing.py`` (z-normalized motifs and shapelets, strict missing policy,
  length-normalized distance, constant windows invalid, rounding snap, tolerance 1e-9).
* Executability is a catalog check (:func:`executable`): a program runs iff its features are
  columns of the target schema; otherwise it is not run and its surrogate is resolved instead.

FMG's legacy rules fire by v1 rule-id matching and are not compiled (:func:`compilable`).

Window-statistic predicates (``baselines/predicates.py``) compile to one aggregate over the observed rows of their
feature (MIN, MAX, AVG, STDDEV_POP, the value at the latest observed bin, REGR_SLOPE on the bin index, the share of
unobserved bins; for status features the share of on bins, the changes between consecutive observed bins and the
latest state), compared with the frozen TRAIN threshold within the generator's relative tolerance.
"""

from __future__ import annotations

import math
from typing import AbstractSet, Mapping, Sequence

import numpy as np

from src.icu.v2.baselines.motif import DEFAULT_EPS, MATCH_ATOL, SNAP_REL, znorm_rows
from src.icu.v2.events import (
    INTERVENTION_FEATURES,
    INTERVENTION_ON_THRESHOLD,
    MISSING_REPEAT_STRIDE,
    OBSERVATION_FEATURES,
    OBSERVATION_WINDOW,
    SPARSE_THRESHOLD,
    _MIN_SPARSE_WINDOW,
)
from src.icu.v2.types import ProgramRecord

SYMBOLIC = ("sequence", "ordered_rule", "unordered_rule")
NUMERIC = ("motif", "shapelet")
PREDICATE = ("predicate",)


def _q(text: str) -> str:
    return "'" + str(text).replace("'", "''") + "'"


def _num(x: float) -> str:
    return repr(float(x))


def compilable(program: ProgramRecord) -> bool:
    if program.kind in PREDICATE:
        return True
    if program.kind in NUMERIC:
        return program.params.get("missing_policy", "strict") == "strict" and bool(program.params.get("normalize", True))
    return program.kind in SYMBOLIC and program.method != "fmg_bridge"


def executable(program: ProgramRecord, schema: AbstractSet[str]) -> bool:
    return frozenset(program.features) <= frozenset(schema)


def edges_rows(bins: Mapping[str, np.ndarray], max_cuts: int = 3) -> list[tuple]:
    """``(feature, c1, ..., c_max)`` rows of the inner edges (None where a feature has fewer)."""
    rows = []
    for feature, edges in bins.items():
        edges = np.asarray(edges, dtype=float)
        if edges.size == 0:
            continue
        inner = [float(x) for x in edges[1:-1]]
        if len(inner) > max_cuts:
            raise ValueError(f"{feature}: {len(inner)} inner edges, more than {max_cuts}")
        rows.append((feature, *inner, *[None] * (max_cuts - len(inner))))
    return rows


def token_sql(max_cuts: int = 3) -> str:
    """CREATE TABLE tokens(episode_id, t, token) from ``obs`` and ``edges``."""
    level = " + ".join(f"(CASE WHEN e.c{i} IS NOT NULL AND g.value >= e.c{i} THEN 1 ELSE 0 END)" for i in range(1, max_cuts + 1))
    interventions = ", ".join(_q(f) for f in sorted(INTERVENTION_FEATURES))
    observation = ", ".join(_q(f) for f in sorted(OBSERVATION_FEATURES))
    w = OBSERVATION_WINDOW
    return f"""
CREATE OR REPLACE TABLE tokens AS
WITH g AS (
    SELECT o.episode_id, o.t, o.feature, o.value, o.value IS NOT NULL AS observed,
           SUM(CASE WHEN o.value IS NOT NULL THEN 1 ELSE 0 END) OVER w6 AS cnt,
           LEAST(o.t + 1, {w}) AS len,
           o.t - COALESCE(MAX(CASE WHEN o.value IS NOT NULL THEN o.t END) OVER wpre, -1) AS streak
    FROM obs o
    WINDOW w6 AS (PARTITION BY o.episode_id, o.feature ORDER BY o.t ROWS BETWEEN {w - 1} PRECEDING AND CURRENT ROW),
           wpre AS (PARTITION BY o.episode_id, o.feature ORDER BY o.t ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW)
)
SELECT g.episode_id, g.t, g.feature || ':q' || CAST({level} AS VARCHAR) AS token
FROM g JOIN edges e ON e.feature = g.feature
WHERE g.observed AND g.feature NOT IN ({interventions})
UNION ALL
SELECT episode_id, t, feature || ':on' FROM g
WHERE observed AND feature IN ({interventions}) AND value > {_num(INTERVENTION_ON_THRESHOLD)}
UNION ALL
SELECT episode_id, t, feature || ':missing' FROM g
WHERE feature IN ({observation}) AND NOT observed AND (streak = 1 OR streak % {MISSING_REPEAT_STRIDE} = 0)
UNION ALL
SELECT episode_id, t, feature || ':sparse_measurement' FROM g
WHERE feature IN ({observation}) AND len >= {_MIN_SPARSE_WINDOW} AND cnt > 0 AND cnt < len AND cnt <= {_num(SPARSE_THRESHOLD)} * len
"""


def _element(items: AbstractSet[str]) -> str:
    listed = ", ".join(_q(x) for x in sorted(items))
    return (f"SELECT episode_id, t FROM tokens WHERE token IN ({listed}) "
            f"GROUP BY episode_id, t HAVING COUNT(DISTINCT token) = {len(items)}")


def _ordered_sql(elements: Sequence[AbstractSet[str]]) -> str:
    parts = [f"e{i} AS ({_element(items)})" for i, items in enumerate(elements)]
    parts.append("s0 AS (SELECT episode_id, MIN(t) AS b FROM e0 GROUP BY episode_id)")
    for i in range(1, len(elements)):
        parts.append(f"s{i} AS (SELECT e{i}.episode_id, MIN(e{i}.t) AS b FROM e{i} JOIN s{i - 1} "
                     f"ON e{i}.episode_id = s{i - 1}.episode_id AND e{i}.t > s{i - 1}.b GROUP BY e{i}.episode_id)")
    return "WITH " + ",\n".join(parts) + f"\nSELECT episode_id FROM s{len(elements) - 1}"


def _unordered_sql(antecedent: AbstractSet[str], consequent: AbstractSet[str]) -> str:
    a = ", ".join(_q(x) for x in sorted(antecedent))
    c = ", ".join(_q(x) for x in sorted(consequent))
    return f"""WITH f AS (SELECT episode_id, token, MIN(t) AS t FROM tokens WHERE token IN ({a}) GROUP BY episode_id, token),
fa AS (SELECT episode_id, MAX(t) AS t FROM f GROUP BY episode_id HAVING COUNT(*) = {len(antecedent)}),
l AS (SELECT episode_id, token, MAX(t) AS t FROM tokens WHERE token IN ({c}) GROUP BY episode_id, token),
lc AS (SELECT episode_id, MIN(t) AS t FROM l GROUP BY episode_id HAVING COUNT(*) = {len(consequent)})
SELECT fa.episode_id FROM fa JOIN lc ON fa.episode_id = lc.episode_id WHERE fa.t < lc.t"""


def _numeric_sql(program: ProgramRecord) -> str:
    params = program.params
    template = np.asarray(params["template"], dtype=float).reshape(1, -1)
    eps = float(params.get("eps", DEFAULT_EPS))
    tz, valid = znorm_rows(template, eps)
    if not valid[0]:
        raise ValueError(f"{program.program_id}: constant template")
    tz = tz[0]
    m = tz.size
    st2 = float(np.sum(tz * tz) / m)
    xs = [f"x{i}" for i in range(m)]
    lead = ", ".join(["value AS x0"] + [f"LEAD(value, {i}) OVER w AS x{i}" for i in range(1, m)])
    mean = "(" + " + ".join(xs) + f") / {m}.0"
    centered_sq = " + ".join(f"(x{i} - mu) * (x{i} - mu)" for i in range(m))
    scale = "GREATEST(1.0, " + ", ".join(f"ABS(x{i})" for i in range(m)) + ")"
    z = [f"((x{i} - mu) / sd)" for i in range(m)]
    sa2 = "(" + " + ".join(f"{zi} * {zi}" for zi in z) + f") / {m}.0"
    dot = "(" + " + ".join(f"{zi} * {_num(tz[i])}" for i, zi in enumerate(z)) + f") / {m}.0"
    threshold = float(params["threshold"]) + MATCH_ATOL
    close = f"""WITH x AS (SELECT episode_id, t, {lead} FROM obs WHERE feature = {_q(params['feature'])}
            WINDOW w AS (PARTITION BY episode_id ORDER BY t)),
v AS (SELECT *, {mean} AS mu FROM x WHERE {' AND '.join(f'{v} IS NOT NULL' for v in xs)}),
s AS (SELECT *, SQRT(({centered_sq}) / {m}.0) AS sd, {scale} AS sc FROM v),
d AS (SELECT episode_id, {sa2} AS sa2, {dot} AS dot FROM s WHERE sd > {_num(eps)} * sc),
r AS (SELECT episode_id, sa2 + {_num(st2)} - 2.0 * dot AS d2, sa2 + {_num(st2)} AS norms FROM d)
SELECT DISTINCT episode_id FROM r
WHERE SQRT(CASE WHEN d2 <= {_num(SNAP_REL)} * norms THEN 0.0 ELSE d2 END) <= {_num(threshold)}"""
    if params.get("direction", "le") == "le":
        return close
    return f"SELECT DISTINCT episode_id FROM obs EXCEPT ({close})"


def _predicate_sql(program: ProgramRecord) -> str:
    from src.icu.v2.baselines.predicates import tolerance

    params = program.params
    f, stat, op = params["feature"], params["stat"], params["op"]
    threshold = float(params["threshold"])
    bound = threshold - tolerance(threshold) if op == "ge" else threshold + tolerance(threshold)
    cmp = ">=" if op == "ge" else "<="
    on = f"(CASE WHEN value > {_num(INTERVENTION_ON_THRESHOLD)} THEN 1.0 ELSE 0.0 END)"
    if stat == "missing":
        return (f"SELECT episode_id FROM obs WHERE feature = {_q(f)} GROUP BY episode_id "
                f"HAVING 1.0 - COUNT(value) * 1.0 / COUNT(*) {cmp} {_num(bound)}")
    if stat == "transitions":
        return f"""WITH o AS (SELECT episode_id, t, {on} AS s FROM obs WHERE feature = {_q(f)} AND value IS NOT NULL),
l AS (SELECT episode_id, s, LAG(s) OVER (PARTITION BY episode_id ORDER BY t) AS p FROM o)
SELECT episode_id FROM l GROUP BY episode_id
HAVING SUM(CASE WHEN p IS NOT NULL AND s <> p THEN 1 ELSE 0 END) {cmp} {_num(bound)}"""
    status = f in INTERVENTION_FEATURES
    expr = {"min": "MIN(value)", "max": "MAX(value)", "mean": "AVG(value)", "std": "STDDEV_POP(value)",
            "last": f"arg_max({on if status else 'value'}, t)", "slope": "REGR_SLOPE(value, t)", "onratio": f"AVG({on})"}[stat]
    # DuckDB orders NaN above every number, so a 0/0 slope must not reach the comparison; a slope needs two observations
    guard = "COUNT(value) >= 2 AND " if stat == "slope" else ""
    return (f"SELECT episode_id FROM obs WHERE feature = {_q(f)} AND value IS NOT NULL GROUP BY episode_id "
            f"HAVING {guard}{expr} IS NOT NULL AND NOT isnan({expr}) AND {expr} {cmp} {_num(bound)}")


def program_sql(program: ProgramRecord) -> str:
    """A query returning the episodes on which ``program`` fires (see the module docstring)."""
    if not compilable(program):
        raise ValueError(f"{program.program_id}: {program.kind} of {program.method} is not compiled to SQL")
    if program.kind in PREDICATE:
        return _predicate_sql(program)
    if program.kind in NUMERIC:
        return _numeric_sql(program)
    if program.kind == "unordered_rule":
        union = lambda sets: frozenset(x for s in sets for x in s)  # noqa: E731
        return _unordered_sql(union(program.elements), union(program.consequent))
    return _ordered_sql(list(program.elements) + list(program.consequent))


def fire(con, programs: Sequence[ProgramRecord], schema: AbstractSet[str]) -> dict[str, frozenset[str]]:
    """Run every executable program on the target's ``obs``/``tokens``; non-executable -> empty."""
    out = {}
    for p in programs:
        if not executable(p, schema):
            out[p.program_id] = frozenset()
            continue
        out[p.program_id] = frozenset(r[0] for r in con.execute(program_sql(p)).fetchall())
    return out


def obs_rows(arrays: Mapping[str, tuple], schema: AbstractSet[str], n_bins: int | None = None):
    """Columns (episode_id, t, feature, value) of the dense long table for ``schema`` features;
    value is NaN where unobserved (masked or non-finite)."""
    ep, tt, ff, vv = [], [], [], []
    for episode_id, (values, mask, names) in arrays.items():
        matrix = np.asarray(values, dtype=float)
        observed = np.isfinite(matrix)
        if mask is not None:
            observed &= ~np.asarray(mask, dtype=bool)
        rows = matrix.shape[0] if n_bins is None else n_bins
        for j, name in enumerate(names):
            if name not in schema:
                continue
            col = np.where(observed[:rows, j], matrix[:rows, j], math.nan)
            ep.extend([episode_id] * rows)
            tt.extend(range(rows))
            ff.extend([name] * rows)
            vv.extend(col.tolist())
    return ep, tt, ff, vv
