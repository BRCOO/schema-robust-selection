"""One query per model: a shared scan for all window statistics (added 2026-11-11, third review, substance item 2).
Additive; ``sql_registry`` and ``sql_models`` are unchanged.

The per-program queries of ``sql_registry`` and the per-statistic CTEs of ``sql_models`` scan ``obs`` once per predicate or
statistic. Here one block computes every needed statistic per (episode, feature) from one scan of ``obs`` restricted to the
features in use and one aggregation, and the predicates or the K-statistics model read it:

* ``fused_predicates_sql(predicates)`` -> ``(episode_id, f0, ..., f{n-1})``, each predicate's 0/1 firing;
* ``fused_registry_score_sql(resolved, coef, intercept)`` -> ``(episode_id, score)``, as ``sql_models.registry_score_sql``
  (programs that are not predicates keep their own query);
* ``fused_statistic_score_sql(columns, defaults, means, scales, coef, intercept)`` -> ``(episode_id, score)``, as
  ``sql_models.statistic_score_sql``;
* ``per_program_score_sql(resolved, coef, intercept, dialect)``: ``registry_score_sql`` in either dialect (the baseline).

Two dialects, as in ``sql_registry``: ``portable`` (DuckDB, PostgreSQL, SQLite: the last observed bin by ROW_NUMBER, two-pass
standard deviation and slope) and ``duckdb`` (arg_max, STDDEV_POP, REGR_SLOPE, no window pass unless transitions are needed).
Semantics follow the per-program queries: statistics over observed rows only, the missing share over all bins of the dense
grid, the last observed bin, a slope only with two observations, status features read as on/off at
``INTERVENTION_ON_THRESHOLD`` (predicates, on-ratio, transitions), the same tolerance-shifted predicate bounds, and the v1
defaults (``COALESCE``) where a statistic is undefined.
"""
from __future__ import annotations

from typing import Sequence

from src.icu.v2.sql_registry import INTERVENTION_FEATURES, INTERVENTION_ON_THRESHOLD, _num, _q, program_sql

__all__ = ["fused_predicates_sql", "fused_registry_score_sql", "fused_statistic_score_sql", "per_program_score_sql", "is_predicate"]

#: v1 kernel statistic (sql_models) -> column of the shared block
_V1 = {"min": "s.vmin", "max": "s.vmax", "mean": "s.vmean", "std": "s.vstd", "last": "s.vlast", "slope": "s.vslope",
       "missing_rate": "1.0 - CAST(s.nobs AS DOUBLE PRECISION) / s.nbins", "transition_count": "CAST(s.ntrans AS DOUBLE PRECISION)",
       "on_ratio": "s.onr"}


def is_predicate(program) -> bool:
    return program is not None and program.kind == "predicate"


def _stat_block(features: Sequence[str], need_last: bool, need_trans: bool, need_moments: bool, dialect: str) -> list[str]:
    """CTEs ending in ``s(episode_id, feature, nbins, nobs, vmin, vmax, vmean, onr[, vlast][, ntrans][, vstd, vslope])``."""
    assert dialect in ("portable", "duckdb"), dialect
    duck = dialect == "duckdb"
    on = _num(INTERVENTION_ON_THRESHOLD)
    state = f"(CASE WHEN value > {on} THEN 1 ELSE 0 END)"
    window = []
    if need_last and not duck:
        window.append("ROW_NUMBER() OVER (PARTITION BY episode_id, feature ORDER BY t DESC) AS rn")
    if need_trans:
        window.append(f"LAG({state}) OVER (PARTITION BY episode_id, feature ORDER BY t) AS p")
    aggs = ["MIN(value) AS vmin", "MAX(value) AS vmax", "AVG(value) AS vmean", f"AVG(CAST({state} AS DOUBLE PRECISION)) AS onr",
            "COUNT(*) AS nobs2"]
    if need_moments and not duck:
        aggs.append("AVG(CAST(t AS DOUBLE PRECISION)) AS tm")
    if need_last:
        aggs.append("arg_max(value, t) AS vlast" if duck else "MAX(CASE WHEN rn = 1 THEN value END) AS vlast")
    if need_trans:
        aggs.append(f"SUM(CASE WHEN p IS NOT NULL AND {state} <> p THEN 1 ELSE 0 END) AS ntrans")
    if need_moments and duck:
        aggs += ["STDDEV_POP(value) AS vstd", "(CASE WHEN COUNT(value) >= 2 THEN REGR_SLOPE(value, t) END) AS vslope"]
    source = f"(SELECT {', '.join(['episode_id', 'feature', 't', 'value', *window])} FROM o WHERE value IS NOT NULL)" if window else \
        "(SELECT episode_id, feature, t, value FROM o WHERE value IS NOT NULL)"
    ctes = [f"o AS (SELECT episode_id, feature, t, value FROM obs WHERE feature IN ({', '.join(_q(f) for f in sorted(set(features)))}))",
            f"w AS {source}",
            "a AS (SELECT episode_id, feature, COUNT(*) AS nbins, COUNT(value) AS nobs FROM o GROUP BY episode_id, feature)",
            f"b AS (SELECT episode_id, feature, {', '.join(aggs)} FROM w GROUP BY episode_id, feature)"]
    cols = ["a.episode_id", "a.feature", "a.nbins", "a.nobs", "b.vmin", "b.vmax", "b.vmean", "b.onr"]
    cols += ["b.vlast"] if need_last else []
    cols += ["b.ntrans"] if need_trans else []
    joins = ["LEFT JOIN b ON a.episode_id = b.episode_id AND a.feature = b.feature"]
    if need_moments and duck:
        cols += ["b.vstd", "b.vslope"]
    elif need_moments:
        ctes.append("c AS (SELECT w.episode_id, w.feature, SQRT(AVG((w.value - b.vmean) * (w.value - b.vmean))) AS vstd, "
                    "SUM((w.t - b.tm) * (w.value - b.vmean)) AS sxy, SUM((w.t - b.tm) * (w.t - b.tm)) AS sxx "
                    "FROM w JOIN b ON w.episode_id = b.episode_id AND w.feature = b.feature GROUP BY w.episode_id, w.feature)")
        cols += ["c.vstd", "(CASE WHEN b.nobs2 >= 2 THEN c.sxy / c.sxx END) AS vslope"]
        joins.append("LEFT JOIN c ON a.episode_id = c.episode_id AND a.feature = c.feature")
    ctes.append(f"s AS (SELECT {', '.join(cols)} FROM a {' '.join(joins)})")
    return ctes


def _cmp_bound(program) -> tuple[str, float]:
    from src.icu.v2.baselines.predicates import tolerance

    threshold = float(program.params["threshold"])
    if program.params["op"] == "ge":
        return ">=", threshold - tolerance(threshold)
    return "<=", threshold + tolerance(threshold)


def _condition(program) -> str:
    f, stat = program.params["feature"], program.params["stat"]
    cmp, bound = _cmp_bound(program)
    on = _num(INTERVENTION_ON_THRESHOLD)
    expr = {"missing": "1.0 - CAST(s.nobs AS DOUBLE PRECISION) / s.nbins", "min": "s.vmin", "max": "s.vmax", "mean": "s.vmean",
            "std": "s.vstd", "slope": "s.vslope", "onratio": "s.onr", "transitions": "s.ntrans",
            "last": (f"(CASE WHEN s.vlast > {on} THEN 1.0 ELSE 0.0 END)" if f in INTERVENTION_FEATURES else "s.vlast")}[stat]
    return f"s.feature = {_q(f)} AND {expr} {cmp} {_num(bound)}"


def fused_predicates_sql(predicates: Sequence, dialect: str = "portable") -> str:
    assert predicates and all(is_predicate(p) for p in predicates)
    stats = {p.params["stat"] for p in predicates}
    ctes = _stat_block([p.params["feature"] for p in predicates], "last" in stats, "transitions" in stats,
                       bool(stats & {"std", "slope"}), dialect)
    fires = [f"MAX(CASE WHEN {_condition(p)} THEN 1 ELSE 0 END) AS f{i}" for i, p in enumerate(predicates)]
    return "WITH " + ",\n".join(ctes) + "\nSELECT s.episode_id, " + ", ".join(fires) + " FROM s GROUP BY s.episode_id"


def fused_registry_score_sql(resolved: Sequence, coef: Sequence[float], intercept: float, dialect: str = "portable") -> str:
    """``resolved[k]`` is the branch coordinate k runs (a ProgramRecord) or None; returns (episode_id, score)."""
    keep = [(k, b, float(c)) for k, (b, c) in enumerate(zip(resolved, coef)) if b is not None and float(c) != 0.0]
    preds = [(k, b, c) for k, b, c in keep if is_predicate(b)]
    others = [(k, b, c) for k, b, c in keep if not is_predicate(b)]
    ctes = ["ep AS (SELECT DISTINCT episode_id FROM obs)"]
    terms, joins = [_num(intercept)], []
    if preds:
        ctes.append("fp AS (" + fused_predicates_sql([b for _, b, _ in preds], dialect) + ")")
        terms += [f"{_num(c)} * COALESCE(fp.f{i}, 0)" for i, (_, _, c) in enumerate(preds)]
        joins.append("LEFT JOIN fp ON fp.episode_id = ep.episode_id")
    for k, b, c in others:
        ctes.append(f"p{k} AS ({program_sql(b, dialect)})")
        terms.append(f"{_num(c)} * (CASE WHEN p{k}.episode_id IS NULL THEN 0 ELSE 1 END)")
        joins.append(f"LEFT JOIN p{k} ON p{k}.episode_id = ep.episode_id")
    return "WITH " + ",\n".join(ctes) + "\nSELECT ep.episode_id, " + " + ".join(terms) + " AS score FROM ep\n" + "\n".join(joins)


def per_program_score_sql(resolved: Sequence, coef: Sequence[float], intercept: float, dialect: str = "portable") -> str:
    """``sql_models.registry_score_sql`` (one CTE, i.e. one scan, per program) in either dialect: the baseline."""
    ctes = ["ep AS (SELECT DISTINCT episode_id FROM obs)"]
    terms, joins = [_num(intercept)], []
    for k, (branch, c) in enumerate(zip(resolved, coef)):
        if branch is None or float(c) == 0.0:
            continue
        ctes.append(f"p{k} AS ({program_sql(branch, dialect)})")
        terms.append(f"{_num(c)} * (CASE WHEN p{k}.episode_id IS NULL THEN 0 ELSE 1 END)")
        joins.append(f"LEFT JOIN p{k} ON p{k}.episode_id = ep.episode_id")
    return "WITH " + ",\n".join(ctes) + "\nSELECT ep.episode_id, " + " + ".join(terms) + " AS score FROM ep\n" + "\n".join(joins)


def fused_statistic_score_sql(columns: Sequence[tuple[str, str]], defaults: Sequence[float], means: Sequence[float],
                              scales: Sequence[float], coef: Sequence[float], intercept: float, dialect: str = "portable") -> str:
    """``columns``: (feature, v1 stat) per model input; returns (episode_id, score) with score = intercept + coef . z."""
    stats = {s for _, s in columns}
    ctes = ["ep AS (SELECT DISTINCT episode_id FROM obs)"]
    ctes += _stat_block([f for f, _ in columns], "last" in stats, "transition_count" in stats, bool(stats & {"std", "slope"}), dialect)
    xs = [f"MAX(CASE WHEN s.feature = {_q(f)} THEN {_V1[st]} END) AS x{j}" for j, (f, st) in enumerate(columns)]
    ctes.append(f"fs AS (SELECT s.episode_id, {', '.join(xs)} FROM s GROUP BY s.episode_id)")
    terms = [_num(intercept)] + [f"{_num(c)} * ((COALESCE(fs.x{j}, {_num(d)}) - {_num(mu)}) / {_num(sd)})"
                                 for j, (d, mu, sd, c) in enumerate(zip(defaults, means, scales, coef))]
    return "WITH " + ",\n".join(ctes) + "\nSELECT ep.episode_id, " + " + ".join(terms) + " AS score FROM ep\nLEFT JOIN fs ON fs.episode_id = ep.episode_id"
