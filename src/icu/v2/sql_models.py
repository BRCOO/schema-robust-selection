"""Score a whole model in SQL (review of 2026-11-09, item 1): one portable query per model returns one score (the logit)
per episode of ``obs``, so registries and window-statistic models can be compared as in-database inference.

* :func:`registry_score_sql`: a registry's resolved programs (``sql_registry.program_sql``) left-joined to the episodes,
  each firing weighted by its probe coefficient (the frozen probe reads 0/1 coordinates; a coordinate with no executable
  branch is 0).
* :func:`statistic_score_sql`: the K-statistics model of ``run_stat_baselines.py``: the v1 handcrafted statistics of the
  chosen columns (min, max, mean, population std, last observed value, least-squares slope and missing rate per feature;
  transition count and on-ratio for status features), the training-median defaults of the v1 kernel (slope 0, missing
  rate 1, transition count and on-ratio 0 when undefined; min/max/mean/std/last their training medians), standardization
  and the logistic weights.

Both use the portable dialect only (no engine-specific function), so DuckDB, PostgreSQL and SQLite run the same text.
"""
from __future__ import annotations

from typing import Sequence

from src.icu.v2.sql_registry import _num, _q, program_sql

V1_STATS = ("min", "max", "mean", "std", "last", "slope", "missing_rate", "transition_count", "on_ratio")


def registry_score_sql(resolved: Sequence, coef: Sequence[float], intercept: float) -> str:
    """``resolved[k]`` is the branch coordinate k runs (a ProgramRecord) or None; returns (episode_id, score)."""
    ctes = ["ep AS (SELECT DISTINCT episode_id FROM obs)"]
    terms, joins = [_num(intercept)], []
    for k, (branch, c) in enumerate(zip(resolved, coef)):
        if branch is None or float(c) == 0.0:
            continue
        ctes.append(f"p{k} AS ({program_sql(branch, 'portable')})")
        terms.append(f"{_num(c)} * (CASE WHEN p{k}.episode_id IS NULL THEN 0 ELSE 1 END)")
        joins.append(f"LEFT JOIN p{k} ON p{k}.episode_id = ep.episode_id")
    return "WITH " + ",\n".join(ctes) + "\nSELECT ep.episode_id, " + " + ".join(terms) + " AS score FROM ep\n" + "\n".join(joins)


def _statistic_ctes(j: int, feature: str, stat: str) -> list[str]:
    f = _q(feature)
    observed = f"FROM obs WHERE feature = {f} AND value IS NOT NULL"
    if stat in ("min", "max", "mean"):
        agg = {"min": "MIN(value)", "max": "MAX(value)", "mean": "AVG(value)"}[stat]
        return [f"f{j} AS (SELECT episode_id, {agg} AS v {observed} GROUP BY episode_id)"]
    if stat == "std":
        return [f"o{j} AS (SELECT episode_id, value {observed})",
                f"m{j} AS (SELECT episode_id, AVG(value) AS mu FROM o{j} GROUP BY episode_id)",
                f"f{j} AS (SELECT o{j}.episode_id, SQRT(AVG((o{j}.value - m{j}.mu) * (o{j}.value - m{j}.mu))) AS v "
                f"FROM o{j} JOIN m{j} ON o{j}.episode_id = m{j}.episode_id GROUP BY o{j}.episode_id)"]
    if stat == "last":
        return [f"o{j} AS (SELECT episode_id, value, ROW_NUMBER() OVER (PARTITION BY episode_id ORDER BY t DESC) AS rn {observed})",
                f"f{j} AS (SELECT episode_id, value AS v FROM o{j} WHERE rn = 1)"]
    if stat == "slope":
        return [f"o{j} AS (SELECT episode_id, t, value {observed})",
                f"m{j} AS (SELECT episode_id, AVG(CAST(t AS DOUBLE PRECISION)) AS tm, AVG(value) AS xm FROM o{j} GROUP BY episode_id HAVING COUNT(*) >= 2)",
                f"f{j} AS (SELECT o{j}.episode_id, SUM((o{j}.t - m{j}.tm) * (o{j}.value - m{j}.xm)) / SUM((o{j}.t - m{j}.tm) * (o{j}.t - m{j}.tm)) AS v "
                f"FROM o{j} JOIN m{j} ON o{j}.episode_id = m{j}.episode_id GROUP BY o{j}.episode_id)"]
    if stat == "missing_rate":
        return [f"f{j} AS (SELECT episode_id, 1.0 - CAST(COUNT(value) AS DOUBLE PRECISION) / COUNT(*) AS v FROM obs WHERE feature = {f} GROUP BY episode_id)"]
    if stat == "transition_count":
        return [f"o{j} AS (SELECT episode_id, t, (CASE WHEN value > 0.5 THEN 1 ELSE 0 END) AS s {observed})",
                f"l{j} AS (SELECT episode_id, s, LAG(s) OVER (PARTITION BY episode_id ORDER BY t) AS p FROM o{j})",
                f"f{j} AS (SELECT episode_id, CAST(SUM(CASE WHEN p IS NOT NULL AND s <> p THEN 1 ELSE 0 END) AS DOUBLE PRECISION) AS v "
                f"FROM l{j} GROUP BY episode_id)"]
    if stat == "on_ratio":
        return [f"f{j} AS (SELECT episode_id, AVG(CAST((CASE WHEN value > 0.5 THEN 1 ELSE 0 END) AS DOUBLE PRECISION)) AS v {observed} GROUP BY episode_id)"]
    raise ValueError(stat)


def statistic_default(stat: str, median: float) -> float:
    """The v1 kernel's value when the statistic is undefined (the feature unobserved or absent)."""
    return {"slope": 0.0, "missing_rate": 1.0, "transition_count": 0.0, "on_ratio": 0.0}.get(stat, float(median))


def statistic_score_sql(columns: Sequence[tuple[str, str]], defaults: Sequence[float], means: Sequence[float], scales: Sequence[float],
                        coef: Sequence[float], intercept: float) -> str:
    """``columns``: (feature, v1 stat) per model input; returns (episode_id, score) with score = intercept + coef . z."""
    ctes = ["ep AS (SELECT DISTINCT episode_id FROM obs)"]
    terms, joins = [_num(intercept)], []
    for j, ((feature, stat), d, mu, sd, c) in enumerate(zip(columns, defaults, means, scales, coef)):
        ctes.extend(_statistic_ctes(j, feature, stat))
        terms.append(f"{_num(c)} * ((COALESCE(f{j}.v, {_num(d)}) - {_num(mu)}) / {_num(sd)})")
        joins.append(f"LEFT JOIN f{j} ON f{j}.episode_id = ep.episode_id")
    return "WITH " + ",\n".join(ctes) + "\nSELECT ep.episode_id, " + " + ".join(terms) + " AS score FROM ep\n" + "\n".join(joins)
