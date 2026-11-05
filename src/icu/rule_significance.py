from __future__ import annotations

import math

import numpy as np


def adjust_p_values_bh(p_values: list[float]) -> list[float]:
    values = [float(value) for value in p_values]
    count = len(values)
    if count == 0:
        return []

    normalized = []
    for idx, value in enumerate(values):
        if not math.isfinite(value):
            bounded = 1.0
        else:
            bounded = min(1.0, max(0.0, value))
        normalized.append((bounded, idx))

    normalized.sort(key=lambda row: row[0])
    q_values = [1.0] * count
    running_min = 1.0
    for reverse_rank in range(count, 0, -1):
        p_value, original_idx = normalized[reverse_rank - 1]
        adjusted = (p_value * float(count)) / float(reverse_rank)
        running_min = min(running_min, adjusted)
        q_values[original_idx] = min(1.0, max(0.0, running_min))
    return q_values


def _smoothed_rate(hits: int, total: int, prior_count: float) -> float:
    prior = max(0.0, float(prior_count))
    return (float(hits) + prior) / max(1.0, float(total) + (2.0 * prior))


def _log2_rate_ratio(
    positive_hits: int,
    positive_total: int,
    control_hits: int,
    control_total: int,
    prior_count: float,
) -> float:
    positive_rate = _smoothed_rate(positive_hits, positive_total, prior_count)
    control_rate = _smoothed_rate(control_hits, control_total, prior_count)
    return math.log2(max(1e-12, positive_rate) / max(1e-12, control_rate))


def _hypergeom_probability(
    positive_hits: int,
    positive_total: int,
    control_hits: int,
    control_total: int,
) -> float:
    total_hits = int(positive_hits + control_hits)
    total = int(positive_total + control_total)
    if total <= 0:
        return 1.0
    return (
        math.comb(int(positive_total), int(positive_hits))
        * math.comb(int(control_total), int(control_hits))
        / float(math.comb(total, total_hits))
    )


def _fisher_exact_two_sided(
    positive_hits: int,
    positive_total: int,
    control_hits: int,
    control_total: int,
) -> float:
    total_hits = int(positive_hits + control_hits)
    min_positive_hits = max(0, total_hits - int(control_total))
    max_positive_hits = min(int(positive_total), total_hits)
    observed_probability = _hypergeom_probability(
        positive_hits,
        positive_total,
        control_hits,
        control_total,
    )
    p_value = 0.0
    for candidate_positive_hits in range(min_positive_hits, max_positive_hits + 1):
        candidate_control_hits = total_hits - candidate_positive_hits
        probability = _hypergeom_probability(
            candidate_positive_hits,
            positive_total,
            candidate_control_hits,
            control_total,
        )
        if probability <= observed_probability + 1e-12:
            p_value += probability
    return min(1.0, max(0.0, float(p_value)))


def _bootstrap_effect_ci(
    positive_hits: int,
    positive_total: int,
    control_hits: int,
    control_total: int,
    prior_count: float,
    n_bootstrap: int,
    random_state: int | None,
) -> tuple[float, float]:
    samples = max(0, int(n_bootstrap))
    if samples <= 0 or positive_total <= 0 or control_total <= 0:
        effect = _log2_rate_ratio(
            positive_hits,
            positive_total,
            control_hits,
            control_total,
            prior_count,
        )
        return effect, effect

    rng = np.random.default_rng(random_state)
    positive_rate = float(positive_hits) / float(max(1, positive_total))
    control_rate = float(control_hits) / float(max(1, control_total))
    effects = []
    for _ in range(samples):
        sampled_positive_hits = int(rng.binomial(int(positive_total), positive_rate))
        sampled_control_hits = int(rng.binomial(int(control_total), control_rate))
        effects.append(
            _log2_rate_ratio(
                sampled_positive_hits,
                positive_total,
                sampled_control_hits,
                control_total,
                prior_count,
            )
        )
    ordered = np.asarray(effects, dtype=float)
    return float(np.percentile(ordered, 2.5)), float(np.percentile(ordered, 97.5))


def estimate_rule_significance(
    positive_hits: int,
    positive_total: int,
    control_hits: int,
    control_total: int,
    n_bootstrap: int = 1000,
    random_state: int | None = None,
    prior_count: float = 0.5,
) -> dict:
    positive_hits = max(0, int(positive_hits))
    positive_total = max(0, int(positive_total))
    control_hits = max(0, int(control_hits))
    control_total = max(0, int(control_total))
    prior = max(0.0, float(prior_count))

    positive_rate = float(positive_hits) / float(max(1, positive_total))
    control_rate = float(control_hits) / float(max(1, control_total))
    effect_size = _log2_rate_ratio(
        positive_hits,
        positive_total,
        control_hits,
        control_total,
        prior,
    )
    ci_low, ci_high = _bootstrap_effect_ci(
        positive_hits,
        positive_total,
        control_hits,
        control_total,
        prior,
        n_bootstrap=n_bootstrap,
        random_state=random_state,
    )
    p_value = _fisher_exact_two_sided(
        positive_hits,
        positive_total,
        control_hits,
        control_total,
    )
    odds_ratio = ((positive_hits + prior) * ((control_total - control_hits) + prior)) / max(
        1e-12,
        ((control_hits + prior) * ((positive_total - positive_hits) + prior)),
    )
    return {
        "positive_rate": positive_rate,
        "control_rate": control_rate,
        "rate_difference": positive_rate - control_rate,
        "effect_size": float(effect_size),
        "odds_ratio": float(odds_ratio),
        "p_value": float(p_value),
        "ci_low": float(min(ci_low, ci_high)),
        "ci_high": float(max(ci_low, ci_high)),
        "bootstrap_samples": int(max(0, int(n_bootstrap))),
    }
