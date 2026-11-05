from __future__ import annotations

import math

from .shock_contrastive_miner import _mechanism_groups_for_features


def _mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(sum(values) / float(len(values)))


def _std(values: list[float]) -> float:
    if len(values) <= 1:
        return 0.0
    mean_value = _mean(values)
    variance = sum((float(value) - mean_value) ** 2 for value in values) / float(len(values))
    return float(math.sqrt(max(0.0, variance)))


def build_program_models(ranked_rules: list[dict], max_models: int = 20) -> list[dict]:
    if not ranked_rules:
        return []

    grouped: dict[tuple[str, ...], list[dict]] = {}
    for rule in ranked_rules:
        anchor_key = tuple(sorted(str(feature) for feature in rule.get("antecedent_features", [])))
        grouped.setdefault(anchor_key, []).append(rule)

    models = []
    for anchor_key, members in grouped.items():
        members_sorted = sorted(
            members,
            key=lambda row: (
                float(row.get("contrastive_score", 0.0)),
                float(row.get("effect_size", 0.0)),
                float(row.get("enrichment", 0.0)),
                str(row.get("rule_id", "")),
            ),
            reverse=True,
        )
        representative = members_sorted[0]
        consequent_union = sorted(
            {
                str(feature)
                for row in members_sorted
                for feature in list(row.get("consequent_features", []))
                if str(feature)
            }
        )
        mechanism_union = sorted(_mechanism_groups_for_features(consequent_union))
        scores = [float(row.get("contrastive_score", 0.0)) for row in members_sorted]
        effect_sizes = [float(row.get("effect_size", 0.0)) for row in members_sorted]
        enrichments = [float(row.get("enrichment", 0.0)) for row in members_sorted]
        lead_times = [float(row.get("lead_time_hours_mean", 0.0)) for row in members_sorted]
        p_values = [min(1.0, max(0.0, float(row.get("p_value", 1.0)))) for row in members_sorted]
        q_values = [min(1.0, max(0.0, float(row.get("q_value", 1.0)))) for row in members_sorted]
        significant_ratio = (
            sum(1 for row in members_sorted if min(1.0, max(0.0, float(row.get("q_value", row.get("p_value", 1.0))))) <= 0.05)
            / float(max(1, len(members_sorted)))
        )
        models.append(
            {
                "model_label": f"{' + '.join(anchor_key) if anchor_key else 'unknown'} mechanism model",
                "anchor_features": list(anchor_key),
                "consequent_feature_union": consequent_union,
                "mechanism_groups": mechanism_union,
                "rule_count": len(members_sorted),
                "representative_rule_id": representative.get("rule_id"),
                "representative_score": float(representative.get("contrastive_score", 0.0)),
                "mean_effect_size": _mean(effect_sizes),
                "effect_size_std": _std(effect_sizes),
                "mean_enrichment": _mean(enrichments),
                "lead_time_hours_mean": _mean(lead_times),
                "score_std": _std(scores),
                "mean_p_value": _mean(p_values),
                "q_value_min": min(q_values) if q_values else 1.0,
                "significant_rule_ratio": float(significant_ratio),
                "member_rule_ids": [row.get("rule_id") for row in members_sorted[:10]],
            }
        )

    ordered = sorted(
        models,
        key=lambda row: (
            float(row.get("representative_score", 0.0)),
            float(row.get("mean_effect_size", 0.0)),
            int(row.get("rule_count", 0)),
        ),
        reverse=True,
    )
    for idx, model in enumerate(ordered[: max(1, int(max_models))], start=1):
        model["program_model_id"] = f"program_model_{idx:03d}"
    return ordered[: max(1, int(max_models))]
