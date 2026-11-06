from __future__ import annotations

import heapq
import math

from .rule_significance import adjust_p_values_bh, estimate_rule_significance


MECHANISM_GROUP_BY_FEATURE: dict[str, str] = {
    "HR": "hemodynamic",
    "SBP": "hemodynamic",
    "DBP": "hemodynamic",
    "MAP": "hemodynamic",
    "RR": "respiratory",
    "SpO2": "respiratory",
    "Lactate": "perfusion_acidbase",
    "pH": "perfusion_acidbase",
    "Creatinine": "renal",
    "urine_output": "renal",
    "Bilirubin_Total": "hepatic",
    "WBC": "hematologic_inflammatory",
    "Platelets": "hematologic_inflammatory",
    "Glucose": "metabolic",
    "Temp": "metabolic_thermoregulation",
    "vasopressor_on": "care_process",
    "ventilation_on": "care_process",
    "sedation_on": "care_process",
}


def _bounded_float(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    return max(float(lower), min(float(upper), float(value)))


def _sparse_measurement_penalty_factor(
    sparse_ratio: float,
    sparse_penalty: float,
    dominance_start_ratio: float = 0.5,
    penalty_exponent: float = 2.0,
) -> float:
    ratio = _bounded_float(sparse_ratio, 0.0, 1.0)
    penalty_floor = _bounded_float(sparse_penalty, 0.0, 1.0)
    start = _bounded_float(dominance_start_ratio, 0.0, 1.0)
    if ratio <= start:
        return 1.0
    if ratio >= 1.0:
        return penalty_floor
    severity = (ratio - start) / max(1e-6, 1.0 - start)
    exponent = max(1e-6, float(penalty_exponent))
    curved_severity = severity**exponent
    return _bounded_float(1.0 - ((1.0 - penalty_floor) * curved_severity), penalty_floor, 1.0)


def _lead_time_quality_components(lead_time_hours: list[float]) -> tuple[float, float]:
    if not lead_time_hours:
        return 0.0, 0.0
    support_factor = min(1.0, float(len(lead_time_hours)) / 3.0)
    if len(lead_time_hours) <= 1:
        sharpness_factor = 0.5
    else:
        mean_hours = sum(lead_time_hours) / len(lead_time_hours)
        variance = sum((value - mean_hours) ** 2 for value in lead_time_hours) / len(lead_time_hours)
        stdev_hours = math.sqrt(max(0.0, variance))
        scale_hours = max(1.0, abs(mean_hours))
        sharpness_factor = 1.0 / (1.0 + (stdev_hours / scale_hours))
    quality_factor = support_factor * sharpness_factor
    return quality_factor, sharpness_factor


def _occurrence_start(occurrence) -> int | None:
    if hasattr(occurrence, "start"):
        try:
            return int(getattr(occurrence, "start"))
        except (TypeError, ValueError):
            return None
    if hasattr(occurrence, "positions"):
        positions = getattr(occurrence, "positions")
        if positions:
            try:
                return int(positions[0])
            except (TypeError, ValueError):
                return None
    if isinstance(occurrence, (tuple, list)) and occurrence:
        try:
            return int(occurrence[0])
        except (TypeError, ValueError):
            return None
    return None


def _estimate_rule_lead_time_steps(
    rule,
    motif_occurrences: dict | None = None,
    episode_length_steps: int | None = None,
) -> float:
    total_steps = int(episode_length_steps) if episode_length_steps is not None else 0
    if motif_occurrences and total_steps > 0:
        episode_end = total_steps - 1
        lead_times = []
        for occurrence in motif_occurrences.get(rule.consequent, []):
            start = _occurrence_start(occurrence)
            if start is None:
                continue
            lead_times.append(float(max(0, episode_end - int(start))))
        if lead_times:
            return float(sum(lead_times) / len(lead_times))
    return float(rule.delta_min + rule.delta_max) / 2.0


def _rule_anchor_signature(rule_row: dict) -> tuple[str, ...]:
    anchors = set()
    for feature in list(rule_row.get("antecedent_features", [])):
        base = str(feature).split(":", 1)[0]
        if base:
            anchors.add(base)
    return tuple(sorted(anchors))


def _rule_mechanism_signature(rule_row: dict) -> tuple[str, ...]:
    return tuple(sorted(_mechanism_groups_for_features(list(rule_row.get("consequent_features", [])))))


def _rule_sort_key(rule_row: dict) -> tuple[float, float, int]:
    return (
        float(rule_row.get("contrastive_score", 0.0)),
        float(rule_row.get("enrichment", 0.0)),
        int(rule_row.get("positive_count", 0)),
    )


def _care_process_ratio_for_features(feature_names: list[str]) -> float:
    if not feature_names:
        return 0.0
    mechanism_groups = []
    for feature in feature_names:
        base = str(feature).split(":", 1)[0]
        if not base:
            continue
        mechanism_groups.append(MECHANISM_GROUP_BY_FEATURE.get(base, f"other:{base}"))
    if not mechanism_groups:
        return 0.0
    care_count = sum(1 for group in mechanism_groups if group == "care_process")
    return float(care_count) / float(len(mechanism_groups))


def _prepare_rule_candidate_pool(
    ordered_rules: list[dict],
    anchor_candidate_top_n: int,
    mechanism_candidate_top_n: int,
    candidate_quality_floor_ratio: float,
    anchor_min_unique: int = 0,
    anchor_max_share: float = 1.0,
    mechanism_min_unique: int = 0,
) -> tuple[list[dict], dict]:
    if not ordered_rules:
        return [], {
            "total_ranked_rules": 0,
            "candidate_pool_size": 0,
            "anchor_bucket_count": 0,
            "mechanism_bucket_count": 0,
            "retained_by_quality_floor_count": 0,
            "retained_by_anchor_topn_count": 0,
            "retained_by_mechanism_topn_count": 0,
            "retained_union_count": 0,
            "quality_floor_ratio": 0.0,
            "quality_floor_threshold": 0.0,
            "candidate_anchor_unique_count": 0,
            "candidate_anchor_max_share": 0.0,
            "candidate_mechanism_unique_count": 0,
            "care_process_dominant_retained_count": 0,
        }

    anchor_top_n = max(0, int(anchor_candidate_top_n))
    mechanism_top_n = max(0, int(mechanism_candidate_top_n))
    quality_ratio = _bounded_float(candidate_quality_floor_ratio, 0.0, 1.0)
    min_anchor_unique = max(0, int(anchor_min_unique))
    max_anchor_share = _bounded_float(anchor_max_share, 0.0, 1.0)
    min_mechanism_unique = max(0, int(mechanism_min_unique))

    anchor_bucket_count = len({_rule_anchor_signature(rule) for rule in ordered_rules})
    mechanism_buckets = set()
    for rule in ordered_rules:
        mechanism_buckets.update(_mechanism_groups_for_features(list(rule.get("consequent_features", []))))
    mechanism_bucket_count = len(mechanism_buckets)

    diagnostics = {
        "total_ranked_rules": int(len(ordered_rules)),
        "candidate_pool_size": int(len(ordered_rules)),
        "anchor_bucket_count": int(anchor_bucket_count),
        "mechanism_bucket_count": int(mechanism_bucket_count),
        "retained_by_quality_floor_count": 0,
        "retained_by_anchor_topn_count": 0,
        "retained_by_mechanism_topn_count": 0,
        "retained_union_count": int(len(ordered_rules)),
        "quality_floor_ratio": float(quality_ratio),
        "quality_floor_threshold": 0.0,
        "candidate_anchor_unique_count": int(anchor_bucket_count),
        "candidate_anchor_max_share": 1.0 if ordered_rules else 0.0,
        "candidate_mechanism_unique_count": int(mechanism_bucket_count),
        "care_process_dominant_retained_count": 0,
    }

    if (
        anchor_top_n <= 0
        and mechanism_top_n <= 0
        and quality_ratio <= 0.0
        and min_anchor_unique <= 0
        and max_anchor_share >= 1.0
        and min_mechanism_unique <= 0
    ):
        return ordered_rules, diagnostics

    quality_keys: set[str] = set()
    anchor_keys: set[str] = set()
    mechanism_keys: set[str] = set()

    top_score = float(ordered_rules[0].get("contrastive_score", 0.0))
    quality_threshold = top_score * quality_ratio
    diagnostics["quality_floor_threshold"] = float(quality_threshold)
    if quality_ratio > 0.0:
        for rule in ordered_rules:
            if float(rule.get("contrastive_score", 0.0)) >= quality_threshold:
                quality_keys.add(str(rule.get("rule_id")))

    if anchor_top_n > 0:
        anchor_counts: dict[tuple[str, ...], int] = {}
        for rule in ordered_rules:
            anchor = _rule_anchor_signature(rule)
            count = anchor_counts.get(anchor, 0)
            if count >= anchor_top_n:
                continue
            anchor_keys.add(str(rule.get("rule_id")))
            anchor_counts[anchor] = count + 1

    if mechanism_top_n > 0:
        mechanism_counts: dict[str, int] = {}
        for rule in ordered_rules:
            groups = _mechanism_groups_for_features(list(rule.get("consequent_features", [])))
            if not groups:
                continue
            for group in sorted(groups):
                count = mechanism_counts.get(group, 0)
                if count >= mechanism_top_n:
                    continue
                mechanism_keys.add(str(rule.get("rule_id")))
                mechanism_counts[group] = count + 1

    candidate_keys = quality_keys.union(anchor_keys).union(mechanism_keys)
    if not candidate_keys:
        candidates = ordered_rules
        candidate_keys = {str(rule.get("rule_id")) for rule in ordered_rules}
    else:
        candidates = [rule for rule in ordered_rules if str(rule.get("rule_id")) in candidate_keys]

    def _candidate_anchor_counts(rows: list[dict]) -> dict[tuple[str, ...], int]:
        counts: dict[tuple[str, ...], int] = {}
        for rule in rows:
            anchor = _rule_anchor_signature(rule)
            counts[anchor] = counts.get(anchor, 0) + 1
        return counts

    def _candidate_mechanism_groups(rows: list[dict]) -> set[str]:
        groups = set()
        for rule in rows:
            groups.update(_mechanism_groups_for_features(list(rule.get("consequent_features", []))))
        return groups

    if min_anchor_unique > 0:
        anchor_counts = _candidate_anchor_counts(candidates)
        while len(anchor_counts) < min_anchor_unique:
            selected_ids = {str(rule.get("rule_id")) for rule in candidates}
            selected_anchors = set(anchor_counts)
            next_rule = None
            for rule in ordered_rules:
                rule_id = str(rule.get("rule_id"))
                if rule_id in selected_ids:
                    continue
                anchor = _rule_anchor_signature(rule)
                if anchor not in selected_anchors:
                    next_rule = rule
                    break
            if next_rule is None:
                break
            candidates.append(next_rule)
            anchor = _rule_anchor_signature(next_rule)
            anchor_counts[anchor] = anchor_counts.get(anchor, 0) + 1

    if min_mechanism_unique > 0:
        current_groups = _candidate_mechanism_groups(candidates)
        while len(current_groups) < min_mechanism_unique:
            selected_ids = {str(rule.get("rule_id")) for rule in candidates}
            next_rule = None
            for rule in ordered_rules:
                rule_id = str(rule.get("rule_id"))
                if rule_id in selected_ids:
                    continue
                groups = _mechanism_groups_for_features(list(rule.get("consequent_features", [])))
                if groups.difference(current_groups):
                    next_rule = rule
                    break
            if next_rule is None:
                break
            candidates.append(next_rule)
            current_groups.update(_mechanism_groups_for_features(list(next_rule.get("consequent_features", []))))

    if max_anchor_share < 1.0 and candidates:
        while True:
            anchor_counts = _candidate_anchor_counts(candidates)
            total = max(1, len(candidates))
            dominant_count = max(anchor_counts.values(), default=0)
            dominant_share = float(dominant_count) / float(total)
            if dominant_share <= max_anchor_share:
                break
            dominant_anchors = {anchor for anchor, count in anchor_counts.items() if count == dominant_count}
            selected_ids = {str(rule.get("rule_id")) for rule in candidates}
            next_rule = None
            for rule in ordered_rules:
                rule_id = str(rule.get("rule_id"))
                if rule_id in selected_ids:
                    continue
                anchor = _rule_anchor_signature(rule)
                if anchor not in dominant_anchors:
                    next_rule = rule
                    break
            if next_rule is not None:
                candidates.append(next_rule)
                continue

            min_anchor_keep = max(1, min_anchor_unique)
            removable_idx = None
            for idx in range(len(candidates) - 1, -1, -1):
                anchor = _rule_anchor_signature(candidates[idx])
                if anchor not in dominant_anchors:
                    continue
                if anchor_counts.get(anchor, 0) <= 1:
                    continue
                remaining_unique = len(anchor_counts) if anchor_counts.get(anchor, 0) > 1 else (len(anchor_counts) - 1)
                if remaining_unique < min_anchor_keep:
                    continue
                removable_idx = idx
                break
            if removable_idx is None:
                break
            candidates.pop(removable_idx)

    diagnostics["candidate_pool_size"] = int(len(candidates))
    diagnostics["retained_by_quality_floor_count"] = int(len(quality_keys))
    diagnostics["retained_by_anchor_topn_count"] = int(len(anchor_keys))
    diagnostics["retained_by_mechanism_topn_count"] = int(len(mechanism_keys))
    diagnostics["retained_union_count"] = int(len(candidate_keys))
    anchor_counts = _candidate_anchor_counts(candidates)
    diagnostics["candidate_anchor_unique_count"] = int(len(anchor_counts))
    diagnostics["candidate_anchor_max_share"] = (
        float(max(anchor_counts.values(), default=0) / max(1, len(candidates))) if candidates else 0.0
    )
    diagnostics["candidate_mechanism_unique_count"] = int(len(_candidate_mechanism_groups(candidates)))
    diagnostics["care_process_dominant_retained_count"] = int(
        sum(1 for rule in candidates if float(rule.get("care_process_ratio", 0.0)) > 0.5)
    )
    return candidates, diagnostics


def _event_id_to_label(event_id: int, event_labels: dict[int, str], feature_names: list[str], n_bins: int) -> str:
    if int(event_id) in event_labels:
        return event_labels[int(event_id)]
    events_per_var = int(n_bins) + 2
    var_idx, offset = divmod(int(event_id), events_per_var)
    if var_idx >= len(feature_names):
        return str(event_id)
    feature_name = feature_names[var_idx]
    if offset < int(n_bins):
        return f"{feature_name}:level_{offset}"
    if offset == int(n_bins):
        return f"{feature_name}:up"
    return f"{feature_name}:down"


def _itemset_to_label(itemset, event_labels: dict[int, str], feature_names: list[str], n_bins: int) -> str:
    labels = [_event_id_to_label(event_id, event_labels, feature_names, n_bins) for event_id in itemset]
    return "{" + ", ".join(labels) + "}"


def _motif_to_label(motif, event_labels: dict[int, str], feature_names: list[str], n_bins: int) -> str:
    return " -> ".join(_itemset_to_label(itemset, event_labels, feature_names, n_bins) for itemset in motif.itemsets)


def extract_rule_records(
    rules,
    event_labels: dict[int, str],
    feature_names: list[str],
    n_bins: int,
    resample_minutes: int = 60,
    motif_occurrences: dict | None = None,
    episode_length_steps: int | None = None,
) -> list[dict]:
    records = []
    for rule in rules:
        antecedent_labels = [
            _event_id_to_label(event_id, event_labels, feature_names, n_bins)
            for itemset in rule.antecedent.itemsets
            for event_id in itemset
        ]
        consequent_labels = [
            _event_id_to_label(event_id, event_labels, feature_names, n_bins)
            for itemset in rule.consequent.itemsets
            for event_id in itemset
        ]
        all_labels = antecedent_labels + consequent_labels
        is_missing_only = bool(all_labels) and all(":missing" in label for label in all_labels)
        sparse_count = sum(1 for label in all_labels if ":sparse_measurement" in label)
        sparse_ratio = (float(sparse_count) / float(len(all_labels))) if all_labels else 0.0
        is_sparse_measurement_dominant = bool(all_labels) and sparse_count * 2 >= len(all_labels)
        records.append(
            {
                "rule_id": f"{_motif_to_label(rule.antecedent, event_labels, feature_names, n_bins)} => {_motif_to_label(rule.consequent, event_labels, feature_names, n_bins)}",
                "antecedent_features": sorted({label.split(":", 1)[0] for label in antecedent_labels}),
                "consequent_features": sorted({label.split(":", 1)[0] for label in consequent_labels}),
                "lead_time_steps": _estimate_rule_lead_time_steps(
                    rule,
                    motif_occurrences=motif_occurrences,
                    episode_length_steps=episode_length_steps,
                ),
                "resample_minutes": int(resample_minutes),
                "is_missing_only": is_missing_only,
                "is_sparse_measurement_dominant": is_sparse_measurement_dominant,
                "sparse_measurement_ratio": sparse_ratio,
            }
        )
    return records


def score_contrastive_rule_records(
    episode_records: list[dict],
    exclude_missing_only: bool = True,
    exclude_sparse_measurement_dominant: bool = False,
    sparse_measurement_penalty: float = 1.0,
    sparse_measurement_penalty_exponent: float = 2.0,
    rule_anchor_saturation_penalty_weight: float = 0.0,
    rule_anchor_saturation_penalty_exponent: float = 1.0,
    rule_mechanism_saturation_penalty_weight: float = 0.0,
    rule_mechanism_saturation_penalty_exponent: float = 1.0,
    lead_time_quality_weight: float = 0.0,
    rule_support_reliability_weight: float = 0.0,
    rule_support_reliability_exponent: float = 0.5,
    min_positive_count: int = 1,
    min_control_count: int = 0,
    allow_zero_control_candidates: bool = False,
    enrichment_rate_prior_count: float = 0.0,
    zero_control_rule_penalty: float = 1.0,
    min_contrastive_score: float = 0.0,
    rule_anchor_candidate_top_n: int = 0,
    rule_mechanism_candidate_top_n: int = 0,
    rule_candidate_quality_floor_ratio: float = 0.0,
    rule_care_process_max_ratio: float = 1.0,
    rule_care_process_penalty: float = 1.0,
    rule_anchor_min_unique: int = 0,
    rule_anchor_max_share: float = 1.0,
    rule_mechanism_min_unique: int = 0,
    significance_weight: float = 0.0,
    significance_prior_count: float = 0.5,
    significance_bootstrap_samples: int = 1000,
    significance_random_state: int | None = None,
    significance_max_q_value: float = 1.0,
    return_candidate_pool_diagnostics: bool = False,
) -> list[dict] | tuple[list[dict], dict]:
    positive_episodes = [record["episode_id"] for record in episode_records if record.get("episode_type") == "positive"]
    control_episodes = [record["episode_id"] for record in episode_records if record.get("episode_type") == "control"]
    positive_total = max(1, len(positive_episodes))
    control_total = max(1, len(control_episodes))
    min_positive = max(0, int(min_positive_count))
    min_control = max(0, int(min_control_count))
    min_score = max(0.0, float(min_contrastive_score))
    sparse_penalty = _bounded_float(sparse_measurement_penalty, 0.0, 1.0)
    sparse_penalty_exponent = max(1e-6, float(sparse_measurement_penalty_exponent))
    anchor_saturation_weight = _bounded_float(rule_anchor_saturation_penalty_weight, 0.0, 1.0)
    anchor_saturation_exponent = max(1e-6, float(rule_anchor_saturation_penalty_exponent))
    mechanism_saturation_weight = _bounded_float(rule_mechanism_saturation_penalty_weight, 0.0, 1.0)
    mechanism_saturation_exponent = max(1e-6, float(rule_mechanism_saturation_penalty_exponent))
    lead_time_weight = _bounded_float(lead_time_quality_weight, 0.0, 1.0)
    support_reliability_weight = _bounded_float(rule_support_reliability_weight, 0.0, 1.0)
    support_reliability_exponent = max(1e-6, float(rule_support_reliability_exponent))
    allow_zero_control = bool(allow_zero_control_candidates)
    rate_prior = max(0.0, float(enrichment_rate_prior_count))
    zero_control_penalty = _bounded_float(zero_control_rule_penalty, 0.0, 1.0)
    care_process_max_ratio = _bounded_float(rule_care_process_max_ratio, 0.0, 1.0)
    care_process_penalty = _bounded_float(rule_care_process_penalty, 0.0, 1.0)
    significance_weight = _bounded_float(significance_weight, 0.0, 1.0)
    significance_prior_count = max(0.0, float(significance_prior_count))
    significance_bootstrap_samples = max(0, int(significance_bootstrap_samples))
    significance_max_q_value = _bounded_float(significance_max_q_value, 0.0, 1.0)
    compute_significance = bool(significance_weight > 0.0 or significance_max_q_value < 1.0)

    aggregated: dict[str, dict] = {}
    for episode in episode_records:
        unique_rules = {}
        for rule in episode.get("rules", []):
            if exclude_missing_only and bool(rule.get("is_missing_only", False)):
                continue
            if exclude_sparse_measurement_dominant and bool(rule.get("is_sparse_measurement_dominant", False)):
                continue
            unique_rules.setdefault(rule["rule_id"], rule)

        for rule_id, rule in unique_rules.items():
            entry = aggregated.setdefault(
                rule_id,
                {
                    "rule_id": rule_id,
                    "antecedent_features": list(rule.get("antecedent_features", [])),
                    "consequent_features": list(rule.get("consequent_features", [])),
                    "is_sparse_measurement_dominant": False,
                    "positive_episode_ids": set(),
                    "control_episode_ids": set(),
                    "lead_time_steps": [],
                    "lead_time_hours": [],
                    "sparse_measurement_ratios": [],
                },
            )
            entry["is_sparse_measurement_dominant"] = bool(entry["is_sparse_measurement_dominant"]) or bool(
                rule.get("is_sparse_measurement_dominant", False)
            )
            sparse_ratio = rule.get("sparse_measurement_ratio")
            if sparse_ratio is None:
                sparse_ratio = 1.0 if bool(rule.get("is_sparse_measurement_dominant", False)) else 0.0
            entry["sparse_measurement_ratios"].append(_bounded_float(float(sparse_ratio), 0.0, 1.0))
            if episode.get("episode_type") == "positive":
                entry["positive_episode_ids"].add(episode["episode_id"])
                lead_steps = float(rule.get("lead_time_steps", 0.0))
                minutes = float(rule.get("resample_minutes", 60))
                entry["lead_time_steps"].append(lead_steps)
                entry["lead_time_hours"].append((lead_steps * minutes) / 60.0)
            elif episode.get("episode_type") == "control":
                entry["control_episode_ids"].add(episode["episode_id"])

    ranked = []
    eps = 1e-6
    for entry in aggregated.values():
        positive_count = len(entry["positive_episode_ids"])
        control_count = len(entry["control_episode_ids"])
        zero_control_candidate = allow_zero_control and control_count == 0 and positive_count >= min_positive
        if positive_count < min_positive:
            continue
        if control_count < min_control and not zero_control_candidate:
            continue
        care_process_ratio = _care_process_ratio_for_features(list(entry.get("consequent_features", [])))
        if care_process_ratio > care_process_max_ratio:
            continue

        if rate_prior > 0.0:
            positive_rate = (positive_count + rate_prior) / (positive_total + (2.0 * rate_prior))
            control_rate = (control_count + rate_prior) / (control_total + (2.0 * rate_prior))
        else:
            positive_rate = positive_count / positive_total
            control_rate = control_count / control_total
        enrichment = (positive_rate + eps) / (control_rate + eps)
        overlap = len(set(entry["antecedent_features"]) & set(entry["consequent_features"]))
        penalty = 1.0 / (1.0 + float(overlap))
        base_contrastive_score = max(0.0, math.log2(enrichment)) * penalty
        sparse_ratio = (
            sum(entry["sparse_measurement_ratios"]) / len(entry["sparse_measurement_ratios"])
            if entry["sparse_measurement_ratios"]
            else 0.0
        )
        sparse_penalty_factor = _sparse_measurement_penalty_factor(
            sparse_ratio,
            sparse_penalty=sparse_penalty,
            penalty_exponent=sparse_penalty_exponent,
        )
        lead_time_quality_factor, lead_time_sharpness_factor = _lead_time_quality_components(entry["lead_time_hours"])
        lead_time_quality_blend_factor = (1.0 - lead_time_weight) + (lead_time_weight * lead_time_quality_factor)
        support_ratio = (positive_count + control_count) / float(max(1, positive_total + control_total))
        support_reliability_base = _bounded_float(support_ratio, 0.0, 1.0) ** support_reliability_exponent
        support_reliability_factor = (
            (1.0 - support_reliability_weight) + (support_reliability_weight * support_reliability_base)
        )
        care_process_penalty_factor = 1.0 - ((1.0 - care_process_penalty) * care_process_ratio)
        if compute_significance:
            significance = estimate_rule_significance(
                positive_hits=positive_count,
                positive_total=positive_total,
                control_hits=control_count,
                control_total=control_total,
                n_bootstrap=significance_bootstrap_samples,
                random_state=significance_random_state,
                prior_count=significance_prior_count,
            )
            ci_width = max(0.0, float(significance["ci_high"]) - float(significance["ci_low"]))
            significance_factor = _bounded_float(
                (1.0 - float(significance["p_value"])) * (1.0 / (1.0 + ci_width)),
                0.0,
                1.0,
            )
        else:
            if significance_prior_count > 0.0:
                sig_positive_rate = (positive_count + significance_prior_count) / (
                    positive_total + (2.0 * significance_prior_count)
                )
                sig_control_rate = (control_count + significance_prior_count) / (
                    control_total + (2.0 * significance_prior_count)
                )
            else:
                sig_positive_rate = positive_count / positive_total
                sig_control_rate = control_count / control_total
            effect_size = math.log2(max(1e-12, sig_positive_rate) / max(1e-12, sig_control_rate))
            odds_ratio = (
                (positive_count + significance_prior_count)
                * ((control_total - control_count) + significance_prior_count)
            ) / max(
                1e-12,
                (control_count + significance_prior_count)
                * ((positive_total - positive_count) + significance_prior_count),
            )
            significance = {
                "effect_size": float(effect_size),
                "odds_ratio": float(odds_ratio),
                "p_value": 1.0,
                "ci_low": float(effect_size),
                "ci_high": float(effect_size),
            }
            significance_factor = 1.0
        contrastive_score = (
            base_contrastive_score
            * sparse_penalty_factor
            * lead_time_quality_blend_factor
            * support_reliability_factor
            * care_process_penalty_factor
        )
        contrastive_score = contrastive_score * ((1.0 - significance_weight) + (significance_weight * significance_factor))
        if zero_control_candidate:
            contrastive_score = contrastive_score * zero_control_penalty
        if contrastive_score < min_score:
            continue
        lead_time_steps_mean = (
            sum(entry["lead_time_steps"]) / len(entry["lead_time_steps"]) if entry["lead_time_steps"] else 0.0
        )
        lead_time_hours_mean = (
            sum(entry["lead_time_hours"]) / len(entry["lead_time_hours"]) if entry["lead_time_hours"] else 0.0
        )
        ranked.append(
            {
                "rule_id": entry["rule_id"],
                "antecedent_features": entry["antecedent_features"],
                "consequent_features": entry["consequent_features"],
                "positive_count": positive_count,
                "control_count": control_count,
                "positive_total": int(positive_total),
                "control_total": int(control_total),
                "positive_rate": positive_rate,
                "control_rate": control_rate,
                "enrichment": enrichment,
                "self_persistence_penalty": penalty,
                "base_contrastive_score": base_contrastive_score,
                "sparse_measurement_penalty_factor": sparse_penalty_factor,
                "sparse_measurement_penalty_exponent": sparse_penalty_exponent,
                "sparse_measurement_ratio": sparse_ratio,
                "lead_time_quality_factor": lead_time_quality_factor,
                "lead_time_sharpness_factor": lead_time_sharpness_factor,
                "lead_time_quality_blend_factor": lead_time_quality_blend_factor,
                "support_ratio": support_ratio,
                "support_reliability_exponent": support_reliability_exponent,
                "support_reliability_factor": support_reliability_factor,
                "care_process_ratio": care_process_ratio,
                "care_process_penalty_factor": care_process_penalty_factor,
                "effect_size": float(significance["effect_size"]),
                "odds_ratio": float(significance["odds_ratio"]),
                "p_value": float(significance["p_value"]),
                "ci_low": float(significance["ci_low"]),
                "ci_high": float(significance["ci_high"]),
                "q_value": 1.0,
                "significant_at_0_05": False,
                "significance_factor": significance_factor,
                "anchor_saturation_ratio": 0.0,
                "anchor_saturation_penalty_factor": 1.0,
                "mechanism_saturation_ratio": 0.0,
                "mechanism_saturation_penalty_factor": 1.0,
                "zero_control_candidate": bool(zero_control_candidate),
                "contrastive_score": contrastive_score,
                "lead_time_steps_mean": lead_time_steps_mean,
                "lead_time_hours_mean": lead_time_hours_mean,
            }
        )

    if ranked and compute_significance:
        q_values = adjust_p_values_bh([float(row.get("p_value", 1.0)) for row in ranked])
        for row, q_value in zip(ranked, q_values):
            row["q_value"] = float(q_value)
            row["significant_at_0_05"] = bool(float(q_value) <= 0.05)
        if significance_max_q_value < 1.0:
            ranked = [
                row
                for row in ranked
                if float(row.get("q_value", 1.0)) <= float(significance_max_q_value)
            ]

    if anchor_saturation_weight > 0.0 and ranked:
        anchor_counts: dict[tuple[str, ...], int] = {}
        for row in ranked:
            anchor = _rule_anchor_signature(row)
            anchor_counts[anchor] = anchor_counts.get(anchor, 0) + 1
        max_count = max(anchor_counts.values(), default=1)
        for row in ranked:
            anchor = _rule_anchor_signature(row)
            saturation_ratio = float(anchor_counts.get(anchor, 0)) / float(max_count)
            saturation_penalty = _bounded_float(
                1.0 - (anchor_saturation_weight * (saturation_ratio**anchor_saturation_exponent)),
                0.0,
                1.0,
            )
            row["anchor_saturation_ratio"] = saturation_ratio
            row["anchor_saturation_penalty_factor"] = saturation_penalty
            row["contrastive_score"] = float(row.get("contrastive_score", 0.0)) * saturation_penalty

    if mechanism_saturation_weight > 0.0 and ranked:
        mechanism_counts: dict[str, int] = {}
        for row in ranked:
            for mechanism in _rule_mechanism_signature(row):
                mechanism_counts[mechanism] = mechanism_counts.get(mechanism, 0) + 1
        max_count = max(mechanism_counts.values(), default=1)
        for row in ranked:
            mechanisms = _rule_mechanism_signature(row)
            if not mechanisms:
                row["mechanism_saturation_ratio"] = 0.0
                row["mechanism_saturation_penalty_factor"] = 1.0
                continue
            saturation_ratio = max(
                float(mechanism_counts.get(mechanism, 0)) / float(max_count)
                for mechanism in mechanisms
            )
            saturation_penalty = _bounded_float(
                1.0 - (mechanism_saturation_weight * (saturation_ratio**mechanism_saturation_exponent)),
                0.0,
                1.0,
            )
            row["mechanism_saturation_ratio"] = saturation_ratio
            row["mechanism_saturation_penalty_factor"] = saturation_penalty
            row["contrastive_score"] = float(row.get("contrastive_score", 0.0)) * saturation_penalty

    ranked.sort(key=_rule_sort_key, reverse=True)
    candidate_pool, candidate_pool_diagnostics = _prepare_rule_candidate_pool(
        ranked,
        anchor_candidate_top_n=rule_anchor_candidate_top_n,
        mechanism_candidate_top_n=rule_mechanism_candidate_top_n,
        candidate_quality_floor_ratio=rule_candidate_quality_floor_ratio,
        anchor_min_unique=rule_anchor_min_unique,
        anchor_max_share=rule_anchor_max_share,
        mechanism_min_unique=rule_mechanism_min_unique,
    )
    if return_candidate_pool_diagnostics:
        return candidate_pool, candidate_pool_diagnostics
    return candidate_pool


def _program_key(rule: dict) -> tuple[tuple[str, ...], tuple[str, ...]]:
    antecedent = tuple(sorted(set(rule.get("antecedent_features", []))))
    consequent = tuple(sorted(set(rule.get("consequent_features", []))))
    return antecedent, consequent


def _program_sort_key(program: dict) -> tuple[float, float, int]:
    return (
        float(program.get("representative_score", 0.0)),
        float(program.get("mean_contrastive_score", 0.0)),
        int(program.get("rule_count", 0)),
    )


def _lead_time_diversity_bonus(program: dict, selected_programs: list[dict], gap_hours: float) -> float:
    if not selected_programs:
        return 0.0
    lead_time = float(program.get("lead_time_hours_mean", 0.0))
    min_diff = min(abs(lead_time - float(row.get("lead_time_hours_mean", 0.0))) for row in selected_programs)
    gap = max(1e-6, float(gap_hours))
    return min(1.0, min_diff / gap)


def _mechanism_groups_for_features(feature_names: list[str]) -> set[str]:
    groups = set()
    for feature in feature_names:
        base = str(feature).split(":", 1)[0]
        if not base:
            continue
        groups.add(MECHANISM_GROUP_BY_FEATURE.get(base, f"other:{base}"))
    return groups


def _program_mechanism_diversity_bonus(program: dict, selected_programs: list[dict]) -> float:
    if not selected_programs:
        return 0.0
    candidate_groups = set(program.get("mechanism_groups", []))
    if not candidate_groups:
        return 0.0

    covered_groups = set()
    for row in selected_programs:
        covered_groups.update(set(row.get("mechanism_groups", [])))
    if not covered_groups:
        return 0.0

    unseen_ratio = len(candidate_groups.difference(covered_groups)) / float(len(candidate_groups))
    if unseen_ratio > 0.0:
        return float(min(1.0, unseen_ratio))

    similarities = []
    for row in selected_programs:
        groups = set(row.get("mechanism_groups", []))
        union = candidate_groups.union(groups)
        if not union:
            continue
        similarities.append(len(candidate_groups.intersection(groups)) / float(len(union)))
    if not similarities:
        return 0.0
    return max(0.0, 1.0 - max(similarities))


def _program_care_process_penalty_value(program: dict) -> float:
    groups = set(program.get("mechanism_groups", []))
    if not groups:
        return 0.0
    return len([group for group in groups if str(group) == "care_process"]) / float(len(groups))


def _program_has_care_process(program: dict) -> bool:
    return any(str(group) == "care_process" for group in set(program.get("mechanism_groups", [])))


def _program_anchor_signature(program: dict) -> tuple[str, ...]:
    return tuple(sorted(str(feature) for feature in program.get("antecedent_features", [])))


def _program_mechanism_signature(program: dict) -> tuple[str, ...]:
    return tuple(sorted(str(group) for group in program.get("mechanism_groups", [])))


def _program_redundancy_penalty(program: dict, selected_programs: list[dict]) -> float:
    if not selected_programs:
        return 0.0

    candidate_tokens = {f"anchor:{feature}" for feature in _program_anchor_signature(program)}
    candidate_tokens.update({f"mechanism:{group}" for group in set(program.get("mechanism_groups", []))})
    if not candidate_tokens:
        return 0.0

    max_overlap = 0.0
    for row in selected_programs:
        selected_tokens = {f"anchor:{feature}" for feature in _program_anchor_signature(row)}
        selected_tokens.update({f"mechanism:{group}" for group in set(row.get("mechanism_groups", []))})
        union = candidate_tokens.union(selected_tokens)
        if not union:
            continue
        overlap = len(candidate_tokens.intersection(selected_tokens)) / float(len(union))
        max_overlap = max(max_overlap, overlap)
    return _bounded_float(max_overlap, 0.0, 1.0)


def _program_anchor_diversity_bonus(program: dict, selected_programs: list[dict]) -> float:
    if not selected_programs:
        return 0.0

    candidate_anchor = set(_program_anchor_signature(program))
    if not candidate_anchor:
        return 0.0

    selected_anchors = set()
    for row in selected_programs:
        selected_anchors.update(set(_program_anchor_signature(row)))

    unseen_ratio = len(candidate_anchor.difference(selected_anchors)) / float(len(candidate_anchor))
    if unseen_ratio > 0.0:
        return _bounded_float(unseen_ratio, 0.0, 1.0)

    similarities = []
    for row in selected_programs:
        anchors = set(_program_anchor_signature(row))
        union = candidate_anchor.union(anchors)
        if not union:
            continue
        similarities.append(len(candidate_anchor.intersection(anchors)) / float(len(union)))
    if not similarities:
        return 0.0
    return _bounded_float(1.0 - max(similarities), 0.0, 1.0)


def _program_coverage_target_bonus(program: dict, selected_programs: list[dict], coverage_target: int) -> float:
    target = max(0, int(coverage_target))
    if target <= 0:
        return 0.0

    covered_groups = set()
    for row in selected_programs:
        covered_groups.update(set(row.get("mechanism_groups", [])))

    covered_count = len(covered_groups)
    if covered_count >= target:
        return 0.0

    candidate_groups = set(program.get("mechanism_groups", []))
    if not candidate_groups:
        return 0.0

    newly_covered = len(candidate_groups.difference(covered_groups))
    if newly_covered <= 0:
        return 0.0

    remaining = max(1, target - covered_count)
    return _bounded_float(newly_covered / float(remaining), 0.0, 1.0)


def _prepare_program_candidate_pool(
    ordered_programs: list[dict],
    max_programs: int,
    anchor_candidate_top_n: int,
    anchor_candidate_quality_ratio: float,
    mechanism_candidate_top_n: int = 0,
    mechanism_min_unique: int = 0,
) -> list[dict]:
    if not ordered_programs:
        return []

    limit = max(1, int(max_programs))
    top_n = max(0, int(anchor_candidate_top_n))
    quality_ratio = _bounded_float(anchor_candidate_quality_ratio, 0.0, 1.0)
    mechanism_top_n = max(0, int(mechanism_candidate_top_n))
    min_mechanism_unique = max(0, int(mechanism_min_unique))
    if top_n <= 0 and quality_ratio <= 0.0 and mechanism_top_n <= 0 and min_mechanism_unique <= 0:
        return ordered_programs

    candidate_keys: set[tuple[tuple[str, ...], tuple[str, ...]]] = set()
    if top_n > 0:
        anchor_counts: dict[tuple[str, ...], int] = {}
        for program in ordered_programs:
            anchor = _program_anchor_signature(program)
            count = anchor_counts.get(anchor, 0)
            if count >= top_n:
                continue
            candidate_keys.add(_program_key(program))
            anchor_counts[anchor] = count + 1

    if quality_ratio > 0.0:
        top_score = float(ordered_programs[0].get("representative_score", 0.0))
        threshold = top_score * quality_ratio
        for program in ordered_programs:
            if float(program.get("representative_score", 0.0)) >= threshold:
                candidate_keys.add(_program_key(program))

    if mechanism_top_n > 0:
        mechanism_counts: dict[str, int] = {}
        for program in ordered_programs:
            groups = sorted(set(program.get("mechanism_groups", [])))
            if not groups:
                continue
            for group in groups:
                count = mechanism_counts.get(group, 0)
                if count >= mechanism_top_n:
                    continue
                candidate_keys.add(_program_key(program))
                mechanism_counts[group] = count + 1

    candidates = [program for program in ordered_programs if _program_key(program) in candidate_keys]

    if min_mechanism_unique > 0:
        selected_keys = {_program_key(program) for program in candidates}
        covered_groups = set()
        for program in candidates:
            covered_groups.update(set(program.get("mechanism_groups", [])))
        while len(covered_groups) < min_mechanism_unique:
            next_program = None
            for program in ordered_programs:
                key = _program_key(program)
                if key in selected_keys:
                    continue
                groups = set(program.get("mechanism_groups", []))
                if groups.difference(covered_groups):
                    next_program = program
                    break
            if next_program is None:
                break
            candidates.append(next_program)
            selected_keys.add(_program_key(next_program))
            covered_groups.update(set(next_program.get("mechanism_groups", [])))

    if len(candidates) < limit:
        seen = {_program_key(program) for program in candidates}
        for program in ordered_programs:
            key = _program_key(program)
            if key in seen:
                continue
            candidates.append(program)
            seen.add(key)
            if len(candidates) >= limit:
                break
    return candidates


def _select_programs_with_lead_time_diversity(
    programs: list[dict],
    max_programs: int,
    lead_time_diversity_weight: float,
    lead_time_diversity_gap_hours: float,
    program_mechanism_diversity_weight: float,
    program_care_process_penalty: float,
    program_redundancy_penalty_weight: float,
    program_coverage_target: int,
    program_coverage_target_weight: float,
    program_anchor_diversity_weight: float,
    program_anchor_candidate_top_n: int,
    program_anchor_candidate_quality_ratio: float,
    program_mechanism_candidate_top_n: int,
    program_mechanism_min_unique: int,
    program_anchor_max_same_count: int,
    program_mechanism_max_same_count: int,
    program_anchor_novelty_target: int,
    program_anchor_novelty_weight: float,
    program_anchor_novelty_rerank_epsilon: float,
    program_diversity_quality_floor_ratio: float,
    program_topk_care_process_max_count: int,
    program_topk_care_process_window: int,
) -> list[dict]:
    ordered_all = sorted(programs, key=_program_sort_key, reverse=True)
    limit = max(1, int(max_programs))
    anchor_candidate_top_n = max(0, int(program_anchor_candidate_top_n))
    anchor_candidate_quality_ratio = _bounded_float(program_anchor_candidate_quality_ratio, 0.0, 1.0)
    mechanism_candidate_top_n = max(0, int(program_mechanism_candidate_top_n))
    mechanism_min_unique = max(0, int(program_mechanism_min_unique))
    anchor_max_same_count = max(0, int(program_anchor_max_same_count))
    mechanism_max_same_count = max(0, int(program_mechanism_max_same_count))
    ordered = _prepare_program_candidate_pool(
        ordered_all,
        max_programs=limit,
        anchor_candidate_top_n=anchor_candidate_top_n,
        anchor_candidate_quality_ratio=anchor_candidate_quality_ratio,
        mechanism_candidate_top_n=mechanism_candidate_top_n,
        mechanism_min_unique=mechanism_min_unique,
    )
    lead_weight = max(0.0, float(lead_time_diversity_weight))
    mechanism_weight = max(0.0, float(program_mechanism_diversity_weight))
    care_penalty_weight = max(0.0, float(program_care_process_penalty))
    redundancy_penalty_weight = max(0.0, float(program_redundancy_penalty_weight))
    coverage_target = max(0, int(program_coverage_target))
    coverage_target_weight = max(0.0, float(program_coverage_target_weight))
    anchor_weight = max(0.0, float(program_anchor_diversity_weight))
    anchor_novelty_target = max(0, int(program_anchor_novelty_target))
    anchor_novelty_weight = max(0.0, float(program_anchor_novelty_weight))
    anchor_novelty_rerank_epsilon = max(0.0, float(program_anchor_novelty_rerank_epsilon))
    quality_floor_ratio = max(0.0, float(program_diversity_quality_floor_ratio))
    topk_care_process_max_count = int(program_topk_care_process_max_count)
    topk_care_process_window = max(1, int(program_topk_care_process_window))
    if (
        lead_weight <= 0.0
        and mechanism_weight <= 0.0
        and care_penalty_weight <= 0.0
        and redundancy_penalty_weight <= 0.0
        and anchor_weight <= 0.0
        and (anchor_novelty_target <= 0 or anchor_novelty_weight <= 0.0)
        and (anchor_novelty_target <= 0 or anchor_novelty_rerank_epsilon <= 0.0)
        and (coverage_target <= 0 or coverage_target_weight <= 0.0)
        and topk_care_process_max_count < 0
    ):
        return ordered[:limit]
    if not ordered:
        return []

    selected: list[dict] = []
    remaining = list(ordered)
    if topk_care_process_max_count >= 0 and topk_care_process_window > 0:
        for idx, candidate in enumerate(remaining):
            if _program_has_care_process(candidate) and topk_care_process_max_count <= 0:
                continue
            selected = [candidate]
            remaining.pop(idx)
            break
    if not selected:
        selected = [remaining.pop(0)]
    top_representative_score = float(ordered[0].get("representative_score", 0.0))
    diversity_quality_floor = top_representative_score * quality_floor_ratio
    while remaining and len(selected) < limit:
        selected_anchor_set = set()
        for row in selected:
            selected_anchor_set.update(set(_program_anchor_signature(row)))
        novelty_needed = anchor_novelty_target > 0 and len(selected_anchor_set) < min(limit, anchor_novelty_target)

        best_entry = None
        scored_candidates = []
        for idx, candidate in enumerate(remaining):
            if topk_care_process_max_count >= 0 and len(selected) < topk_care_process_window:
                current_topk_care_process_count = sum(
                    1
                    for row in selected[:topk_care_process_window]
                    if _program_has_care_process(row)
                )
                if _program_has_care_process(candidate) and current_topk_care_process_count >= topk_care_process_max_count:
                    continue
            if anchor_max_same_count > 0:
                candidate_anchor_signature = _program_anchor_signature(candidate)
                current_same_anchor_count = sum(
                    1 for row in selected if _program_anchor_signature(row) == candidate_anchor_signature
                )
                if current_same_anchor_count >= anchor_max_same_count:
                    continue
            if mechanism_max_same_count > 0:
                candidate_mechanism_signature = _program_mechanism_signature(candidate)
                current_same_mechanism_count = sum(
                    1 for row in selected if _program_mechanism_signature(row) == candidate_mechanism_signature
                )
                if current_same_mechanism_count >= mechanism_max_same_count:
                    continue
            candidate_score = float(candidate.get("representative_score", 0.0))
            diversity_allowed = candidate_score >= diversity_quality_floor
            candidate_anchor_set = set(_program_anchor_signature(candidate))
            adds_new_anchor = bool(candidate_anchor_set.difference(selected_anchor_set))
            anchor_novelty_adjustment = 0.0
            if diversity_allowed and novelty_needed and anchor_novelty_weight > 0.0:
                anchor_novelty_adjustment = anchor_novelty_weight if adds_new_anchor else -anchor_novelty_weight
            bonus = _lead_time_diversity_bonus(
                candidate,
                selected_programs=selected,
                gap_hours=lead_time_diversity_gap_hours,
            ) if diversity_allowed else 0.0
            mechanism_bonus = (
                _program_mechanism_diversity_bonus(candidate, selected_programs=selected) if diversity_allowed else 0.0
            )
            coverage_bonus = (
                _program_coverage_target_bonus(
                    candidate,
                    selected_programs=selected,
                    coverage_target=coverage_target,
                )
                if diversity_allowed
                else 0.0
            )
            anchor_bonus = (
                _program_anchor_diversity_bonus(candidate, selected_programs=selected) if diversity_allowed else 0.0
            )
            care_penalty = _program_care_process_penalty_value(candidate) if diversity_allowed else 0.0
            redundancy_penalty = _program_redundancy_penalty(candidate, selected_programs=selected) if diversity_allowed else 0.0
            combined_score = candidate_score + (lead_weight * bonus) + (mechanism_weight * mechanism_bonus) + (
                coverage_target_weight * coverage_bonus
            ) + (anchor_weight * anchor_bonus) - (care_penalty_weight * care_penalty) - (
                redundancy_penalty_weight * redundancy_penalty
            ) + anchor_novelty_adjustment
            key = (combined_score, *_program_sort_key(candidate))
            entry = {
                "idx": idx,
                "key": key,
                "combined_score": combined_score,
                "diversity_allowed": diversity_allowed,
                "adds_new_anchor": adds_new_anchor,
            }
            scored_candidates.append(entry)
            if best_entry is None or key > best_entry["key"]:
                best_entry = entry

        if best_entry is None:
            break

        chosen_entry = best_entry
        if novelty_needed and anchor_novelty_rerank_epsilon > 0.0:
            best_combined_score = float(best_entry["combined_score"])
            safe_frontier = [
                row
                for row in scored_candidates
                if row["diversity_allowed"]
                and (best_combined_score - float(row["combined_score"])) <= (anchor_novelty_rerank_epsilon + 1e-12)
            ]
            novelty_frontier = [row for row in safe_frontier if row["adds_new_anchor"]]
            if novelty_frontier:
                chosen_entry = max(novelty_frontier, key=lambda row: row["key"])

        selected.append(remaining.pop(int(chosen_entry["idx"])))
    return selected[:limit]


def consolidate_contrastive_programs(
    ranked_rules: list[dict],
    max_programs: int = 20,
    lead_time_diversity_weight: float = 0.0,
    lead_time_diversity_gap_hours: float = 1.0,
    program_mechanism_diversity_weight: float = 0.0,
    program_care_process_penalty: float = 0.1,
    program_redundancy_penalty_weight: float = 0.0,
    program_coverage_target: int = 0,
    program_coverage_target_weight: float = 0.0,
    program_anchor_diversity_weight: float = 0.0,
    program_anchor_candidate_top_n: int = 0,
    program_anchor_candidate_quality_ratio: float = 0.0,
    program_mechanism_candidate_top_n: int = 0,
    program_mechanism_min_unique: int = 0,
    program_anchor_max_same_count: int = 0,
    program_mechanism_max_same_count: int = 0,
    program_anchor_novelty_target: int = 0,
    program_anchor_novelty_weight: float = 0.0,
    program_anchor_novelty_rerank_epsilon: float = 0.0,
    program_diversity_quality_floor_ratio: float = 0.0,
    program_topk_care_process_max_count: int = -1,
    program_topk_care_process_window: int = 5,
    min_program_representative_score: float = 0.0,
) -> list[dict]:
    if not ranked_rules:
        return []

    grouped: dict[tuple[tuple[str, ...], tuple[str, ...]], list[dict]] = {}
    for rule in ranked_rules:
        key = _program_key(rule)
        grouped.setdefault(key, []).append(rule)

    programs = []
    representative_component_keys = (
        "base_contrastive_score",
        "self_persistence_penalty",
        "sparse_measurement_penalty_factor",
        "lead_time_quality_blend_factor",
        "support_reliability_factor",
        "care_process_penalty_factor",
        "significance_factor",
        "anchor_saturation_penalty_factor",
        "mechanism_saturation_penalty_factor",
    )
    def _member_sort_key(row: dict) -> tuple:
        return (
            float(row.get("contrastive_score", 0.0)),
            float(row.get("enrichment", 0.0)),
            row.get("rule_id", ""),
        )

    for (antecedent, consequent), members in grouped.items():
        representative = max(members, key=_member_sort_key)
        member_top_ids = [
            row.get("rule_id")
            for row in heapq.nlargest(5, members, key=_member_sort_key)
        ]
        member_count = max(1, len(members))
        lead_steps_sum = sum(float(row.get("lead_time_steps_mean", 0.0)) for row in members)
        lead_hours_sum = sum(float(row.get("lead_time_hours_mean", 0.0)) for row in members)
        score_sum = sum(float(row.get("contrastive_score", 0.0)) for row in members)
        enrichment_sum = sum(float(row.get("enrichment", 0.0)) for row in members)
        component_values: dict[str, float] = {}
        for key in representative_component_keys:
            if key in representative:
                component_values[key] = float(representative.get(key, 0.0))
        component_values["final_contrastive_score"] = float(representative.get("contrastive_score", 0.0))
        support_snapshot = {
            "positive_count": int(representative.get("positive_count", 0)),
            "control_count": int(representative.get("control_count", 0)),
            "positive_total": int(representative.get("positive_total", 0)),
            "control_total": int(representative.get("control_total", 0)),
            "support_ratio": float(representative.get("support_ratio", 0.0)),
            "care_process_ratio": float(representative.get("care_process_ratio", 0.0)),
            "zero_control_candidate": bool(representative.get("zero_control_candidate", False)),
        }
        antecedent_label = " + ".join(antecedent) if antecedent else "unknown"
        consequent_label = " + ".join(consequent) if consequent else "unknown"
        programs.append(
            {
                "program_label": f"{antecedent_label} -> {consequent_label}",
                "antecedent_features": list(antecedent),
                "consequent_features": list(consequent),
                "mechanism_groups": sorted(_mechanism_groups_for_features(list(consequent))),
                "rule_count": member_count,
                "representative_rule_id": representative.get("rule_id"),
                "representative_score": float(representative.get("contrastive_score", 0.0)),
                "representative_enrichment": float(representative.get("enrichment", 0.0)),
                "mean_contrastive_score": score_sum / member_count,
                "mean_enrichment": enrichment_sum / member_count,
                "lead_time_steps_mean": lead_steps_sum / member_count,
                "lead_time_hours_mean": lead_hours_sum / member_count,
                "member_rule_ids": member_top_ids,
                "representative_score_components": component_values,
                "representative_score_support": support_snapshot,
            }
        )

    min_repr_score = max(0.0, float(min_program_representative_score))
    if min_repr_score > 0.0:
        programs = [row for row in programs if float(row.get("representative_score", 0.0)) >= min_repr_score]
        if not programs:
            return []

    selected_programs = _select_programs_with_lead_time_diversity(
        programs,
        max_programs=max_programs,
        lead_time_diversity_weight=lead_time_diversity_weight,
        lead_time_diversity_gap_hours=lead_time_diversity_gap_hours,
        program_mechanism_diversity_weight=program_mechanism_diversity_weight,
        program_care_process_penalty=program_care_process_penalty,
        program_redundancy_penalty_weight=program_redundancy_penalty_weight,
        program_coverage_target=program_coverage_target,
        program_coverage_target_weight=program_coverage_target_weight,
        program_anchor_diversity_weight=program_anchor_diversity_weight,
        program_anchor_candidate_top_n=program_anchor_candidate_top_n,
        program_anchor_candidate_quality_ratio=program_anchor_candidate_quality_ratio,
        program_mechanism_candidate_top_n=program_mechanism_candidate_top_n,
        program_mechanism_min_unique=program_mechanism_min_unique,
        program_anchor_max_same_count=program_anchor_max_same_count,
        program_mechanism_max_same_count=program_mechanism_max_same_count,
        program_anchor_novelty_target=program_anchor_novelty_target,
        program_anchor_novelty_weight=program_anchor_novelty_weight,
        program_anchor_novelty_rerank_epsilon=program_anchor_novelty_rerank_epsilon,
        program_diversity_quality_floor_ratio=program_diversity_quality_floor_ratio,
        program_topk_care_process_max_count=program_topk_care_process_max_count,
        program_topk_care_process_window=program_topk_care_process_window,
    )
    for idx, program in enumerate(selected_programs[: max(1, int(max_programs))], start=1):
        program["program_id"] = f"program_{idx:03d}"
    return selected_programs[: max(1, int(max_programs))]
