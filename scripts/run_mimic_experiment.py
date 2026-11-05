import argparse
import glob
import hashlib
import json
import math
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ROOT_STR = str(ROOT)
if ROOT_STR not in sys.path:
    sys.path.insert(0, ROOT_STR)

from nonstationary_miner import NonStationaryRuleMiner
from src.icu.shock_contrastive_miner import (
    MECHANISM_GROUP_BY_FEATURE,
    consolidate_contrastive_programs,
    extract_rule_records,
    score_contrastive_rule_records,
)
from src.icu.anchor_segmenter import segment_episode_relative_to_anchor, summarize_anchor_segments
from src.icu.program_model import build_program_models
from src.icu.rule_stability import compute_rule_stability, evaluate_stability_gates, summarize_robustness_matrix


CORE_OBS_FEATURES = {
    "HR",
    "RR",
    "SpO2",
    "SBP",
    "DBP",
    "MAP",
    "Temp",
}

INTERVENTION_FEATURES = {
    "vasopressor_on",
    "ventilation_on",
    "sedation_on",
}

PROGRAM_MECHANISM_GROUP_MODES = {
    "coarse",
    "coarse_plus_feature",
}

PROGRAM_SCORE_CALIBRATION_MODES = {
    "none",
    "affine_quantile",
}


def _normalize_adaptive_domain(value: str | None) -> str:
    text = str(value or "").strip().lower()
    if not text:
        return ""
    text = re.sub(r"[^a-z0-9]+", " ", text)
    tokens = [token for token in text.split() if token]
    return "_".join(tokens)


def _load_stay_specs(input_dir: Path) -> list[dict]:
    shock_index_path = input_dir / "shock_episode_index.json"
    if shock_index_path.exists():
        return json.loads(shock_index_path.read_text(encoding="utf-8"))

    index_path = input_dir / "mimic_iv_index.json"
    if index_path.exists():
        return json.loads(index_path.read_text(encoding="utf-8"))

    specs = []
    for npy_path in sorted(input_dir.glob("mimic_iv_stay_*.npy")):
        meta_path = npy_path.with_suffix(".json")
        if not meta_path.exists():
            continue
        specs.append(json.loads(meta_path.read_text(encoding="utf-8")))
    return specs


def _limit_specs_with_type_balance(
    specs: list[dict],
    limit: int,
    sampling_seed: int | None = None,
    enforce_unique_stays: bool = False,
) -> list[dict]:
    if limit <= 0 or len(specs) <= int(limit):
        return specs

    positives = [row for row in specs if row.get("episode_type") == "positive"]
    controls = [row for row in specs if row.get("episode_type") == "control"]
    if not positives or not controls:
        if sampling_seed is not None:
            rng = np.random.default_rng(int(sampling_seed))
            shuffled = list(specs)
            rng.shuffle(shuffled)
            return shuffled[: int(limit)]
        return specs[: int(limit)]

    if sampling_seed is not None:
        rng = np.random.default_rng(int(sampling_seed))
        positives = list(positives)
        controls = list(controls)
        rng.shuffle(positives)
        rng.shuffle(controls)
        specs_remaining = list(specs)
        rng.shuffle(specs_remaining)
    else:
        specs_remaining = specs

    control_target = min(len(controls), max(1, int(limit) // 2))
    positive_target = min(len(positives), int(limit) - control_target)

    selected: list[dict] = []
    selected_ids: set[int] = set()
    seen_stay_ids: set[int] = set()

    def _can_use_row(row: dict, require_new_stay: bool) -> bool:
        if id(row) in selected_ids:
            return False
        if not enforce_unique_stays:
            return True
        stay_id = row.get("stay_id")
        if stay_id is None:
            return True
        try:
            normalized_stay_id = int(stay_id)
        except Exception:
            return True
        if require_new_stay and normalized_stay_id in seen_stay_ids:
            return False
        return True

    def _append_rows(candidates: list[dict], target_count: int, require_new_stay: bool) -> None:
        for row in candidates:
            if len(selected) >= int(limit) or target_count <= 0:
                break
            if not _can_use_row(row, require_new_stay=require_new_stay):
                continue
            selected.append(row)
            selected_ids.add(id(row))
            stay_id = row.get("stay_id")
            if stay_id is not None:
                try:
                    seen_stay_ids.add(int(stay_id))
                except Exception:
                    pass
            target_count -= 1

    _append_rows(controls, control_target, require_new_stay=True)
    _append_rows(positives, positive_target, require_new_stay=True)
    if len(selected) < int(limit):
        _append_rows(specs_remaining, int(limit) - len(selected), require_new_stay=True)

    # Fallback: if unique-stay constraint cannot fill limit, allow repeated stays.
    if len(selected) < int(limit):
        _append_rows(specs_remaining, int(limit) - len(selected), require_new_stay=False)

    return selected[: int(limit)]


def _load_shadow_training_episode_ids(path: Path | str | None) -> set[str]:
    if path is None:
        return set()
    from src.icu.comparators.data_contract import load_split_manifest

    manifest = load_split_manifest(path)
    split = manifest.get("split", {})
    train_ids = split.get("train", []) if isinstance(split, dict) else []
    return {str(value).strip() for value in train_ids if str(value).strip()}


def _load_profile_overrides(path: Path | str | None) -> dict[str, object]:
    if path is None or str(path).strip() == "":
        return {}
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("profile json must be an object")
    return payload


def _load_rule_mechanism_confidence_domain_thresholds(path: Path | str | None) -> dict[str, dict[str, float]]:
    if path is None or str(path).strip() == "":
        return {}
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("rule mechanism confidence thresholds json must be an object")
    out: dict[str, dict[str, float]] = {}
    for raw_domain, raw_cfg in payload.items():
        domain = _normalize_adaptive_domain(str(raw_domain))
        if str(raw_domain).strip().lower() == "default":
            domain = "default"
        if not domain or not isinstance(raw_cfg, dict):
            continue
        cfg = dict(raw_cfg)
        row: dict[str, float] = {}
        if "min_support_ratio" in cfg:
            row["min_support_ratio"] = float(cfg["min_support_ratio"])
        if "min_reliable_rules" in cfg:
            row["min_reliable_rules"] = float(cfg["min_reliable_rules"])
        if "min_expected_gain" in cfg:
            row["min_expected_gain"] = float(cfg["min_expected_gain"])
        if row:
            out[domain] = row
    return out


def _resolve_rule_mechanism_confidence_thresholds(
    domain: str,
    domain_thresholds: dict[str, dict[str, float]] | None,
    default_min_support_ratio: float,
    default_min_reliable_rules: int,
    default_min_expected_gain: int,
) -> dict[str, object]:
    resolved = {
        "domain": str(domain or ""),
        "source": "runtime_defaults",
        "min_support_ratio": float(default_min_support_ratio),
        "min_reliable_rules": int(default_min_reliable_rules),
        "min_expected_gain": int(default_min_expected_gain),
    }
    threshold_map = dict(domain_thresholds or {})
    if not threshold_map:
        return resolved

    normalized_domain = _normalize_adaptive_domain(domain)
    chosen = None
    source = "runtime_defaults"
    if normalized_domain and normalized_domain in threshold_map:
        chosen = dict(threshold_map.get(normalized_domain, {}))
        source = f"domain:{normalized_domain}"
    elif "default" in threshold_map:
        chosen = dict(threshold_map.get("default", {}))
        source = "domain:default"

    if not chosen:
        return resolved

    if "min_support_ratio" in chosen:
        resolved["min_support_ratio"] = float(chosen["min_support_ratio"])
    if "min_reliable_rules" in chosen:
        resolved["min_reliable_rules"] = int(chosen["min_reliable_rules"])
    if "min_expected_gain" in chosen:
        resolved["min_expected_gain"] = int(chosen["min_expected_gain"])
    resolved["source"] = source
    return resolved


def _apply_profile_overrides(args: argparse.Namespace, overrides: dict[str, object] | None) -> dict[str, list[str]]:
    applied: list[str] = []
    ignored: list[str] = []
    for raw_key, value in dict(overrides or {}).items():
        key = str(raw_key)
        if hasattr(args, key):
            setattr(args, key, value)
            applied.append(key)
        else:
            ignored.append(key)
    return {
        "applied": sorted(applied),
        "ignored": sorted(ignored),
    }


def _filter_shadow_training_records(records: list, training_episode_ids: set[str] | None) -> list:
    allowed = {str(value).strip() for value in set(training_episode_ids or set()) if str(value).strip()}
    if not allowed:
        return list(records)
    filtered = []
    for record in records:
        episode_id = str(getattr(record, "episode_id", "")).strip()
        if episode_id in allowed:
            filtered.append(record)
    return filtered


def _fit_shadow_predictive_feature_weights(
    input_dir: Path,
    limit: int = 0,
    seed: int = 42,
    training_episode_ids: set[str] | None = None,
) -> dict[str, float]:
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        from src.icu.comparators.data_contract import build_episode_records
        from src.icu.comparators.feature_builder import build_handcrafted_feature_matrix
    except Exception:
        return {}

    max_records = max(0, int(limit))
    records = build_episode_records(
        input_dir,
        limit=max_records,
        episode_types=("positive", "control"),
    )
    records = _filter_shadow_training_records(records, training_episode_ids)
    if len(records) < 8:
        return {}

    bundle = build_handcrafted_feature_matrix(records)
    X = np.asarray(bundle.get("X", []), dtype=float)
    y = np.asarray(bundle.get("y", []), dtype=int)
    feature_names = list(bundle.get("feature_names", []))
    if X.ndim != 2 or X.shape[0] < 8 or X.shape[1] <= 0:
        return {}
    if y.size != X.shape[0] or np.unique(y).size < 2:
        return {}
    if len(feature_names) != X.shape[1]:
        return {}

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model = LogisticRegression(
        max_iter=1000,
        random_state=int(seed),
        class_weight="balanced",
    )
    model.fit(X_scaled, y)
    coef = np.asarray(model.coef_, dtype=float)
    if coef.ndim != 2 or coef.shape[0] == 0:
        return {}
    abs_coef = np.abs(coef[0])
    if abs_coef.size != len(feature_names):
        return {}

    per_feature: dict[str, float] = {}
    for name, value in zip(feature_names, abs_coef):
        key = str(name).split("__", 1)[0]
        per_feature[key] = per_feature.get(key, 0.0) + float(value)

    if not per_feature:
        return {}
    max_value = max(per_feature.values())
    if max_value <= 0.0:
        return {}
    return {key: float(value / max_value) for key, value in per_feature.items()}


def _apply_shadow_predictive_program_rerank(
    programs: list[dict],
    feature_weights: dict[str, float],
    blend_weight: float,
    max_adjustment: float,
    quality_floor_ratio: float,
    exclude_care_process: bool,
) -> list[dict]:
    if not programs:
        return []

    weight = max(0.0, min(1.0, float(blend_weight)))
    max_adj = max(0.0, float(max_adjustment))
    if weight <= 0.0 or max_adj <= 0.0 or not feature_weights:
        return [dict(row) for row in programs]

    top_score = max(float(row.get("representative_score", 0.0)) for row in programs)
    floor_ratio = max(0.0, min(1.0, float(quality_floor_ratio)))
    score_floor = float(top_score * floor_ratio) if floor_ratio > 0.0 else -float("inf")

    enriched: list[dict] = []
    for idx, row in enumerate(programs):
        out = dict(row)
        antecedent = [str(v) for v in list(row.get("antecedent_features", []))]
        consequent = [str(v) for v in list(row.get("consequent_features", []))]
        keys = list(dict.fromkeys(antecedent + consequent))
        proxy_values = [float(feature_weights.get(key, 0.0)) for key in keys]
        proxy = float(sum(proxy_values) / max(1, len(proxy_values))) if proxy_values else 0.0
        base_score = float(row.get("representative_score", 0.0))
        is_care_process = bool("care_process" in _program_mechanism_groups(row))
        eligible = bool(base_score >= score_floor) and not (bool(exclude_care_process) and is_care_process)
        bonus = float(weight * max_adj * proxy) if eligible else 0.0
        out["shadow_predictive_proxy"] = float(proxy)
        out["shadow_predictive_bonus"] = float(bonus)
        out["shadow_predictive_adjusted_score"] = float(base_score + bonus)
        out["shadow_predictive_boost_eligible"] = bool(eligible)
        out["shadow_predictive_is_care_process"] = bool(is_care_process)
        out["_shadow_rank_index"] = int(idx)
        enriched.append(out)

    enriched.sort(
        key=lambda row: (
            float(row.get("shadow_predictive_adjusted_score", 0.0)),
            float(row.get("representative_score", 0.0)),
            -int(row.get("_shadow_rank_index", 0)),
        ),
        reverse=True,
    )

    for row in enriched:
        row.pop("_shadow_rank_index", None)
    return enriched


def _repair_top_window_group_diversity(
    programs: list[dict],
    window_size: int,
    target_unique_groups: int,
    fixed_prefix: int,
    forbidden_groups: set[str] | None = None,
) -> tuple[list[dict], dict]:
    if not programs:
        return [], {
            "enabled": False,
            "reason": "no_programs",
        }

    window = max(1, int(window_size))
    target = max(0, int(target_unique_groups))
    prefix = max(0, int(fixed_prefix))
    if target <= 0:
        return [dict(row) for row in programs], {
            "enabled": False,
            "reason": "target_zero",
            "window_size": int(window),
            "target_unique_groups": int(target),
            "fixed_prefix": int(prefix),
        }

    rows = [dict(row) for row in programs]
    banned = {str(group) for group in set(forbidden_groups or set()) if str(group)}
    active_window = min(len(rows), int(window))
    active_prefix = min(active_window, int(prefix))

    head = rows[:active_prefix]
    tail_pool = rows[active_prefix:]

    existing_groups = set()
    for row in head:
        existing_groups.update(_program_mechanism_groups(row))

    chosen = list(head)
    chosen_ids = {id(row) for row in chosen}
    picks_added = 0

    while len(chosen) < active_window and tail_pool:
        current_unique = len(existing_groups)
        need_more_diversity = current_unique < int(target)
        best_idx = None
        best_key = None
        for idx, row in enumerate(tail_pool):
            groups = _program_mechanism_groups(row)
            adds_new = len(groups.difference(existing_groups)) > 0
            has_forbidden = bool(groups.intersection(banned))
            rep = float(row.get("representative_score", 0.0))
            if need_more_diversity:
                key = (1 if not has_forbidden else 0, 1 if adds_new else 0, rep)
            else:
                key = (1 if not has_forbidden else 0, rep)
            if best_key is None or key > best_key:
                best_key = key
                best_idx = idx
        if best_idx is None:
            break
        picked = tail_pool.pop(best_idx)
        chosen.append(picked)
        chosen_ids.add(id(picked))
        existing_groups.update(_program_mechanism_groups(picked))
        if len(chosen) > active_prefix:
            picks_added += 1

    leftovers = [row for row in rows if id(row) not in chosen_ids]
    repaired = chosen + leftovers

    before_groups = set()
    for row in rows[:active_window]:
        before_groups.update(_program_mechanism_groups(row))
    after_groups = set()
    for row in repaired[:active_window]:
        after_groups.update(_program_mechanism_groups(row))

    metadata = {
        "enabled": True,
        "window_size": int(active_window),
        "target_unique_groups": int(target),
        "fixed_prefix": int(active_prefix),
        "forbidden_groups": sorted(banned),
        "before_unique_groups": int(len(before_groups)),
        "after_unique_groups": int(len(after_groups)),
        "repair_applied": bool(len(after_groups) > len(before_groups) or picks_added > 0),
        "picks_added": int(picks_added),
    }
    return repaired, metadata


def _enforce_topk_care_process_cap(
    programs: list[dict],
    top_k_window: int,
    max_care_process_count: int,
) -> tuple[list[dict], dict]:
    rows = [dict(row) for row in programs]
    window = min(len(rows), max(1, int(top_k_window)))
    cap = int(max_care_process_count)
    if cap < 0:
        return rows, {
            "enabled": False,
            "reason": "cap_disabled",
            "window_size": int(window),
            "max_care_process_count": int(cap),
        }

    def _is_care(row: dict) -> bool:
        return "care_process" in _program_mechanism_groups(row)

    before_count = sum(1 for row in rows[:window] if _is_care(row))
    if before_count <= cap:
        return rows, {
            "enabled": True,
            "repair_applied": False,
            "window_size": int(window),
            "max_care_process_count": int(cap),
            "before_count": int(before_count),
            "after_count": int(before_count),
            "swaps": 0,
        }

    swaps = 0
    while True:
        current_count = sum(1 for row in rows[:window] if _is_care(row))
        if current_count <= cap:
            break

        care_indices = [idx for idx in range(window) if _is_care(rows[idx])]
        if not care_indices:
            break
        # Prefer replacing the weakest in-window care-process program first.
        care_idx = min(care_indices, key=lambda idx: float(rows[idx].get("representative_score", 0.0)))

        candidate_indices = [idx for idx in range(window, len(rows)) if not _is_care(rows[idx])]
        if not candidate_indices:
            break
        # Pull in the strongest available non-care-process candidate.
        replacement_idx = max(candidate_indices, key=lambda idx: float(rows[idx].get("representative_score", 0.0)))
        rows[care_idx], rows[replacement_idx] = rows[replacement_idx], rows[care_idx]
        swaps += 1

    after_count = sum(1 for row in rows[:window] if _is_care(row))
    return rows, {
        "enabled": True,
        "repair_applied": bool(swaps > 0),
        "window_size": int(window),
        "max_care_process_count": int(cap),
        "before_count": int(before_count),
        "after_count": int(after_count),
        "swaps": int(swaps),
    }


def _select_topk_with_constraints(
    programs: list[dict],
    top_window_size: int,
    top_window_unique_groups_target: int,
    top_window_fixed_prefix: int,
    top5_unique_groups_target: int,
    top5_fixed_prefix: int,
    topk_care_process_window: int,
    topk_care_process_max_count: int,
    top5_max_same_anchor_count: int,
    candidate_pool_size: int = 0,
    quality_floor_ratio: float = 0.0,
    two_stage_swap_budget: int = 0,
) -> tuple[list[dict], dict]:
    rows = [dict(row) for row in programs]
    if not rows:
        return [], {
            "enabled": False,
            "reason": "no_programs",
        }

    active_window = min(
        len(rows),
        max(1, int(top_window_size), int(topk_care_process_window), 5),
    )
    top5_window = min(5, active_window)
    topk_window = min(active_window, max(1, int(topk_care_process_window)))
    target_top5_groups = max(0, int(top5_unique_groups_target))
    target_top_window_groups = max(0, int(top_window_unique_groups_target))
    anchor_cap = max(0, int(top5_max_same_anchor_count))
    care_cap_enabled = int(topk_care_process_max_count) >= 0
    care_cap = int(topk_care_process_max_count)
    fixed_prefix = min(
        active_window,
        max(0, int(top_window_fixed_prefix), int(top5_fixed_prefix)),
    )
    quality_floor = max(0.0, min(1.0, float(quality_floor_ratio)))
    top_score = max(float(row.get("representative_score", 0.0)) for row in rows)
    score_floor = float(top_score * quality_floor) if quality_floor > 0.0 else -float("inf")
    max_pool = max(0, int(candidate_pool_size))
    if max_pool <= 0:
        max_pool = max(active_window, min(len(rows), active_window * 3))
    swap_budget = max(0, int(two_stage_swap_budget))

    selected: list[dict] = []
    selected_idx_set: set[int] = set()
    selected_indices: list[int] = []
    seen_top5_groups: set[str] = set()
    seen_top_window_groups: set[str] = set()
    top5_anchor_counts: dict[tuple[str, ...], int] = {}
    topk_care_count = 0
    relax_level_used = 0
    stage2_swap_count = 0

    def _is_care(row: dict) -> bool:
        return "care_process" in _program_mechanism_groups(row)

    def _append_candidate(idx: int, row: dict) -> None:
        nonlocal topk_care_count
        selected.append(row)
        selected_idx_set.add(int(idx))
        selected_indices.append(int(idx))
        rank = len(selected) - 1
        groups = _program_mechanism_groups(row)
        seen_top_window_groups.update(groups)
        if rank < top5_window:
            seen_top5_groups.update(groups)
            anchor = _program_anchor_signature(row)
            top5_anchor_counts[anchor] = top5_anchor_counts.get(anchor, 0) + 1
        if rank < topk_window and _is_care(row):
            topk_care_count += 1

    for idx in range(fixed_prefix):
        _append_candidate(idx, rows[idx])

    ranked_indices_by_score = sorted(
        range(len(rows)),
        key=lambda idx: (
            float(rows[idx].get("representative_score", 0.0)),
            -int(idx),
        ),
        reverse=True,
    )
    pool_indices = [idx for idx in ranked_indices_by_score if float(rows[idx].get("representative_score", 0.0)) >= score_floor]
    if len(pool_indices) < active_window:
        for idx in ranked_indices_by_score:
            if idx not in pool_indices:
                pool_indices.append(idx)
            if len(pool_indices) >= active_window:
                break
    pool_indices = pool_indices[:max_pool]
    if len(pool_indices) < active_window:
        for idx in ranked_indices_by_score:
            if idx not in pool_indices:
                pool_indices.append(idx)
            if len(pool_indices) >= active_window:
                break

    for pos in range(len(selected), active_window):
        candidate_rows: list[tuple[int, dict, tuple]] = []
        for idx in pool_indices:
            if idx in selected_idx_set:
                continue
            row = rows[idx]
            groups = _program_mechanism_groups(row)
            rep = float(row.get("representative_score", 0.0))
            is_care = _is_care(row)
            anchor = _program_anchor_signature(row)
            anchor_count_after = int(top5_anchor_counts.get(anchor, 0)) + (1 if pos < top5_window else 0)
            care_count_after = int(topk_care_count) + (1 if pos < topk_window and is_care else 0)
            adds_top5 = len(groups.difference(seen_top5_groups)) if pos < top5_window else 0
            adds_top_window = len(groups.difference(seen_top_window_groups))

            # relax=0: enforce care cap + anchor cap
            # relax=1: enforce care cap only
            # relax=2: enforce anchor cap only
            # relax=3: no hard constraint
            for relax in range(4):
                anchor_violation = pos < top5_window and anchor_count_after > anchor_cap
                care_violation = care_cap_enabled and pos < topk_window and care_count_after > care_cap
                if relax == 0 and (anchor_violation or care_violation):
                    continue
                if relax == 1 and care_violation:
                    continue
                if relax == 2 and anchor_violation:
                    continue
                needed_top5 = max(0, target_top5_groups - len(seen_top5_groups)) if pos < top5_window else 0
                needed_top_window = max(0, target_top_window_groups - len(seen_top_window_groups))
                diversity_top5_priority = 1 if needed_top5 > 0 and adds_top5 > 0 else 0
                diversity_top_window_priority = 1 if needed_top_window > 0 and adds_top_window > 0 else 0
                score_tuple = (
                    -int(relax),
                    int(diversity_top5_priority),
                    int(adds_top5),
                    int(diversity_top_window_priority),
                    int(adds_top_window),
                    0 if is_care else 1,
                    -int(anchor_count_after),
                    float(rep),
                    -int(idx),
                )
                candidate_rows.append((idx, row, score_tuple))
                break

        if not candidate_rows:
            for idx, row in enumerate(rows):
                if idx in selected_idx_set:
                    continue
                groups = _program_mechanism_groups(row)
                rep = float(row.get("representative_score", 0.0))
                is_care = _is_care(row)
                anchor = _program_anchor_signature(row)
                anchor_count_after = int(top5_anchor_counts.get(anchor, 0)) + (1 if pos < top5_window else 0)
                care_count_after = int(topk_care_count) + (1 if pos < topk_window and is_care else 0)
                adds_top5 = len(groups.difference(seen_top5_groups)) if pos < top5_window else 0
                adds_top_window = len(groups.difference(seen_top_window_groups))
                for relax in range(4):
                    anchor_violation = pos < top5_window and anchor_count_after > anchor_cap
                    care_violation = care_cap_enabled and pos < topk_window and care_count_after > care_cap
                    if relax == 0 and (anchor_violation or care_violation):
                        continue
                    if relax == 1 and care_violation:
                        continue
                    if relax == 2 and anchor_violation:
                        continue
                    needed_top5 = max(0, target_top5_groups - len(seen_top5_groups)) if pos < top5_window else 0
                    needed_top_window = max(0, target_top_window_groups - len(seen_top_window_groups))
                    diversity_top5_priority = 1 if needed_top5 > 0 and adds_top5 > 0 else 0
                    diversity_top_window_priority = 1 if needed_top_window > 0 and adds_top_window > 0 else 0
                    score_tuple = (
                        -int(relax),
                        int(diversity_top5_priority),
                        int(adds_top5),
                        int(diversity_top_window_priority),
                        int(adds_top_window),
                        0 if is_care else 1,
                        -int(anchor_count_after),
                        float(rep),
                        -int(idx),
                    )
                    candidate_rows.append((idx, row, score_tuple))
                    break

        if not candidate_rows:
            break

        best_idx, best_row, best_score = max(candidate_rows, key=lambda item: item[2])
        used_relax = int(-best_score[0])
        relax_level_used = max(relax_level_used, used_relax)
        _append_candidate(best_idx, best_row)

    ordered_indices = selected_indices + [idx for idx in range(len(rows)) if idx not in selected_idx_set]

    def _build_ranked(order_indices: list[int]) -> list[dict]:
        return [rows[idx] for idx in order_indices]

    def _summarize_rank_metrics(order_indices: list[int]) -> dict:
        ranked_rows = _build_ranked(order_indices)
        local_top5 = ranked_rows[:top5_window]
        local_top_window = ranked_rows[:active_window]
        local_top5_groups = {g for row in local_top5 for g in _program_mechanism_groups(row)}
        local_top_window_groups = {g for row in local_top_window for g in _program_mechanism_groups(row)}
        local_topk_care = sum(1 for row in local_top_window[:topk_window] if _is_care(row))
        local_anchor_counts: dict[str, int] = {}
        for row in local_top5:
            key = " + ".join(_program_anchor_signature(row))
            local_anchor_counts[key] = local_anchor_counts.get(key, 0) + 1
        local_top5_scores = [float(row.get("representative_score", 0.0)) for row in local_top5]
        local_score_min = float(min(local_top5_scores)) if local_top5_scores else 0.0
        local_score_mean = float(sum(local_top5_scores) / max(1, len(local_top5_scores))) if local_top5_scores else 0.0
        local_quality_checks = {
            "top5_unique_groups_ge_target": {
                "observed": int(len(local_top5_groups)),
                "target": int(target_top5_groups),
                "pass": int(len(local_top5_groups)) >= int(target_top5_groups),
            },
            "top_window_unique_groups_ge_target": {
                "observed": int(len(local_top_window_groups)),
                "target": int(target_top_window_groups),
                "pass": int(len(local_top_window_groups)) >= int(target_top_window_groups),
            },
            "topk_care_process_count_le_max": {
                "observed": int(local_topk_care),
                "target": int(care_cap),
                "pass": (not care_cap_enabled) or int(local_topk_care) <= int(care_cap),
            },
            "top5_max_same_anchor_count_le_max": {
                "observed": int(max(local_anchor_counts.values(), default=0)),
                "target": int(anchor_cap),
                "pass": int(max(local_anchor_counts.values(), default=0)) <= int(anchor_cap),
            },
        }
        return {
            "top5_groups_count": int(len(local_top5_groups)),
            "top_window_groups_count": int(len(local_top_window_groups)),
            "topk_care_count": int(local_topk_care),
            "top5_anchor_counts": local_anchor_counts,
            "top5_max_anchor_count": int(max(local_anchor_counts.values(), default=0)),
            "top5_score_min": float(local_score_min),
            "top5_score_mean": float(local_score_mean),
            "quality_checks": local_quality_checks,
            "all_constraints_pass": all(bool(check.get("pass", False)) for check in local_quality_checks.values()),
        }

    def _objective(metrics: dict) -> tuple:
        passed_checks = int(sum(1 for check in metrics["quality_checks"].values() if bool(check.get("pass", False))))
        return (
            int(metrics["all_constraints_pass"]),
            passed_checks,
            int(metrics["top5_groups_count"]),
            int(metrics["top_window_groups_count"]),
            -int(metrics["topk_care_count"]),
            -int(metrics["top5_max_anchor_count"]),
            float(metrics["top5_score_min"]),
            float(metrics["top5_score_mean"]),
        )

    current_metrics = _summarize_rank_metrics(ordered_indices)
    if swap_budget > 0 and not bool(current_metrics.get("all_constraints_pass", False)):
        active_indices = list(ordered_indices[:active_window])
        tail_indices = list(ordered_indices[active_window:])
        for _ in range(swap_budget):
            current_obj = _objective(current_metrics)
            best_trial: tuple[tuple, int, int, dict] | None = None
            for pos in range(fixed_prefix, len(active_indices)):
                for tail_pos, tail_idx in enumerate(tail_indices):
                    tail_row = rows[tail_idx]
                    if float(tail_row.get("representative_score", 0.0)) < score_floor:
                        continue
                    trial_active = list(active_indices)
                    trial_tail = list(tail_indices)
                    replaced_idx = int(trial_active[pos])
                    trial_active[pos] = int(tail_idx)
                    trial_tail[tail_pos] = replaced_idx
                    trial_indices = trial_active + trial_tail
                    trial_metrics = _summarize_rank_metrics(trial_indices)
                    trial_obj = _objective(trial_metrics)
                    if trial_obj <= current_obj:
                        continue
                    if best_trial is None or trial_obj > best_trial[0]:
                        best_trial = (trial_obj, pos, tail_pos, trial_metrics)
            if best_trial is None:
                break
            _, best_pos, best_tail_pos, best_metrics = best_trial
            swap_in_idx = int(tail_indices[best_tail_pos])
            swap_out_idx = int(active_indices[best_pos])
            active_indices[best_pos] = swap_in_idx
            tail_indices[best_tail_pos] = swap_out_idx
            current_metrics = best_metrics
            stage2_swap_count += 1
        ordered_indices = active_indices + tail_indices
    # Top-5 rescue for reachable-but-drop cases: directly optimize top-5 mechanism coverage.
    current_metrics = _summarize_rank_metrics(ordered_indices)
    top5_rescue_meta = {
        "enabled": False,
        "reason": "not_needed_or_not_applicable",
        "before_top5_groups": int(current_metrics.get("top5_groups_count", 0)),
        "after_top5_groups": int(current_metrics.get("top5_groups_count", 0)),
        "target_top5_groups": int(target_top5_groups),
        "swap_count": 0,
        "attempt_budget": 0,
        "min_candidate_score": None,
        "under_floor_novel_group_swap_count": 0,
    }
    if int(target_top5_groups) > 0:
        current_top5_groups = int(current_metrics.get("top5_groups_count", 0))
        shortfall = int(target_top5_groups) - int(current_top5_groups)
        allow_start = max(0, int(fixed_prefix))
        if shortfall <= 0:
            top5_rescue_meta["enabled"] = True
            top5_rescue_meta["reason"] = "target_already_met"
        elif allow_start >= int(top5_window):
            top5_rescue_meta["enabled"] = False
            top5_rescue_meta["reason"] = "no_swappable_slots_in_top5"
        else:
            active_indices = list(ordered_indices[:active_window])
            tail_indices = list(ordered_indices[active_window:])
            if not tail_indices:
                top5_rescue_meta["enabled"] = False
                top5_rescue_meta["reason"] = "empty_tail_candidates"
            else:
                attempt_budget = max(1, int(swap_budget) if int(swap_budget) > 0 else int(top5_window))
                if math.isfinite(float(score_floor)):
                    min_candidate_score = float(score_floor) * 0.75
                else:
                    min_candidate_score = -float("inf")
                rescue_swaps = 0
                under_floor_novel_group_swaps = 0
                top5_rescue_meta["enabled"] = True
                top5_rescue_meta["reason"] = "applied"
                top5_rescue_meta["attempt_budget"] = int(attempt_budget)
                top5_rescue_meta["min_candidate_score"] = float(min_candidate_score)
                top5_rescue_meta["before_top5_groups"] = int(current_top5_groups)
                for _ in range(attempt_budget):
                    base_indices = active_indices + tail_indices
                    base_metrics = _summarize_rank_metrics(base_indices)
                    base_top5_groups = int(base_metrics.get("top5_groups_count", 0))
                    if base_top5_groups >= int(target_top5_groups):
                        current_metrics = base_metrics
                        break
                    base_checks = dict(base_metrics.get("quality_checks", {}))
                    base_top5_group_counts: dict[str, int] = {}
                    for idx in base_indices[:top5_window]:
                        for group in _program_mechanism_groups(rows[idx]):
                            base_top5_group_counts[group] = base_top5_group_counts.get(group, 0) + 1
                    best_trial = None
                    best_trial_score = None
                    for tail_pos, tail_idx in enumerate(tail_indices):
                        candidate_row = rows[tail_idx]
                        candidate_score = float(candidate_row.get("representative_score", 0.0))
                        candidate_groups = _program_mechanism_groups(candidate_row)
                        if not candidate_groups:
                            continue
                        for pos in range(int(allow_start), int(top5_window)):
                            trial_active = list(active_indices)
                            trial_tail = list(tail_indices)
                            replaced_idx = int(trial_active[pos])
                            replaced_row = rows[replaced_idx]
                            replaced_is_care = _is_care(replaced_row)
                            candidate_is_care = _is_care(candidate_row)
                            candidate_novel_groups = candidate_groups.difference(base_top5_group_counts.keys())
                            allow_under_floor_novel_group_swap = (
                                candidate_score < float(min_candidate_score)
                                and bool(candidate_novel_groups)
                                and not candidate_is_care
                            )
                            if candidate_score < float(min_candidate_score) and not allow_under_floor_novel_group_swap:
                                continue
                            trial_active[pos] = int(tail_idx)
                            trial_tail[tail_pos] = int(replaced_idx)
                            trial_indices = trial_active + trial_tail
                            trial_metrics = _summarize_rank_metrics(trial_indices)
                            trial_top5_groups = int(trial_metrics.get("top5_groups_count", 0))
                            if trial_top5_groups <= base_top5_groups:
                                continue
                            if allow_under_floor_novel_group_swap:
                                base_top5_score_min = float(base_metrics.get("top5_score_min", 0.0))
                                trial_top5_score_min = float(trial_metrics.get("top5_score_min", 0.0))
                                if trial_top5_score_min < base_top5_score_min * 0.75:
                                    continue
                            trial_checks = dict(trial_metrics.get("quality_checks", {}))
                            # Never trade away already-satisfied hard checks when rescuing top-5 coverage.
                            regression = False
                            for check_name, base_check in base_checks.items():
                                if check_name in {"top5_unique_groups_ge_target", "top_window_unique_groups_ge_target"}:
                                    continue
                                if bool(base_check.get("pass", False)) and not bool(
                                    dict(trial_checks.get(check_name, {})).get("pass", False)
                                ):
                                    regression = True
                                    break
                            if regression:
                                continue
                            trial_score = (
                                int(trial_top5_groups),
                                int(trial_metrics.get("all_constraints_pass", False)),
                                int(sum(1 for check in dict(trial_metrics.get("quality_checks", {})).values() if bool(check.get("pass", False)))),
                                int(trial_metrics.get("top_window_groups_count", 0)),
                                1 if replaced_is_care and not candidate_is_care else 0,
                                int(len(candidate_novel_groups)),
                                0 if candidate_is_care else 1,
                                1 if allow_under_floor_novel_group_swap else 0,
                                float(trial_metrics.get("top5_score_min", 0.0)),
                                float(candidate_score),
                            )
                            if best_trial_score is None or trial_score > best_trial_score:
                                best_trial_score = trial_score
                                best_trial = (
                                    trial_active,
                                    trial_tail,
                                    trial_metrics,
                                    bool(allow_under_floor_novel_group_swap),
                                )
                    if best_trial is None:
                        break
                    active_indices, tail_indices, current_metrics, used_under_floor_novel_group_swap = best_trial
                    rescue_swaps += 1
                    if used_under_floor_novel_group_swap:
                        under_floor_novel_group_swaps += 1
                ordered_indices = active_indices + tail_indices
                current_metrics = _summarize_rank_metrics(ordered_indices)
                top5_rescue_meta["after_top5_groups"] = int(current_metrics.get("top5_groups_count", 0))
                top5_rescue_meta["swap_count"] = int(rescue_swaps)
                top5_rescue_meta["under_floor_novel_group_swap_count"] = int(under_floor_novel_group_swaps)
                if rescue_swaps == 0 and int(top5_rescue_meta["before_top5_groups"]) < int(target_top5_groups):
                    top5_rescue_meta["reason"] = "no_improving_swap_found"
    # Coverage rescue for reachable-but-drop cases: only swap positions outside top-5.
    current_metrics = _summarize_rank_metrics(ordered_indices)
    coverage_rescue_meta = {
        "enabled": False,
        "reason": "not_needed_or_not_applicable",
        "before_top_window_groups": int(current_metrics.get("top_window_groups_count", 0)),
        "after_top_window_groups": int(current_metrics.get("top_window_groups_count", 0)),
        "target_top_window_groups": int(target_top_window_groups),
        "swap_count": 0,
        "attempt_budget": 0,
        "min_candidate_score": None,
        "under_floor_novel_group_swap_count": 0,
    }
    if int(target_top_window_groups) > 0:
        current_top_window_groups = int(current_metrics.get("top_window_groups_count", 0))
        shortfall = int(target_top_window_groups) - int(current_top_window_groups)
        allow_start = max(int(fixed_prefix), int(top5_window))
        if shortfall <= 0:
            coverage_rescue_meta["enabled"] = True
            coverage_rescue_meta["reason"] = "target_already_met"
        elif allow_start >= int(active_window):
            coverage_rescue_meta["enabled"] = False
            coverage_rescue_meta["reason"] = "no_swappable_slots_beyond_top5"
        else:
            active_indices = list(ordered_indices[:active_window])
            tail_indices = list(ordered_indices[active_window:])
            if not tail_indices:
                coverage_rescue_meta["enabled"] = False
                coverage_rescue_meta["reason"] = "empty_tail_candidates"
            else:
                attempt_budget = max(1, int(swap_budget) if int(swap_budget) > 0 else int(active_window))
                if math.isfinite(float(score_floor)):
                    min_candidate_score = float(score_floor) * 0.75
                else:
                    min_candidate_score = -float("inf")
                rescue_swaps = 0
                under_floor_novel_group_swaps = 0
                coverage_rescue_meta["enabled"] = True
                coverage_rescue_meta["reason"] = "applied"
                coverage_rescue_meta["attempt_budget"] = int(attempt_budget)
                coverage_rescue_meta["min_candidate_score"] = float(min_candidate_score)
                coverage_rescue_meta["before_top_window_groups"] = int(current_top_window_groups)
                for _ in range(attempt_budget):
                    base_indices = active_indices + tail_indices
                    base_metrics = _summarize_rank_metrics(base_indices)
                    base_groups = int(base_metrics.get("top_window_groups_count", 0))
                    if base_groups >= int(target_top_window_groups):
                        current_metrics = base_metrics
                        break
                    base_checks = dict(base_metrics.get("quality_checks", {}))
                    base_top_window_group_counts: dict[str, int] = {}
                    for idx in base_indices[:active_window]:
                        for group in _program_mechanism_groups(rows[idx]):
                            base_top_window_group_counts[group] = base_top_window_group_counts.get(group, 0) + 1
                    best_trial = None
                    best_trial_score = None
                    for tail_pos, tail_idx in enumerate(tail_indices):
                        candidate_row = rows[tail_idx]
                        candidate_score = float(candidate_row.get("representative_score", 0.0))
                        candidate_groups = _program_mechanism_groups(candidate_row)
                        if not candidate_groups:
                            continue
                        for pos in range(int(allow_start), int(active_window)):
                            trial_active = list(active_indices)
                            trial_tail = list(tail_indices)
                            replaced_idx = int(trial_active[pos])
                            replaced_row = rows[replaced_idx]
                            replaced_is_care = _is_care(replaced_row)
                            candidate_is_care = _is_care(candidate_row)
                            candidate_novel_groups = candidate_groups.difference(base_top_window_group_counts.keys())
                            allow_under_floor_novel_group_swap = (
                                candidate_score < float(min_candidate_score)
                                and bool(candidate_novel_groups)
                                and replaced_is_care
                                and not candidate_is_care
                            )
                            if candidate_score < float(min_candidate_score) and not allow_under_floor_novel_group_swap:
                                continue
                            trial_active[pos] = int(tail_idx)
                            trial_tail[tail_pos] = int(replaced_idx)
                            trial_indices = trial_active + trial_tail
                            trial_metrics = _summarize_rank_metrics(trial_indices)
                            trial_groups = int(trial_metrics.get("top_window_groups_count", 0))
                            if trial_groups <= base_groups:
                                continue
                            trial_checks = dict(trial_metrics.get("quality_checks", {}))
                            regression = False
                            for check_name, base_check in base_checks.items():
                                if check_name == "top_window_unique_groups_ge_target":
                                    continue
                                if bool(base_check.get("pass", False)) and not bool(
                                    dict(trial_checks.get(check_name, {})).get("pass", False)
                                ):
                                    regression = True
                                    break
                            if regression:
                                continue
                            trial_score = (
                                int(trial_groups),
                                int(trial_metrics.get("all_constraints_pass", False)),
                                int(sum(1 for check in trial_checks.values() if bool(check.get("pass", False)))),
                                1 if replaced_is_care and not candidate_is_care else 0,
                                int(len(candidate_novel_groups)),
                                0 if candidate_is_care else 1,
                                1 if allow_under_floor_novel_group_swap else 0,
                                float(trial_metrics.get("top5_score_min", 0.0)),
                                float(candidate_score),
                            )
                            if best_trial_score is None or trial_score > best_trial_score:
                                best_trial_score = trial_score
                                best_trial = (
                                    trial_active,
                                    trial_tail,
                                    trial_metrics,
                                    bool(allow_under_floor_novel_group_swap),
                                )
                    if best_trial is None:
                        break
                    active_indices, tail_indices, current_metrics, used_under_floor_novel_group_swap = best_trial
                    rescue_swaps += 1
                    if used_under_floor_novel_group_swap:
                        under_floor_novel_group_swaps += 1
                ordered_indices = active_indices + tail_indices
                current_metrics = _summarize_rank_metrics(ordered_indices)
                coverage_rescue_meta["after_top_window_groups"] = int(current_metrics.get("top_window_groups_count", 0))
                coverage_rescue_meta["swap_count"] = int(rescue_swaps)
                coverage_rescue_meta["under_floor_novel_group_swap_count"] = int(under_floor_novel_group_swaps)
                if rescue_swaps == 0 and int(coverage_rescue_meta["before_top_window_groups"]) < int(target_top_window_groups):
                    coverage_rescue_meta["reason"] = "no_improving_swap_found"

    ranked = _build_ranked(ordered_indices)

    before_top5 = rows[:top5_window]
    before_top_window = rows[:active_window]
    after_top5 = ranked[:top5_window]
    after_top_window = ranked[:active_window]

    before_top5_groups = {g for row in before_top5 for g in _program_mechanism_groups(row)}
    after_top5_groups = {g for row in after_top5 for g in _program_mechanism_groups(row)}
    before_top_window_groups = {g for row in before_top_window for g in _program_mechanism_groups(row)}
    after_top_window_groups = {g for row in after_top_window for g in _program_mechanism_groups(row)}
    before_topk_care = sum(1 for row in before_top_window[:topk_window] if _is_care(row))
    after_topk_care = sum(1 for row in after_top_window[:topk_window] if _is_care(row))
    rank_metrics = _summarize_rank_metrics(ordered_indices)
    after_anchor_counts = dict(rank_metrics["top5_anchor_counts"])
    quality_checks = dict(rank_metrics["quality_checks"])

    metadata = {
        "enabled": True,
        "strategy": "two_stage_constrained_rerank",
        "window_size": int(active_window),
        "top5_window_size": int(top5_window),
        "topk_window_size": int(topk_window),
        "selected_indices": [int(v) for v in ordered_indices[:active_window]],
        "fixed_prefix": int(fixed_prefix),
        "relax_level_used": int(relax_level_used),
        "candidate_pool_size_effective": int(max_pool),
        "quality_floor_ratio": float(quality_floor),
        "quality_floor_score": float(score_floor),
        "second_stage_swap_budget": int(swap_budget),
        "second_stage_swap_count": int(stage2_swap_count),
        "targets": {
            "top5_unique_groups_target": int(target_top5_groups),
            "top_window_unique_groups_target": int(target_top_window_groups),
            "topk_care_process_max_count": int(care_cap),
            "top5_max_same_anchor_count": int(anchor_cap),
        },
        "before": {
            "top5_unique_groups": int(len(before_top5_groups)),
            "top_window_unique_groups": int(len(before_top_window_groups)),
            "topk_care_process_count": int(before_topk_care),
        },
        "after": {
            "top5_unique_groups": int(len(after_top5_groups)),
            "top_window_unique_groups": int(len(after_top_window_groups)),
            "topk_care_process_count": int(after_topk_care),
            "top5_score_min": float(rank_metrics["top5_score_min"]),
            "top5_score_mean": float(rank_metrics["top5_score_mean"]),
            "top5_anchor_counts": after_anchor_counts,
        },
        "quality_checks": quality_checks,
        "all_constraints_pass": bool(rank_metrics["all_constraints_pass"]),
        "top5_rescue": top5_rescue_meta,
        "coverage_rescue": coverage_rescue_meta,
    }
    return ranked, metadata


def _enforce_program_mechanism_coverage_lower_bound(
    programs: list[dict],
    window_size: int,
    min_unique_groups: int,
    quality_floor_ratio: float = 0.0,
) -> tuple[list[dict], dict]:
    rows = [dict(row) for row in programs]
    if not rows:
        return [], {"enabled": False, "reason": "no_programs"}
    active_window = min(len(rows), max(1, int(window_size)))
    target_unique = max(0, int(min_unique_groups))
    if target_unique <= 0:
        return rows, {
            "enabled": False,
            "reason": "target_disabled",
            "window_size": int(active_window),
            "target_unique_groups": int(target_unique),
        }
    top_score = max(float(row.get("representative_score", 0.0)) for row in rows)
    floor_ratio = max(0.0, min(1.0, float(quality_floor_ratio)))
    score_floor = float(top_score * floor_ratio) if floor_ratio > 0.0 else -float("inf")

    def _groups_of(row: dict) -> set[str]:
        return set(_program_mechanism_groups(row))

    before_groups = set()
    for row in rows[:active_window]:
        before_groups.update(_groups_of(row))

    swap_count = 0
    max_swaps = max(0, min(len(rows), active_window * 4))
    for _ in range(max_swaps):
        top_groups = set()
        for row in rows[:active_window]:
            top_groups.update(_groups_of(row))
        if len(top_groups) >= target_unique:
            break

        outside_candidates: list[tuple[int, dict, set[str], float]] = []
        for idx in range(active_window, len(rows)):
            row = rows[idx]
            rep = float(row.get("representative_score", 0.0))
            if rep < score_floor:
                continue
            groups = _groups_of(row)
            gain = groups.difference(top_groups)
            if not gain:
                continue
            outside_candidates.append((idx, row, gain, rep))
        if not outside_candidates:
            break
        outside_candidates.sort(key=lambda item: (len(item[2]), item[3], -item[0]), reverse=True)
        src_idx, src_row, _, _ = outside_candidates[0]

        group_counts: dict[str, int] = {}
        for row in rows[:active_window]:
            for g in _groups_of(row):
                group_counts[g] = group_counts.get(g, 0) + 1
        replace_idx = None
        replace_key = None
        for idx in range(active_window):
            row = rows[idx]
            row_groups = _groups_of(row)
            uniqueness_cost = sum(1 for g in row_groups if group_counts.get(g, 0) <= 1)
            rep = float(row.get("representative_score", 0.0))
            key = (uniqueness_cost, rep, -idx)
            if replace_key is None or key < replace_key:
                replace_key = key
                replace_idx = idx
        if replace_idx is None:
            break
        rows[replace_idx], rows[src_idx] = rows[src_idx], rows[replace_idx]
        swap_count += 1

    after_groups = set()
    for row in rows[:active_window]:
        after_groups.update(_groups_of(row))
    metadata = {
        "enabled": True,
        "window_size": int(active_window),
        "target_unique_groups": int(target_unique),
        "quality_floor_ratio": float(floor_ratio),
        "quality_floor_score": float(score_floor),
        "before_unique_groups": int(len(before_groups)),
        "after_unique_groups": int(len(after_groups)),
        "swap_count": int(swap_count),
        "coverage_pass": int(len(after_groups)) >= int(target_unique),
    }
    return rows, metadata


def _evaluate_transfer_target_against_baseline(
    mechanism_diversity: dict,
    baseline_top5_unique_groups: int = -1,
    baseline_top10_unique_groups: int = -1,
    baseline_top5_score_min: float = float("nan"),
    max_drop_top5_unique_groups: int = 999,
    max_drop_top10_unique_groups: int = 999,
    max_drop_top5_score_min: float = 999.0,
) -> dict:
    top5 = int(mechanism_diversity.get("top5_unique_groups", 0))
    top10 = int(mechanism_diversity.get("top10_unique_groups", 0))
    score_min = float(mechanism_diversity.get("top5_score_min", 0.0))
    enabled = int(baseline_top5_unique_groups) >= 0 and int(baseline_top10_unique_groups) >= 0 and not math.isnan(float(baseline_top5_score_min))
    if not enabled:
        return {
            "enabled": False,
            "pass": False,
            "delta_top5_unique_groups": 0,
            "delta_top10_unique_groups": 0,
            "delta_top5_score_min": 0.0,
        }
    delta_top5 = int(top5 - int(baseline_top5_unique_groups))
    delta_top10 = int(top10 - int(baseline_top10_unique_groups))
    delta_score = float(score_min - float(baseline_top5_score_min))
    passed = (
        delta_top5 >= (-1 * int(max_drop_top5_unique_groups))
        and delta_top10 >= (-1 * int(max_drop_top10_unique_groups))
        and delta_score >= (-1.0 * float(max_drop_top5_score_min))
    )
    return {
        "enabled": True,
        "pass": bool(passed),
        "delta_top5_unique_groups": int(delta_top5),
        "delta_top10_unique_groups": int(delta_top10),
        "delta_top5_score_min": float(delta_score),
    }


def _rule_has_care_process(row: dict) -> bool:
    consequents = _normalized_feature_names(list(row.get("consequent_features", [])))
    if consequents.intersection(INTERVENTION_FEATURES):
        return True
    for feature in consequents:
        if MECHANISM_GROUP_BY_FEATURE.get(feature, "") == "care_process":
            return True
    return False


def _apply_shadow_predictive_rule_rerank(
    rules: list[dict],
    feature_weights: dict[str, float],
    blend_weight: float,
    max_adjustment: float,
    quality_floor_ratio: float,
    exclude_care_process: bool,
    apply_adjusted_score: bool,
) -> list[dict]:
    if not rules:
        return []

    weight = max(0.0, min(1.0, float(blend_weight)))
    max_adj = max(0.0, float(max_adjustment))
    if weight <= 0.0 or max_adj <= 0.0 or not feature_weights:
        return [dict(row) for row in rules]

    top_score = max(float(row.get("contrastive_score", 0.0)) for row in rules)
    floor_ratio = max(0.0, min(1.0, float(quality_floor_ratio)))
    score_floor = float(top_score * floor_ratio) if floor_ratio > 0.0 else -float("inf")

    enriched: list[dict] = []
    for idx, row in enumerate(rules):
        out = dict(row)
        antecedent = [str(v) for v in list(row.get("antecedent_features", []))]
        consequent = [str(v) for v in list(row.get("consequent_features", []))]
        keys = list(dict.fromkeys(antecedent + consequent))
        proxy_values = [float(feature_weights.get(key, 0.0)) for key in keys]
        proxy = float(sum(proxy_values) / max(1, len(proxy_values))) if proxy_values else 0.0
        base_score = float(row.get("contrastive_score", 0.0))
        is_care_process = bool(_rule_has_care_process(row))
        eligible = bool(base_score >= score_floor) and not (bool(exclude_care_process) and is_care_process)
        bonus = float(weight * max_adj * proxy) if eligible else 0.0
        adjusted = float(base_score + bonus)

        out["shadow_predictive_rule_proxy"] = float(proxy)
        out["shadow_predictive_rule_bonus"] = float(bonus)
        out["shadow_predictive_rule_adjusted_score"] = float(adjusted)
        out["shadow_predictive_rule_boost_eligible"] = bool(eligible)
        out["shadow_predictive_rule_is_care_process"] = bool(is_care_process)
        if bool(apply_adjusted_score):
            out["base_contrastive_score_before_shadow_rule"] = float(base_score)
            out["contrastive_score"] = float(adjusted)
        out["_shadow_rank_index"] = int(idx)
        enriched.append(out)

    enriched.sort(
        key=lambda row: (
            float(row.get("shadow_predictive_rule_adjusted_score", 0.0)),
            float(row.get("contrastive_score", 0.0)),
            -int(row.get("_shadow_rank_index", 0)),
        ),
        reverse=True,
    )

    for row in enriched:
        row.pop("_shadow_rank_index", None)
    return enriched


def _infer_spec_window_hours(spec: dict) -> int | None:
    shape = list(spec.get("shape") or [])
    resample_minutes = spec.get("resample_minutes", 60)
    if not shape:
        return None
    try:
        return int(round((int(shape[0]) - 1) * float(resample_minutes) / 60.0))
    except (TypeError, ValueError):
        return None


def _filter_episode_specs(specs: list[dict], anchor_kind: str | None = None, window_hours: int | None = None) -> list[dict]:
    anchor_filter = str(anchor_kind or "").strip().lower()
    window_filter = None if window_hours is None else int(window_hours)
    filtered = []
    for spec in specs:
        if not isinstance(spec, dict):
            continue
        if anchor_filter and str(spec.get("anchor_kind", "")).strip().lower() != anchor_filter:
            continue
        if window_filter is not None and _infer_spec_window_hours(spec) != window_filter:
            continue
        filtered.append(spec)
    return filtered

def _default_miner_config() -> dict:
    return {
        "eventizer_kind": "icu",
        "n_bins": 4,
        "eventizer_window": 6,
        "quantile_stride": 1,
        "change_threshold": 0.25,
        "event_support_min": None,
        "event_support_max": None,
        "seg_window": 6,
        "seg_threshold": 1.0,
        "max_length": 3,
        "gap_max": 2,
        "top_M0": 50,
        "top_candidates": 200,
        "top_M1": 50,
        "top_M2": 100,
        "top_K": 20,
        "report_print_top_n": 0,
        "report_compute_ranking": False,
        "print_stage_timing": False,
    }


def _json_safe(value):
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def _compute_sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _compute_input_data_fingerprint(input_path: Path) -> str:
    if not input_path.exists():
        return ""
    hasher = hashlib.sha256()
    file_paths = sorted([path for path in input_path.rglob("*") if path.is_file()])
    for path in file_paths:
        try:
            rel = str(path.relative_to(input_path)).replace("\\", "/")
        except ValueError:
            rel = str(path).replace("\\", "/")
        stat = path.stat()
        hasher.update(rel.encode("utf-8"))
        hasher.update(str(int(stat.st_size)).encode("utf-8"))
        hasher.update(str(int(stat.st_mtime_ns)).encode("utf-8"))
    return hasher.hexdigest()


def _resolve_git_commit(root: Path) -> str:
    try:
        output = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(root),
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return str(output).strip()
    except Exception:
        return ""


def _is_low_information_episode(
    feature_meta: dict,
    min_core_observed_channels: int,
    min_core_observed_points: int,
) -> bool:
    missing_mask = feature_meta.get("missing_mask")
    feature_names = list(feature_meta.get("feature_names", []))
    if missing_mask is None or not feature_names:
        return False

    mask = np.asarray(missing_mask, dtype=bool)
    if mask.ndim != 2 or mask.shape[1] != len(feature_names):
        return False

    core_indices = [idx for idx, name in enumerate(feature_names) if name in CORE_OBS_FEATURES]
    if not core_indices:
        return False

    observed = (~mask[:, core_indices]).sum(axis=0)
    observed_channels = int((observed > 0).sum())
    observed_points = int(observed.sum())
    if observed_channels < int(min_core_observed_channels):
        return True
    if observed_points < int(min_core_observed_points):
        return True
    return False


def _summarize_series(values: list[float]) -> dict:
    if not values:
        return {"count": 0}
    ordered = sorted(float(v) for v in values)
    n = len(ordered)

    def quantile(prob: float) -> float:
        if n == 1:
            return ordered[0]
        index = int(round((n - 1) * float(prob)))
        return ordered[max(0, min(n - 1, index))]

    return {
        "count": n,
        "min": ordered[0],
        "p25": quantile(0.25),
        "p50": quantile(0.50),
        "p75": quantile(0.75),
        "p90": quantile(0.90),
        "max": ordered[-1],
        "mean": float(sum(ordered) / n),
    }


def _normalized_feature_names(feature_names: list[str]) -> set[str]:
    return {str(name).split(":", 1)[0] for name in feature_names}


def _is_physiologic_rule(rule: dict) -> bool:
    consequents = _normalized_feature_names(list(rule.get("consequent_features", [])))
    if not consequents:
        return False
    return consequents.isdisjoint(INTERVENTION_FEATURES)


def _collect_lead_time_hours(
    contrastive_inputs: list[dict],
    episode_type: str | None = None,
    physiologic_only: bool = False,
) -> list[float]:
    values = []
    for episode in contrastive_inputs:
        if episode_type is not None and str(episode.get("episode_type")) != str(episode_type):
            continue
        for rule in episode.get("rules", []):
            if physiologic_only and not _is_physiologic_rule(rule):
                continue
            lead_time_steps = float(rule.get("lead_time_steps", 0.0))
            resample_minutes = float(rule.get("resample_minutes", 60))
            values.append((lead_time_steps * resample_minutes) / 60.0)
    return values


def _episode_anchor_segments(X: np.ndarray, resample_minutes: int) -> list[dict]:
    matrix = np.asarray(X, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] == 0:
        return []
    lead_hours = [((matrix.shape[0] - idx) * float(resample_minutes)) / 60.0 for idx in range(matrix.shape[0])]
    deltas = np.zeros(matrix.shape[0], dtype=float)
    if matrix.shape[0] > 1:
        deltas[1:] = np.mean(np.abs(np.diff(matrix, axis=0)), axis=1)
    instability = np.cumsum(deltas)
    if instability.size > 0 and float(instability.max()) > 0.0:
        instability = instability / float(instability.max())
    return segment_episode_relative_to_anchor(
        lead_hours=lead_hours,
        instability_score=instability.tolist(),
        min_phase_len=2,
    )


def _program_interpretability_tag(program: dict) -> str:
    consequents = _normalized_feature_names(list(program.get("consequent_features", [])))
    if not consequents:
        return "unclear"
    has_intervention = not consequents.isdisjoint(INTERVENTION_FEATURES)
    has_physiology = bool(consequents.difference(INTERVENTION_FEATURES))
    if has_intervention and has_physiology:
        return "mixed-physiology-and-care-process"
    if has_intervention:
        return "care-process-linked"
    return "physiology-linked"


def _program_anchor_signature(program: dict) -> tuple[str, ...]:
    return tuple(sorted(_normalized_feature_names(list(program.get("antecedent_features", [])))))


def _program_mechanism_groups(program: dict) -> set[str]:
    groups = {str(group) for group in program.get("mechanism_groups", []) if str(group)}
    if groups:
        return groups

    derived = set()
    for feature in _normalized_feature_names(list(program.get("consequent_features", []))):
        derived.add(MECHANISM_GROUP_BY_FEATURE.get(feature, f"other:{feature}"))
    return derived


def _resolve_program_mechanism_group_mode(mode: str | None) -> str:
    resolved = str(mode or "coarse").strip().lower()
    if resolved in PROGRAM_MECHANISM_GROUP_MODES:
        return resolved
    return "coarse"


def _apply_program_mechanism_group_mode(
    programs: list[dict],
    mode: str,
    feature_score_floor_ratio: float = 0.0,
) -> list[dict]:
    resolved_mode = _resolve_program_mechanism_group_mode(mode)
    if resolved_mode == "coarse":
        return list(programs)

    ratio = max(0.0, min(1.0, float(feature_score_floor_ratio)))
    top_score = 0.0
    for program in list(programs):
        try:
            score = float(program.get("representative_score", 0.0) or 0.0)
        except (TypeError, ValueError):
            score = 0.0
        if score > top_score:
            top_score = score
    score_floor = top_score * ratio if ratio > 0.0 and top_score > 0.0 else None

    remapped: list[dict] = []
    for program in list(programs):
        row = dict(program)
        groups = set(_program_mechanism_groups(row))
        try:
            score = float(row.get("representative_score", 0.0) or 0.0)
        except (TypeError, ValueError):
            score = 0.0
        add_feature_groups = score_floor is None or score >= score_floor
        if add_feature_groups:
            for feature in sorted(_normalized_feature_names(list(row.get("consequent_features", [])))):
                coarse_group = MECHANISM_GROUP_BY_FEATURE.get(feature, f"other:{feature}")
                groups.add(f"{coarse_group}::{feature}")
        row["mechanism_groups"] = sorted(groups)
        remapped.append(row)
    return remapped


def _resolve_program_score_calibration_mode(mode: str | None) -> str:
    resolved = str(mode or "none").strip().lower()
    if resolved in PROGRAM_SCORE_CALIBRATION_MODES:
        return resolved
    return "none"


def _quantile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    qq = max(0.0, min(1.0, float(q)))
    arr = np.asarray(values, dtype=float)
    return float(np.quantile(arr, qq))


def _apply_program_score_calibration(
    programs: list[dict],
    mode: str,
    quantile_low: float,
    quantile_high: float,
    target_low: float,
    target_high: float,
    blend_weight: float,
) -> tuple[list[dict], dict]:
    rows = [dict(row) for row in list(programs)]
    resolved_mode = _resolve_program_score_calibration_mode(mode)
    if not rows:
        return rows, {
            "enabled": False,
            "mode": str(resolved_mode),
            "reason": "empty_programs",
        }
    if resolved_mode == "none":
        for row in rows:
            raw_score = float(row.get("representative_score", 0.0) or 0.0)
            row["representative_score_raw"] = float(raw_score)
            row["representative_score_calibrated"] = float(raw_score)
            row["representative_score_calibration_delta"] = 0.0
            row["representative_score_is_calibrated"] = False
        return rows, {
            "enabled": False,
            "mode": "none",
            "reason": "disabled",
        }

    q_low = max(0.0, min(1.0, float(quantile_low)))
    q_high = max(0.0, min(1.0, float(quantile_high)))
    if q_high <= q_low:
        q_low, q_high = 0.5, 0.9
    t_low = float(target_low)
    t_high = float(target_high)
    if not (t_high > t_low > 0.0):
        for row in rows:
            raw_score = float(row.get("representative_score", 0.0) or 0.0)
            row["representative_score_raw"] = float(raw_score)
            row["representative_score_calibrated"] = float(raw_score)
            row["representative_score_calibration_delta"] = 0.0
            row["representative_score_is_calibrated"] = False
        return rows, {
            "enabled": False,
            "mode": str(resolved_mode),
            "reason": "invalid_target_range",
            "target_low": float(t_low),
            "target_high": float(t_high),
        }

    weight = max(0.0, min(1.0, float(blend_weight)))
    raw_scores = [float(row.get("representative_score", 0.0) or 0.0) for row in rows]
    obs_low = _quantile(raw_scores, q_low)
    obs_high = _quantile(raw_scores, q_high)
    if obs_high <= obs_low + 1e-12:
        for row in rows:
            raw_score = float(row.get("representative_score", 0.0) or 0.0)
            row["representative_score_raw"] = float(raw_score)
            row["representative_score_calibrated"] = float(raw_score)
            row["representative_score_calibration_delta"] = 0.0
            row["representative_score_is_calibrated"] = False
        return rows, {
            "enabled": False,
            "mode": str(resolved_mode),
            "reason": "degenerate_observed_quantiles",
            "observed_low": float(obs_low),
            "observed_high": float(obs_high),
        }

    slope = (t_high - t_low) / (obs_high - obs_low)
    intercept = t_low - (slope * obs_low)
    calibrated_scores: list[float] = []
    blended_scores: list[float] = []
    for row in rows:
        raw_score = float(row.get("representative_score", 0.0) or 0.0)
        calibrated = max(0.0, float((slope * raw_score) + intercept))
        blended = float(((1.0 - weight) * raw_score) + (weight * calibrated))
        row["representative_score_raw"] = float(raw_score)
        row["representative_score_calibrated"] = float(calibrated)
        row["representative_score_calibration_delta"] = float(blended - raw_score)
        row["representative_score_is_calibrated"] = True
        row["representative_score"] = float(blended)
        calibrated_scores.append(float(calibrated))
        blended_scores.append(float(blended))

    return rows, {
        "enabled": True,
        "mode": str(resolved_mode),
        "quantile_low": float(q_low),
        "quantile_high": float(q_high),
        "target_low": float(t_low),
        "target_high": float(t_high),
        "blend_weight": float(weight),
        "observed_low": float(obs_low),
        "observed_high": float(obs_high),
        "slope": float(slope),
        "intercept": float(intercept),
        "raw_score_summary": _summarize_series(raw_scores),
        "calibrated_score_summary": _summarize_series(calibrated_scores),
        "blended_score_summary": _summarize_series(blended_scores),
    }


def _wilson_lower_bound(success_count: int, total_count: int, z_score: float) -> float:
    total = max(0, int(total_count))
    success = max(0, int(success_count))
    if total <= 0:
        return 0.0
    success = min(success, total)
    z = max(0.0, float(z_score))
    if z <= 0.0:
        return float(success) / float(total)
    p_hat = float(success) / float(total)
    z2 = z * z
    denom = 1.0 + (z2 / float(total))
    center = p_hat + (z2 / (2.0 * float(total)))
    margin_term = (p_hat * (1.0 - p_hat) / float(total)) + (z2 / (4.0 * float(total) * float(total)))
    margin = z * math.sqrt(max(0.0, margin_term))
    return max(0.0, min(1.0, float((center - margin) / denom)))


def _apply_program_support_confidence_adjustment(
    programs: list[dict],
    weight: float,
    z_score: float,
    floor: float,
    exponent: float,
) -> tuple[list[dict], dict]:
    rows = [dict(row) for row in list(programs)]
    blend_weight = max(0.0, min(1.0, float(weight)))
    confidence_floor = max(0.0, min(1.0, float(floor)))
    confidence_exponent = max(1e-6, float(exponent))
    confidence_z_score = max(0.0, float(z_score))
    if not rows:
        return rows, {
            "enabled": False,
            "reason": "empty_programs",
            "weight": float(blend_weight),
        }
    if blend_weight <= 0.0:
        for row in rows:
            score_value = float(row.get("representative_score", 0.0) or 0.0)
            row["representative_score_support_confidence"] = 1.0
            row["representative_score_support_confidence_factor"] = 1.0
            row["representative_score_before_support_confidence"] = float(score_value)
            row["representative_score_support_confidence_delta"] = 0.0
        return rows, {
            "enabled": False,
            "reason": "weight_zero",
            "weight": float(blend_weight),
            "z_score": float(confidence_z_score),
            "floor": float(confidence_floor),
            "exponent": float(confidence_exponent),
        }

    confidence_values: list[float] = []
    factor_values: list[float] = []
    before_scores: list[float] = []
    after_scores: list[float] = []
    for row in rows:
        score_before = float(row.get("representative_score", 0.0) or 0.0)
        support = dict(row.get("representative_score_support", {}))
        positive_count = int(support.get("positive_count", 0) or 0)
        control_count = int(support.get("control_count", 0) or 0)
        positive_total = int(support.get("positive_total", 0) or 0)
        control_total = int(support.get("control_total", 0) or 0)
        support_hits = max(0, positive_count) + max(0, control_count)
        support_total = max(0, positive_total) + max(0, control_total)
        confidence = _wilson_lower_bound(support_hits, support_total, confidence_z_score)
        confidence_shaped = max(0.0, min(1.0, float(confidence))) ** confidence_exponent
        confidence_factor = confidence_floor + ((1.0 - confidence_floor) * confidence_shaped)
        score_factor = (1.0 - blend_weight) + (blend_weight * confidence_factor)
        score_after = float(score_before * score_factor)
        components = dict(row.get("representative_score_components", {}))
        components["support_confidence"] = float(confidence)
        components["support_confidence_factor"] = float(confidence_factor)
        row["representative_score_components"] = components
        row["representative_score_support_confidence"] = float(confidence)
        row["representative_score_support_confidence_factor"] = float(score_factor)
        row["representative_score_before_support_confidence"] = float(score_before)
        row["representative_score_support_confidence_delta"] = float(score_after - score_before)
        row["representative_score"] = float(score_after)
        confidence_values.append(float(confidence))
        factor_values.append(float(score_factor))
        before_scores.append(float(score_before))
        after_scores.append(float(score_after))

    return rows, {
        "enabled": True,
        "weight": float(blend_weight),
        "z_score": float(confidence_z_score),
        "floor": float(confidence_floor),
        "exponent": float(confidence_exponent),
        "confidence_summary": _summarize_series(confidence_values),
        "factor_summary": _summarize_series(factor_values),
        "score_before_summary": _summarize_series(before_scores),
        "score_after_summary": _summarize_series(after_scores),
    }


def _summarize_contrastive_qc(contrastive_inputs: list[dict], programs: list[dict], top_k: int = 5) -> dict:
    positive_count = sum(1 for episode in contrastive_inputs if episode.get("episode_type") == "positive")
    control_count = sum(1 for episode in contrastive_inputs if episode.get("episode_type") == "control")
    top_limit = max(1, int(top_k))
    top_programs_window = list(programs[:top_limit])
    top10_window = list(programs[:10])

    top_programs = []
    for rank, program in enumerate(top_programs_window, start=1):
        score_raw = float(program.get("representative_score_raw", program.get("representative_score", 0.0)))
        score_value = float(program.get("representative_score", 0.0))
        score_calibrated = float(program.get("representative_score_calibrated", score_value))
        top_programs.append(
            {
                "rank": rank,
                "program_id": program.get("program_id"),
                "program_label": program.get("program_label"),
                "rule_count": int(program.get("rule_count", 0)),
                "representative_score": float(score_value),
                "representative_score_raw": float(score_raw),
                "representative_score_calibrated": float(score_calibrated),
                "representative_score_calibration_delta": float(score_value - score_raw),
                "antecedent_features": list(program.get("antecedent_features", [])),
                "consequent_features": list(program.get("consequent_features", [])),
                "representative_score_components": dict(program.get("representative_score_components", {})),
                "representative_score_support": dict(program.get("representative_score_support", {})),
                "interpretability_tag": _program_interpretability_tag(program),
            }
        )

    top5_group_counts: dict[str, int] = {}
    for program in top_programs_window:
        for group in _program_mechanism_groups(program):
            top5_group_counts[group] = top5_group_counts.get(group, 0) + 1

    top5_anchor_counts: dict[tuple[str, ...], int] = {}
    for program in top_programs_window:
        anchor = _program_anchor_signature(program)
        top5_anchor_counts[anchor] = top5_anchor_counts.get(anchor, 0) + 1

    all_unique_groups = len({group for program in programs for group in _program_mechanism_groups(program)})
    top5_unique_groups = len({group for program in top_programs_window for group in _program_mechanism_groups(program)})
    top10_unique_groups = len({group for program in top10_window for group in _program_mechanism_groups(program)})
    top5_score_min = min(
        (float(program.get("representative_score", 0.0)) for program in top_programs_window),
        default=0.0,
    )
    top5_score_min_raw = min(
        (
            float(program.get("representative_score_raw", program.get("representative_score", 0.0)))
            for program in top_programs_window
        ),
        default=0.0,
    )
    top5_care_process_count = sum(
        1 for program in top_programs_window if "care_process" in _program_mechanism_groups(program)
    )
    top5_unique_anchor_count = len(top5_anchor_counts)
    top5_max_same_anchor_count = max(top5_anchor_counts.values(), default=0)
    best_score_per_anchor: dict[str, float] = {}
    for program in programs:
        anchor_tuple = _program_anchor_signature(program)
        anchor_label = " + ".join(anchor_tuple) if anchor_tuple else "unknown"
        score = float(program.get("representative_score", 0.0))
        current = best_score_per_anchor.get(anchor_label)
        if current is None or score > current:
            best_score_per_anchor[anchor_label] = score
    anchors_above_score_082_count = sum(1 for score in best_score_per_anchor.values() if score >= 0.82)

    return {
        "episode_counts": {"positive": int(positive_count), "control": int(control_count)},
        "lead_time_hours": {
            "all": _summarize_series(_collect_lead_time_hours(contrastive_inputs)),
            "positive": _summarize_series(_collect_lead_time_hours(contrastive_inputs, episode_type="positive")),
            "control": _summarize_series(_collect_lead_time_hours(contrastive_inputs, episode_type="control")),
            "physiologic_consequents_only": _summarize_series(
                _collect_lead_time_hours(contrastive_inputs, physiologic_only=True)
            ),
        },
        "mechanism_diversity": {
            "all_unique_groups": int(all_unique_groups),
            "top5_unique_groups_ceiling": int(min(max(0, int(top_limit)), max(0, int(all_unique_groups)))),
            "top10_unique_groups_ceiling": int(min(10, max(0, int(all_unique_groups)))),
            "top5_unique_groups": int(top5_unique_groups),
            "top10_unique_groups": int(top10_unique_groups),
            "top5_score_min": float(top5_score_min),
            "top5_score_min_raw": float(top5_score_min_raw),
            "top5_care_process_count": int(top5_care_process_count),
            "top5_unique_anchor_count": int(top5_unique_anchor_count),
            "top5_max_same_anchor_count": int(top5_max_same_anchor_count),
            "anchors_above_score_082_count": int(anchors_above_score_082_count),
            "best_score_per_anchor": {
                key: float(value)
                for key, value in sorted(best_score_per_anchor.items(), key=lambda item: item[0])
            },
            "top5_group_counts": dict(sorted(top5_group_counts.items(), key=lambda item: item[0])),
        },
        "top_program_spotcheck": top_programs,
    }


def _summarize_statistical_evidence(rules: list[dict], top_k: int = 10) -> dict:
    ranked = list(rules)
    top_limit = max(1, int(top_k))
    if not ranked:
        return {
            "rule_count": 0,
            "top_k": int(top_limit),
            "p_value": {"count": 0},
            "q_value": {"count": 0},
            "effect_size": {"count": 0},
            "ci_width": {"count": 0},
            "significance_counts": {
                "p_le_0_05": 0,
                "q_le_0_05": 0,
                "q_eq_1_0": 0,
                "topk_q_le_0_05": 0,
            },
            "top_rules": [],
        }

    top_rules = ranked[:top_limit]
    p_values = [min(1.0, max(0.0, float(row.get("p_value", 1.0)))) for row in ranked]
    q_values = [min(1.0, max(0.0, float(row.get("q_value", 1.0)))) for row in ranked]
    effect_sizes = [float(row.get("effect_size", 0.0)) for row in ranked]
    ci_widths = [
        max(0.0, float(row.get("ci_high", 0.0)) - float(row.get("ci_low", 0.0)))
        for row in ranked
    ]

    return {
        "rule_count": int(len(ranked)),
        "top_k": int(top_limit),
        "p_value": _summarize_series(p_values),
        "q_value": _summarize_series(q_values),
        "effect_size": _summarize_series(effect_sizes),
        "ci_width": _summarize_series(ci_widths),
        "significance_counts": {
            "p_le_0_05": int(sum(1 for value in p_values if value <= 0.05)),
            "q_le_0_05": int(sum(1 for value in q_values if value <= 0.05)),
            "q_eq_1_0": int(sum(1 for value in q_values if value >= 1.0)),
            "topk_q_le_0_05": int(
                sum(
                    1
                    for row in top_rules
                    if min(1.0, max(0.0, float(row.get("q_value", row.get("p_value", 1.0))))) <= 0.05
                )
            ),
        },
        "top_rules": [
            {
                "rank": int(idx),
                "rule_id": row.get("rule_id"),
                "contrastive_score": float(row.get("contrastive_score", 0.0)),
                "effect_size": float(row.get("effect_size", 0.0)),
                "p_value": min(1.0, max(0.0, float(row.get("p_value", 1.0)))),
                "q_value": min(1.0, max(0.0, float(row.get("q_value", row.get("p_value", 1.0))))),
                "ci_low": float(row.get("ci_low", 0.0)),
                "ci_high": float(row.get("ci_high", 0.0)),
            }
            for idx, row in enumerate(top_rules, start=1)
        ],
    }


def _build_statistical_null_benchmark(
    contrastive_inputs: list[dict],
    scoring_kwargs: dict,
    observed_rules: list[dict],
    permutation_count: int,
    random_seed: int,
    top_k: int,
) -> dict:
    requested = max(0, int(permutation_count))
    top_limit = max(1, int(top_k))
    if requested <= 0:
        return {
            "enabled": False,
            "permutation_count": 0,
            "top_k": int(top_limit),
            "observed": {},
            "null_distribution": {},
            "empirical_p_values": {},
        }

    episodes = list(contrastive_inputs)
    labels = [str(episode.get("episode_type", "")) for episode in episodes]
    if len(episodes) <= 1 or len(set(labels)) <= 1:
        return {
            "enabled": False,
            "permutation_count": 0,
            "top_k": int(top_limit),
            "observed": {},
            "null_distribution": {},
            "empirical_p_values": {},
            "reason": "insufficient_label_variation",
        }

    rng = np.random.default_rng(int(random_seed))
    observed_window = list(observed_rules[:top_limit])
    observed_topk_score_mean = float(
        sum(float(row.get("contrastive_score", 0.0)) for row in observed_window) / max(1, len(observed_window))
    )
    observed_top1_score = float(observed_window[0].get("contrastive_score", 0.0)) if observed_window else 0.0
    observed_topk_q05_ratio = float(
        sum(
            1
            for row in observed_window
            if min(1.0, max(0.0, float(row.get("q_value", row.get("p_value", 1.0))))) <= 0.05
        )
        / max(1, len(observed_window))
    )

    null_topk_score_means: list[float] = []
    null_top1_scores: list[float] = []
    null_topk_q05_ratios: list[float] = []

    base_kwargs = dict(scoring_kwargs)
    base_kwargs["return_candidate_pool_diagnostics"] = False

    for perm_idx in range(requested):
        shuffled_labels = list(labels)
        rng.shuffle(shuffled_labels)
        permuted_inputs = []
        for episode, shuffled_label in zip(episodes, shuffled_labels):
            copied = dict(episode)
            copied["episode_type"] = shuffled_label
            permuted_inputs.append(copied)

        base_kwargs["significance_random_state"] = int(random_seed + perm_idx + 1)
        permuted_scored = score_contrastive_rule_records(permuted_inputs, **base_kwargs)
        if isinstance(permuted_scored, tuple):
            permuted_rules = list(permuted_scored[0])
        else:
            permuted_rules = list(permuted_scored)
        permuted_window = list(permuted_rules[:top_limit])
        null_topk_score_means.append(
            float(sum(float(row.get("contrastive_score", 0.0)) for row in permuted_window) / max(1, len(permuted_window)))
        )
        null_top1_scores.append(float(permuted_window[0].get("contrastive_score", 0.0)) if permuted_window else 0.0)
        null_topk_q05_ratios.append(
            float(
                sum(
                    1
                    for row in permuted_window
                    if min(1.0, max(0.0, float(row.get("q_value", row.get("p_value", 1.0))))) <= 0.05
                )
                / max(1, len(permuted_window))
            )
        )

    null_count = max(1, len(null_topk_score_means))
    empirical_topk_score_mean = float(
        (1 + sum(1 for value in null_topk_score_means if value >= observed_topk_score_mean)) / (null_count + 1)
    )
    empirical_top1_score = float(
        (1 + sum(1 for value in null_top1_scores if value >= observed_top1_score)) / (null_count + 1)
    )
    empirical_topk_q05_ratio = float(
        (1 + sum(1 for value in null_topk_q05_ratios if value >= observed_topk_q05_ratio)) / (null_count + 1)
    )

    return {
        "enabled": True,
        "permutation_count": int(requested),
        "top_k": int(top_limit),
        "observed": {
            "topk_score_mean": float(observed_topk_score_mean),
            "top1_score": float(observed_top1_score),
            "topk_q05_ratio": float(observed_topk_q05_ratio),
        },
        "null_distribution": {
            "topk_score_mean": _summarize_series(null_topk_score_means),
            "top1_score": _summarize_series(null_top1_scores),
            "topk_q05_ratio": _summarize_series(null_topk_q05_ratios),
        },
        "empirical_p_values": {
            "topk_score_mean_ge_null": float(empirical_topk_score_mean),
            "top1_score_ge_null": float(empirical_top1_score),
            "topk_q05_ratio_ge_null": float(empirical_topk_q05_ratio),
        },
    }


def _summarize_actionability_bridge(
    rules: list[dict],
    programs: list[dict],
    top_k: int = 5,
    min_lead_hours: float = 2.0,
) -> dict:
    top_limit = max(1, int(top_k))
    top_rules = list(rules[:top_limit])
    top_programs = list(programs[:top_limit])

    lead_hours = [float(row.get("lead_time_hours_mean", 0.0)) for row in top_rules]
    lead_summary = _summarize_series(lead_hours)
    lead_mean = float(lead_summary.get("mean", 0.0)) if lead_summary.get("count", 0) else 0.0
    lead_std = 0.0
    if len(lead_hours) > 1:
        variance = sum((float(value) - lead_mean) ** 2 for value in lead_hours) / float(len(lead_hours))
        lead_std = float(math.sqrt(max(0.0, variance)))
    lead_cv = float(lead_std / lead_mean) if lead_mean > 0.0 else 0.0
    lead_above_threshold_ratio = float(
        sum(1 for value in lead_hours if float(value) >= float(min_lead_hours)) / max(1, len(lead_hours))
    )

    rule_anchor_counts: dict[tuple[str, ...], int] = {}
    mechanism_groups = set()
    for row in top_rules:
        anchor = tuple(sorted(_normalized_feature_names(list(row.get("antecedent_features", [])))))
        rule_anchor_counts[anchor] = rule_anchor_counts.get(anchor, 0) + 1
        for feature in _normalized_feature_names(list(row.get("consequent_features", []))):
            mechanism_groups.add(MECHANISM_GROUP_BY_FEATURE.get(feature, f"other:{feature}"))
    anchor_unique_count = int(len(rule_anchor_counts))
    anchor_max_share = float(max(rule_anchor_counts.values(), default=0) / max(1, len(top_rules)))

    program_signature_counts: dict[tuple[str, ...], int] = {}
    care_process_count = 0
    for program in top_programs:
        signature = tuple(sorted(_normalized_feature_names(list(program.get("consequent_features", [])))))
        program_signature_counts[signature] = program_signature_counts.get(signature, 0) + 1
        if "care_process" in _program_mechanism_groups(program):
            care_process_count += 1
    duplicate_programs = int(
        sum((count - 1) for count in program_signature_counts.values() if int(count) > 1)
    )
    redundancy_ratio = float(duplicate_programs / max(1, len(top_programs)))
    care_process_ratio = float(care_process_count / max(1, len(top_programs)))

    if lead_above_threshold_ratio >= 0.8 and anchor_max_share <= 0.6 and redundancy_ratio <= 0.2:
        status = "green"
    elif lead_above_threshold_ratio >= 0.5 and anchor_max_share <= 0.8 and redundancy_ratio <= 0.4:
        status = "yellow"
    else:
        status = "red"

    return {
        "top_k": int(top_limit),
        "min_lead_hours_threshold": float(min_lead_hours),
        "lead_time_hours": {
            **lead_summary,
            "std": float(lead_std),
            "cv": float(lead_cv),
            "above_threshold_ratio": float(lead_above_threshold_ratio),
        },
        "anchor_consistency": {
            "topk_unique_anchor_count": int(anchor_unique_count),
            "topk_max_anchor_share": float(anchor_max_share),
        },
        "mechanism_consistency": {
            "topk_unique_mechanism_groups": int(len(mechanism_groups)),
        },
        "program_redundancy": {
            "duplicate_program_count": int(duplicate_programs),
            "redundancy_ratio": float(redundancy_ratio),
        },
        "care_process_exposure": {
            "topk_care_process_count": int(care_process_count),
            "topk_care_process_ratio": float(care_process_ratio),
        },
        "actionability_status": status,
    }


_ALGORITHM_READY_FAILURE_PRIORITY = [
    "top5_care_process_count_le_max",
    "top5_score_min_ge_threshold",
    "top5_unique_groups_ge_min",
    "top10_unique_groups_ge_min",
    "top5_max_same_anchor_count_le_max",
    "top5_unique_anchor_count_ge_min",
    "contrastive_programs_count_ge_reference_floor",
    "contrastive_rules_count_ge_reference_floor",
]

_ALGORITHM_READY_SUGGESTION_POLICY = {
    "top5_care_process_count_le_max": [
        {
            "knob": "program_care_process_penalty",
            "action": "increase",
            "step": 0.05,
            "rationale": "prioritize physiologic programs over care_process proxies",
        },
        {
            "knob": "program_topk_care_process_max_count",
            "action": "set",
            "value": 0,
            "rationale": "keep top5 care-process leak strictly capped",
        },
    ],
    "top5_max_same_anchor_count_le_max": [
        {
            "knob": "program_anchor_max_same_count",
            "action": "decrease",
            "step": 1,
            "rationale": "reduce anchor over-concentration in top programs",
        }
    ],
    "top5_unique_anchor_count_ge_min": [
        {
            "knob": "program_anchor_novelty_target",
            "action": "increase",
            "step": 1,
            "rationale": "push top5 to cover more anchor types",
        }
    ],
    "top5_unique_groups_ge_min": [
        {
            "knob": "program_top5_unique_groups_target",
            "action": "increase",
            "step": 1,
            "rationale": "raise top5 diversity target directly",
        }
    ],
    "top10_unique_groups_ge_min": [
        {
            "knob": "program_top_window_unique_groups_target",
            "action": "increase",
            "step": 1,
            "rationale": "expand top10 mechanism diversity pressure",
        }
    ],
    "top5_score_min_ge_threshold": [
        {
            "knob": "program_diversity_quality_floor_ratio",
            "action": "increase",
            "step": 0.02,
            "rationale": "avoid low-quality swaps while enforcing diversity",
        },
        {
            "knob": "shadow_predictive_quality_floor_ratio",
            "action": "increase",
            "step": 0.02,
            "rationale": "bound reranking degradation under shift",
        },
    ],
    "contrastive_rules_count_ge_reference_floor": [
        {
            "knob": "miner_top_candidates",
            "action": "increase",
            "step": 0.2,
            "rationale": "widen candidate frontier to recover rule count",
        }
    ],
    "contrastive_programs_count_ge_reference_floor": [
        {
            "knob": "motif_verify_top_k",
            "action": "increase",
            "step": 0.2,
            "rationale": "retain more valid motifs before program composition",
        }
    ],
}


def _failure_rank_key(name: str) -> int:
    try:
        return _ALGORITHM_READY_FAILURE_PRIORITY.index(name)
    except ValueError:
        return len(_ALGORITHM_READY_FAILURE_PRIORITY)


def _build_algorithm_ready_failure_attribution(
    quality_checks: dict[str, dict],
    reference_checks: dict[str, dict],
    quality_failures: list[str],
    reference_failures: list[str],
    profile: str,
) -> dict:
    merged_failures = list(quality_failures) + list(reference_failures)
    ordered_failures = sorted(merged_failures, key=_failure_rank_key)
    primary_blocker = ordered_failures[0] if ordered_failures else None
    secondary_blockers = ordered_failures[1:] if len(ordered_failures) > 1 else []

    evidence_rows: list[dict] = []
    for name in ordered_failures:
        if name in quality_checks:
            check = dict(quality_checks[name])
            threshold = check.get("threshold")
        else:
            check = dict(reference_checks.get(name, {}))
            threshold = check.get("required_min", check.get("threshold"))
        observed = check.get("observed")
        evidence_rows.append(
            {
                "check": name,
                "observed": observed,
                "threshold": threshold,
                "enabled": bool(check.get("enabled", True)),
                "pass": bool(check.get("pass", False)),
            }
        )

    suggestions: list[dict] = []
    seen_knobs: set[str] = set()
    for failure_name in ordered_failures:
        for suggestion in _ALGORITHM_READY_SUGGESTION_POLICY.get(failure_name, []):
            knob = str(suggestion.get("knob"))
            if knob in seen_knobs:
                continue
            # Locked profile must keep strict constraints; suggestions only tune upstream ranking behavior.
            if profile == "locked" and knob in {"algorithm_ready_top5_care_process_max", "algorithm_ready_top5_unique_groups_min"}:
                continue
            suggestions.append(dict(suggestion))
            seen_knobs.add(knob)

    return {
        "quality_failures": quality_failures,
        "reference_count_failures": reference_failures,
        "primary_blocker": primary_blocker,
        "secondary_blockers": secondary_blockers,
        "evidence": evidence_rows,
        "recommended_next_knobs": suggestions,
    }


def _evaluate_algorithm_ready_gates(
    summary: dict,
    gate_profile: str = "exploratory",
    baseline_rules_count: int = 0,
    baseline_programs_count: int = 0,
    rule_count_min_ratio: float = 0.0,
    program_count_min_ratio: float = 0.0,
    rule_count_min_abs: int = 0,
    program_count_min_abs: int = 0,
    top5_care_process_max: int = 0,
    top5_max_same_anchor_max: int = 2,
    top5_unique_anchor_min: int = 3,
    top5_unique_groups_min: int = 5,
    top10_unique_groups_min: int = 6,
    top5_score_min_threshold: float = 0.82,
) -> dict:
    profile = str(gate_profile or "exploratory").strip().lower()
    if profile not in {"exploratory", "locked"}:
        profile = "exploratory"

    mechanism_diversity = dict(summary.get("contrastive_qc", {}).get("mechanism_diversity", {}))
    observed_top5_care = int(mechanism_diversity.get("top5_care_process_count", 0))
    observed_top5_max_same_anchor = int(mechanism_diversity.get("top5_max_same_anchor_count", 0))
    observed_top5_unique_anchor = int(mechanism_diversity.get("top5_unique_anchor_count", 0))
    observed_top5_unique_groups = int(mechanism_diversity.get("top5_unique_groups", 0))
    observed_top10_unique_groups = int(mechanism_diversity.get("top10_unique_groups", 0))
    observed_top5_score_min = float(mechanism_diversity.get("top5_score_min", 0.0))

    quality_checks = {
        "top5_care_process_count_le_max": {
            "observed": observed_top5_care,
            "threshold": int(top5_care_process_max),
            "pass": observed_top5_care <= int(top5_care_process_max),
        },
        "top5_max_same_anchor_count_le_max": {
            "observed": observed_top5_max_same_anchor,
            "threshold": int(top5_max_same_anchor_max),
            "pass": observed_top5_max_same_anchor <= int(top5_max_same_anchor_max),
        },
        "top5_unique_anchor_count_ge_min": {
            "observed": observed_top5_unique_anchor,
            "threshold": int(top5_unique_anchor_min),
            "pass": observed_top5_unique_anchor >= int(top5_unique_anchor_min),
        },
        "top5_unique_groups_ge_min": {
            "observed": observed_top5_unique_groups,
            "threshold": int(top5_unique_groups_min),
            "pass": observed_top5_unique_groups >= int(top5_unique_groups_min),
        },
        "top10_unique_groups_ge_min": {
            "observed": observed_top10_unique_groups,
            "threshold": int(top10_unique_groups_min),
            "pass": observed_top10_unique_groups >= int(top10_unique_groups_min),
        },
        "top5_score_min_ge_threshold": {
            "observed": observed_top5_score_min,
            "threshold": float(top5_score_min_threshold),
            "pass": observed_top5_score_min >= float(top5_score_min_threshold),
        },
    }
    quality_failures = [name for name, check in quality_checks.items() if not bool(check.get("pass", False))]
    quality_all_pass = not quality_failures

    observed_rule_count = int(len(summary.get("contrastive_rules", [])))
    observed_program_count = int(len(summary.get("contrastive_programs", [])))
    baseline_rules_count = max(0, int(baseline_rules_count))
    baseline_programs_count = max(0, int(baseline_programs_count))
    rule_count_min_abs = max(0, int(rule_count_min_abs))
    program_count_min_abs = max(0, int(program_count_min_abs))
    rule_count_min_ratio = max(0.0, float(rule_count_min_ratio))
    program_count_min_ratio = max(0.0, float(program_count_min_ratio))

    rule_ratio_floor = int(math.ceil(float(baseline_rules_count) * rule_count_min_ratio)) if baseline_rules_count > 0 else 0
    program_ratio_floor = (
        int(math.ceil(float(baseline_programs_count) * program_count_min_ratio))
        if baseline_programs_count > 0
        else 0
    )
    required_rule_count = max(rule_count_min_abs, rule_ratio_floor)
    required_program_count = max(program_count_min_abs, program_ratio_floor)

    rule_reference_enabled = baseline_rules_count > 0 or rule_count_min_abs > 0
    program_reference_enabled = baseline_programs_count > 0 or program_count_min_abs > 0
    rule_reference_pass = observed_rule_count >= required_rule_count if rule_reference_enabled else True
    program_reference_pass = observed_program_count >= required_program_count if program_reference_enabled else True

    reference_checks = {
        "contrastive_rules_count_ge_reference_floor": {
            "enabled": bool(rule_reference_enabled),
            "observed": int(observed_rule_count),
            "baseline": int(baseline_rules_count),
            "min_abs": int(rule_count_min_abs),
            "min_ratio": float(rule_count_min_ratio),
            "required_min": int(required_rule_count),
            "pass": bool(rule_reference_pass),
        },
        "contrastive_programs_count_ge_reference_floor": {
            "enabled": bool(program_reference_enabled),
            "observed": int(observed_program_count),
            "baseline": int(baseline_programs_count),
            "min_abs": int(program_count_min_abs),
            "min_ratio": float(program_count_min_ratio),
            "required_min": int(required_program_count),
            "pass": bool(program_reference_pass),
        },
    }
    reference_enabled = bool(rule_reference_enabled or program_reference_enabled)
    reference_failures = [
        name
        for name, check in reference_checks.items()
        if bool(check.get("enabled", False)) and not bool(check.get("pass", False))
    ]
    reference_all_pass = not reference_failures

    include_reference_in_overall = profile == "locked"
    overall_pass = bool(quality_all_pass and (reference_all_pass if include_reference_in_overall else True))
    failure_attribution = _build_algorithm_ready_failure_attribution(
        quality_checks=quality_checks,
        reference_checks=reference_checks,
        quality_failures=quality_failures,
        reference_failures=reference_failures,
        profile=profile,
    )
    return {
        "profile": profile,
        "quality": {
            "all_pass": bool(quality_all_pass),
            "checks": quality_checks,
            "failed": quality_failures,
        },
        "reference_counts": {
            "enabled": bool(reference_enabled),
            "included_in_overall": bool(include_reference_in_overall),
            "all_pass": bool(reference_all_pass),
            "checks": reference_checks,
            "failed": reference_failures,
        },
        "overall_pass": bool(overall_pass),
        "failure_attribution": failure_attribution,
    }


def _load_cached_contrastive_inputs(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"contrastive inputs cache not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"contrastive inputs cache must be a JSON list: {path}")
    normalized = []
    for entry in payload:
        if not isinstance(entry, dict):
            continue
        normalized.append(
            {
                "episode_id": entry.get("episode_id"),
                "episode_type": entry.get("episode_type"),
                "rules": list(entry.get("rules", [])),
            }
        )
    return normalized


def _load_stability_run_summaries(path_glob: str) -> list[dict]:
    rows: list[dict] = []
    for path_str in sorted(glob.glob(str(path_glob))):
        path = Path(path_str)
        if not path.exists():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            rows.append(payload)
    return rows


def _load_adaptive_history_risk_samples(path_glob: str | None, history_domain: str | None = None) -> list[dict]:
    if not path_glob:
        return []
    samples: list[dict] = []
    expected_domain = _normalize_adaptive_domain(history_domain)
    for path_str in sorted(glob.glob(str(path_glob))):
        path = Path(path_str)
        if not path.exists():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            continue
        filters = dict(payload.get("contrastive_filters", {}))
        domain_value = (
            payload.get("stability_auto_adaptive_domain")
            if payload.get("stability_auto_adaptive_domain") is not None
            else filters.get("stability_auto_adaptive_domain")
        )
        domain_value_normalized = _normalize_adaptive_domain(domain_value)
        if expected_domain and domain_value_normalized != expected_domain:
            continue

        metric_candidates = [
            payload.get("stability_auto_adaptive_metrics"),
            filters.get("stability_auto_adaptive_metrics"),
        ]
        for candidate in metric_candidates:
            if not isinstance(candidate, dict):
                continue
            value = candidate.get("risk_score")
            if value is None:
                continue
            try:
                score = float(value)
            except (TypeError, ValueError):
                continue
            if np.isfinite(score):
                run_count_value = candidate.get("run_count")
                pair_count_value = candidate.get("pair_count")
                try:
                    run_count = max(0, int(run_count_value))
                except (TypeError, ValueError):
                    run_count = 0
                try:
                    pair_count = max(0, int(pair_count_value))
                except (TypeError, ValueError):
                    pair_count = 0
                samples.append(
                    {
                        "risk_score": float(max(0.0, min(1.0, score))),
                        "run_count": int(run_count),
                        "pair_count": int(pair_count),
                    }
                )
                break
    return samples


def _load_adaptive_history_risk_scores(path_glob: str | None, history_domain: str | None = None) -> list[float]:
    samples = _load_adaptive_history_risk_samples(path_glob, history_domain=history_domain)
    return [float(row.get("risk_score", 0.0)) for row in samples]

def _filter_adaptive_history_outliers(scores: list[float], iqr_multiplier: float) -> dict:
    normalized = [max(0.0, min(1.0, float(v))) for v in list(scores or []) if np.isfinite(float(v))]
    raw_count = len(normalized)
    multiplier = max(0.0, float(iqr_multiplier))
    if raw_count < 4 or multiplier <= 0.0:
        kept_indices = [int(idx) for idx in range(raw_count)]
        return {
            "scores": list(normalized),
            "raw_count": int(raw_count),
            "filtered_count": int(raw_count),
            "removed_count": 0,
            "applied": False,
            "lower_bound": None,
            "upper_bound": None,
            "kept_indices": list(kept_indices),
        }

    values = np.asarray(normalized, dtype=float)
    q1 = float(np.quantile(values, 0.25))
    q3 = float(np.quantile(values, 0.75))
    iqr = max(0.0, q3 - q1)
    lower = q1 - (multiplier * iqr)
    upper = q3 + (multiplier * iqr)

    kept_indices = [
        int(idx)
        for idx, value in enumerate(normalized)
        if (value >= (lower - 1e-12) and value <= (upper + 1e-12))
    ]
    filtered = [normalized[idx] for idx in kept_indices]
    if not filtered:
        filtered = list(normalized)
        kept_indices = [int(idx) for idx in range(raw_count)]

    return {
        "scores": list(filtered),
        "raw_count": int(raw_count),
        "filtered_count": int(len(filtered)),
        "removed_count": int(max(0, raw_count - len(filtered))),
        "applied": True,
        "lower_bound": float(lower),
        "upper_bound": float(upper),
        "kept_indices": list(kept_indices),
    }


def _adaptive_history_recency_weights(count: int, half_life: float) -> list[float]:
    n = max(0, int(count))
    if n <= 0:
        return []
    hl = max(0.0, float(half_life))
    if hl <= 0.0:
        return [1.0 for _ in range(n)]
    weights: list[float] = []
    for idx in range(n):
        age = float((n - 1) - idx)
        weights.append(float(0.5 ** (age / hl)))
    return weights


def _adaptive_history_best_drift_split(scores: list[float], min_segment: int = 2) -> dict:
    normalized = [max(0.0, min(1.0, float(v))) for v in list(scores or []) if np.isfinite(float(v))]
    n = len(normalized)
    min_seg = max(1, int(min_segment))
    if n < (2 * min_seg):
        return {
            "applied": False,
            "split_index": 0,
            "old_median": None,
            "recent_median": None,
            "gap": 0.0,
        }

    best_gap = -1.0
    best_split = min_seg
    best_old = None
    best_recent = None
    for split_index in range(min_seg, n - min_seg + 1):
        old_values = np.asarray(normalized[:split_index], dtype=float)
        recent_values = np.asarray(normalized[split_index:], dtype=float)
        old_median = float(np.quantile(old_values, 0.50))
        recent_median = float(np.quantile(recent_values, 0.50))
        gap = abs(float(recent_median) - float(old_median))
        if gap > best_gap:
            best_gap = float(gap)
            best_split = int(split_index)
            best_old = float(old_median)
            best_recent = float(recent_median)

    return {
        "applied": True,
        "split_index": int(best_split),
        "old_median": (float(best_old) if best_old is not None else None),
        "recent_median": (float(best_recent) if best_recent is not None else None),
        "gap": float(max(0.0, best_gap)),
    }


def _weighted_quantile(values: list[float], weights: list[float], quantile: float) -> float:
    if not values:
        return 0.0
    if not weights or len(weights) != len(values):
        return float(np.quantile(np.asarray(values, dtype=float), max(0.0, min(1.0, float(quantile)))))
    q = max(0.0, min(1.0, float(quantile)))
    pairs = sorted(zip(values, weights), key=lambda item: float(item[0]))
    ordered_values = np.asarray([float(value) for value, _ in pairs], dtype=float)
    ordered_weights = np.asarray([max(0.0, float(weight)) for _, weight in pairs], dtype=float)
    if float(ordered_weights.sum()) <= 0.0:
        return float(np.quantile(ordered_values, q))
    cumulative = np.cumsum(ordered_weights)
    target = q * float(cumulative[-1])
    index = int(np.searchsorted(cumulative, target, side="left"))
    index = max(0, min(len(ordered_values) - 1, index))
    return float(ordered_values[index])


def _topk_rule_ids_for_consensus(run_summary: dict, top_k: int) -> set[str]:
    rules = list(run_summary.get("contrastive_rules", []))[: max(0, int(top_k))]
    return {str(row.get("rule_id")) for row in rules if str(row.get("rule_id"))}


def _compute_rule_consensus_frequency(run_summaries: list[dict], top_k: int) -> dict[str, float]:
    normalized = [row for row in list(run_summaries or []) if isinstance(row, dict)]
    run_count = len(normalized)
    if run_count <= 0:
        return {}
    counts: dict[str, int] = {}
    for run in normalized:
        for rule_id in _topk_rule_ids_for_consensus(run, top_k=top_k):
            counts[rule_id] = counts.get(rule_id, 0) + 1
    return {rule_id: float(count) / float(run_count) for rule_id, count in counts.items()}


def _jaccard_ratio(left: set[str], right: set[str]) -> float:
    union = left.union(right)
    if not union:
        return 1.0
    return float(len(left.intersection(right))) / float(len(union))


def _pairwise_jaccard_distribution(values: list[set[str]]) -> list[float]:
    count = len(values)
    if count <= 1:
        return [1.0]
    scores = []
    for i in range(count):
        for j in range(i + 1, count):
            scores.append(_jaccard_ratio(values[i], values[j]))
    return scores or [1.0]


def _summarize_jaccard_distribution(scores: list[float]) -> dict:
    values = np.asarray(list(scores or [1.0]), dtype=float)
    return {
        "count": int(values.size),
        "mean": float(values.mean()),
        "std": float(values.std()),
        "min": float(values.min()),
        "p25": float(np.quantile(values, 0.25)),
        "p50": float(np.quantile(values, 0.50)),
        "p75": float(np.quantile(values, 0.75)),
        "max": float(values.max()),
    }


def _topk_anchor_features_for_adaptive(run_summary: dict, top_k: int) -> set[str]:
    rules = list(run_summary.get("contrastive_rules", []))[: max(0, int(top_k))]
    anchors: set[str] = set()
    for rule in rules:
        for feature in list(rule.get("antecedent_features", [])):
            base = str(feature).split(":", 1)[0]
            if base:
                anchors.add(base)
    return anchors


def _topk_mechanism_groups_for_adaptive(run_summary: dict, top_k: int) -> set[str]:
    rules = list(run_summary.get("contrastive_rules", []))[: max(0, int(top_k))]
    groups: set[str] = set()
    for rule in rules:
        for feature in list(rule.get("consequent_features", [])):
            base = str(feature).split(":", 1)[0]
            if not base:
                continue
            groups.add(MECHANISM_GROUP_BY_FEATURE.get(base, f"other:{base}"))
    return groups


def _compute_adaptive_auto_profile_metrics(run_summaries: list[dict], top_k: int) -> dict:
    normalized = [row for row in list(run_summaries or []) if isinstance(row, dict)]
    k = max(1, int(top_k))

    rule_sets = [_topk_rule_ids_for_consensus(row, k) for row in normalized]
    anchor_sets = [_topk_anchor_features_for_adaptive(row, k) for row in normalized]
    mechanism_sets = [_topk_mechanism_groups_for_adaptive(row, k) for row in normalized]

    rule_summary = _summarize_jaccard_distribution(_pairwise_jaccard_distribution(rule_sets))
    anchor_summary = _summarize_jaccard_distribution(_pairwise_jaccard_distribution(anchor_sets))
    mechanism_summary = _summarize_jaccard_distribution(_pairwise_jaccard_distribution(mechanism_sets))

    weakest_mean = min(
        float(rule_summary.get("mean", 1.0)),
        float(anchor_summary.get("mean", 1.0)),
        float(mechanism_summary.get("mean", 1.0)),
    )
    weakest_p25 = min(
        float(rule_summary.get("p25", 1.0)),
        float(anchor_summary.get("p25", 1.0)),
        float(mechanism_summary.get("p25", 1.0)),
    )
    max_std = max(
        float(rule_summary.get("std", 0.0)),
        float(anchor_summary.get("std", 0.0)),
        float(mechanism_summary.get("std", 0.0)),
    )

    risk_score = max(
        0.0,
        min(
            1.0,
            (0.55 * (1.0 - weakest_mean))
            + (0.35 * (1.0 - weakest_p25))
            + (0.10 * max_std),
        ),
    )
    run_count = len(normalized)
    confidence = max(0.0, min(1.0, float(run_count - 2) / 6.0))
    robust_threshold = max(0.0, min(1.0, 0.50 - (0.06 * confidence)))
    balanced_threshold = max(0.0, min(1.0, 0.20 - (0.05 * confidence)))

    pair_count = 0 if run_count <= 1 else int(run_count * (run_count - 1) / 2)
    return {
        "run_count": int(run_count),
        "pair_count": int(pair_count),
        "top_k": int(k),
        "rule_jaccard": rule_summary,
        "anchor_jaccard": anchor_summary,
        "mechanism_jaccard": mechanism_summary,
        "risk_score": float(risk_score),
        "robust_threshold": float(robust_threshold),
        "balanced_threshold": float(balanced_threshold),
        "confidence": float(confidence),
    }


def _apply_rule_consensus_boost(
    ranked_rules: list[dict],
    consensus_frequency: dict[str, float],
    boost_weight: float,
    rerank_epsilon: float = 0.0,
    quality_floor_ratio: float = 0.0,
) -> list[dict]:
    if not ranked_rules:
        return []
    weight = max(0.0, min(1.0, float(boost_weight)))
    if weight <= 0.0 or not consensus_frequency:
        return list(ranked_rules)

    top_score = max(float(row.get("contrastive_score", 0.0)) for row in ranked_rules)
    epsilon = max(0.0, float(rerank_epsilon))
    quality_ratio = max(0.0, min(1.0, float(quality_floor_ratio)))
    quality_floor = top_score * quality_ratio

    boosted = []
    for row in ranked_rules:
        next_row = dict(row)
        rule_id = str(next_row.get("rule_id"))
        consensus_support = float(consensus_frequency.get(rule_id, 0.0))
        base_score = float(next_row.get("contrastive_score", 0.0))

        within_quality_floor = base_score >= quality_floor
        within_epsilon = (top_score - base_score) <= (epsilon + 1e-12) if epsilon > 0.0 else True
        consensus_boost_eligible = bool(within_quality_floor and within_epsilon)

        adjusted_score = base_score + (weight * consensus_support) if consensus_boost_eligible else base_score
        next_row["consensus_support"] = consensus_support
        next_row["consensus_boost_eligible"] = consensus_boost_eligible
        next_row["consensus_adjusted_score"] = adjusted_score
        boosted.append(next_row)
    boosted.sort(
        key=lambda row: (
            float(row.get("consensus_adjusted_score", row.get("contrastive_score", 0.0))),
            float(row.get("contrastive_score", 0.0)),
            float(row.get("enrichment", 0.0)),
            str(row.get("rule_id", "")),
        ),
        reverse=True,
    )
    return boosted


def _rule_anchor_signature_from_rule(rule_row: dict) -> tuple[str, ...]:
    anchor = tuple(sorted(_normalized_feature_names(list(rule_row.get("antecedent_features", [])))))
    return anchor if anchor else ("unknown",)


def _rule_mechanism_groups_from_rule(rule_row: dict) -> tuple[str, ...]:
    groups = set()
    for feature in list(rule_row.get("consequent_features", [])):
        base = str(feature).split(":", 1)[0]
        if not base:
            continue
        groups.add(MECHANISM_GROUP_BY_FEATURE.get(base, f"other:{base}"))
    if not groups:
        return ("unknown",)
    return tuple(sorted(groups))


def _recover_rule_candidates_for_low_mechanism_coverage(
    contrastive_inputs: list[dict],
    ranked_rules: list[dict],
    candidate_pool_diagnostics: dict | None,
    scoring_kwargs: dict,
    recovery_enabled: bool,
    trigger_top_k: int,
    min_unique_groups: int,
    recovery_min_control_count: int,
    recovery_allow_zero_control_candidates: bool,
    recovery_zero_control_penalty: float,
    recovery_support_reliability_weight: float,
    recovery_support_reliability_exponent: float,
    recovery_significance_max_q_value: float,
) -> tuple[list[dict], dict, dict]:
    base_rules = list(ranked_rules or [])
    base_pool = dict(candidate_pool_diagnostics or {})
    top_k = max(1, int(trigger_top_k))
    target_unique = max(0, int(min_unique_groups))

    def _topk_group_count(rows: list[dict], k: int) -> int:
        covered: set[str] = set()
        for row in rows[:k]:
            covered.update(set(_rule_mechanism_groups_from_rule(row)))
        return int(len(covered))

    before_count = _topk_group_count(base_rules, top_k)
    meta = {
        "enabled": bool(recovery_enabled),
        "trigger_top_k": int(top_k),
        "min_unique_groups": int(target_unique),
        "triggered": bool(False),
        "applied": bool(False),
        "reason": "disabled",
        "covered_groups_before_count": int(before_count),
        "covered_groups_after_count": int(before_count),
    }

    if not recovery_enabled:
        return base_rules, base_pool, meta
    if target_unique <= 0:
        meta["reason"] = "target_disabled"
        return base_rules, base_pool, meta
    if before_count >= target_unique:
        meta["reason"] = "already_sufficient_coverage"
        return base_rules, base_pool, meta
    if not contrastive_inputs:
        meta["reason"] = "empty_contrastive_inputs"
        return base_rules, base_pool, meta

    meta["triggered"] = bool(True)
    recovery_kwargs = dict(scoring_kwargs or {})
    current_min_control = int(recovery_kwargs.get("min_control_count", 0))
    current_allow_zero = bool(recovery_kwargs.get("allow_zero_control_candidates", False))
    current_zero_control_penalty = float(recovery_kwargs.get("zero_control_rule_penalty", 1.0))
    current_support_weight = float(recovery_kwargs.get("rule_support_reliability_weight", 0.0))
    current_support_exponent = float(recovery_kwargs.get("rule_support_reliability_exponent", 0.5))
    current_significance_max_q = float(recovery_kwargs.get("significance_max_q_value", 1.0))

    recovery_kwargs["min_control_count"] = int(min(current_min_control, int(recovery_min_control_count)))
    recovery_kwargs["allow_zero_control_candidates"] = bool(
        current_allow_zero or bool(recovery_allow_zero_control_candidates)
    )
    recovery_kwargs["zero_control_rule_penalty"] = float(
        min(current_zero_control_penalty, max(0.0, min(1.0, float(recovery_zero_control_penalty))))
    )
    recovery_kwargs["rule_support_reliability_weight"] = float(
        max(current_support_weight, max(0.0, min(1.0, float(recovery_support_reliability_weight))))
    )
    recovery_kwargs["rule_support_reliability_exponent"] = float(
        max(current_support_exponent, max(1e-6, float(recovery_support_reliability_exponent)))
    )
    recovery_kwargs["significance_max_q_value"] = float(
        max(
            current_significance_max_q,
            max(0.0, min(1.0, float(recovery_significance_max_q_value))),
        )
    )
    recovery_kwargs["return_candidate_pool_diagnostics"] = True

    rescored = score_contrastive_rule_records(contrastive_inputs, **recovery_kwargs)
    if isinstance(rescored, tuple):
        recovered_rules, recovered_pool = rescored
    else:
        recovered_rules = list(rescored)
        recovered_pool = {}
    after_count = _topk_group_count(recovered_rules, top_k)

    meta.update(
        {
            "recovery_min_control_count_applied": int(recovery_kwargs.get("min_control_count", 0)),
            "recovery_allow_zero_control_candidates_applied": bool(
                recovery_kwargs.get("allow_zero_control_candidates", False)
            ),
            "recovery_zero_control_penalty_applied": float(
                recovery_kwargs.get("zero_control_rule_penalty", 1.0)
            ),
            "recovery_support_reliability_weight_applied": float(
                recovery_kwargs.get("rule_support_reliability_weight", 0.0)
            ),
            "recovery_support_reliability_exponent_applied": float(
                recovery_kwargs.get("rule_support_reliability_exponent", 0.5)
            ),
            "recovery_significance_max_q_value_applied": float(
                recovery_kwargs.get("significance_max_q_value", 1.0)
            ),
            "covered_groups_after_count": int(after_count),
        }
    )

    if after_count > before_count:
        meta["applied"] = bool(True)
        meta["reason"] = "applied_improved_coverage"
        return list(recovered_rules), dict(recovered_pool or {}), meta

    meta["reason"] = "rejected_no_coverage_gain"
    return base_rules, base_pool, meta


def _apply_rule_anchor_diversity_rerank(
    ranked_rules: list[dict],
    diversity_weight: float,
    top_k: int,
    rerank_epsilon: float = 0.0,
    quality_floor_ratio: float = 0.0,
) -> list[dict]:
    if not ranked_rules:
        return []
    weight = max(0.0, float(diversity_weight))
    if weight <= 0.0:
        return list(ranked_rules)

    top_limit = max(1, int(top_k))
    top_score = max(float(row.get("contrastive_score", 0.0)) for row in ranked_rules)
    epsilon = max(0.0, float(rerank_epsilon))
    quality_ratio = max(0.0, min(1.0, float(quality_floor_ratio)))
    quality_floor = top_score * quality_ratio

    anchor_counts: dict[tuple[str, ...], int] = {}
    reranked: list[dict] = []
    remaining = [dict(row) for row in ranked_rules]
    greedy_window = min(len(remaining), top_limit)

    for idx in range(greedy_window):
        best_idx = 0
        best_key = None
        best_payload = None

        for candidate_idx, candidate in enumerate(remaining):
            base_score = float(candidate.get("contrastive_score", 0.0))
            anchor = _rule_anchor_signature_from_rule(candidate)
            current_anchor_count = int(anchor_counts.get(anchor, 0))

            within_top_window = idx < top_limit
            within_quality_floor = base_score >= quality_floor
            within_epsilon = (top_score - base_score) <= (epsilon + 1e-12) if epsilon > 0.0 else True
            eligible = bool(within_top_window and within_quality_floor and within_epsilon)

            novelty_bonus = 0.0
            if eligible:
                novelty_bonus = max(0.0, 1.0 - (float(current_anchor_count) / float(top_limit)))
            adjusted_score = base_score + (weight * novelty_bonus)

            sort_key = (
                float(adjusted_score),
                float(base_score),
                float(candidate.get("enrichment", 0.0)),
                str(candidate.get("rule_id", "")),
            )
            if best_key is None or sort_key > best_key:
                best_key = sort_key
                best_idx = candidate_idx
                best_payload = {
                    "anchor": anchor,
                    "count_before": current_anchor_count,
                    "eligible": eligible,
                    "bonus": novelty_bonus,
                    "adjusted_score": adjusted_score,
                }

        selected = remaining.pop(best_idx)
        anchor = best_payload["anchor"] if best_payload is not None else _rule_anchor_signature_from_rule(selected)
        selected["anchor_signature"] = list(anchor)
        selected["anchor_diversity_count_before"] = int(best_payload["count_before"] if best_payload else 0)
        selected["anchor_diversity_boost_eligible"] = bool(best_payload["eligible"] if best_payload else False)
        selected["anchor_diversity_bonus"] = float(best_payload["bonus"] if best_payload else 0.0)
        selected["anchor_diversity_adjusted_score"] = float(
            best_payload["adjusted_score"] if best_payload else float(selected.get("contrastive_score", 0.0))
        )

        anchor_counts[anchor] = int(anchor_counts.get(anchor, 0)) + 1
        reranked.append(selected)

    if remaining:
        tail = sorted(
            remaining,
            key=lambda row: (
                float(row.get("contrastive_score", 0.0)),
                float(row.get("enrichment", 0.0)),
                str(row.get("rule_id", "")),
            ),
            reverse=True,
        )
        for selected in tail:
            anchor = _rule_anchor_signature_from_rule(selected)
            count_before = int(anchor_counts.get(anchor, 0))
            selected["anchor_signature"] = list(anchor)
            selected["anchor_diversity_count_before"] = int(count_before)
            selected["anchor_diversity_boost_eligible"] = False
            selected["anchor_diversity_bonus"] = 0.0
            selected["anchor_diversity_adjusted_score"] = float(selected.get("contrastive_score", 0.0))
            anchor_counts[anchor] = count_before + 1
            reranked.append(selected)

    return reranked


def _apply_rule_mechanism_diversity_rerank(
    ranked_rules: list[dict],
    diversity_weight: float,
    top_k: int,
    rerank_epsilon: float = 0.0,
    quality_floor_ratio: float = 0.0,
    coverage_target: int = 0,
    two_stage_swap_budget: int = 0,
    stage1_min_support_ratio: float = 0.0,
    return_metadata: bool = False,
) -> list[dict] | tuple[list[dict], dict]:
    if not ranked_rules:
        empty_meta = {
            "top_k": max(1, int(top_k)),
            "coverage_target": max(0, int(coverage_target)),
            "covered_groups_before": [],
            "covered_groups_after": [],
            "covered_groups_before_count": 0,
            "covered_groups_after_count": 0,
            "coverage_target_met": bool(max(0, int(coverage_target)) == 0),
            "topk_set_swapped_count": 0,
            "topk_order_changed_positions": 0,
            "stage1_applied": False,
            "stage1_swap_count": 0,
            "stage2_applied": False,
        }
        if return_metadata:
            return [], empty_meta
        return []
    weight = max(0.0, float(diversity_weight))
    top_limit = max(1, int(top_k))
    target = max(0, int(coverage_target))

    def _topk_groups(rows: list[dict], k: int) -> set[str]:
        groups: set[str] = set()
        for row in rows[:k]:
            groups.update(set(_rule_mechanism_groups_from_rule(row)))
        return groups

    original_top_rule_ids = [str(row.get("rule_id", "")) for row in ranked_rules[:top_limit]]
    covered_before = _topk_groups(ranked_rules, top_limit)

    top_score = max(float(row.get("contrastive_score", 0.0)) for row in ranked_rules)
    epsilon = max(0.0, float(rerank_epsilon))
    quality_ratio = max(0.0, min(1.0, float(quality_floor_ratio)))
    quality_floor = top_score * quality_ratio
    stage1_budget = max(0, int(two_stage_swap_budget))
    stage1_support_floor = max(0.0, min(1.0, float(stage1_min_support_ratio)))

    working = [dict(row) for row in ranked_rules]

    def _eligible_for_stage1(row: dict) -> bool:
        base_score = float(row.get("contrastive_score", 0.0))
        within_quality_floor = base_score >= quality_floor
        within_epsilon = (top_score - base_score) <= (epsilon + 1e-12) if epsilon > 0.0 else True
        support_ratio = float(row.get("support_ratio", 0.0))
        return bool(within_quality_floor and within_epsilon and support_ratio >= stage1_support_floor)

    stage1_swap_count = 0
    if stage1_budget > 0 and len(working) > top_limit:
        stage1_selected = list(working[:top_limit])
        stage1_tail = list(working[top_limit:])
        for _ in range(stage1_budget):
            covered_before_stage1 = _topk_groups(stage1_selected, top_limit)
            if target > 0 and len(covered_before_stage1) >= target:
                break

            best_in_idx = None
            best_in_key = None
            for idx, candidate in enumerate(stage1_tail):
                if not _eligible_for_stage1(candidate):
                    continue
                groups = set(_rule_mechanism_groups_from_rule(candidate))
                novelty = len(groups.difference(covered_before_stage1))
                if novelty <= 0:
                    continue
                key = (
                    int(novelty),
                    float(candidate.get("contrastive_score", 0.0)),
                    float(candidate.get("enrichment", 0.0)),
                    -int(idx),
                )
                if best_in_key is None or key > best_in_key:
                    best_in_key = key
                    best_in_idx = idx

            if best_in_idx is None:
                break

            best_out_idx = None
            best_out_key = None
            best_gain = 0
            for idx, current in enumerate(stage1_selected):
                current_groups = set(_rule_mechanism_groups_from_rule(current))
                trial = list(stage1_selected)
                trial[idx] = stage1_tail[best_in_idx]
                covered_after_trial = _topk_groups(trial, top_limit)
                gain = int(len(covered_after_trial) - len(covered_before_stage1))
                if gain < 0:
                    continue
                current_base = float(current.get("contrastive_score", 0.0))
                in_base = float(stage1_tail[best_in_idx].get("contrastive_score", 0.0))
                score_drop = max(0.0, current_base - in_base)
                key = (
                    int(gain),
                    -float(score_drop),
                    -float(current_base),
                    -int(len(current_groups)),
                    -int(idx),
                )
                if best_out_key is None or key > best_out_key:
                    best_out_key = key
                    best_out_idx = idx
                    best_gain = gain

            if best_out_idx is None or best_gain <= 0:
                break

            outgoing = stage1_selected[best_out_idx]
            incoming = stage1_tail[best_in_idx]
            stage1_selected[best_out_idx] = incoming
            stage1_tail[best_in_idx] = outgoing
            stage1_swap_count += 1

        working = stage1_selected + stage1_tail

    if weight <= 0.0:
        reranked = list(working)
        reranked_top_rule_ids = [str(row.get("rule_id", "")) for row in reranked[:top_limit]]
        covered_after = _topk_groups(reranked, top_limit)
        swapped_count = sum(1 for rid in reranked_top_rule_ids if rid not in set(original_top_rule_ids))
        order_changed_positions = sum(
            1 for before_id, after_id in zip(original_top_rule_ids, reranked_top_rule_ids) if before_id != after_id
        )
        metadata = {
            "top_k": int(top_limit),
            "coverage_target": int(target),
            "covered_groups_before": sorted(covered_before),
            "covered_groups_after": sorted(covered_after),
            "covered_groups_before_count": int(len(covered_before)),
            "covered_groups_after_count": int(len(covered_after)),
            "coverage_target_met": bool(target <= 0 or len(covered_after) >= target),
            "topk_set_swapped_count": int(swapped_count),
            "topk_order_changed_positions": int(order_changed_positions),
            "stage1_applied": bool(stage1_budget > 0),
            "stage1_swap_count": int(stage1_swap_count),
            "stage2_applied": bool(False),
        }
        if return_metadata:
            return reranked, metadata
        return reranked

    covered_groups: set[str] = set()
    reranked: list[dict] = []
    remaining = [dict(row) for row in working]

    for idx in range(len(remaining)):
        best_idx = 0
        best_key = None
        best_payload = None
        for candidate_idx, candidate in enumerate(remaining):
            base_score = float(candidate.get("contrastive_score", 0.0))
            groups = set(_rule_mechanism_groups_from_rule(candidate))
            newly_covered = groups.difference(covered_groups)
            newly_covered_count = int(len(newly_covered))

            within_top_window = idx < top_limit
            within_quality_floor = base_score >= quality_floor
            within_epsilon = (top_score - base_score) <= (epsilon + 1e-12) if epsilon > 0.0 else True
            eligible = bool(within_top_window and within_quality_floor and within_epsilon)

            novelty_bonus = 0.0
            if eligible and newly_covered_count > 0:
                if target > 0:
                    remaining_target = max(1, target - len(covered_groups))
                    novelty_bonus = max(0.0, min(1.0, float(newly_covered_count) / float(remaining_target)))
                else:
                    novelty_bonus = max(0.0, min(1.0, float(newly_covered_count) / max(1.0, float(len(groups)))))
            adjusted_score = base_score + (weight * novelty_bonus)
            sort_key = (
                float(adjusted_score),
                float(base_score),
                float(candidate.get("enrichment", 0.0)),
                str(candidate.get("rule_id", "")),
            )
            if best_key is None or sort_key > best_key:
                best_key = sort_key
                best_idx = candidate_idx
                best_payload = {
                    "groups": tuple(sorted(groups)),
                    "newly_covered_count": int(newly_covered_count),
                    "eligible": bool(eligible),
                    "bonus": float(novelty_bonus),
                    "adjusted_score": float(adjusted_score),
                }

        selected = remaining.pop(best_idx)
        selected_groups = tuple(best_payload["groups"]) if best_payload is not None else _rule_mechanism_groups_from_rule(selected)
        selected["mechanism_signature"] = list(selected_groups)
        selected["mechanism_diversity_new_group_count"] = int(best_payload["newly_covered_count"] if best_payload else 0)
        selected["mechanism_diversity_boost_eligible"] = bool(best_payload["eligible"] if best_payload else False)
        selected["mechanism_diversity_bonus"] = float(best_payload["bonus"] if best_payload else 0.0)
        selected["mechanism_diversity_adjusted_score"] = float(
            best_payload["adjusted_score"] if best_payload else float(selected.get("contrastive_score", 0.0))
        )
        covered_groups.update(set(selected_groups))
        reranked.append(selected)

    reranked_top_rule_ids = [str(row.get("rule_id", "")) for row in reranked[:top_limit]]
    covered_after = _topk_groups(reranked, top_limit)
    swapped_count = sum(1 for rid in reranked_top_rule_ids if rid not in set(original_top_rule_ids))
    order_changed_positions = sum(
        1 for before_id, after_id in zip(original_top_rule_ids, reranked_top_rule_ids) if before_id != after_id
    )
    metadata = {
        "top_k": int(top_limit),
        "coverage_target": int(target),
        "covered_groups_before": sorted(covered_before),
        "covered_groups_after": sorted(covered_after),
        "covered_groups_before_count": int(len(covered_before)),
        "covered_groups_after_count": int(len(covered_after)),
        "coverage_target_met": bool(target <= 0 or len(covered_after) >= target),
        "topk_set_swapped_count": int(swapped_count),
        "topk_order_changed_positions": int(order_changed_positions),
        "stage1_applied": bool(stage1_budget > 0),
        "stage1_swap_count": int(stage1_swap_count),
        "stage2_applied": bool(weight > 0.0),
    }
    if return_metadata:
        return reranked, metadata
    return reranked


def _evaluate_rule_mechanism_diversity_confidence_gate(
    ranked_rules: list[dict],
    top_k: int,
    rerank_epsilon: float,
    quality_floor_ratio: float,
    coverage_target: int,
    min_reliable_support_ratio: float,
    min_reliable_rules: int,
    min_expected_coverage_gain: int,
) -> dict:
    top_limit = max(1, int(top_k))
    target = max(0, int(coverage_target))
    min_support = max(0.0, min(1.0, float(min_reliable_support_ratio)))
    min_reliable = max(1, int(min_reliable_rules))
    min_gain = max(1, int(min_expected_coverage_gain))
    epsilon = max(0.0, float(rerank_epsilon))

    if not ranked_rules:
        return {
            "enabled": False,
            "reason": "empty_rule_list",
            "top_k": int(top_limit),
            "coverage_target": int(target),
            "quality_floor_ratio": float(max(0.0, min(1.0, float(quality_floor_ratio)))),
            "rerank_epsilon": float(epsilon),
            "eligible_pool_size": 0,
            "reliable_candidate_count": 0,
            "expected_coverage_gain": 0,
            "min_reliable_support_ratio": float(min_support),
            "min_reliable_rules": int(min_reliable),
            "min_expected_coverage_gain": int(min_gain),
            "covered_groups_before_count": 0,
            "max_possible_groups_after_count": 0,
            "score_gap_max_in_pool": None,
            "confidence_score": 0.0,
        }

    top_score = max(float(row.get("contrastive_score", 0.0)) for row in ranked_rules)
    quality_ratio = max(0.0, min(1.0, float(quality_floor_ratio)))
    quality_floor = float(top_score * quality_ratio)
    covered_before: set[str] = set()
    for row in ranked_rules[:top_limit]:
        covered_before.update(set(_rule_mechanism_groups_from_rule(row)))

    eligible_pool: list[dict] = []
    for row in ranked_rules:
        base_score = float(row.get("contrastive_score", 0.0))
        within_quality_floor = bool(base_score >= quality_floor)
        within_epsilon = bool((top_score - base_score) <= (epsilon + 1e-12)) if epsilon > 0.0 else True
        if within_quality_floor and within_epsilon:
            eligible_pool.append(row)

    reliable_candidates: list[dict] = []
    max_possible_groups = set(covered_before)
    for row in eligible_pool:
        support_ratio = float(row.get("support_ratio", 0.0))
        if support_ratio >= min_support:
            reliable_candidates.append(row)
            max_possible_groups.update(set(_rule_mechanism_groups_from_rule(row)))

    expected_gain = max(0, int(len(max_possible_groups) - len(covered_before)))
    target_gap = max(0, int(target - len(covered_before))) if target > 0 else 0
    required_gain = max(min_gain, target_gap)

    reasons: list[str] = []
    if len(reliable_candidates) < min_reliable:
        reasons.append("insufficient_reliable_support")
    if expected_gain < required_gain:
        reasons.append("insufficient_expected_coverage_gain")
    if len(eligible_pool) <= top_limit:
        reasons.append("no_effective_swap_capacity")

    support_confidence = min(1.0, float(len(reliable_candidates)) / float(max(1, min_reliable)))
    gain_confidence = min(1.0, float(expected_gain) / float(max(1, required_gain)))
    confidence_score = float(max(0.0, min(1.0, 0.5 * support_confidence + 0.5 * gain_confidence)))

    score_gap_max = None
    if eligible_pool:
        score_gap_max = max(
            float(top_score - float(row.get("contrastive_score", 0.0)))
            for row in eligible_pool
        )

    return {
        "enabled": bool(len(reasons) == 0),
        "reason": "ok" if not reasons else "+".join(reasons),
        "top_k": int(top_limit),
        "coverage_target": int(target),
        "quality_floor_ratio": float(quality_ratio),
        "rerank_epsilon": float(epsilon),
        "eligible_pool_size": int(len(eligible_pool)),
        "reliable_candidate_count": int(len(reliable_candidates)),
        "expected_coverage_gain": int(expected_gain),
        "required_coverage_gain": int(required_gain),
        "min_reliable_support_ratio": float(min_support),
        "min_reliable_rules": int(min_reliable),
        "min_expected_coverage_gain": int(min_gain),
        "covered_groups_before_count": int(len(covered_before)),
        "max_possible_groups_after_count": int(len(max_possible_groups)),
        "score_gap_max_in_pool": None if score_gap_max is None else float(score_gap_max),
        "confidence_score": float(confidence_score),
    }


def _build_stability_tuning_profiles(stability_metrics: dict, current_filters: dict | None = None) -> dict:
    filters = dict(current_filters or {})

    rule_j = float(stability_metrics.get("topk_rule_jaccard_mean", 1.0))
    anchor_j = float(stability_metrics.get("topk_anchor_jaccard_mean", 1.0))
    mechanism_j = float(stability_metrics.get("topk_mechanism_jaccard_mean", 1.0))

    current_support_weight = float(filters.get("rule_support_reliability_weight", 0.0))
    current_support_exponent = float(filters.get("rule_support_reliability_exponent", 0.5))
    current_consensus_weight = float(filters.get("rule_consensus_boost_weight", 0.0))
    current_consensus_top_k = int(filters.get("rule_consensus_top_k", 10))
    current_consensus_epsilon = float(filters.get("rule_consensus_rerank_epsilon", 0.0))
    current_consensus_floor = float(filters.get("rule_consensus_quality_floor_ratio", 0.0))
    current_anchor_diversity = float(filters.get("program_anchor_diversity_weight", 0.0))
    current_mechanism_diversity = float(filters.get("program_mechanism_diversity_weight", 0.0))
    current_anchor_cap = int(filters.get("program_anchor_max_same_count", 0))
    current_mechanism_cap = int(filters.get("program_mechanism_max_same_count", 0))

    robust_filters = {
        "rule_support_reliability_weight": round(max(current_support_weight, 0.5 if rule_j < 0.75 else 0.35), 3),
        "rule_support_reliability_exponent": round(max(current_support_exponent, 1.0), 3),
        "rule_consensus_boost_weight": round(max(current_consensus_weight, 0.25 if rule_j < 0.75 else 0.15), 3),
        "rule_consensus_top_k": max(current_consensus_top_k, 10),
        "rule_consensus_rerank_epsilon": round(max(current_consensus_epsilon, 0.03), 3),
        "rule_consensus_quality_floor_ratio": round(max(current_consensus_floor, 0.95), 3),
        "program_anchor_max_same_count": 1 if (current_anchor_cap <= 0 or current_anchor_cap > 1) else current_anchor_cap,
        "program_anchor_diversity_weight": round(max(current_anchor_diversity, 0.3 if anchor_j < 0.8 else 0.2), 3),
        "program_mechanism_max_same_count": 1 if (current_mechanism_cap <= 0 or current_mechanism_cap > 1) else current_mechanism_cap,
        "program_mechanism_diversity_weight": round(max(current_mechanism_diversity, 0.35 if mechanism_j < 0.8 else 0.3), 3),
    }

    balanced_filters = {
        "rule_support_reliability_weight": round(max(current_support_weight, 0.3 if rule_j < 0.75 else 0.2), 3),
        "rule_support_reliability_exponent": round(max(current_support_exponent, 0.8), 3),
        "rule_consensus_boost_weight": round(max(current_consensus_weight, 0.2 if rule_j < 0.75 else 0.1), 3),
        "rule_consensus_top_k": max(current_consensus_top_k, 10),
        "rule_consensus_rerank_epsilon": round(max(current_consensus_epsilon, 0.06), 3),
        "rule_consensus_quality_floor_ratio": round(max(current_consensus_floor, 0.9), 3),
        "program_anchor_max_same_count": 1 if (current_anchor_cap <= 0 or current_anchor_cap > 1) else current_anchor_cap,
        "program_anchor_diversity_weight": round(max(current_anchor_diversity, 0.2), 3),
        "program_mechanism_max_same_count": 1 if (current_mechanism_cap <= 0 or current_mechanism_cap > 1) else current_mechanism_cap,
        "program_mechanism_diversity_weight": round(max(current_mechanism_diversity, 0.3), 3),
    }

    aggressive_filters = {
        "rule_support_reliability_weight": round(max(current_support_weight, 0.15), 3),
        "rule_support_reliability_exponent": round(max(current_support_exponent, 0.6), 3),
        "rule_consensus_boost_weight": round(max(current_consensus_weight, 0.35), 3),
        "rule_consensus_top_k": max(current_consensus_top_k, 20),
        "rule_consensus_rerank_epsilon": round(max(current_consensus_epsilon, 0.12), 3),
        "rule_consensus_quality_floor_ratio": round(max(current_consensus_floor, 0.75), 3),
        "program_anchor_max_same_count": 2 if current_anchor_cap <= 0 else min(current_anchor_cap, 2),
        "program_anchor_diversity_weight": round(max(current_anchor_diversity, 0.15), 3),
        "program_mechanism_max_same_count": 2 if current_mechanism_cap <= 0 else min(current_mechanism_cap, 2),
        "program_mechanism_diversity_weight": round(max(current_mechanism_diversity, 0.2), 3),
    }

    return {
        "robust": {
            "objective": "maximize_stability",
            "recommended_filters": robust_filters,
        },
        "balanced": {
            "objective": "tradeoff_stability_and_novelty",
            "recommended_filters": balanced_filters,
        },
        "aggressive": {
            "objective": "favor_discovery_with_guardrails",
            "recommended_filters": aggressive_filters,
        },
    }


def _suggest_stability_tuning(stability_metrics: dict, current_filters: dict | None = None) -> dict:
    filters = dict(current_filters or {})
    suggestions: dict[str, float | int] = {}
    rationale: list[str] = []

    rule_j = float(stability_metrics.get("topk_rule_jaccard_mean", 1.0))
    anchor_j = float(stability_metrics.get("topk_anchor_jaccard_mean", 1.0))
    mechanism_j = float(stability_metrics.get("topk_mechanism_jaccard_mean", 1.0))

    if rule_j < 0.6:
        current_weight = float(filters.get("rule_support_reliability_weight", 0.0))
        current_exp = float(filters.get("rule_support_reliability_exponent", 0.5))
        current_consensus_weight = float(filters.get("rule_consensus_boost_weight", 0.0))
        current_consensus_epsilon = float(filters.get("rule_consensus_rerank_epsilon", 0.0))
        current_consensus_floor = float(filters.get("rule_consensus_quality_floor_ratio", 0.0))

        suggestions["rule_support_reliability_weight"] = round(max(current_weight, min(1.0, current_weight + 0.2)), 3)
        suggestions["rule_support_reliability_exponent"] = round(max(current_exp, min(2.0, current_exp + 0.2)), 3)
        suggestions["rule_consensus_boost_weight"] = round(max(current_consensus_weight, 0.2), 3)
        suggestions["rule_consensus_rerank_epsilon"] = round(max(current_consensus_epsilon, 0.05), 3)
        suggestions["rule_consensus_quality_floor_ratio"] = round(max(current_consensus_floor, 0.9), 3)
        rationale.append("Top-K rule overlap is low; strengthen reliability and consensus safeguards.")

    if anchor_j < 0.7:
        current_anchor_cap = int(filters.get("program_anchor_max_same_count", 0))
        if current_anchor_cap <= 0 or current_anchor_cap > 1:
            suggestions["program_anchor_max_same_count"] = 1
        current_anchor_diversity = float(filters.get("program_anchor_diversity_weight", 0.0))
        suggestions["program_anchor_diversity_weight"] = round(max(current_anchor_diversity, 0.2), 3)
        rationale.append("Anchor composition is unstable; tighten anchor caps and diversity pressure.")

    if mechanism_j < 0.7:
        current_mechanism_cap = int(filters.get("program_mechanism_max_same_count", 0))
        if current_mechanism_cap <= 0 or current_mechanism_cap > 1:
            suggestions["program_mechanism_max_same_count"] = 1
        current_mechanism_diversity = float(filters.get("program_mechanism_diversity_weight", 0.0))
        suggestions["program_mechanism_diversity_weight"] = round(max(current_mechanism_diversity, 0.3), 3)
        rationale.append("Mechanism composition is unstable; tighten mechanism caps and diversity pressure.")

    profiles = _build_stability_tuning_profiles(stability_metrics=stability_metrics, current_filters=filters)

    status = "actionable" if suggestions else "stable"
    if not rationale:
        rationale.append("Current stability metrics look healthy; keep current settings.")

    return {
        "status": status,
        "observed_metrics": {
            "topk_rule_jaccard_mean": rule_j,
            "topk_anchor_jaccard_mean": anchor_j,
            "topk_mechanism_jaccard_mean": mechanism_j,
        },
        "recommended_filters": suggestions,
        "profiles": profiles,
        "rationale": rationale,
    }


def run_mimic_experiment(
    input_dir: Path | str,
    out_dir: Path | str,
    limit: int = 0,
    episode_sampling_seed: int | None = None,
    episode_sampling_unique_stays: bool = False,
    filter_anchor_kind: str | None = None,
    filter_window_hours: int | None = None,
    balance_episode_types: bool = True,
    min_core_observed_channels: int = 2,
    min_core_observed_points: int = 6,
    min_episode_timesteps: int = 1,
    min_rule_positive_count: int = 2,
    min_rule_control_count: int = 1,
    allow_zero_control_candidates: bool = False,
    enrichment_rate_prior_count: float = 0.0,
    zero_control_rule_penalty: float = 1.0,
    miner_top_candidates: int | None = None,
    miner_top_m1: int | None = None,
    miner_top_m2: int | None = None,
    motif_verify_top_k: int | None = None,
    motif_min_support_ratio: float | None = None,
    rule_top_k: int | None = None,
    exclude_sparse_measurement_dominant: bool = False,
    sparse_measurement_penalty: float = 0.35,
    sparse_measurement_penalty_exponent: float = 2.0,
    rule_anchor_saturation_penalty_weight: float = 0.0,
    rule_anchor_saturation_penalty_exponent: float = 1.0,
    rule_mechanism_saturation_penalty_weight: float = 0.0,
    rule_mechanism_saturation_penalty_exponent: float = 1.0,
    rule_leadtime_quality_weight: float = 0.0,
    rule_support_reliability_weight: float = 0.0,
    rule_support_reliability_exponent: float = 0.5,
    rule_consensus_boost_weight: float = 0.0,
    rule_consensus_top_k: int = 10,
    rule_consensus_rerank_epsilon: float = 0.0,
    rule_consensus_quality_floor_ratio: float = 0.0,
    rule_anchor_diversity_weight: float = 0.0,
    rule_anchor_diversity_top_k: int = 10,
    rule_anchor_diversity_rerank_epsilon: float = 0.0,
    rule_anchor_diversity_quality_floor_ratio: float = 0.0,
    rule_mechanism_diversity_weight: float = 0.0,
    rule_mechanism_diversity_top_k: int = 10,
    rule_mechanism_diversity_rerank_epsilon: float = 0.0,
    rule_mechanism_diversity_quality_floor_ratio: float = 0.0,
    rule_mechanism_diversity_target: int = 0,
    rule_mechanism_diversity_two_stage_swap_budget: int = 0,
    rule_mechanism_diversity_stage1_min_support_ratio: float = 0.0,
    rule_mechanism_diversity_confidence_gate_enabled: bool = False,
    rule_mechanism_diversity_confidence_min_support_ratio: float = 0.05,
    rule_mechanism_diversity_confidence_min_reliable_rules: int = 2,
    rule_mechanism_diversity_confidence_min_expected_gain: int = 1,
    rule_mechanism_diversity_confidence_domain_thresholds_path: Path | str | None = None,
    rule_low_coverage_recovery_enabled: bool = False,
    rule_low_coverage_recovery_trigger_top_k: int = 10,
    rule_low_coverage_recovery_min_unique_groups: int = 0,
    rule_low_coverage_recovery_min_control_count: int = 0,
    rule_low_coverage_recovery_allow_zero_control_candidates: bool = False,
    rule_low_coverage_recovery_zero_control_penalty: float = 0.75,
    rule_low_coverage_recovery_support_reliability_weight: float = 0.35,
    rule_low_coverage_recovery_support_reliability_exponent: float = 1.0,
    rule_low_coverage_recovery_significance_max_q_value: float = 1.0,
    min_rule_contrastive_score: float = 0.0,
    rule_anchor_candidate_top_n: int = 0,
    rule_mechanism_candidate_top_n: int = 0,
    rule_candidate_quality_floor_ratio: float = 0.0,
    rule_care_process_max_ratio: float = 1.0,
    rule_care_process_penalty: float = 1.0,
    rule_anchor_min_unique: int = 0,
    rule_anchor_max_share: float = 1.0,
    rule_mechanism_min_unique: int = 0,
    rule_significance_weight: float = 0.0,
    rule_significance_prior_count: float = 0.5,
    rule_significance_bootstrap_samples: int = 1000,
    rule_significance_random_state: int | None = None,
    rule_significance_max_q_value: float = 1.0,
    statistical_null_permutations: int = 0,
    statistical_null_random_seed: int = 42,
    statistical_null_top_k: int = 10,
    actionability_top_k: int = 5,
    actionability_min_lead_hours: float = 2.0,
    program_leadtime_diversity_weight: float = 0.0,
    program_leadtime_diversity_gap_hours: float = 1.0,
    program_mechanism_diversity_weight: float = 0.3,
    program_mechanism_max_same_count: int = 0,
    program_care_process_penalty: float = 0.1,
    program_redundancy_penalty_weight: float = 0.0,
    program_coverage_target: int = 0,
    program_coverage_target_weight: float = 0.0,
    program_anchor_diversity_weight: float = 0.0,
    program_anchor_candidate_top_n: int = 0,
    program_anchor_candidate_quality_ratio: float = 0.0,
    program_mechanism_candidate_top_n: int = 0,
    program_mechanism_min_unique: int = 0,
    program_mechanism_group_mode: str = "coarse",
    program_mechanism_feature_score_floor_ratio: float = 0.0,
    program_score_calibration_mode: str = "none",
    program_score_calibration_quantile_low: float = 0.5,
    program_score_calibration_quantile_high: float = 0.9,
    program_score_calibration_target_low: float = 0.0,
    program_score_calibration_target_high: float = 0.0,
    program_score_calibration_blend_weight: float = 1.0,
    program_support_confidence_weight: float = 0.0,
    program_support_confidence_z_score: float = 1.96,
    program_support_confidence_floor: float = 0.6,
    program_support_confidence_exponent: float = 1.0,
    program_anchor_max_same_count: int = 0,
    program_anchor_novelty_target: int = 0,
    program_anchor_novelty_weight: float = 0.0,
    program_anchor_novelty_rerank_epsilon: float = 0.0,
    program_diversity_quality_floor_ratio: float = 0.0,
    program_topk_care_process_max_count: int = -1,
    program_topk_care_process_window: int = 5,
    min_program_representative_score: float = 0.0,
    program_top5_unique_groups_target: int = 0,
    program_top5_fixed_prefix: int = 0,
    program_top_window_unique_groups_target: int = 0,
    program_top_window_size: int = 10,
    program_top_window_fixed_prefix: int = 5,
    program_constrained_candidate_pool_size: int = 0,
    program_constrained_quality_floor_ratio: float = 0.0,
    program_constrained_two_stage_swap_budget: int = 0,
    program_constraint_autotune_steps: int = 0,
    program_constraint_autotune_group_step: int = 1,
    program_constraint_autotune_pool_step: int = 5,
    program_preselect_coverage_min_unique_groups: int = 0,
    program_preselect_coverage_window: int = 20,
    program_preselect_quality_floor_ratio: float = 0.0,
    transfer_baseline_top5_unique_groups: int = -1,
    transfer_baseline_top10_unique_groups: int = -1,
    transfer_baseline_top5_score_min: float = float("nan"),
    transfer_max_drop_top5_unique_groups: int = 999,
    transfer_max_drop_top10_unique_groups: int = 999,
    transfer_max_drop_top5_score_min: float = 999.0,
    shadow_rule_blend_weight: float = 0.0,
    shadow_rule_max_adjustment: float = 0.0,
    shadow_rule_quality_floor_ratio: float = 0.9,
    shadow_rule_exclude_care_process: bool = True,
    shadow_rule_apply_adjusted_score: bool = False,
    shadow_predictive_blend_weight: float = 0.0,
    shadow_predictive_max_adjustment: float = 0.03,
    shadow_predictive_quality_floor_ratio: float = 0.9,
    shadow_predictive_exclude_care_process: bool = True,
    shadow_predictive_training_limit: int = 0,
    shadow_predictive_seed: int = 42,
    shadow_predictive_split_manifest: Path | str | None = None,
    algorithm_ready_gate_profile: str = "exploratory",
    algorithm_ready_baseline_rules_count: int = 0,
    algorithm_ready_baseline_programs_count: int = 0,
    algorithm_ready_rule_count_min_ratio: float = 0.0,
    algorithm_ready_program_count_min_ratio: float = 0.0,
    algorithm_ready_rule_count_min_abs: int = 0,
    algorithm_ready_program_count_min_abs: int = 0,
    algorithm_ready_top5_care_process_max: int = 0,
    algorithm_ready_top5_max_same_anchor_max: int = 2,
    algorithm_ready_top5_unique_anchor_min: int = 3,
    algorithm_ready_top5_unique_groups_min: int = 5,
    algorithm_ready_top10_unique_groups_min: int = 6,
    algorithm_ready_top5_score_min_threshold: float = 0.82,
    cache_contrastive_inputs: bool = True,
    stop_after_contrastive_cache: bool = False,
    reuse_contrastive_inputs: Path | str | None = None,
    stability_cache_glob: str | None = None,
    stability_top_k: int = 10,
    stability_rule_jaccard_min: float = 0.0,
    stability_anchor_jaccard_min: float = 0.0,
    stability_mechanism_jaccard_min: float = 0.0,
    stability_tuning_profile: str = "none",
    stability_auto_preset: str = "custom",
    stability_auto_robust_rule_jaccard_max: float = 0.45,
    stability_auto_robust_anchor_jaccard_max: float = 0.6,
    stability_auto_robust_mechanism_jaccard_max: float = 0.6,
    stability_auto_balanced_rule_jaccard_max: float = 0.75,
    stability_auto_balanced_anchor_jaccard_max: float = 0.75,
    stability_auto_balanced_mechanism_jaccard_max: float = 0.75,
    stability_auto_robust_sensitivity: float = 1.0,
    stability_auto_balanced_sensitivity: float = 1.0,
    stability_auto_adaptive_history_glob: str | None = None,
    stability_auto_adaptive_history_domain: str = "",
    stability_auto_adaptive_history_domain_auto: bool = True,
    stability_auto_adaptive_history_min_samples: int = 3,
    stability_auto_adaptive_history_min_run_count: int = 0,
    stability_auto_adaptive_history_min_pair_count: int = 0,
    stability_auto_adaptive_history_warmup_runs: int = 10,
    stability_auto_adaptive_history_robust_quantile: float = 0.8,
    stability_auto_adaptive_history_balanced_quantile: float = 0.5,
    stability_auto_adaptive_history_outlier_iqr_multiplier: float = 0.0,
    stability_auto_adaptive_history_recency_half_life: float = 0.0,
    stability_auto_adaptive_history_dispersion_scale: float = 0.0,
    stability_auto_adaptive_history_drift_scale: float = 0.0,
    stability_auto_adaptive_history_drift_min_segment: int = 2,
    stability_auto_adaptive_history_reliability_strength: float = 0.0,
    stability_auto_adaptive_history_confidence_calibration: str = "none",
) -> dict:
    input_path = Path(input_dir)
    output_path = Path(out_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    miner_config = _default_miner_config()
    if miner_top_candidates is not None and int(miner_top_candidates) > 0:
        miner_config["top_candidates"] = int(miner_top_candidates)
    if miner_top_m1 is not None and int(miner_top_m1) > 0:
        miner_config["top_M1"] = int(miner_top_m1)
    if miner_top_m2 is not None and int(miner_top_m2) > 0:
        miner_config["top_M2"] = int(miner_top_m2)
    if motif_verify_top_k is not None and int(motif_verify_top_k) > 0:
        miner_config["motif_verify_top_k"] = int(motif_verify_top_k)
    if motif_min_support_ratio is not None:
        miner_config["motif_min_support_ratio"] = float(motif_min_support_ratio)
    if rule_top_k is not None and int(rule_top_k) > 0:
        miner_config["rule_top_k"] = int(rule_top_k)
    runs = []
    processed_stays = []
    skipped_stays = []
    skipped_low_information = []
    contrastive_inputs = []
    contrastive_inputs_source = None
    anchor_segment_records = []
    stability_runs = _load_stability_run_summaries(str(stability_cache_glob)) if stability_cache_glob else []

    adaptive_history_min_samples = max(1, int(stability_auto_adaptive_history_min_samples))
    adaptive_history_min_run_count = max(0, int(stability_auto_adaptive_history_min_run_count))
    adaptive_history_min_pair_count = max(0, int(stability_auto_adaptive_history_min_pair_count))
    adaptive_history_warmup_runs = max(1, int(stability_auto_adaptive_history_warmup_runs))
    adaptive_history_robust_quantile = max(0.0, min(1.0, float(stability_auto_adaptive_history_robust_quantile)))
    adaptive_history_balanced_quantile = max(0.0, min(1.0, float(stability_auto_adaptive_history_balanced_quantile)))
    adaptive_history_outlier_iqr_multiplier = max(0.0, float(stability_auto_adaptive_history_outlier_iqr_multiplier))
    adaptive_history_recency_half_life = max(0.0, float(stability_auto_adaptive_history_recency_half_life))
    adaptive_history_dispersion_scale = max(0.0, float(stability_auto_adaptive_history_dispersion_scale))
    adaptive_history_drift_scale = max(0.0, float(stability_auto_adaptive_history_drift_scale))
    adaptive_history_drift_min_segment = max(1, int(stability_auto_adaptive_history_drift_min_segment))
    adaptive_history_reliability_strength = max(0.0, min(1.0, float(stability_auto_adaptive_history_reliability_strength)))
    confidence_calibration_mode = str(stability_auto_adaptive_history_confidence_calibration or "none").strip().lower()
    if confidence_calibration_mode not in {"none", "auto"}:
        confidence_calibration_mode = "none"
    adaptive_history_domain = _normalize_adaptive_domain(stability_auto_adaptive_history_domain)
    adaptive_history_domain_source = "explicit" if adaptive_history_domain else "none"
    if (not adaptive_history_domain) and bool(stability_auto_adaptive_history_domain_auto):
        inferred_domain = _normalize_adaptive_domain(input_path.name)
        if inferred_domain:
            adaptive_history_domain = inferred_domain
            adaptive_history_domain_source = "auto_inferred_from_input_dir"

    mechanism_confidence_domain = _normalize_adaptive_domain(input_path.name)
    mechanism_confidence_threshold_map = _load_rule_mechanism_confidence_domain_thresholds(
        rule_mechanism_diversity_confidence_domain_thresholds_path
    )
    mechanism_confidence_threshold_resolved = _resolve_rule_mechanism_confidence_thresholds(
        domain=mechanism_confidence_domain,
        domain_thresholds=mechanism_confidence_threshold_map,
        default_min_support_ratio=float(rule_mechanism_diversity_confidence_min_support_ratio),
        default_min_reliable_rules=int(rule_mechanism_diversity_confidence_min_reliable_rules),
        default_min_expected_gain=int(rule_mechanism_diversity_confidence_min_expected_gain),
    )

    reuse_contrastive_path = Path(reuse_contrastive_inputs) if reuse_contrastive_inputs else None
    if reuse_contrastive_path is not None:
        contrastive_inputs = _load_cached_contrastive_inputs(reuse_contrastive_path)
        if limit and limit > 0:
            if balance_episode_types:
                contrastive_inputs = _limit_specs_with_type_balance(
                    contrastive_inputs,
                    limit=int(limit),
                    sampling_seed=episode_sampling_seed,
                    enforce_unique_stays=bool(episode_sampling_unique_stays),
                )
            else:
                contrastive_inputs = contrastive_inputs[: int(limit)]
        contrastive_inputs_source = str(reuse_contrastive_path)
        specs = []
    else:
        specs = _load_stay_specs(input_path)
        specs = _filter_episode_specs(specs, anchor_kind=filter_anchor_kind, window_hours=filter_window_hours)
        if limit and limit > 0:
            if balance_episode_types:
                specs = _limit_specs_with_type_balance(
                    specs,
                    limit=int(limit),
                    sampling_seed=episode_sampling_seed,
                    enforce_unique_stays=bool(episode_sampling_unique_stays),
                )
            else:
                specs = specs[: int(limit)]

    if not reuse_contrastive_path:
        for meta in specs:
            stay_id = int(meta["stay_id"])
            episode_id = meta.get("episode_id")
            if episode_id:
                npy_path = input_path / f"shock_episode_{episode_id}.npy"
            else:
                npy_path = input_path / f"mimic_iv_stay_{stay_id}.npy"
            if not npy_path.exists():
                skipped_stays.append(stay_id)
                continue

            X = np.load(npy_path)
            if X.ndim != 2 or X.shape[0] == 0:
                skipped_stays.append(stay_id)
                continue
            if int(X.shape[0]) < max(1, int(min_episode_timesteps)):
                skipped_low_information.append(
                    {
                        "stay_id": stay_id,
                        "episode_id": episode_id,
                        "episode_type": meta.get("episode_type"),
                        "reason": "episode_too_short",
                        "observed_timesteps": int(X.shape[0]),
                        "min_episode_timesteps": int(max(1, int(min_episode_timesteps))),
                    }
                )
                continue

            feature_meta = dict(meta)
            missing_mask_file = meta.get("missing_mask_file")
            if missing_mask_file:
                feature_meta["missing_mask"] = np.load(input_path / missing_mask_file)

            if _is_low_information_episode(
                feature_meta,
                min_core_observed_channels=min_core_observed_channels,
                min_core_observed_points=min_core_observed_points,
            ):
                skipped_low_information.append(
                    {
                        "stay_id": stay_id,
                        "episode_id": episode_id,
                        "episode_type": meta.get("episode_type"),
                    }
                )
                continue

            miner = NonStationaryRuleMiner(**miner_config)
            result = miner.fit((X, feature_meta))
            run_record = {
                "stay_id": stay_id,
                "episode_id": episode_id,
                "episode_type": meta.get("episode_type"),
                "shape": [int(X.shape[0]), int(X.shape[1])],
                "eventizer_name": result["eventizer_name"],
                "regimes": _json_safe(result["regimes"]),
                "summary": _json_safe(result["summary"]),
                "timings": _json_safe(result["timings"]),
                "motif_count": int(len(result["motifs"])),
                "rule_count": int(len(result["all_rules"])),
                "metadata": _json_safe(meta),
            }
            if episode_id:
                result_name = f"shock_episode_{episode_id}_result.json"
            else:
                result_name = f"mimic_iv_stay_{stay_id}_result.json"
            with open(output_path / result_name, "w", encoding="utf-8") as handle:
                json.dump(run_record, handle, indent=2)
            runs.append(run_record)
            processed_stays.append(stay_id)
            if episode_id:
                episode_anchor_segments = _episode_anchor_segments(X, int(meta.get("resample_minutes", 60)))
                anchor_segment_records.append(
                    {
                        "episode_id": episode_id,
                        "episode_type": meta.get("episode_type"),
                        "segments": episode_anchor_segments,
                    }
                )
                contrastive_inputs.append(
                    {
                        "episode_id": episode_id,
                        "episode_type": meta.get("episode_type"),
                        "anchor_segments": episode_anchor_segments,
                        "rules": extract_rule_records(
                            result["all_rules"],
                            event_labels=result.get("event_labels", {}),
                            feature_names=list(meta.get("feature_names", [])),
                            n_bins=int(miner_config["n_bins"]),
                            resample_minutes=int(meta.get("resample_minutes", 60)),
                            motif_occurrences=result.get("motif_occurrences", {}),
                            episode_length_steps=int(X.shape[0]),
                        ),
                    }
                )
        if contrastive_inputs and bool(cache_contrastive_inputs):
            cache_path = output_path / "contrastive_inputs.json"
            with open(cache_path, "w", encoding="utf-8") as handle:
                json.dump(_json_safe(contrastive_inputs), handle, indent=2)
            contrastive_inputs_source = str(cache_path)
        elif contrastive_inputs:
            contrastive_inputs_source = "not_cached:no_cache_contrastive_inputs"
        if bool(stop_after_contrastive_cache):
            manifest = {
                "status": "contrastive_cache_complete",
                "input_dir": str(input_path),
                "output_dir": str(output_path),
                "run_count": int(len(runs)),
                "contrastive_input_count": int(len(contrastive_inputs)),
                "contrastive_inputs_source": contrastive_inputs_source,
                "cache_written": bool(cache_contrastive_inputs),
                "requires_resume_summary": True,
            }
            with open(output_path / "contrastive_cache_manifest.json", "w", encoding="utf-8") as handle:
                json.dump(_json_safe(manifest), handle, indent=2)
            return manifest

    auto_preset = str(stability_auto_preset or "custom").strip().lower()
    if auto_preset in {"off", "disable", "disabled", "none"}:
        auto_preset = "custom"
    preset_map = {
        "strict": {
            "robust_rule": 0.6,
            "robust_anchor": 0.75,
            "robust_mechanism": 0.75,
            "balanced_rule": 0.85,
            "balanced_anchor": 0.85,
            "balanced_mechanism": 0.85,
            "robust_sensitivity": 1.2,
            "balanced_sensitivity": 1.1,
        },
        "normal": {
            "robust_rule": 0.45,
            "robust_anchor": 0.6,
            "robust_mechanism": 0.6,
            "balanced_rule": 0.75,
            "balanced_anchor": 0.75,
            "balanced_mechanism": 0.75,
            "robust_sensitivity": 1.0,
            "balanced_sensitivity": 1.0,
        },
        "loose": {
            "robust_rule": 0.2,
            "robust_anchor": 0.2,
            "robust_mechanism": 0.2,
            "balanced_rule": 0.3,
            "balanced_anchor": 0.3,
            "balanced_mechanism": 0.3,
            "robust_sensitivity": 1.0,
            "balanced_sensitivity": 1.0,
        },
    }
    applied_auto_preset = "custom"
    if auto_preset in preset_map:
        preset = preset_map[auto_preset]
        stability_auto_robust_rule_jaccard_max = float(preset["robust_rule"])
        stability_auto_robust_anchor_jaccard_max = float(preset["robust_anchor"])
        stability_auto_robust_mechanism_jaccard_max = float(preset["robust_mechanism"])
        stability_auto_balanced_rule_jaccard_max = float(preset["balanced_rule"])
        stability_auto_balanced_anchor_jaccard_max = float(preset["balanced_anchor"])
        stability_auto_balanced_mechanism_jaccard_max = float(preset["balanced_mechanism"])
        stability_auto_robust_sensitivity = float(preset["robust_sensitivity"])
        stability_auto_balanced_sensitivity = float(preset["balanced_sensitivity"])
        applied_auto_preset = auto_preset
    elif auto_preset == "adaptive":
        applied_auto_preset = "adaptive"

    profile_selection = str(stability_tuning_profile or "none").strip().lower()
    if profile_selection in {"off", "disable", "disabled"}:
        profile_selection = "none"
    resolved_profile = profile_selection
    applied_profile = "none"
    adaptive_auto_profile_metrics: dict = {}
    if stability_runs:
        profile_metrics = compute_rule_stability(stability_runs, top_k=int(stability_top_k))
        profile_catalog = _build_stability_tuning_profiles(
            stability_metrics=profile_metrics,
            current_filters={
                "rule_support_reliability_weight": rule_support_reliability_weight,
                "rule_support_reliability_exponent": rule_support_reliability_exponent,
                "rule_consensus_boost_weight": rule_consensus_boost_weight,
                "rule_consensus_top_k": rule_consensus_top_k,
                "rule_consensus_rerank_epsilon": rule_consensus_rerank_epsilon,
                "rule_consensus_quality_floor_ratio": rule_consensus_quality_floor_ratio,
                "program_anchor_max_same_count": program_anchor_max_same_count,
                "program_anchor_diversity_weight": program_anchor_diversity_weight,
                "program_mechanism_max_same_count": program_mechanism_max_same_count,
                "program_mechanism_diversity_weight": program_mechanism_diversity_weight,
            },
        )
        if profile_selection == "auto":
            if applied_auto_preset == "adaptive":
                adaptive_auto_profile_metrics = _compute_adaptive_auto_profile_metrics(
                    stability_runs,
                    top_k=int(stability_top_k),
                )
                risk_score = float(adaptive_auto_profile_metrics.get("risk_score", 0.0))
                robust_threshold = float(adaptive_auto_profile_metrics.get("robust_threshold", 1.0))
                balanced_threshold = float(adaptive_auto_profile_metrics.get("balanced_threshold", 1.0))

                stability_run_count = int(adaptive_auto_profile_metrics.get("run_count", 0))
                stability_pair_count = int(adaptive_auto_profile_metrics.get("pair_count", 0))
                history_quality_gate_pass = bool(
                    (stability_run_count >= adaptive_history_min_run_count)
                    and (stability_pair_count >= adaptive_history_min_pair_count)
                )
                adaptive_auto_profile_metrics["history_quality_gate_pass"] = bool(history_quality_gate_pass)
                adaptive_auto_profile_metrics["history_quality_gate_run_count"] = int(stability_run_count)
                adaptive_auto_profile_metrics["history_quality_gate_pair_count"] = int(stability_pair_count)
                adaptive_auto_profile_metrics["history_quality_gate_min_run_count"] = int(adaptive_history_min_run_count)
                adaptive_auto_profile_metrics["history_quality_gate_min_pair_count"] = int(adaptive_history_min_pair_count)

                history_domain_used = adaptive_history_domain
                history_domain_fallback_to_global = False
                history_samples: list[dict] = []
                if history_quality_gate_pass:
                    history_samples = _load_adaptive_history_risk_samples(
                        stability_auto_adaptive_history_glob,
                        history_domain=history_domain_used,
                    )
                    if (
                        adaptive_history_domain_source == "auto_inferred_from_input_dir"
                        and history_domain_used
                        and len(history_samples) < adaptive_history_min_samples
                    ):
                        fallback_samples = _load_adaptive_history_risk_samples(
                            stability_auto_adaptive_history_glob,
                            history_domain=None,
                        )
                        if len(fallback_samples) > len(history_samples):
                            history_samples = fallback_samples
                            history_domain_used = ""
                            history_domain_fallback_to_global = True
                history_scores = [float(row.get("risk_score", 0.0)) for row in history_samples]
                history_filter = _filter_adaptive_history_outliers(
                    history_scores,
                    adaptive_history_outlier_iqr_multiplier,
                )
                history_scores = list(history_filter.get("scores", history_scores))
                history_kept_indices = list(history_filter.get("kept_indices", list(range(len(history_scores)))))
                history_samples_filtered = [
                    history_samples[idx]
                    for idx in history_kept_indices
                    if idx >= 0 and idx < len(history_samples)
                ]
                if len(history_samples_filtered) != len(history_scores):
                    history_samples_filtered = [
                        {"risk_score": float(score), "run_count": 0, "pair_count": 0}
                        for score in history_scores
                    ]
                history_raw_count = int(history_filter.get("raw_count", len(history_scores)))
                history_filtered_count = int(history_filter.get("filtered_count", len(history_scores)))
                history_removed_count = int(history_filter.get("removed_count", 0))
                adaptive_auto_profile_metrics["history_risk_count_raw"] = int(history_raw_count)
                adaptive_auto_profile_metrics["history_risk_count"] = int(history_filtered_count)
                adaptive_auto_profile_metrics["history_outlier_filter_applied"] = bool(history_filter.get("applied", False))
                adaptive_auto_profile_metrics["history_outlier_filter_removed_count"] = int(history_removed_count)
                adaptive_auto_profile_metrics["history_outlier_filter_iqr_multiplier"] = float(adaptive_history_outlier_iqr_multiplier)
                if history_filter.get("lower_bound") is not None:
                    adaptive_auto_profile_metrics["history_outlier_filter_lower_bound"] = float(history_filter.get("lower_bound"))
                if history_filter.get("upper_bound") is not None:
                    adaptive_auto_profile_metrics["history_outlier_filter_upper_bound"] = float(history_filter.get("upper_bound"))
                adaptive_auto_profile_metrics["history_glob"] = str(stability_auto_adaptive_history_glob or "")
                adaptive_auto_profile_metrics["history_domain"] = str(history_domain_used)
                adaptive_auto_profile_metrics["history_domain_requested"] = str(adaptive_history_domain)
                adaptive_auto_profile_metrics["history_domain_source"] = str(adaptive_history_domain_source)
                adaptive_auto_profile_metrics["history_domain_fallback_to_global"] = bool(history_domain_fallback_to_global)
                adaptive_auto_profile_metrics["history_min_samples"] = int(adaptive_history_min_samples)
                adaptive_auto_profile_metrics["history_warmup_runs"] = int(adaptive_history_warmup_runs)
                adaptive_auto_profile_metrics["history_robust_quantile"] = float(adaptive_history_robust_quantile)
                adaptive_auto_profile_metrics["history_balanced_quantile"] = float(adaptive_history_balanced_quantile)
                adaptive_auto_profile_metrics["history_recency_half_life"] = float(adaptive_history_recency_half_life)
                adaptive_auto_profile_metrics["history_dispersion_scale"] = float(adaptive_history_dispersion_scale)
                adaptive_auto_profile_metrics["history_drift_scale"] = float(adaptive_history_drift_scale)
                adaptive_auto_profile_metrics["history_drift_min_segment"] = int(adaptive_history_drift_min_segment)
                adaptive_auto_profile_metrics["history_reliability_strength"] = float(adaptive_history_reliability_strength)
                adaptive_auto_profile_metrics["history_confidence_calibration_mode"] = str(confidence_calibration_mode)
                quality_confidence = 1.0 if history_quality_gate_pass else 0.0
                adaptive_auto_profile_metrics["history_confidence_quality_gate"] = float(quality_confidence)
                adaptive_auto_profile_metrics["history_confidence_support"] = 0.0
                adaptive_auto_profile_metrics["history_confidence_distribution"] = 0.0
                adaptive_auto_profile_metrics["history_confidence_total"] = 0.0
                adaptive_auto_profile_metrics["history_confidence_total_upper_bound"] = float(
                    quality_confidence * 0.0
                )
                adaptive_auto_profile_metrics["history_confidence_total_gap_to_upper_bound"] = 0.0
                adaptive_auto_profile_metrics["history_confidence_dispersion_sensitivity"] = 0.0
                adaptive_auto_profile_metrics["history_confidence_drift_sensitivity"] = 0.0
                adaptive_auto_profile_metrics["history_confidence_calibration_applied"] = bool(False)
                adaptive_auto_profile_metrics["history_confidence_calibration_auto_dispersion_applied"] = bool(False)
                adaptive_auto_profile_metrics["history_confidence_calibration_auto_drift_applied"] = bool(False)
                adaptive_auto_profile_metrics["history_dispersion_scale_effective"] = float(adaptive_history_dispersion_scale)
                adaptive_auto_profile_metrics["history_drift_scale_effective"] = float(adaptive_history_drift_scale)

                if history_quality_gate_pass and len(history_scores) >= adaptive_history_min_samples:
                    recency_weighting_applied = bool(adaptive_history_recency_half_life > 0.0)
                    raw_support_count = float(len(history_scores))

                    if recency_weighting_applied:
                        recency_weights = _adaptive_history_recency_weights(
                            len(history_scores),
                            adaptive_history_recency_half_life,
                        )
                        history_recency_weight_sum = float(sum(recency_weights))
                    else:
                        recency_weights = [1.0 for _ in history_scores]
                        history_recency_weight_sum = 0.0

                    reliability_strength = max(0.0, min(1.0, float(adaptive_history_reliability_strength)))
                    reliability_weighting_applied = bool(reliability_strength > 0.0)
                    reliability_weights: list[float] = []
                    for sample in history_samples_filtered:
                        if not reliability_weighting_applied:
                            reliability_weights.append(1.0)
                            continue
                        sample_run_count = max(0, int(sample.get("run_count", 0)))
                        sample_pair_count = max(0, int(sample.get("pair_count", 0)))
                        reliability_components: list[float] = []
                        if stability_run_count > 0:
                            reliability_components.append(
                                max(0.0, min(1.0, float(sample_run_count) / float(stability_run_count)))
                            )
                        if stability_pair_count > 0:
                            reliability_components.append(
                                max(0.0, min(1.0, float(sample_pair_count) / float(stability_pair_count)))
                            )
                        if reliability_components:
                            evidence_reliability = float(sum(reliability_components) / float(len(reliability_components)))
                        else:
                            evidence_reliability = 1.0
                        reliability_weights.append(
                            max(
                                0.0,
                                min(
                                    1.0,
                                    ((1.0 - reliability_strength) + (reliability_strength * float(evidence_reliability))),
                                ),
                            )
                        )

                    if len(reliability_weights) != len(history_scores):
                        reliability_weights = [1.0 for _ in history_scores]
                        reliability_weighting_applied = False

                    combined_weights = [
                        max(0.0, float(recency_weights[idx]) * float(reliability_weights[idx]))
                        for idx in range(len(history_scores))
                    ]
                    combined_weight_sum = float(sum(combined_weights))
                    if combined_weight_sum <= 0.0:
                        combined_weights = [1.0 for _ in history_scores]
                        combined_weight_sum = float(len(combined_weights))

                    if recency_weighting_applied or reliability_weighting_applied:
                        robust_hist = _weighted_quantile(history_scores, combined_weights, adaptive_history_robust_quantile)
                        balanced_hist = _weighted_quantile(history_scores, combined_weights, adaptive_history_balanced_quantile)
                    else:
                        robust_hist = float(np.quantile(np.asarray(history_scores, dtype=float), adaptive_history_robust_quantile))
                        balanced_hist = float(np.quantile(np.asarray(history_scores, dtype=float), adaptive_history_balanced_quantile))

                    history_weight_square_sum = float(sum(float(weight) * float(weight) for weight in combined_weights))
                    mix_support_count = float(raw_support_count)
                    if history_weight_square_sum > 0.0:
                        mix_support_count = float((combined_weight_sum * combined_weight_sum) / history_weight_square_sum)

                    history_iqr = 0.0
                    if len(history_scores) >= 2:
                        history_values = np.asarray(history_scores, dtype=float)
                        history_iqr = float(np.quantile(history_values, 0.75) - np.quantile(history_values, 0.25))
                    mix_support_count = max(0.0, min(raw_support_count, float(mix_support_count)))
                    mix_base = max(0.0, min(1.0, float(mix_support_count) / float(adaptive_history_warmup_runs)))
                    effective_dispersion_scale = float(adaptive_history_dispersion_scale)
                    auto_dispersion_scale_applied = False
                    if (
                        confidence_calibration_mode == "auto"
                        and effective_dispersion_scale <= 0.0
                        and history_iqr > 0.0
                    ):
                        effective_dispersion_scale = float(history_iqr)
                        auto_dispersion_scale_applied = True
                    dispersion_confidence = 1.0
                    if effective_dispersion_scale > 0.0:
                        dispersion_confidence = 1.0 / (
                            1.0 + (max(0.0, history_iqr) / float(effective_dispersion_scale))
                        )
                        dispersion_confidence = max(0.0, min(1.0, float(dispersion_confidence)))
                    drift_confidence = 1.0
                    drift_gap = 0.0
                    drift_old_median = None
                    drift_recent_median = None
                    drift_search_applied = False
                    drift_split_index = 0
                    if adaptive_history_drift_scale > 0.0 or confidence_calibration_mode == "auto":
                        drift_split = _adaptive_history_best_drift_split(
                            history_scores,
                            min_segment=adaptive_history_drift_min_segment,
                        )
                        drift_search_applied = bool(drift_split.get("applied", False))
                        drift_split_index = int(drift_split.get("split_index", 0) or 0)
                        if drift_search_applied:
                            drift_old_median = drift_split.get("old_median")
                            drift_recent_median = drift_split.get("recent_median")
                            drift_gap = float(drift_split.get("gap", 0.0) or 0.0)
                    effective_drift_scale = float(adaptive_history_drift_scale)
                    auto_drift_scale_applied = False
                    if (
                        confidence_calibration_mode == "auto"
                        and effective_drift_scale <= 0.0
                        and drift_gap > 0.0
                    ):
                        effective_drift_scale = float(drift_gap)
                        auto_drift_scale_applied = True
                    if effective_drift_scale > 0.0 and drift_search_applied:
                        drift_confidence = 1.0 / (
                            1.0 + (max(0.0, drift_gap) / float(effective_drift_scale))
                        )
                        drift_confidence = max(0.0, min(1.0, float(drift_confidence)))
                    support_confidence = max(0.0, min(1.0, float(mix_base)))
                    distribution_confidence = max(
                        0.0,
                        min(1.0, float(dispersion_confidence) * float(drift_confidence)),
                    )
                    history_confidence_total = max(
                        0.0,
                        min(
                            1.0,
                            float(quality_confidence) * float(support_confidence) * float(distribution_confidence),
                        ),
                    )
                    history_confidence_total_upper_bound = max(
                        0.0,
                        min(1.0, float(quality_confidence) * float(support_confidence)),
                    )
                    history_confidence_total_gap_to_upper_bound = max(
                        0.0,
                        float(history_confidence_total_upper_bound) - float(history_confidence_total),
                    )
                    dispersion_sensitivity = 0.0
                    if effective_dispersion_scale > 0.0:
                        dispersion_sensitivity = float(dispersion_confidence) * (1.0 - float(dispersion_confidence))
                    drift_sensitivity = 0.0
                    if effective_drift_scale > 0.0 and drift_search_applied:
                        drift_sensitivity = float(drift_confidence) * (1.0 - float(drift_confidence))
                    mix = float(history_confidence_total)
                    adaptive_auto_profile_metrics["history_recency_weighting_applied"] = bool(recency_weighting_applied)
                    adaptive_auto_profile_metrics["history_recency_weight_sum"] = float(history_recency_weight_sum)
                    adaptive_auto_profile_metrics["history_reliability_weighting_applied"] = bool(reliability_weighting_applied)
                    adaptive_auto_profile_metrics["history_reliability_weight_sum"] = float(sum(reliability_weights))
                    adaptive_auto_profile_metrics["history_reliability_weight_min"] = float(min(reliability_weights))
                    adaptive_auto_profile_metrics["history_reliability_weight_max"] = float(max(reliability_weights))
                    adaptive_auto_profile_metrics["history_recency_weight_square_sum"] = float(history_weight_square_sum)
                    adaptive_auto_profile_metrics["history_mix_support_count_raw"] = float(raw_support_count)
                    adaptive_auto_profile_metrics["history_mix_support_count"] = float(mix_support_count)
                    adaptive_auto_profile_metrics["history_mix_base"] = float(mix_base)
                    adaptive_auto_profile_metrics["history_dispersion_iqr"] = float(history_iqr)
                    adaptive_auto_profile_metrics["history_dispersion_confidence"] = float(dispersion_confidence)
                    adaptive_auto_profile_metrics["history_drift_gap"] = float(drift_gap)
                    adaptive_auto_profile_metrics["history_drift_confidence"] = float(drift_confidence)
                    adaptive_auto_profile_metrics["history_confidence_support"] = float(support_confidence)
                    adaptive_auto_profile_metrics["history_confidence_distribution"] = float(distribution_confidence)
                    adaptive_auto_profile_metrics["history_confidence_total"] = float(history_confidence_total)
                    adaptive_auto_profile_metrics["history_confidence_total_upper_bound"] = float(
                        history_confidence_total_upper_bound
                    )
                    adaptive_auto_profile_metrics["history_confidence_total_gap_to_upper_bound"] = float(
                        history_confidence_total_gap_to_upper_bound
                    )
                    adaptive_auto_profile_metrics["history_confidence_dispersion_sensitivity"] = float(
                        max(0.0, min(1.0, dispersion_sensitivity))
                    )
                    adaptive_auto_profile_metrics["history_confidence_drift_sensitivity"] = float(
                        max(0.0, min(1.0, drift_sensitivity))
                    )
                    adaptive_auto_profile_metrics["history_confidence_calibration_auto_dispersion_applied"] = bool(
                        auto_dispersion_scale_applied
                    )
                    adaptive_auto_profile_metrics["history_confidence_calibration_auto_drift_applied"] = bool(
                        auto_drift_scale_applied
                    )
                    adaptive_auto_profile_metrics["history_confidence_calibration_applied"] = bool(
                        auto_dispersion_scale_applied or auto_drift_scale_applied
                    )
                    adaptive_auto_profile_metrics["history_dispersion_scale_effective"] = float(effective_dispersion_scale)
                    adaptive_auto_profile_metrics["history_drift_scale_effective"] = float(effective_drift_scale)
                    adaptive_auto_profile_metrics["history_drift_search_applied"] = bool(drift_search_applied)
                    adaptive_auto_profile_metrics["history_drift_split_index"] = int(drift_split_index)
                    if drift_old_median is not None:
                        adaptive_auto_profile_metrics["history_drift_old_median"] = float(drift_old_median)
                    if drift_recent_median is not None:
                        adaptive_auto_profile_metrics["history_drift_recent_median"] = float(drift_recent_median)
                    robust_threshold = ((1.0 - mix) * robust_threshold) + (mix * robust_hist)
                    balanced_threshold = ((1.0 - mix) * balanced_threshold) + (mix * balanced_hist)
                    robust_threshold = max(0.0, min(1.0, robust_threshold))
                    balanced_threshold = max(0.0, min(1.0, balanced_threshold))
                    if balanced_threshold > robust_threshold:
                        balanced_threshold = max(0.0, robust_threshold - 0.01)
                    adaptive_auto_profile_metrics["history_mix"] = float(mix)
                    adaptive_auto_profile_metrics["history_robust_threshold"] = float(robust_hist)
                    adaptive_auto_profile_metrics["history_balanced_threshold"] = float(balanced_hist)

                if "history_recency_weighting_applied" not in adaptive_auto_profile_metrics:
                    adaptive_auto_profile_metrics["history_recency_weighting_applied"] = bool(False)
                    adaptive_auto_profile_metrics["history_recency_weight_sum"] = 0.0
                    adaptive_auto_profile_metrics["history_recency_weight_square_sum"] = 0.0
                if "history_reliability_weighting_applied" not in adaptive_auto_profile_metrics:
                    adaptive_auto_profile_metrics["history_reliability_weighting_applied"] = bool(False)
                    adaptive_auto_profile_metrics["history_reliability_weight_sum"] = 0.0
                    adaptive_auto_profile_metrics["history_reliability_weight_min"] = 1.0
                    adaptive_auto_profile_metrics["history_reliability_weight_max"] = 1.0

                adaptive_auto_profile_metrics["robust_threshold"] = float(robust_threshold)
                adaptive_auto_profile_metrics["balanced_threshold"] = float(balanced_threshold)
                if risk_score >= robust_threshold:
                    resolved_profile = "robust"
                elif risk_score >= balanced_threshold:
                    resolved_profile = "balanced"
                else:
                    resolved_profile = "aggressive"
            else:
                rule_j = float(profile_metrics.get("topk_rule_jaccard_mean", 1.0))
                anchor_j = float(profile_metrics.get("topk_anchor_jaccard_mean", 1.0))
                mechanism_j = float(profile_metrics.get("topk_mechanism_jaccard_mean", 1.0))
                robust_sensitivity = max(0.0, float(stability_auto_robust_sensitivity))
                balanced_sensitivity = max(0.0, float(stability_auto_balanced_sensitivity))
                robust_rule_max = max(0.0, min(1.0, float(stability_auto_robust_rule_jaccard_max) * robust_sensitivity))
                robust_anchor_max = max(0.0, min(1.0, float(stability_auto_robust_anchor_jaccard_max) * robust_sensitivity))
                robust_mechanism_max = max(0.0, min(1.0, float(stability_auto_robust_mechanism_jaccard_max) * robust_sensitivity))
                balanced_rule_max = max(0.0, min(1.0, float(stability_auto_balanced_rule_jaccard_max) * balanced_sensitivity))
                balanced_anchor_max = max(0.0, min(1.0, float(stability_auto_balanced_anchor_jaccard_max) * balanced_sensitivity))
                balanced_mechanism_max = max(0.0, min(1.0, float(stability_auto_balanced_mechanism_jaccard_max) * balanced_sensitivity))
                if rule_j < robust_rule_max or anchor_j < robust_anchor_max or mechanism_j < robust_mechanism_max:
                    resolved_profile = "robust"
                elif (
                    rule_j < balanced_rule_max
                    or anchor_j < balanced_anchor_max
                    or mechanism_j < balanced_mechanism_max
                ):
                    resolved_profile = "balanced"
                else:
                    resolved_profile = "aggressive"
    else:
        profile_catalog = {}
        if profile_selection == "auto":
            resolved_profile = "none"

    if resolved_profile in profile_catalog:
        profile_filters = dict(profile_catalog[resolved_profile].get("recommended_filters", {}))
        rule_support_reliability_weight = float(
            profile_filters.get("rule_support_reliability_weight", rule_support_reliability_weight)
        )
        rule_support_reliability_exponent = float(
            profile_filters.get("rule_support_reliability_exponent", rule_support_reliability_exponent)
        )
        rule_consensus_boost_weight = float(profile_filters.get("rule_consensus_boost_weight", rule_consensus_boost_weight))
        rule_consensus_top_k = int(profile_filters.get("rule_consensus_top_k", rule_consensus_top_k))
        rule_consensus_rerank_epsilon = float(
            profile_filters.get("rule_consensus_rerank_epsilon", rule_consensus_rerank_epsilon)
        )
        rule_consensus_quality_floor_ratio = float(
            profile_filters.get("rule_consensus_quality_floor_ratio", rule_consensus_quality_floor_ratio)
        )
        rule_anchor_diversity_weight = float(
            profile_filters.get("rule_anchor_diversity_weight", rule_anchor_diversity_weight)
        )
        rule_anchor_diversity_top_k = int(
            profile_filters.get("rule_anchor_diversity_top_k", rule_anchor_diversity_top_k)
        )
        rule_anchor_diversity_rerank_epsilon = float(
            profile_filters.get("rule_anchor_diversity_rerank_epsilon", rule_anchor_diversity_rerank_epsilon)
        )
        rule_anchor_diversity_quality_floor_ratio = float(
            profile_filters.get(
                "rule_anchor_diversity_quality_floor_ratio",
                rule_anchor_diversity_quality_floor_ratio,
            )
        )
        rule_mechanism_diversity_weight = float(
            profile_filters.get("rule_mechanism_diversity_weight", rule_mechanism_diversity_weight)
        )
        rule_mechanism_diversity_top_k = int(
            profile_filters.get("rule_mechanism_diversity_top_k", rule_mechanism_diversity_top_k)
        )
        rule_mechanism_diversity_rerank_epsilon = float(
            profile_filters.get(
                "rule_mechanism_diversity_rerank_epsilon",
                rule_mechanism_diversity_rerank_epsilon,
            )
        )
        rule_mechanism_diversity_quality_floor_ratio = float(
            profile_filters.get(
                "rule_mechanism_diversity_quality_floor_ratio",
                rule_mechanism_diversity_quality_floor_ratio,
            )
        )
        rule_mechanism_diversity_target = int(
            profile_filters.get("rule_mechanism_diversity_target", rule_mechanism_diversity_target)
        )
        rule_mechanism_diversity_two_stage_swap_budget = int(
            profile_filters.get(
                "rule_mechanism_diversity_two_stage_swap_budget",
                rule_mechanism_diversity_two_stage_swap_budget,
            )
        )
        rule_mechanism_diversity_stage1_min_support_ratio = float(
            profile_filters.get(
                "rule_mechanism_diversity_stage1_min_support_ratio",
                rule_mechanism_diversity_stage1_min_support_ratio,
            )
        )
        rule_mechanism_diversity_confidence_gate_enabled = bool(
            profile_filters.get(
                "rule_mechanism_diversity_confidence_gate_enabled",
                rule_mechanism_diversity_confidence_gate_enabled,
            )
        )
        rule_mechanism_diversity_confidence_min_support_ratio = float(
            profile_filters.get(
                "rule_mechanism_diversity_confidence_min_support_ratio",
                rule_mechanism_diversity_confidence_min_support_ratio,
            )
        )
        rule_mechanism_diversity_confidence_min_reliable_rules = int(
            profile_filters.get(
                "rule_mechanism_diversity_confidence_min_reliable_rules",
                rule_mechanism_diversity_confidence_min_reliable_rules,
            )
        )
        rule_mechanism_diversity_confidence_min_expected_gain = int(
            profile_filters.get(
                "rule_mechanism_diversity_confidence_min_expected_gain",
                rule_mechanism_diversity_confidence_min_expected_gain,
            )
        )
        rule_low_coverage_recovery_enabled = bool(
            profile_filters.get(
                "rule_low_coverage_recovery_enabled",
                rule_low_coverage_recovery_enabled,
            )
        )
        rule_low_coverage_recovery_trigger_top_k = int(
            profile_filters.get(
                "rule_low_coverage_recovery_trigger_top_k",
                rule_low_coverage_recovery_trigger_top_k,
            )
        )
        rule_low_coverage_recovery_min_unique_groups = int(
            profile_filters.get(
                "rule_low_coverage_recovery_min_unique_groups",
                rule_low_coverage_recovery_min_unique_groups,
            )
        )
        rule_low_coverage_recovery_min_control_count = int(
            profile_filters.get(
                "rule_low_coverage_recovery_min_control_count",
                rule_low_coverage_recovery_min_control_count,
            )
        )
        rule_low_coverage_recovery_allow_zero_control_candidates = bool(
            profile_filters.get(
                "rule_low_coverage_recovery_allow_zero_control_candidates",
                rule_low_coverage_recovery_allow_zero_control_candidates,
            )
        )
        rule_low_coverage_recovery_zero_control_penalty = float(
            profile_filters.get(
                "rule_low_coverage_recovery_zero_control_penalty",
                rule_low_coverage_recovery_zero_control_penalty,
            )
        )
        rule_low_coverage_recovery_support_reliability_weight = float(
            profile_filters.get(
                "rule_low_coverage_recovery_support_reliability_weight",
                rule_low_coverage_recovery_support_reliability_weight,
            )
        )
        rule_low_coverage_recovery_support_reliability_exponent = float(
            profile_filters.get(
                "rule_low_coverage_recovery_support_reliability_exponent",
                rule_low_coverage_recovery_support_reliability_exponent,
            )
        )
        rule_low_coverage_recovery_significance_max_q_value = float(
            profile_filters.get(
                "rule_low_coverage_recovery_significance_max_q_value",
                rule_low_coverage_recovery_significance_max_q_value,
            )
        )
        program_anchor_max_same_count = int(
            profile_filters.get("program_anchor_max_same_count", program_anchor_max_same_count)
        )
        program_anchor_diversity_weight = float(
            profile_filters.get("program_anchor_diversity_weight", program_anchor_diversity_weight)
        )
        program_mechanism_max_same_count = int(
            profile_filters.get("program_mechanism_max_same_count", program_mechanism_max_same_count)
        )
        program_mechanism_diversity_weight = float(
            profile_filters.get("program_mechanism_diversity_weight", program_mechanism_diversity_weight)
        )
        program_mechanism_group_mode = str(
            profile_filters.get("program_mechanism_group_mode", program_mechanism_group_mode)
        )
        program_mechanism_feature_score_floor_ratio = float(
            profile_filters.get(
                "program_mechanism_feature_score_floor_ratio",
                program_mechanism_feature_score_floor_ratio,
            )
        )
        program_score_calibration_mode = str(
            profile_filters.get("program_score_calibration_mode", program_score_calibration_mode)
        )
        program_score_calibration_quantile_low = float(
            profile_filters.get("program_score_calibration_quantile_low", program_score_calibration_quantile_low)
        )
        program_score_calibration_quantile_high = float(
            profile_filters.get("program_score_calibration_quantile_high", program_score_calibration_quantile_high)
        )
        program_score_calibration_target_low = float(
            profile_filters.get("program_score_calibration_target_low", program_score_calibration_target_low)
        )
        program_score_calibration_target_high = float(
            profile_filters.get("program_score_calibration_target_high", program_score_calibration_target_high)
        )
        program_score_calibration_blend_weight = float(
            profile_filters.get("program_score_calibration_blend_weight", program_score_calibration_blend_weight)
        )
        program_support_confidence_weight = float(
            profile_filters.get("program_support_confidence_weight", program_support_confidence_weight)
        )
        program_support_confidence_z_score = float(
            profile_filters.get("program_support_confidence_z_score", program_support_confidence_z_score)
        )
        program_support_confidence_floor = float(
            profile_filters.get("program_support_confidence_floor", program_support_confidence_floor)
        )
        program_support_confidence_exponent = float(
            profile_filters.get("program_support_confidence_exponent", program_support_confidence_exponent)
        )
        applied_profile = resolved_profile

    program_mechanism_group_mode = _resolve_program_mechanism_group_mode(program_mechanism_group_mode)
    program_mechanism_feature_score_floor_ratio = max(
        0.0,
        min(1.0, float(program_mechanism_feature_score_floor_ratio)),
    )
    program_score_calibration_mode = _resolve_program_score_calibration_mode(program_score_calibration_mode)
    program_score_calibration_quantile_low = max(0.0, min(1.0, float(program_score_calibration_quantile_low)))
    program_score_calibration_quantile_high = max(0.0, min(1.0, float(program_score_calibration_quantile_high)))
    program_score_calibration_blend_weight = max(0.0, min(1.0, float(program_score_calibration_blend_weight)))
    program_support_confidence_weight = max(0.0, min(1.0, float(program_support_confidence_weight)))
    program_support_confidence_z_score = max(0.0, float(program_support_confidence_z_score))
    program_support_confidence_floor = max(0.0, min(1.0, float(program_support_confidence_floor)))
    program_support_confidence_exponent = max(1e-6, float(program_support_confidence_exponent))

    summary = {
        "input_dir": str(input_path),
        "output_dir": str(output_path),
        "processed_stays": processed_stays,
        "skipped_stays": skipped_stays,
        "skipped_low_information": skipped_low_information,
        "run_count": len(runs),
        "miner_config": miner_config,
        "runs": runs,
    }
    requested_topk_care_process_max_count = int(program_topk_care_process_max_count)
    effective_topk_care_process_max_count = requested_topk_care_process_max_count
    if (
        effective_topk_care_process_max_count < 0
        and str(algorithm_ready_gate_profile or "").strip().lower() == "locked"
    ):
        effective_topk_care_process_max_count = int(algorithm_ready_top5_care_process_max)
    effective_topk_care_process_window = max(1, int(program_topk_care_process_window))
    if contrastive_inputs:
        summary["contrastive_inputs_source"] = contrastive_inputs_source
        summary["contrastive_input_count"] = int(len(contrastive_inputs))
        summary["anchor_segment_summary"] = summarize_anchor_segments(anchor_segment_records)
        summary["contrastive_filters"] = {
            "exclude_missing_only": True,
            "min_rule_positive_count": int(min_rule_positive_count),
            "min_rule_control_count": int(min_rule_control_count),
            "min_episode_timesteps": int(max(1, int(min_episode_timesteps))),
            "episode_sampling_seed": None if episode_sampling_seed is None else int(episode_sampling_seed),
            "episode_sampling_unique_stays": bool(episode_sampling_unique_stays),
            "allow_zero_control_candidates": bool(allow_zero_control_candidates),
            "enrichment_rate_prior_count": float(enrichment_rate_prior_count),
            "zero_control_rule_penalty": float(zero_control_rule_penalty),
            "exclude_sparse_measurement_dominant": bool(exclude_sparse_measurement_dominant),
            "sparse_measurement_penalty": float(sparse_measurement_penalty),
            "sparse_measurement_penalty_exponent": float(sparse_measurement_penalty_exponent),
            "rule_anchor_saturation_penalty_weight": float(rule_anchor_saturation_penalty_weight),
            "rule_anchor_saturation_penalty_exponent": float(rule_anchor_saturation_penalty_exponent),
            "rule_mechanism_saturation_penalty_weight": float(rule_mechanism_saturation_penalty_weight),
            "rule_mechanism_saturation_penalty_exponent": float(rule_mechanism_saturation_penalty_exponent),
            "rule_leadtime_quality_weight": float(rule_leadtime_quality_weight),
            "stability_tuning_profile": str(applied_profile),
            "stability_auto_preset": str(applied_auto_preset),
            "rule_support_reliability_weight": float(rule_support_reliability_weight),
            "rule_support_reliability_exponent": float(rule_support_reliability_exponent),
            "rule_consensus_boost_weight": float(rule_consensus_boost_weight),
            "rule_consensus_top_k": int(rule_consensus_top_k),
            "rule_consensus_rerank_epsilon": float(rule_consensus_rerank_epsilon),
            "rule_consensus_quality_floor_ratio": float(rule_consensus_quality_floor_ratio),
            "rule_anchor_diversity_weight": float(rule_anchor_diversity_weight),
            "rule_anchor_diversity_top_k": int(rule_anchor_diversity_top_k),
            "rule_anchor_diversity_rerank_epsilon": float(rule_anchor_diversity_rerank_epsilon),
            "rule_anchor_diversity_quality_floor_ratio": float(rule_anchor_diversity_quality_floor_ratio),
            "rule_mechanism_diversity_weight": float(rule_mechanism_diversity_weight),
            "rule_mechanism_diversity_top_k": int(rule_mechanism_diversity_top_k),
            "rule_mechanism_diversity_rerank_epsilon": float(rule_mechanism_diversity_rerank_epsilon),
            "rule_mechanism_diversity_quality_floor_ratio": float(rule_mechanism_diversity_quality_floor_ratio),
            "rule_mechanism_diversity_target": int(rule_mechanism_diversity_target),
            "rule_mechanism_diversity_two_stage_swap_budget": int(rule_mechanism_diversity_two_stage_swap_budget),
            "rule_mechanism_diversity_stage1_min_support_ratio": float(
                rule_mechanism_diversity_stage1_min_support_ratio
            ),
            "rule_mechanism_diversity_confidence_gate_enabled": bool(
                rule_mechanism_diversity_confidence_gate_enabled
            ),
            "rule_mechanism_diversity_confidence_min_support_ratio": float(
                rule_mechanism_diversity_confidence_min_support_ratio
            ),
            "rule_mechanism_diversity_confidence_min_reliable_rules": int(
                rule_mechanism_diversity_confidence_min_reliable_rules
            ),
            "rule_mechanism_diversity_confidence_min_expected_gain": int(
                rule_mechanism_diversity_confidence_min_expected_gain
            ),
            "rule_mechanism_diversity_confidence_domain_thresholds_path": (
                str(rule_mechanism_diversity_confidence_domain_thresholds_path)
                if rule_mechanism_diversity_confidence_domain_thresholds_path
                else None
            ),
            "rule_mechanism_diversity_confidence_domain": str(mechanism_confidence_domain),
            "rule_mechanism_diversity_confidence_threshold_source": str(
                mechanism_confidence_threshold_resolved.get("source", "runtime_defaults")
            ),
            "rule_mechanism_diversity_confidence_min_support_ratio_effective": float(
                mechanism_confidence_threshold_resolved.get(
                    "min_support_ratio",
                    rule_mechanism_diversity_confidence_min_support_ratio,
                )
            ),
            "rule_mechanism_diversity_confidence_min_reliable_rules_effective": int(
                mechanism_confidence_threshold_resolved.get(
                    "min_reliable_rules",
                    rule_mechanism_diversity_confidence_min_reliable_rules,
                )
            ),
            "rule_mechanism_diversity_confidence_min_expected_gain_effective": int(
                mechanism_confidence_threshold_resolved.get(
                    "min_expected_gain",
                    rule_mechanism_diversity_confidence_min_expected_gain,
                )
            ),
            "rule_low_coverage_recovery_enabled": bool(rule_low_coverage_recovery_enabled),
            "rule_low_coverage_recovery_trigger_top_k": int(rule_low_coverage_recovery_trigger_top_k),
            "rule_low_coverage_recovery_min_unique_groups": int(rule_low_coverage_recovery_min_unique_groups),
            "rule_low_coverage_recovery_min_control_count": int(rule_low_coverage_recovery_min_control_count),
            "rule_low_coverage_recovery_allow_zero_control_candidates": bool(
                rule_low_coverage_recovery_allow_zero_control_candidates
            ),
            "rule_low_coverage_recovery_zero_control_penalty": float(rule_low_coverage_recovery_zero_control_penalty),
            "rule_low_coverage_recovery_support_reliability_weight": float(
                rule_low_coverage_recovery_support_reliability_weight
            ),
            "rule_low_coverage_recovery_support_reliability_exponent": float(
                rule_low_coverage_recovery_support_reliability_exponent
            ),
            "rule_low_coverage_recovery_significance_max_q_value": float(
                max(0.0, min(1.0, float(rule_low_coverage_recovery_significance_max_q_value)))
            ),
            "stability_auto_robust_rule_jaccard_max": float(stability_auto_robust_rule_jaccard_max),
            "stability_auto_robust_anchor_jaccard_max": float(stability_auto_robust_anchor_jaccard_max),
            "stability_auto_robust_mechanism_jaccard_max": float(stability_auto_robust_mechanism_jaccard_max),
            "stability_auto_balanced_rule_jaccard_max": float(stability_auto_balanced_rule_jaccard_max),
            "stability_auto_balanced_anchor_jaccard_max": float(stability_auto_balanced_anchor_jaccard_max),
            "stability_auto_balanced_mechanism_jaccard_max": float(stability_auto_balanced_mechanism_jaccard_max),
            "stability_auto_robust_sensitivity": float(stability_auto_robust_sensitivity),
            "stability_auto_balanced_sensitivity": float(stability_auto_balanced_sensitivity),
            "stability_auto_adaptive_history_glob": str(stability_auto_adaptive_history_glob or ""),
            "stability_auto_adaptive_domain": str(adaptive_history_domain),
            "stability_auto_adaptive_domain_source": str(adaptive_history_domain_source),
            "stability_auto_adaptive_history_domain_auto": bool(stability_auto_adaptive_history_domain_auto),
            "stability_auto_adaptive_history_min_samples": int(adaptive_history_min_samples),
            "stability_auto_adaptive_history_min_run_count": int(adaptive_history_min_run_count),
            "stability_auto_adaptive_history_min_pair_count": int(adaptive_history_min_pair_count),
            "stability_auto_adaptive_history_warmup_runs": int(adaptive_history_warmup_runs),
            "stability_auto_adaptive_history_robust_quantile": float(adaptive_history_robust_quantile),
            "stability_auto_adaptive_history_balanced_quantile": float(adaptive_history_balanced_quantile),
            "stability_auto_adaptive_history_outlier_iqr_multiplier": float(adaptive_history_outlier_iqr_multiplier),
            "stability_auto_adaptive_history_recency_half_life": float(adaptive_history_recency_half_life),
            "stability_auto_adaptive_history_dispersion_scale": float(adaptive_history_dispersion_scale),
            "stability_auto_adaptive_history_drift_scale": float(adaptive_history_drift_scale),
            "stability_auto_adaptive_history_drift_min_segment": int(adaptive_history_drift_min_segment),
            "stability_auto_adaptive_history_reliability_strength": float(adaptive_history_reliability_strength),
            "stability_auto_adaptive_history_confidence_calibration": str(confidence_calibration_mode),
            "stability_auto_adaptive_metrics": (
                dict(adaptive_auto_profile_metrics)
                if applied_auto_preset == "adaptive"
                else {}
            ),
            "min_rule_contrastive_score": float(min_rule_contrastive_score),
            "rule_anchor_candidate_top_n": int(rule_anchor_candidate_top_n),
            "rule_mechanism_candidate_top_n": int(rule_mechanism_candidate_top_n),
            "rule_candidate_quality_floor_ratio": float(rule_candidate_quality_floor_ratio),
            "rule_care_process_max_ratio": float(rule_care_process_max_ratio),
            "rule_care_process_penalty": float(rule_care_process_penalty),
            "rule_anchor_min_unique": int(rule_anchor_min_unique),
            "rule_anchor_max_share": float(rule_anchor_max_share),
            "rule_mechanism_min_unique": int(rule_mechanism_min_unique),
            "rule_significance_weight": float(rule_significance_weight),
            "rule_significance_prior_count": float(rule_significance_prior_count),
            "rule_significance_bootstrap_samples": int(rule_significance_bootstrap_samples),
            "rule_significance_random_state": None if rule_significance_random_state is None else int(rule_significance_random_state),
            "rule_significance_max_q_value": float(rule_significance_max_q_value),
            "statistical_null_permutations": int(max(0, int(statistical_null_permutations))),
            "statistical_null_random_seed": int(statistical_null_random_seed),
            "statistical_null_top_k": int(max(1, int(statistical_null_top_k))),
            "actionability_top_k": int(max(1, int(actionability_top_k))),
            "actionability_min_lead_hours": float(max(0.0, float(actionability_min_lead_hours))),
            "program_leadtime_diversity_weight": float(program_leadtime_diversity_weight),
            "program_leadtime_diversity_gap_hours": float(program_leadtime_diversity_gap_hours),
            "program_mechanism_diversity_weight": float(program_mechanism_diversity_weight),
            "program_mechanism_max_same_count": int(program_mechanism_max_same_count),
            "program_care_process_penalty": float(program_care_process_penalty),
            "program_redundancy_penalty_weight": float(program_redundancy_penalty_weight),
            "program_coverage_target": int(program_coverage_target),
            "program_coverage_target_weight": float(program_coverage_target_weight),
            "program_anchor_diversity_weight": float(program_anchor_diversity_weight),
            "program_anchor_candidate_top_n": int(program_anchor_candidate_top_n),
            "program_anchor_candidate_quality_ratio": float(program_anchor_candidate_quality_ratio),
            "program_mechanism_candidate_top_n": int(program_mechanism_candidate_top_n),
            "program_mechanism_min_unique": int(program_mechanism_min_unique),
            "program_mechanism_group_mode": str(program_mechanism_group_mode),
            "program_mechanism_feature_score_floor_ratio": float(program_mechanism_feature_score_floor_ratio),
            "program_score_calibration_mode": str(program_score_calibration_mode),
            "program_score_calibration_quantile_low": float(program_score_calibration_quantile_low),
            "program_score_calibration_quantile_high": float(program_score_calibration_quantile_high),
            "program_score_calibration_target_low": float(program_score_calibration_target_low),
            "program_score_calibration_target_high": float(program_score_calibration_target_high),
            "program_score_calibration_blend_weight": float(program_score_calibration_blend_weight),
            "program_support_confidence_weight": float(program_support_confidence_weight),
            "program_support_confidence_z_score": float(program_support_confidence_z_score),
            "program_support_confidence_floor": float(program_support_confidence_floor),
            "program_support_confidence_exponent": float(program_support_confidence_exponent),
            "program_anchor_max_same_count": int(program_anchor_max_same_count),
            "program_anchor_novelty_target": int(program_anchor_novelty_target),
            "program_anchor_novelty_weight": float(program_anchor_novelty_weight),
            "program_anchor_novelty_rerank_epsilon": float(program_anchor_novelty_rerank_epsilon),
            "program_diversity_quality_floor_ratio": float(program_diversity_quality_floor_ratio),
            "program_topk_care_process_max_count": int(requested_topk_care_process_max_count),
            "program_topk_care_process_max_count_effective": int(effective_topk_care_process_max_count),
            "program_topk_care_process_window": int(effective_topk_care_process_window),
            "min_program_representative_score": float(min_program_representative_score),
            "program_top5_unique_groups_target": int(program_top5_unique_groups_target),
            "program_top5_fixed_prefix": int(program_top5_fixed_prefix),
            "program_top_window_unique_groups_target": int(program_top_window_unique_groups_target),
            "program_top_window_size": int(program_top_window_size),
            "program_top_window_fixed_prefix": int(program_top_window_fixed_prefix),
            "program_constrained_candidate_pool_size": int(program_constrained_candidate_pool_size),
            "program_constrained_quality_floor_ratio": float(program_constrained_quality_floor_ratio),
            "program_constrained_two_stage_swap_budget": int(program_constrained_two_stage_swap_budget),
            "program_constraint_autotune_steps": int(program_constraint_autotune_steps),
            "program_constraint_autotune_group_step": int(program_constraint_autotune_group_step),
            "program_constraint_autotune_pool_step": int(program_constraint_autotune_pool_step),
            "program_preselect_coverage_min_unique_groups": int(program_preselect_coverage_min_unique_groups),
            "program_preselect_coverage_window": int(program_preselect_coverage_window),
            "program_preselect_quality_floor_ratio": float(program_preselect_quality_floor_ratio),
            "transfer_baseline_top5_unique_groups": int(transfer_baseline_top5_unique_groups),
            "transfer_baseline_top10_unique_groups": int(transfer_baseline_top10_unique_groups),
            "transfer_baseline_top5_score_min": float(transfer_baseline_top5_score_min) if not math.isnan(float(transfer_baseline_top5_score_min)) else None,
            "transfer_max_drop_top5_unique_groups": int(transfer_max_drop_top5_unique_groups),
            "transfer_max_drop_top10_unique_groups": int(transfer_max_drop_top10_unique_groups),
            "transfer_max_drop_top5_score_min": float(transfer_max_drop_top5_score_min),
            "shadow_rule_blend_weight": float(shadow_rule_blend_weight),
            "shadow_rule_max_adjustment": float(shadow_rule_max_adjustment),
            "shadow_rule_quality_floor_ratio": float(shadow_rule_quality_floor_ratio),
            "shadow_rule_exclude_care_process": bool(shadow_rule_exclude_care_process),
            "shadow_rule_apply_adjusted_score": bool(shadow_rule_apply_adjusted_score),
            "shadow_predictive_blend_weight": float(shadow_predictive_blend_weight),
            "shadow_predictive_max_adjustment": float(shadow_predictive_max_adjustment),
            "shadow_predictive_quality_floor_ratio": float(shadow_predictive_quality_floor_ratio),
            "shadow_predictive_exclude_care_process": bool(shadow_predictive_exclude_care_process),
            "shadow_predictive_training_limit": int(shadow_predictive_training_limit),
            "shadow_predictive_seed": int(shadow_predictive_seed),
            "shadow_predictive_split_manifest": str(shadow_predictive_split_manifest) if shadow_predictive_split_manifest else None,
            "algorithm_ready_gate_profile": str(algorithm_ready_gate_profile),
            "algorithm_ready_baseline_rules_count": int(algorithm_ready_baseline_rules_count),
            "algorithm_ready_baseline_programs_count": int(algorithm_ready_baseline_programs_count),
            "algorithm_ready_rule_count_min_ratio": float(algorithm_ready_rule_count_min_ratio),
            "algorithm_ready_program_count_min_ratio": float(algorithm_ready_program_count_min_ratio),
            "algorithm_ready_rule_count_min_abs": int(algorithm_ready_rule_count_min_abs),
            "algorithm_ready_program_count_min_abs": int(algorithm_ready_program_count_min_abs),
            "algorithm_ready_top5_care_process_max": int(algorithm_ready_top5_care_process_max),
            "algorithm_ready_top5_max_same_anchor_max": int(algorithm_ready_top5_max_same_anchor_max),
            "algorithm_ready_top5_unique_anchor_min": int(algorithm_ready_top5_unique_anchor_min),
            "algorithm_ready_top5_unique_groups_min": int(algorithm_ready_top5_unique_groups_min),
            "algorithm_ready_top10_unique_groups_min": int(algorithm_ready_top10_unique_groups_min),
            "algorithm_ready_top5_score_min_threshold": float(algorithm_ready_top5_score_min_threshold),
        }
        rule_scoring_kwargs = {
            "exclude_missing_only": True,
            "exclude_sparse_measurement_dominant": exclude_sparse_measurement_dominant,
            "sparse_measurement_penalty": sparse_measurement_penalty,
            "sparse_measurement_penalty_exponent": sparse_measurement_penalty_exponent,
            "rule_anchor_saturation_penalty_weight": rule_anchor_saturation_penalty_weight,
            "rule_anchor_saturation_penalty_exponent": rule_anchor_saturation_penalty_exponent,
            "rule_mechanism_saturation_penalty_weight": rule_mechanism_saturation_penalty_weight,
            "rule_mechanism_saturation_penalty_exponent": rule_mechanism_saturation_penalty_exponent,
            "lead_time_quality_weight": rule_leadtime_quality_weight,
            "rule_support_reliability_weight": rule_support_reliability_weight,
            "rule_support_reliability_exponent": rule_support_reliability_exponent,
            "min_positive_count": min_rule_positive_count,
            "min_control_count": min_rule_control_count,
            "allow_zero_control_candidates": allow_zero_control_candidates,
            "enrichment_rate_prior_count": enrichment_rate_prior_count,
            "zero_control_rule_penalty": zero_control_rule_penalty,
            "min_contrastive_score": min_rule_contrastive_score,
            "rule_anchor_candidate_top_n": rule_anchor_candidate_top_n,
            "rule_mechanism_candidate_top_n": rule_mechanism_candidate_top_n,
            "rule_candidate_quality_floor_ratio": rule_candidate_quality_floor_ratio,
            "rule_care_process_max_ratio": rule_care_process_max_ratio,
            "rule_care_process_penalty": rule_care_process_penalty,
            "rule_anchor_min_unique": rule_anchor_min_unique,
            "rule_anchor_max_share": rule_anchor_max_share,
            "rule_mechanism_min_unique": rule_mechanism_min_unique,
            "significance_weight": rule_significance_weight,
            "significance_prior_count": rule_significance_prior_count,
            "significance_bootstrap_samples": rule_significance_bootstrap_samples,
            "significance_random_state": rule_significance_random_state,
            "significance_max_q_value": rule_significance_max_q_value,
            "return_candidate_pool_diagnostics": True,
        }
        scored = score_contrastive_rule_records(
            contrastive_inputs,
            **rule_scoring_kwargs,
        )
        if isinstance(scored, tuple):
            contrastive_rules, rule_candidate_pool = scored
        else:
            contrastive_rules = list(scored)
            rule_candidate_pool = {}
        (
            contrastive_rules,
            rule_candidate_pool,
            rule_low_coverage_recovery_meta,
        ) = _recover_rule_candidates_for_low_mechanism_coverage(
            contrastive_inputs=contrastive_inputs,
            ranked_rules=contrastive_rules,
            candidate_pool_diagnostics=rule_candidate_pool,
            scoring_kwargs=rule_scoring_kwargs,
            recovery_enabled=bool(rule_low_coverage_recovery_enabled),
            trigger_top_k=int(rule_low_coverage_recovery_trigger_top_k),
            min_unique_groups=int(rule_low_coverage_recovery_min_unique_groups),
            recovery_min_control_count=int(rule_low_coverage_recovery_min_control_count),
            recovery_allow_zero_control_candidates=bool(rule_low_coverage_recovery_allow_zero_control_candidates),
            recovery_zero_control_penalty=float(rule_low_coverage_recovery_zero_control_penalty),
            recovery_support_reliability_weight=float(rule_low_coverage_recovery_support_reliability_weight),
            recovery_support_reliability_exponent=float(rule_low_coverage_recovery_support_reliability_exponent),
            recovery_significance_max_q_value=float(rule_low_coverage_recovery_significance_max_q_value),
        )
        summary["rule_low_coverage_recovery"] = dict(rule_low_coverage_recovery_meta)
        consensus_weight = max(0.0, min(1.0, float(rule_consensus_boost_weight)))
        consensus_top_k = max(1, int(rule_consensus_top_k))
        consensus_rerank_epsilon = max(0.0, float(rule_consensus_rerank_epsilon))
        consensus_quality_floor_ratio = max(0.0, min(1.0, float(rule_consensus_quality_floor_ratio)))
        consensus_frequency = (
            _compute_rule_consensus_frequency(stability_runs, top_k=consensus_top_k)
            if stability_runs and consensus_weight > 0.0
            else {}
        )
        if consensus_frequency:
            contrastive_rules = _apply_rule_consensus_boost(
                contrastive_rules,
                consensus_frequency=consensus_frequency,
                boost_weight=consensus_weight,
                rerank_epsilon=consensus_rerank_epsilon,
                quality_floor_ratio=consensus_quality_floor_ratio,
            )
            summary["rule_consensus_rerank"] = {
                "enabled": True,
                "weight": float(consensus_weight),
                "top_k": int(consensus_top_k),
                "rerank_epsilon": float(consensus_rerank_epsilon),
                "quality_floor_ratio": float(consensus_quality_floor_ratio),
                "consensus_rule_count": int(len(consensus_frequency)),
            }
        elif consensus_weight > 0.0:
            summary["rule_consensus_rerank"] = {
                "enabled": False,
                "weight": float(consensus_weight),
                "top_k": int(consensus_top_k),
                "rerank_epsilon": float(consensus_rerank_epsilon),
                "quality_floor_ratio": float(consensus_quality_floor_ratio),
                "consensus_rule_count": 0,
            }

        anchor_diversity_weight = max(0.0, float(rule_anchor_diversity_weight))
        anchor_diversity_top_k = max(1, int(rule_anchor_diversity_top_k))
        anchor_diversity_rerank_epsilon = max(0.0, float(rule_anchor_diversity_rerank_epsilon))
        anchor_diversity_quality_floor_ratio = max(
            0.0,
            min(1.0, float(rule_anchor_diversity_quality_floor_ratio)),
        )
        if anchor_diversity_weight > 0.0:
            contrastive_rules = _apply_rule_anchor_diversity_rerank(
                contrastive_rules,
                diversity_weight=anchor_diversity_weight,
                top_k=anchor_diversity_top_k,
                rerank_epsilon=anchor_diversity_rerank_epsilon,
                quality_floor_ratio=anchor_diversity_quality_floor_ratio,
            )
            summary["rule_anchor_diversity_rerank"] = {
                "enabled": True,
                "weight": float(anchor_diversity_weight),
                "top_k": int(anchor_diversity_top_k),
                "rerank_epsilon": float(anchor_diversity_rerank_epsilon),
                "quality_floor_ratio": float(anchor_diversity_quality_floor_ratio),
            }
        else:
            summary["rule_anchor_diversity_rerank"] = {
                "enabled": False,
                "weight": float(anchor_diversity_weight),
                "top_k": int(anchor_diversity_top_k),
                "rerank_epsilon": float(anchor_diversity_rerank_epsilon),
                "quality_floor_ratio": float(anchor_diversity_quality_floor_ratio),
            }

        mechanism_diversity_weight = max(0.0, float(rule_mechanism_diversity_weight))
        mechanism_diversity_top_k = max(1, int(rule_mechanism_diversity_top_k))
        mechanism_diversity_rerank_epsilon = max(0.0, float(rule_mechanism_diversity_rerank_epsilon))
        mechanism_diversity_quality_floor_ratio = max(
            0.0,
            min(1.0, float(rule_mechanism_diversity_quality_floor_ratio)),
        )
        mechanism_diversity_target = max(0, int(rule_mechanism_diversity_target))
        mechanism_gate_meta = _evaluate_rule_mechanism_diversity_confidence_gate(
            contrastive_rules,
            top_k=mechanism_diversity_top_k,
            rerank_epsilon=mechanism_diversity_rerank_epsilon,
            quality_floor_ratio=mechanism_diversity_quality_floor_ratio,
            coverage_target=mechanism_diversity_target,
            min_reliable_support_ratio=float(
                mechanism_confidence_threshold_resolved.get(
                    "min_support_ratio",
                    rule_mechanism_diversity_confidence_min_support_ratio,
                )
            ),
            min_reliable_rules=int(
                mechanism_confidence_threshold_resolved.get(
                    "min_reliable_rules",
                    rule_mechanism_diversity_confidence_min_reliable_rules,
                )
            ),
            min_expected_coverage_gain=int(
                mechanism_confidence_threshold_resolved.get(
                    "min_expected_gain",
                    rule_mechanism_diversity_confidence_min_expected_gain,
                )
            ),
        )
        mechanism_gate_enabled = bool(rule_mechanism_diversity_confidence_gate_enabled)
        mechanism_rerank_allowed = bool(
            mechanism_diversity_weight > 0.0
            and (not mechanism_gate_enabled or bool(mechanism_gate_meta.get("enabled", False)))
        )
        if mechanism_rerank_allowed:
            contrastive_rules, mechanism_rerank_meta = _apply_rule_mechanism_diversity_rerank(
                contrastive_rules,
                diversity_weight=mechanism_diversity_weight,
                top_k=mechanism_diversity_top_k,
                rerank_epsilon=mechanism_diversity_rerank_epsilon,
                quality_floor_ratio=mechanism_diversity_quality_floor_ratio,
                coverage_target=mechanism_diversity_target,
                two_stage_swap_budget=rule_mechanism_diversity_two_stage_swap_budget,
                stage1_min_support_ratio=rule_mechanism_diversity_stage1_min_support_ratio,
                return_metadata=True,
            )
            summary["rule_mechanism_diversity_rerank"] = {
                "enabled": True,
                "weight": float(mechanism_diversity_weight),
                "top_k": int(mechanism_diversity_top_k),
                "rerank_epsilon": float(mechanism_diversity_rerank_epsilon),
                "quality_floor_ratio": float(mechanism_diversity_quality_floor_ratio),
                "coverage_target": int(mechanism_diversity_target),
                "two_stage_swap_budget": int(rule_mechanism_diversity_two_stage_swap_budget),
                "stage1_min_support_ratio": float(rule_mechanism_diversity_stage1_min_support_ratio),
                "confidence_gate_enabled": bool(mechanism_gate_enabled),
                "confidence_threshold_source": str(mechanism_confidence_threshold_resolved.get("source", "runtime_defaults")),
                "confidence_threshold_domain": str(mechanism_confidence_domain),
                "confidence_gate": dict(mechanism_gate_meta),
                **dict(mechanism_rerank_meta),
            }
        else:
            _, mechanism_rerank_meta = _apply_rule_mechanism_diversity_rerank(
                contrastive_rules,
                diversity_weight=mechanism_diversity_weight,
                top_k=mechanism_diversity_top_k,
                rerank_epsilon=mechanism_diversity_rerank_epsilon,
                quality_floor_ratio=mechanism_diversity_quality_floor_ratio,
                coverage_target=mechanism_diversity_target,
                two_stage_swap_budget=rule_mechanism_diversity_two_stage_swap_budget,
                stage1_min_support_ratio=rule_mechanism_diversity_stage1_min_support_ratio,
                return_metadata=True,
            )
            disabled_reason = "weight_is_zero"
            if mechanism_diversity_weight > 0.0 and mechanism_gate_enabled:
                disabled_reason = str(mechanism_gate_meta.get("reason", "confidence_gate_blocked"))
            summary["rule_mechanism_diversity_rerank"] = {
                "enabled": False,
                "reason": disabled_reason,
                "weight": float(mechanism_diversity_weight),
                "top_k": int(mechanism_diversity_top_k),
                "rerank_epsilon": float(mechanism_diversity_rerank_epsilon),
                "quality_floor_ratio": float(mechanism_diversity_quality_floor_ratio),
                "coverage_target": int(mechanism_diversity_target),
                "two_stage_swap_budget": int(rule_mechanism_diversity_two_stage_swap_budget),
                "stage1_min_support_ratio": float(rule_mechanism_diversity_stage1_min_support_ratio),
                "confidence_gate_enabled": bool(mechanism_gate_enabled),
                "confidence_threshold_source": str(mechanism_confidence_threshold_resolved.get("source", "runtime_defaults")),
                "confidence_threshold_domain": str(mechanism_confidence_domain),
                "confidence_gate": dict(mechanism_gate_meta),
                **dict(mechanism_rerank_meta),
            }

        shadow_training_limit = int(shadow_predictive_training_limit)
        if shadow_training_limit <= 0:
            shadow_training_limit = int(limit) if int(limit) > 0 else int(len(summary.get("runs", [])))
        shadow_training_episode_ids = _load_shadow_training_episode_ids(shadow_predictive_split_manifest)
        shadow_feature_weights: dict[str, float] = {}
        shadow_rule_weight = max(0.0, min(1.0, float(shadow_rule_blend_weight)))
        shadow_program_weight = max(0.0, min(1.0, float(shadow_predictive_blend_weight)))
        if shadow_rule_weight > 0.0 or shadow_program_weight > 0.0:
            shadow_feature_weights = _fit_shadow_predictive_feature_weights(
                input_path,
                limit=max(0, int(shadow_training_limit)),
                seed=int(shadow_predictive_seed),
                training_episode_ids=shadow_training_episode_ids,
            )

        if shadow_rule_weight > 0.0:
            if shadow_feature_weights:
                contrastive_rules = _apply_shadow_predictive_rule_rerank(
                    contrastive_rules,
                    feature_weights=shadow_feature_weights,
                    blend_weight=shadow_rule_weight,
                    max_adjustment=float(shadow_rule_max_adjustment),
                    quality_floor_ratio=float(shadow_rule_quality_floor_ratio),
                    exclude_care_process=bool(shadow_rule_exclude_care_process),
                    apply_adjusted_score=bool(shadow_rule_apply_adjusted_score),
                )
                summary["shadow_predictive_rule_rerank"] = {
                    "enabled": True,
                    "blend_weight": float(shadow_rule_weight),
                    "max_adjustment": float(max(0.0, float(shadow_rule_max_adjustment))),
                    "quality_floor_ratio": float(max(0.0, min(1.0, float(shadow_rule_quality_floor_ratio)))),
                    "exclude_care_process": bool(shadow_rule_exclude_care_process),
                    "apply_adjusted_score": bool(shadow_rule_apply_adjusted_score),
                    "training_limit": int(max(0, int(shadow_training_limit))),
                    "seed": int(shadow_predictive_seed),
                    "training_episode_id_count": int(len(shadow_training_episode_ids)),
                    "split_manifest": str(shadow_predictive_split_manifest) if shadow_predictive_split_manifest else None,
                    "feature_weight_count": int(len(shadow_feature_weights)),
                }
            else:
                summary["shadow_predictive_rule_rerank"] = {
                    "enabled": False,
                    "reason": "feature_weight_fit_failed_or_insufficient_data",
                    "blend_weight": float(shadow_rule_weight),
                    "max_adjustment": float(max(0.0, float(shadow_rule_max_adjustment))),
                    "quality_floor_ratio": float(max(0.0, min(1.0, float(shadow_rule_quality_floor_ratio)))),
                    "exclude_care_process": bool(shadow_rule_exclude_care_process),
                    "apply_adjusted_score": bool(shadow_rule_apply_adjusted_score),
                    "training_limit": int(max(0, int(shadow_training_limit))),
                    "seed": int(shadow_predictive_seed),
                    "training_episode_id_count": int(len(shadow_training_episode_ids)),
                    "split_manifest": str(shadow_predictive_split_manifest) if shadow_predictive_split_manifest else None,
                }
        else:
            summary["shadow_predictive_rule_rerank"] = {
                "enabled": False,
                "reason": "blend_weight_zero",
                "blend_weight": float(shadow_rule_weight),
                "max_adjustment": float(max(0.0, float(shadow_rule_max_adjustment))),
                "quality_floor_ratio": float(max(0.0, min(1.0, float(shadow_rule_quality_floor_ratio)))),
                "exclude_care_process": bool(shadow_rule_exclude_care_process),
                "apply_adjusted_score": bool(shadow_rule_apply_adjusted_score),
            }

        summary["contrastive_rules"] = contrastive_rules
        summary["contrastive_rule_candidate_pool"] = rule_candidate_pool
        summary["statistical_evidence"] = {
            "rule_level": _summarize_statistical_evidence(
                summary["contrastive_rules"],
                top_k=max(1, int(statistical_null_top_k)),
            ),
            "null_benchmark": _build_statistical_null_benchmark(
                contrastive_inputs=contrastive_inputs,
                scoring_kwargs=rule_scoring_kwargs,
                observed_rules=summary["contrastive_rules"],
                permutation_count=max(0, int(statistical_null_permutations)),
                random_seed=int(statistical_null_random_seed),
                top_k=max(1, int(statistical_null_top_k)),
            ),
        }
        summary["contrastive_programs"] = consolidate_contrastive_programs(
            summary["contrastive_rules"],
            lead_time_diversity_weight=program_leadtime_diversity_weight,
            lead_time_diversity_gap_hours=program_leadtime_diversity_gap_hours,
            program_mechanism_diversity_weight=program_mechanism_diversity_weight,
            program_mechanism_max_same_count=program_mechanism_max_same_count,
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
            program_anchor_novelty_target=program_anchor_novelty_target,
            program_anchor_novelty_weight=program_anchor_novelty_weight,
            program_anchor_novelty_rerank_epsilon=program_anchor_novelty_rerank_epsilon,
            program_diversity_quality_floor_ratio=program_diversity_quality_floor_ratio,
            program_topk_care_process_max_count=effective_topk_care_process_max_count,
            program_topk_care_process_window=effective_topk_care_process_window,
            min_program_representative_score=min_program_representative_score,
        )
        shadow_weight = shadow_program_weight
        if shadow_weight > 0.0:
            if shadow_feature_weights:
                summary["contrastive_programs"] = _apply_shadow_predictive_program_rerank(
                    summary["contrastive_programs"],
                    feature_weights=shadow_feature_weights,
                    blend_weight=shadow_weight,
                    max_adjustment=float(shadow_predictive_max_adjustment),
                    quality_floor_ratio=float(shadow_predictive_quality_floor_ratio),
                    exclude_care_process=bool(shadow_predictive_exclude_care_process),
                )
                summary["shadow_predictive_rerank"] = {
                    "enabled": True,
                    "blend_weight": float(shadow_weight),
                    "max_adjustment": float(max(0.0, float(shadow_predictive_max_adjustment))),
                    "quality_floor_ratio": float(max(0.0, min(1.0, float(shadow_predictive_quality_floor_ratio)))),
                    "exclude_care_process": bool(shadow_predictive_exclude_care_process),
                    "training_limit": int(max(0, int(shadow_training_limit))),
                    "seed": int(shadow_predictive_seed),
                    "training_episode_id_count": int(len(shadow_training_episode_ids)),
                    "split_manifest": str(shadow_predictive_split_manifest) if shadow_predictive_split_manifest else None,
                    "feature_weight_count": int(len(shadow_feature_weights)),
                    "feature_weights": {
                        key: float(shadow_feature_weights[key])
                        for key in sorted(shadow_feature_weights)
                    },
                }
            else:
                summary["shadow_predictive_rerank"] = {
                    "enabled": False,
                    "reason": "feature_weight_fit_failed_or_insufficient_data",
                    "blend_weight": float(shadow_weight),
                    "exclude_care_process": bool(shadow_predictive_exclude_care_process),
                    "training_limit": int(max(0, int(shadow_training_limit))),
                    "seed": int(shadow_predictive_seed),
                    "training_episode_id_count": int(len(shadow_training_episode_ids)),
                    "split_manifest": str(shadow_predictive_split_manifest) if shadow_predictive_split_manifest else None,
                }
        else:
            summary["shadow_predictive_rerank"] = {
                "enabled": False,
                "reason": "blend_weight_zero",
                "blend_weight": float(shadow_weight),
                "exclude_care_process": bool(shadow_predictive_exclude_care_process),
            }

        summary["contrastive_programs"] = _apply_program_mechanism_group_mode(
            summary["contrastive_programs"],
            mode=str(program_mechanism_group_mode),
            feature_score_floor_ratio=float(program_mechanism_feature_score_floor_ratio),
        )
        summary["contrastive_programs"], score_calibration_meta = _apply_program_score_calibration(
            summary["contrastive_programs"],
            mode=str(program_score_calibration_mode),
            quantile_low=float(program_score_calibration_quantile_low),
            quantile_high=float(program_score_calibration_quantile_high),
            target_low=float(program_score_calibration_target_low),
            target_high=float(program_score_calibration_target_high),
            blend_weight=float(program_score_calibration_blend_weight),
        )
        summary["program_score_calibration"] = score_calibration_meta
        summary["contrastive_programs"], support_confidence_meta = _apply_program_support_confidence_adjustment(
            summary["contrastive_programs"],
            weight=float(program_support_confidence_weight),
            z_score=float(program_support_confidence_z_score),
            floor=float(program_support_confidence_floor),
            exponent=float(program_support_confidence_exponent),
        )
        summary["program_support_confidence_adjustment"] = support_confidence_meta

        preselected_programs, preselect_meta = _enforce_program_mechanism_coverage_lower_bound(
            summary["contrastive_programs"],
            window_size=int(program_preselect_coverage_window),
            min_unique_groups=int(program_preselect_coverage_min_unique_groups),
            quality_floor_ratio=float(program_preselect_quality_floor_ratio),
        )
        summary["contrastive_programs"] = preselected_programs
        summary["program_preselect_coverage"] = preselect_meta

        def _evaluate_program_gate_snapshot(ranked_programs: list[dict]) -> tuple[dict, dict]:
            local_qc = _summarize_contrastive_qc(
                contrastive_inputs,
                ranked_programs,
                top_k=5,
            )
            local_summary = {
                "contrastive_rules": summary["contrastive_rules"],
                "contrastive_programs": ranked_programs,
                "contrastive_qc": local_qc,
            }
            local_gates = _evaluate_algorithm_ready_gates(
                local_summary,
                gate_profile=algorithm_ready_gate_profile,
                baseline_rules_count=algorithm_ready_baseline_rules_count,
                baseline_programs_count=algorithm_ready_baseline_programs_count,
                rule_count_min_ratio=algorithm_ready_rule_count_min_ratio,
                program_count_min_ratio=algorithm_ready_program_count_min_ratio,
                rule_count_min_abs=algorithm_ready_rule_count_min_abs,
                program_count_min_abs=algorithm_ready_program_count_min_abs,
                top5_care_process_max=algorithm_ready_top5_care_process_max,
                top5_max_same_anchor_max=algorithm_ready_top5_max_same_anchor_max,
                top5_unique_anchor_min=algorithm_ready_top5_unique_anchor_min,
                top5_unique_groups_min=algorithm_ready_top5_unique_groups_min,
                top10_unique_groups_min=algorithm_ready_top10_unique_groups_min,
                top5_score_min_threshold=algorithm_ready_top5_score_min_threshold,
            )
            return local_qc, local_gates

        selector_kwargs = {
            "top_window_size": int(program_top_window_size),
            "top_window_unique_groups_target": int(program_top_window_unique_groups_target),
            "top_window_fixed_prefix": int(program_top_window_fixed_prefix),
            "top5_unique_groups_target": int(program_top5_unique_groups_target),
            "top5_fixed_prefix": int(program_top5_fixed_prefix),
            "topk_care_process_window": int(effective_topk_care_process_window),
            "topk_care_process_max_count": int(effective_topk_care_process_max_count),
            "top5_max_same_anchor_count": int(algorithm_ready_top5_max_same_anchor_max),
            "candidate_pool_size": int(program_constrained_candidate_pool_size),
            "quality_floor_ratio": float(program_constrained_quality_floor_ratio),
            "two_stage_swap_budget": int(program_constrained_two_stage_swap_budget),
        }
        constrained_programs, selector_meta = _select_topk_with_constraints(
            summary["contrastive_programs"],
            **selector_kwargs,
        )
        constrained_qc, constrained_gates = _evaluate_program_gate_snapshot(constrained_programs)
        selector_search_trace: list[dict] = []
        best_programs = constrained_programs
        best_selector_meta = selector_meta
        best_qc = constrained_qc
        best_gates = constrained_gates

        def _selector_candidate_score(qc_row: dict, gate_row: dict, selector_row: dict) -> tuple:
            mechanism = dict(qc_row.get("mechanism_diversity", {}))
            quality = dict(gate_row.get("quality", {}))
            quality_checks = dict(quality.get("checks", {}))
            transfer_target = _evaluate_transfer_target_against_baseline(
                mechanism,
                baseline_top5_unique_groups=int(transfer_baseline_top5_unique_groups),
                baseline_top10_unique_groups=int(transfer_baseline_top10_unique_groups),
                baseline_top5_score_min=float(transfer_baseline_top5_score_min),
                max_drop_top5_unique_groups=int(transfer_max_drop_top5_unique_groups),
                max_drop_top10_unique_groups=int(transfer_max_drop_top10_unique_groups),
                max_drop_top5_score_min=float(transfer_max_drop_top5_score_min),
            )
            pass_count = int(sum(1 for check in quality_checks.values() if bool(check.get("pass", False))))
            transfer_enabled = bool(transfer_target.get("enabled", False))
            transfer_pass = bool(transfer_target.get("pass", False))
            return (
                int(transfer_pass) if transfer_enabled else -1,
                int(bool(gate_row.get("overall_pass", False))),
                int(bool(selector_row.get("all_constraints_pass", False))),
                pass_count,
                int(transfer_target.get("delta_top10_unique_groups", 0)),
                int(transfer_target.get("delta_top5_unique_groups", 0)),
                float(transfer_target.get("delta_top5_score_min", 0.0)),
                int(mechanism.get("top5_unique_groups", 0)),
                int(mechanism.get("top10_unique_groups", 0)),
                -int(mechanism.get("top5_care_process_count", 0)),
                -int(mechanism.get("top5_max_same_anchor_count", 0)),
                float(mechanism.get("top5_score_min", 0.0)),
            )

        autotune_steps = max(0, int(program_constraint_autotune_steps))
        autotune_group_step = max(1, int(program_constraint_autotune_group_step))
        autotune_pool_step = max(1, int(program_constraint_autotune_pool_step))
        initial_transfer_target = _evaluate_transfer_target_against_baseline(
            dict(best_qc.get("mechanism_diversity", {})),
            baseline_top5_unique_groups=int(transfer_baseline_top5_unique_groups),
            baseline_top10_unique_groups=int(transfer_baseline_top10_unique_groups),
            baseline_top5_score_min=float(transfer_baseline_top5_score_min),
            max_drop_top5_unique_groups=int(transfer_max_drop_top5_unique_groups),
            max_drop_top10_unique_groups=int(transfer_max_drop_top10_unique_groups),
            max_drop_top5_score_min=float(transfer_max_drop_top5_score_min),
        )
        should_autotune = bool(
            autotune_steps > 0
            and (
                (not bool(best_gates.get("overall_pass", False)))
                or (
                    bool(initial_transfer_target.get("enabled", False))
                    and (not bool(initial_transfer_target.get("pass", False)))
                )
            )
        )
        if should_autotune:
            base_selector_kwargs = dict(selector_kwargs)
            for tune_round in range(autotune_steps):
                failure = dict(best_gates.get("failure_attribution", {}))
                failed = set([str(v) for v in list(failure.get("quality_failures", [])) + list(failure.get("reference_count_failures", []))])
                current_transfer = _evaluate_transfer_target_against_baseline(
                    dict(best_qc.get("mechanism_diversity", {})),
                    baseline_top5_unique_groups=int(transfer_baseline_top5_unique_groups),
                    baseline_top10_unique_groups=int(transfer_baseline_top10_unique_groups),
                    baseline_top5_score_min=float(transfer_baseline_top5_score_min),
                    max_drop_top5_unique_groups=int(transfer_max_drop_top5_unique_groups),
                    max_drop_top10_unique_groups=int(transfer_max_drop_top10_unique_groups),
                    max_drop_top5_score_min=float(transfer_max_drop_top5_score_min),
                )
                if bool(current_transfer.get("enabled", False)) and not bool(current_transfer.get("pass", False)):
                    if int(current_transfer.get("delta_top10_unique_groups", 0)) < 0:
                        failed.add("transfer_top10_drop")
                    if int(current_transfer.get("delta_top5_unique_groups", 0)) < 0:
                        failed.add("transfer_top5_drop")
                    if float(current_transfer.get("delta_top5_score_min", 0.0)) < 0.0:
                        failed.add("transfer_top5_score_drop")
                trial_kwargs: list[tuple[str, dict]] = []
                if "top5_unique_groups_ge_min" in failed or "transfer_top5_drop" in failed:
                    cfg = dict(base_selector_kwargs)
                    cfg["top5_unique_groups_target"] = int(cfg["top5_unique_groups_target"]) + autotune_group_step
                    trial_kwargs.append(("raise_top5_groups_target", cfg))
                if (
                    "top10_unique_groups_ge_min" in failed
                    or "top5_unique_groups_ge_min" in failed
                    or "transfer_top10_drop" in failed
                ):
                    cfg = dict(base_selector_kwargs)
                    cfg["top_window_unique_groups_target"] = int(cfg["top_window_unique_groups_target"]) + autotune_group_step
                    cfg["candidate_pool_size"] = max(
                        int(cfg.get("candidate_pool_size", 0)),
                        int(base_selector_kwargs.get("top_window_size", 10)) + autotune_pool_step,
                    )
                    trial_kwargs.append(("raise_top_window_groups_target", cfg))
                if "transfer_top10_drop" in failed:
                    cfg = dict(base_selector_kwargs)
                    cfg["quality_floor_ratio"] = max(0.0, float(cfg.get("quality_floor_ratio", 0.0)) - 0.15)
                    cfg["candidate_pool_size"] = int(cfg.get("candidate_pool_size", 0)) + (autotune_pool_step * 4)
                    cfg["two_stage_swap_budget"] = int(cfg.get("two_stage_swap_budget", 0)) + 40
                    trial_kwargs.append(("relax_quality_floor_for_top10", cfg))
                if "transfer_top5_score_drop" in failed:
                    cfg = dict(base_selector_kwargs)
                    cfg["quality_floor_ratio"] = min(1.0, float(cfg.get("quality_floor_ratio", 0.0)) + 0.05)
                    trial_kwargs.append(("raise_quality_floor_ratio", cfg))
                cfg = dict(base_selector_kwargs)
                cfg["candidate_pool_size"] = int(cfg.get("candidate_pool_size", 0)) + autotune_pool_step
                trial_kwargs.append(("expand_candidate_pool", cfg))
                cfg = dict(base_selector_kwargs)
                cfg["two_stage_swap_budget"] = int(cfg.get("two_stage_swap_budget", 0)) + 10
                trial_kwargs.append(("increase_swap_budget", cfg))

                best_trial = None
                best_trial_score = None
                for trial_name, trial_cfg in trial_kwargs:
                    trial_programs, trial_meta = _select_topk_with_constraints(summary["contrastive_programs"], **trial_cfg)
                    trial_qc, trial_gates = _evaluate_program_gate_snapshot(trial_programs)
                    trial_transfer = _evaluate_transfer_target_against_baseline(
                        dict(trial_qc.get("mechanism_diversity", {})),
                        baseline_top5_unique_groups=int(transfer_baseline_top5_unique_groups),
                        baseline_top10_unique_groups=int(transfer_baseline_top10_unique_groups),
                        baseline_top5_score_min=float(transfer_baseline_top5_score_min),
                        max_drop_top5_unique_groups=int(transfer_max_drop_top5_unique_groups),
                        max_drop_top10_unique_groups=int(transfer_max_drop_top10_unique_groups),
                        max_drop_top5_score_min=float(transfer_max_drop_top5_score_min),
                    )
                    trial_score = _selector_candidate_score(trial_qc, trial_gates, trial_meta)
                    selector_search_trace.append(
                        {
                            "round": int(tune_round + 1),
                            "trial": trial_name,
                            "params": dict(trial_cfg),
                            "score": list(trial_score),
                            "overall_pass": bool(trial_gates.get("overall_pass", False)),
                            "selector_all_constraints_pass": bool(trial_meta.get("all_constraints_pass", False)),
                            "transfer_target": trial_transfer,
                        }
                    )
                    if best_trial_score is None or trial_score > best_trial_score:
                        best_trial_score = trial_score
                        best_trial = (trial_programs, trial_meta, trial_qc, trial_gates, trial_cfg)
                if best_trial is None:
                    break
                current_score = _selector_candidate_score(best_qc, best_gates, best_selector_meta)
                if best_trial_score is not None and best_trial_score > current_score:
                    best_programs, best_selector_meta, best_qc, best_gates, accepted_cfg = best_trial
                    base_selector_kwargs = dict(accepted_cfg)
                else:
                    break
                latest_transfer = _evaluate_transfer_target_against_baseline(
                    dict(best_qc.get("mechanism_diversity", {})),
                    baseline_top5_unique_groups=int(transfer_baseline_top5_unique_groups),
                    baseline_top10_unique_groups=int(transfer_baseline_top10_unique_groups),
                    baseline_top5_score_min=float(transfer_baseline_top5_score_min),
                    max_drop_top5_unique_groups=int(transfer_max_drop_top5_unique_groups),
                    max_drop_top10_unique_groups=int(transfer_max_drop_top10_unique_groups),
                    max_drop_top5_score_min=float(transfer_max_drop_top5_score_min),
                )
                if bool(best_gates.get("overall_pass", False)) and (
                    (not bool(latest_transfer.get("enabled", False)))
                    or bool(latest_transfer.get("pass", False))
                ):
                    break

        if selector_search_trace:
            final_transfer_target = _evaluate_transfer_target_against_baseline(
                dict(best_qc.get("mechanism_diversity", {})),
                baseline_top5_unique_groups=int(transfer_baseline_top5_unique_groups),
                baseline_top10_unique_groups=int(transfer_baseline_top10_unique_groups),
                baseline_top5_score_min=float(transfer_baseline_top5_score_min),
                max_drop_top5_unique_groups=int(transfer_max_drop_top5_unique_groups),
                max_drop_top10_unique_groups=int(transfer_max_drop_top10_unique_groups),
                max_drop_top5_score_min=float(transfer_max_drop_top5_score_min),
            )
            best_selector_meta["autotune"] = {
                "enabled": True,
                "steps_requested": int(autotune_steps),
                "group_step": int(autotune_group_step),
                "pool_step": int(autotune_pool_step),
                "attempts": selector_search_trace,
                "final_overall_pass": bool(best_gates.get("overall_pass", False)),
                "final_transfer_target": final_transfer_target,
            }
        else:
            final_transfer_target = _evaluate_transfer_target_against_baseline(
                dict(best_qc.get("mechanism_diversity", {})),
                baseline_top5_unique_groups=int(transfer_baseline_top5_unique_groups),
                baseline_top10_unique_groups=int(transfer_baseline_top10_unique_groups),
                baseline_top5_score_min=float(transfer_baseline_top5_score_min),
                max_drop_top5_unique_groups=int(transfer_max_drop_top5_unique_groups),
                max_drop_top10_unique_groups=int(transfer_max_drop_top10_unique_groups),
                max_drop_top5_score_min=float(transfer_max_drop_top5_score_min),
            )
            best_selector_meta["autotune"] = {
                "enabled": bool(autotune_steps > 0),
                "steps_requested": int(autotune_steps),
                "group_step": int(autotune_group_step),
                "pool_step": int(autotune_pool_step),
                "attempts": [],
                "final_overall_pass": bool(best_gates.get("overall_pass", False)),
                "final_transfer_target": final_transfer_target,
            }

        summary["contrastive_programs"] = best_programs
        summary["program_constrained_selection"] = best_selector_meta
        # Backward-compatibility placeholders for downstream readers.
        summary["program_top_window_repair"] = {"enabled": False, "reason": "replaced_by_program_constrained_selection"}
        summary["program_top5_repair"] = {"enabled": False, "reason": "replaced_by_program_constrained_selection"}
        summary["program_topk_care_process_cap_repair"] = {"enabled": False, "reason": "replaced_by_program_constrained_selection"}
        summary["program_top5_repair_post_care_cap"] = {"enabled": False, "reason": "replaced_by_program_constrained_selection"}
        summary["contrastive_program_models"] = build_program_models(
            summary["contrastive_rules"],
            max_models=max(1, len(summary["contrastive_programs"])) if summary["contrastive_programs"] else 1,
        )
        summary["actionability_bridge"] = _summarize_actionability_bridge(
            summary["contrastive_rules"],
            summary["contrastive_programs"],
            top_k=max(1, int(actionability_top_k)),
            min_lead_hours=max(0.0, float(actionability_min_lead_hours)),
        )
        summary["contrastive_qc"] = best_qc
        summary["algorithm_ready_gates"] = best_gates
    if stability_runs:
        stability_metrics = compute_rule_stability(stability_runs, top_k=int(stability_top_k))
        robustness_summary = summarize_robustness_matrix(stability_runs, top_k=int(stability_top_k))
        stability_gates = evaluate_stability_gates(
            stability_metrics,
            {
                "topk_rule_jaccard_mean_min": float(stability_rule_jaccard_min),
                "topk_anchor_jaccard_mean_min": float(stability_anchor_jaccard_min),
                "topk_mechanism_jaccard_mean_min": float(stability_mechanism_jaccard_min),
            },
        )
        summary["stability"] = {
            "source_glob": str(stability_cache_glob),
            "metrics": stability_metrics,
            "gates": stability_gates,
            "pass": bool(stability_gates.get("all_pass", False)),
        }
        summary["robustness"] = robustness_summary
        summary["stability_metrics"] = stability_metrics
        summary["stability_gate_pass"] = bool(stability_gates.get("all_pass", False))
        summary["stability_tuning_suggestions"] = _suggest_stability_tuning(
            stability_metrics=stability_metrics,
            current_filters=summary.get("contrastive_filters", {}),
        )

    effective_config_payload = {
        "miner_config": summary.get("miner_config", {}),
        "contrastive_filters": summary.get("contrastive_filters", {}),
        "input_dir": str(input_path),
    }
    summary["run_fingerprint"] = {
        "git_commit": _resolve_git_commit(ROOT),
        "config_sha256": _compute_sha256_text(json.dumps(_json_safe(effective_config_payload), sort_keys=True)),
        "input_data_sha256": _compute_input_data_fingerprint(input_path),
        "episode_sampling_seed": None if episode_sampling_seed is None else int(episode_sampling_seed),
        "episode_sampling_unique_stays": bool(episode_sampling_unique_stays),
    }
    with open(output_path / "mimic_iv_summary.json", "w", encoding="utf-8") as handle:
        json.dump(_json_safe(summary), handle, indent=2)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the non-stationary rule miner on processed MIMIC-IV stays.")
    parser.add_argument("--input", required=True, help="Directory containing processed mimic_iv_stay_*.npy/json files.")
    parser.add_argument("--out-dir", default="artifacts/results/mimic_iv", help="Directory to write result JSON files.")
    parser.add_argument("--profile-json", default=None, help="Optional JSON file containing parser-argument overrides. Unknown keys are ignored.")
    parser.add_argument("--limit", type=int, default=0, help="Optional maximum number of stays to process.")
    parser.add_argument("--episode-sampling-seed", type=int, default=None, help="Optional seed for balanced episode subsampling when --limit is active.")
    parser.add_argument("--episode-sampling-unique-stays", action="store_true", help="When limiting episodes, prefer no repeated stay_id in the sampled subset (falls back to repeats only if unique stays are insufficient).")
    parser.add_argument("--filter-anchor-kind", default=None, help="Optional anchor_kind filter for shock episodes.")
    parser.add_argument("--filter-window-hours", type=int, default=None, help="Optional episode window-length filter in hours.")
    parser.add_argument("--disable-episode-type-balance", action="store_true", help="Use naive first-N limiting instead of balancing positive/control episodes.")
    parser.add_argument("--min-core-observed-channels", type=int, default=2, help="Minimum observed core channels for an episode.")
    parser.add_argument("--min-core-observed-points", type=int, default=6, help="Minimum observed points across core channels for an episode.")
    parser.add_argument("--min-episode-timesteps", type=int, default=1, help="Minimum required sequence length (timesteps); shorter episodes are skipped as low-information.")
    parser.add_argument("--min-rule-positive-count", type=int, default=2, help="Minimum positive-episode support for a scored rule.")
    parser.add_argument("--min-rule-control-count", type=int, default=1, help="Minimum control-episode support for a scored rule.")
    parser.add_argument("--allow-zero-control-candidates", action="store_true", help="Relax candidate generation to include rules with zero control support (score smoothed by prior/penalty knobs).")
    parser.add_argument("--enrichment-rate-prior-count", type=float, default=0.0, help="Additive prior count for positive/control rates when computing enrichment (0 keeps legacy formula).")
    parser.add_argument("--zero-control-rule-penalty", type=float, default=1.0, help="Additional multiplicative penalty [0,1] applied to zero-control candidate rules.")
    parser.add_argument("--miner-top-candidates", type=int, default=None, help="Override miner top_candidates to widen/narrow motif candidate generation.")
    parser.add_argument("--miner-top-m1", type=int, default=None, help="Override miner top_M1 (rule antecedent frontier width).")
    parser.add_argument("--miner-top-m2", type=int, default=None, help="Override miner top_M2 to widen/narrow stage-2 motif retention.")
    parser.add_argument("--motif-verify-top-k", type=int, default=None, help="Override motif verification top-k frontier (wider verification pass).")
    parser.add_argument("--motif-min-support-ratio", type=float, default=None, help="Override motif minimum support ratio; None keeps miner default.")
    parser.add_argument("--rule-top-k", type=int, default=None, help="Override rule miner top-k retention for candidate rule generation.")
    parser.add_argument("--hard-exclude-sparse-measurement-dominant", action="store_true", help="Hard-exclude sparse-measurement-dominant rules before scoring.")
    parser.add_argument("--include-sparse-measurement-dominant", action="store_true", help="Compatibility flag: keep sparse-measurement-dominant rules in scoring.")
    parser.add_argument("--sparse-measurement-penalty", type=float, default=0.35, help="Soft penalty factor [0,1] applied to sparse-measurement-dominant rules.")
    parser.add_argument("--sparse-measurement-penalty-exponent", type=float, default=2.0, help="Nonlinear exponent (>1 relaxes moderate sparse ratios while keeping high-ratio penalties).")
    parser.add_argument("--rule-anchor-saturation-penalty-weight", type=float, default=0.0, help="Soft penalty weight for saturated antecedent anchors in rule-level ranking.")
    parser.add_argument("--rule-anchor-saturation-penalty-exponent", type=float, default=1.0, help="Exponent controlling how aggressively anchor saturation is penalized.")
    parser.add_argument("--rule-mechanism-saturation-penalty-weight", type=float, default=0.0, help="Soft penalty weight for saturated consequent mechanism groups in rule-level ranking.")
    parser.add_argument("--rule-mechanism-saturation-penalty-exponent", type=float, default=1.0, help="Exponent controlling how aggressively mechanism saturation is penalized.")
    parser.add_argument("--rule-leadtime-quality-weight", type=float, default=0.0, help="Blend weight [0,1] for lead-time information quality in rule scoring.")
    parser.add_argument("--rule-support-reliability-weight", type=float, default=0.0, help="Blend weight [0,1] for penalizing low-support rule instability in scoring.")
    parser.add_argument("--rule-support-reliability-exponent", type=float, default=0.5, help="Nonlinear exponent for support reliability curve (higher => stronger low-support penalty).")
    parser.add_argument("--rule-consensus-boost-weight", type=float, default=0.0, help="Blend weight [0,1] for boosting rules repeatedly seen in cached stability runs.")
    parser.add_argument("--rule-consensus-top-k", type=int, default=10, help="Top-K window from cached runs used to compute per-rule consensus frequency.")
    parser.add_argument("--rule-consensus-rerank-epsilon", type=float, default=0.0, help="Consensus rerank safety window: only near-top rules within this score gap can receive consensus boost.")
    parser.add_argument("--rule-consensus-quality-floor-ratio", type=float, default=0.0, help="Consensus rerank quality floor ratio [0,1]: only rules above (top score * ratio) can receive consensus boost.")
    parser.add_argument("--rule-anchor-diversity-weight", type=float, default=0.0, help="Non-negative blend weight for anchor-diversity reranking in top rule window.")
    parser.add_argument("--rule-anchor-diversity-top-k", type=int, default=10, help="Top-K rule window targeted by anchor-diversity reranking.")
    parser.add_argument("--rule-anchor-diversity-rerank-epsilon", type=float, default=0.0, help="Anchor-diversity rerank safety window: only near-top rules within this score gap can be reranked.")
    parser.add_argument("--rule-anchor-diversity-quality-floor-ratio", type=float, default=0.0, help="Anchor-diversity quality floor ratio [0,1]: only rules above (top score * ratio) can be reranked.")
    parser.add_argument("--rule-mechanism-diversity-weight", type=float, default=0.0, help="Non-negative blend weight for mechanism-diversity reranking in top rule window.")
    parser.add_argument("--rule-mechanism-diversity-top-k", type=int, default=10, help="Top-K rule window targeted by mechanism-diversity reranking.")
    parser.add_argument("--rule-mechanism-diversity-rerank-epsilon", type=float, default=0.0, help="Mechanism-diversity rerank safety window: only near-top rules within this score gap can be reranked.")
    parser.add_argument("--rule-mechanism-diversity-quality-floor-ratio", type=float, default=0.0, help="Mechanism-diversity quality floor ratio [0,1]: only rules above (top score * ratio) can be reranked.")
    parser.add_argument("--rule-mechanism-diversity-target", type=int, default=0, help="Optional mechanism-group coverage target within top-K rule rerank window (0 disables explicit target).")
    parser.add_argument("--rule-mechanism-diversity-two-stage-swap-budget", type=int, default=0, help="Stage-1 feasible swap budget before mechanism rerank scoring (0 disables two-stage mode).")
    parser.add_argument("--rule-mechanism-diversity-stage1-min-support-ratio", type=float, default=0.0, help="Minimum support_ratio required for stage-1 feasible-swap candidates.")
    parser.add_argument("--rule-mechanism-diversity-confidence-gate-enabled", action="store_true", help="Enable confidence-aware gate before mechanism-diversity rerank; gate can auto-disable rerank when evidence is weak.")
    parser.add_argument("--rule-mechanism-diversity-confidence-min-support-ratio", type=float, default=0.05, help="Confidence gate minimum support_ratio for a rule to be considered reliable.")
    parser.add_argument("--rule-mechanism-diversity-confidence-min-reliable-rules", type=int, default=2, help="Confidence gate minimum reliable-rule count required in rerank-eligible pool.")
    parser.add_argument("--rule-mechanism-diversity-confidence-min-expected-gain", type=int, default=1, help="Confidence gate minimum expected mechanism-group coverage gain required to enable rerank.")
    parser.add_argument("--rule-mechanism-diversity-confidence-domain-thresholds-path", type=str, default=None, help="Optional JSON path for domain-aware confidence thresholds keyed by input-domain (e.g., hirid/eicu/default).")
    parser.add_argument("--rule-low-coverage-recovery-enabled", action="store_true", help="Enable fallback rule-candidate recovery when top-k mechanism coverage is below threshold.")
    parser.add_argument("--rule-low-coverage-recovery-trigger-top-k", type=int, default=10, help="Top-K window used to detect low mechanism coverage for fallback recovery.")
    parser.add_argument("--rule-low-coverage-recovery-min-unique-groups", type=int, default=0, help="Coverage trigger threshold; recovery runs when top-k unique program groups are below this value.")
    parser.add_argument("--rule-low-coverage-recovery-min-control-count", type=int, default=0, help="Fallback minimum control support used during low-coverage recovery scoring pass.")
    parser.add_argument("--rule-low-coverage-recovery-allow-zero-control-candidates", action="store_true", help="Allow zero-control candidates in fallback recovery scoring pass.")
    parser.add_argument("--rule-low-coverage-recovery-zero-control-penalty", type=float, default=0.75, help="Penalty applied to zero-control candidates during fallback recovery scoring pass.")
    parser.add_argument("--rule-low-coverage-recovery-support-reliability-weight", type=float, default=0.35, help="Support-reliability penalty weight applied in fallback recovery scoring pass.")
    parser.add_argument("--rule-low-coverage-recovery-support-reliability-exponent", type=float, default=1.0, help="Support-reliability penalty exponent applied in fallback recovery scoring pass.")
    parser.add_argument("--rule-low-coverage-recovery-significance-max-q-value", type=float, default=1.0, help="Relaxed maximum q-value [0,1] allowed during fallback recovery scoring pass.")
    parser.add_argument("--min-rule-contrastive-score", type=float, default=0.0, help="Minimum contrastive score required for a rule to be reported.")
    parser.add_argument("--rule-anchor-candidate-top-n", type=int, default=0, help="Candidate-pool relaxation: retain top-N rules per antecedent anchor before final rule sorting.")
    parser.add_argument("--rule-mechanism-candidate-top-n", type=int, default=0, help="Candidate-pool relaxation: retain top-N rules per mechanism group (mapped from consequent features).")
    parser.add_argument("--rule-candidate-quality-floor-ratio", type=float, default=0.0, help="Candidate-pool quality floor ratio [0,1]: retain rules with score >= (global top score * ratio).")
    parser.add_argument("--rule-care-process-max-ratio", type=float, default=1.0, help="Hard candidate filter [0,1]: exclude rules whose consequent care-process ratio exceeds this threshold.")
    parser.add_argument("--rule-care-process-penalty", type=float, default=1.0, help="Soft candidate penalty factor [0,1] blended by consequent care-process ratio.")
    parser.add_argument("--rule-anchor-min-unique", type=int, default=0, help="Candidate-pool coverage guardrail: minimum unique antecedent anchors retained when available.")
    parser.add_argument("--rule-anchor-max-share", type=float, default=1.0, help="Candidate-pool dominance guardrail [0,1]: max share allowed for the most frequent antecedent anchor.")
    parser.add_argument("--rule-mechanism-min-unique", type=int, default=0, help="Candidate-pool coverage guardrail: minimum unique consequent mechanism groups retained when available.")
    parser.add_argument("--rule-significance-weight", type=float, default=0.0, help="Blend weight [0,1] for statistical evidence in rule scoring.")
    parser.add_argument("--rule-significance-prior-count", type=float, default=0.5, help="Prior count used when computing smoothed rule effect sizes.")
    parser.add_argument("--rule-significance-bootstrap-samples", type=int, default=1000, help="Bootstrap sample count for rule effect-size confidence intervals.")
    parser.add_argument("--rule-significance-random-state", type=int, default=None, help="Optional random seed for rule significance bootstrap.")
    parser.add_argument("--rule-significance-max-q-value", type=float, default=1.0, help="Optional BH-FDR q-value filter threshold in [0,1]. 1.0 disables filtering.")
    parser.add_argument("--statistical-null-permutations", type=int, default=0, help="Permutation count for label-shuffle null benchmark (0 disables).")
    parser.add_argument("--statistical-null-random-seed", type=int, default=42, help="Random seed used for permutation null benchmark.")
    parser.add_argument("--statistical-null-top-k", type=int, default=10, help="Top-K window used in statistical evidence and null benchmark summaries.")
    parser.add_argument("--actionability-top-k", type=int, default=5, help="Top-K rules/programs used for actionability bridge metrics.")
    parser.add_argument("--actionability-min-lead-hours", type=float, default=2.0, help="Lead-time threshold (hours) used in actionability bridge metrics.")
    parser.add_argument("--program-leadtime-diversity-weight", type=float, default=0.0, help="Weight for promoting lead-time-diverse program selection.")
    parser.add_argument("--program-leadtime-diversity-gap-hours", type=float, default=1.0, help="Lead-time gap (hours) used when computing diversity bonus.")
    parser.add_argument("--program-mechanism-diversity-weight", type=float, default=0.3, help="Weight for promoting clinical mechanism diversity in program selection.")
    parser.add_argument("--program-mechanism-max-same-count", type=int, default=0, help="Hard top-K guardrail: max programs allowed per identical mechanism-group signature (0 disables).")
    parser.add_argument("--program-care-process-penalty", type=float, default=0.1, help="Soft penalty for care-process-driven programs during candidate ranking.")
    parser.add_argument("--program-redundancy-penalty-weight", type=float, default=0.0, help="Soft penalty weight for redundant same-anchor mechanism clusters during candidate ranking.")
    parser.add_argument("--program-coverage-target", type=int, default=0, help="Target number of mechanism groups to cover (soft target, no hard quota).")
    parser.add_argument("--program-coverage-target-weight", type=float, default=0.0, help="Soft bonus weight for candidates that add uncovered mechanism groups before target is reached.")
    parser.add_argument("--program-anchor-diversity-weight", type=float, default=0.0, help="Soft bonus weight for introducing new antecedent anchors in selected programs.")
    parser.add_argument("--program-anchor-candidate-top-n", type=int, default=0, help="Candidate-generation relaxation: keep top-N programs per anchor before final top-K assembly.")
    parser.add_argument("--program-anchor-candidate-quality-ratio", type=float, default=0.0, help="Candidate-generation quality ratio [0,1]: always keep candidates above (global top score * ratio).")
    parser.add_argument("--program-mechanism-candidate-top-n", type=int, default=0, help="Candidate-generation relaxation: keep top-N programs per mechanism-group bucket before final top-K assembly.")
    parser.add_argument("--program-mechanism-min-unique", type=int, default=0, help="Candidate-pool coverage guardrail: minimum unique program groups retained in program candidates when available.")
    parser.add_argument("--program-mechanism-group-mode", choices=sorted(PROGRAM_MECHANISM_GROUP_MODES), default="coarse", help="Mechanism-group coverage mode used for program-level diversity/gating metrics.")
    parser.add_argument("--program-mechanism-feature-score-floor-ratio", type=float, default=0.0, help="When using coarse_plus_feature mode, only programs with representative_score >= (top score * ratio) receive feature-level mechanism tags.")
    parser.add_argument("--program-score-calibration-mode", choices=sorted(PROGRAM_SCORE_CALIBRATION_MODES), default="none", help="Optional post-aggregation calibration for representative program scores.")
    parser.add_argument("--program-score-calibration-quantile-low", type=float, default=0.5, help="Observed low quantile used by affine program-score calibration.")
    parser.add_argument("--program-score-calibration-quantile-high", type=float, default=0.9, help="Observed high quantile used by affine program-score calibration.")
    parser.add_argument("--program-score-calibration-target-low", type=float, default=0.0, help="Target score mapped from low quantile when affine program-score calibration is enabled.")
    parser.add_argument("--program-score-calibration-target-high", type=float, default=0.0, help="Target score mapped from high quantile when affine program-score calibration is enabled.")
    parser.add_argument("--program-score-calibration-blend-weight", type=float, default=1.0, help="Blend weight between raw and calibrated representative score (0=raw, 1=calibrated).")
    parser.add_argument("--program-support-confidence-weight", type=float, default=0.0, help="Blend weight [0,1] for support-confidence-aware score attenuation at program level (0 disables).")
    parser.add_argument("--program-support-confidence-z-score", type=float, default=1.96, help="Wilson confidence z-score used for program support lower bound.")
    parser.add_argument("--program-support-confidence-floor", type=float, default=0.6, help="Minimum support-confidence factor in [0,1] before blending.")
    parser.add_argument("--program-support-confidence-exponent", type=float, default=1.0, help="Exponent applied to Wilson lower-bound confidence before floor/blend.")
    parser.add_argument("--program-anchor-max-same-count", type=int, default=0, help="Hard top-K guardrail: max programs allowed per identical antecedent anchor signature (0 disables).")
    parser.add_argument("--program-anchor-novelty-target", type=int, default=0, help="Desired number of unique anchors in top-K assembly (soft target).")
    parser.add_argument("--program-anchor-novelty-weight", type=float, default=0.0, help="Soft adjustment weight for preferring novel anchors while target is unmet.")
    parser.add_argument("--program-anchor-novelty-rerank-epsilon", type=float, default=0.0, help="Within this combined-score epsilon and quality window, rerank toward candidates adding new anchors.")
    parser.add_argument("--program-diversity-quality-floor-ratio", type=float, default=0.0, help="When >0, candidates below (top representative score * ratio) do not receive diversity adjustments.")
    parser.add_argument("--program-topk-care-process-max-count", type=int, default=-1, help="Hard top-k care-process cap. -1 disables. 0 enforces zero care-process programs in the guarded window.")
    parser.add_argument("--program-topk-care-process-window", type=int, default=5, help="Window size for top-k care-process hard cap.")
    parser.add_argument("--min-program-representative-score", type=float, default=0.0, help="Minimum representative score required for a program to be retained.")
    parser.add_argument("--program-top-window-unique-groups-target", type=int, default=0, help="Optional minimum unique mechanism-group target for a repaired top window; 0 disables repair.")
    parser.add_argument("--program-top-window-size", type=int, default=10, help="Window size used by top-window mechanism diversity repair.")
    parser.add_argument("--program-top-window-fixed-prefix", type=int, default=5, help="Fixed top-prefix protected from swaps during top-window repair.")
    parser.add_argument("--program-top5-unique-groups-target", type=int, default=0, help="Optional minimum unique mechanism-group target for repaired top-5 programs; 0 disables repair.")
    parser.add_argument("--program-top5-fixed-prefix", type=int, default=0, help="Fixed top-prefix protected from swaps during top-5 repair.")
    parser.add_argument("--program-constrained-candidate-pool-size", type=int, default=0, help="Stage-1 constrained selector candidate pool size (0 auto = max(window, 3*window)).")
    parser.add_argument("--program-constrained-quality-floor-ratio", type=float, default=0.0, help="Stage-2 swap quality floor ratio [0,1]: only candidates with representative_score >= top_score*ratio can be swapped in.")
    parser.add_argument("--program-constrained-two-stage-swap-budget", type=int, default=0, help="Max local-swap iterations for stage-2 constrained reranking (0 disables stage-2 swaps).")
    parser.add_argument("--program-constraint-autotune-steps", type=int, default=0, help="When >0 and gates fail, run failure-attribution-guided micro-tuning rounds on constrained selector params.")
    parser.add_argument("--program-constraint-autotune-group-step", type=int, default=1, help="Per-round increment for diversity targets during constrained-selector autotune.")
    parser.add_argument("--program-constraint-autotune-pool-step", type=int, default=5, help="Per-round increment for constrained candidate-pool size during autotune.")
    parser.add_argument("--program-preselect-coverage-min-unique-groups", type=int, default=0, help="Hard lower-bound preselection target: minimum unique program groups in preselect window (0 disables).")
    parser.add_argument("--program-preselect-coverage-window", type=int, default=20, help="Window size used by preselection coverage hard-lower-bound pass.")
    parser.add_argument("--program-preselect-quality-floor-ratio", type=float, default=0.0, help="Preselection quality floor ratio [0,1] relative to top representative score.")
    parser.add_argument("--transfer-baseline-top5-unique-groups", type=int, default=-1, help="Optional transfer objective baseline top5_unique_groups; -1 disables transfer-target objective.")
    parser.add_argument("--transfer-baseline-top10-unique-groups", type=int, default=-1, help="Optional transfer objective baseline top10_unique_groups; -1 disables transfer-target objective.")
    parser.add_argument("--transfer-baseline-top5-score-min", type=float, default=float('nan'), help="Optional transfer objective baseline top5_score_min; NaN disables transfer-target objective.")
    parser.add_argument("--transfer-max-drop-top5-unique-groups", type=int, default=999, help="Transfer-target tolerance: max allowed drop for top5_unique_groups.")
    parser.add_argument("--transfer-max-drop-top10-unique-groups", type=int, default=999, help="Transfer-target tolerance: max allowed drop for top10_unique_groups.")
    parser.add_argument("--transfer-max-drop-top5-score-min", type=float, default=999.0, help="Transfer-target tolerance: max allowed drop for top5_score_min.")
    parser.add_argument("--shadow-rule-blend-weight", type=float, default=0.0, help="Optional [0,1] blend weight for shadow predictive tie-break on contrastive rules.")
    parser.add_argument("--shadow-rule-max-adjustment", type=float, default=0.0, help="Maximum absolute score bonus added by shadow rule tie-break when enabled.")
    parser.add_argument("--shadow-rule-quality-floor-ratio", type=float, default=0.9, help="Only rules with contrastive_score >= top_score*ratio receive shadow rule bonus.")
    parser.add_argument("--shadow-rule-allow-care-process-boost", action="store_true", help="Allow shadow rule bonus on care-process-linked rules. Disabled by default.")
    parser.add_argument("--shadow-rule-apply-adjusted-score", action="store_true", help="Write shadow-adjusted score back into contrastive_score for downstream program consolidation.")
    parser.add_argument("--shadow-predictive-blend-weight", type=float, default=0.0, help="Optional [0,1] blend weight for shadow predictive tie-break during program ranking (0 disables).")
    parser.add_argument("--shadow-predictive-max-adjustment", type=float, default=0.03, help="Maximum absolute score bonus added by shadow predictive tie-break when enabled.")
    parser.add_argument("--shadow-predictive-quality-floor-ratio", type=float, default=0.9, help="Only programs with representative_score >= top_score*ratio receive shadow predictive bonus.")
    parser.add_argument("--shadow-predictive-allow-care-process-boost", action="store_true", help="Allow shadow predictive bonus to apply to care-process-linked programs. Disabled by default to reduce gate-regression risk.")
    parser.add_argument("--shadow-predictive-training-limit", type=int, default=0, help="Optional max episode count used to fit shadow predictive feature weights (0 auto-uses run limit or run count).")
    parser.add_argument("--shadow-predictive-seed", type=int, default=42, help="Random seed for shadow predictive feature-weight fitting.")
    parser.add_argument("--shadow-predictive-split-manifest", default=None, help="Optional split manifest JSON; shadow predictive fitting is constrained to split.train episode IDs.")
    parser.add_argument("--algorithm-ready-gate-profile", choices=["exploratory", "locked"], default="exploratory", help="Gate profile: exploratory ignores baseline-volume checks in overall pass; locked includes them.")
    parser.add_argument("--algorithm-ready-baseline-rules-count", type=int, default=0, help="Reference baseline rule count for locked-profile volume checks.")
    parser.add_argument("--algorithm-ready-baseline-programs-count", type=int, default=0, help="Reference baseline program count for locked-profile volume checks.")
    parser.add_argument("--algorithm-ready-rule-count-min-ratio", type=float, default=0.0, help="Locked-profile rule-count floor ratio against baseline count.")
    parser.add_argument("--algorithm-ready-program-count-min-ratio", type=float, default=0.0, help="Locked-profile program-count floor ratio against baseline count.")
    parser.add_argument("--algorithm-ready-rule-count-min-abs", type=int, default=0, help="Absolute minimum retained rule count required by volume checks.")
    parser.add_argument("--algorithm-ready-program-count-min-abs", type=int, default=0, help="Absolute minimum retained program count required by volume checks.")
    parser.add_argument("--algorithm-ready-top5-care-process-max", type=int, default=0, help="Quality gate: max allowed care-process programs in top-5.")
    parser.add_argument("--algorithm-ready-top5-max-same-anchor-max", type=int, default=2, help="Quality gate: max allowed same-anchor concentration in top-5.")
    parser.add_argument("--algorithm-ready-top5-unique-anchor-min", type=int, default=3, help="Quality gate: minimum unique anchor count in top-5.")
    parser.add_argument("--algorithm-ready-top5-unique-groups-min", type=int, default=5, help="Quality gate: minimum unique program groups in top-5.")
    parser.add_argument("--algorithm-ready-top10-unique-groups-min", type=int, default=6, help="Quality gate: minimum unique program groups in top-10.")
    parser.add_argument("--algorithm-ready-top5-score-min-threshold", type=float, default=0.82, help="Quality gate: minimum acceptable top-5 representative-score floor.")
    parser.add_argument("--reuse-contrastive-inputs", default=None, help="Optional path to cached contrastive_inputs.json. When provided, skip per-episode mining and reuse cached episode-level rule records.")
    parser.add_argument("--no-cache-contrastive-inputs", action="store_true", help="Do not write the large contrastive_inputs.json cache after mining; summary diagnostics are still computed in memory.")
    parser.add_argument("--stop-after-contrastive-cache", action="store_true", help="Stop after per-episode mining and contrastive_inputs.json cache creation; use the streaming resume script to build the final summary.")
    parser.add_argument("--stability-cache-glob", default=None, help="Optional glob for cached run summaries used to compute cross-run stability metrics.")
    parser.add_argument("--stability-top-k", type=int, default=10, help="Top-K rules used for stability computation.")
    parser.add_argument("--stability-rule-jaccard-min", type=float, default=0.0, help="Minimum rule-ID Jaccard mean gate for stability pass.")
    parser.add_argument("--stability-anchor-jaccard-min", type=float, default=0.0, help="Minimum anchor-set Jaccard mean gate for stability pass.")
    parser.add_argument("--stability-mechanism-jaccard-min", type=float, default=0.0, help="Minimum mechanism-group Jaccard mean gate for stability pass.")
    parser.add_argument("--stability-tuning-profile", choices=["none", "auto", "balanced", "robust", "aggressive"], default="none", help="One-click apply a generated stability tuning profile when stability cache runs are available.")
    parser.add_argument("--stability-auto-preset", choices=["custom", "strict", "normal", "loose", "adaptive"], default="custom", help="Shortcut preset for AUTO thresholds/sensitivities; adaptive learns risk from cached stability-run distributions.")
    parser.add_argument("--stability-auto-robust-rule-jaccard-max", type=float, default=0.45, help="AUTO profile threshold: below this rule-Jaccard uses robust profile.")
    parser.add_argument("--stability-auto-robust-anchor-jaccard-max", type=float, default=0.6, help="AUTO profile threshold: below this anchor-Jaccard uses robust profile.")
    parser.add_argument("--stability-auto-robust-mechanism-jaccard-max", type=float, default=0.6, help="AUTO profile threshold: below this mechanism-Jaccard uses robust profile.")
    parser.add_argument("--stability-auto-balanced-rule-jaccard-max", type=float, default=0.75, help="AUTO profile threshold: below this rule-Jaccard uses balanced profile (when robust is not triggered).")
    parser.add_argument("--stability-auto-balanced-anchor-jaccard-max", type=float, default=0.75, help="AUTO profile threshold: below this anchor-Jaccard uses balanced profile (when robust is not triggered).")
    parser.add_argument("--stability-auto-balanced-mechanism-jaccard-max", type=float, default=0.75, help="AUTO profile threshold: below this mechanism-Jaccard uses balanced profile (when robust is not triggered).")
    parser.add_argument("--stability-auto-robust-sensitivity", type=float, default=1.0, help="AUTO profile multiplier for robust thresholds (0 disables robust auto triggering; >1 makes robust easier to trigger).")
    parser.add_argument("--stability-auto-balanced-sensitivity", type=float, default=1.0, help="AUTO profile multiplier for balanced thresholds (0 disables balanced auto triggering; >1 makes balanced easier to trigger).")
    parser.add_argument("--stability-auto-adaptive-history-glob", default=None, help="Optional glob of previous summary JSON files for adaptive AUTO risk-threshold calibration.")
    parser.add_argument("--stability-auto-adaptive-history-domain", default="", help="Optional domain tag for grouping adaptive history calibration; when set, only matching historical domain summaries are used.")
    parser.add_argument("--disable-stability-auto-adaptive-history-domain-auto", action="store_true", help="Disable automatic adaptive-history domain inference from input directory name when explicit domain is not provided.")
    parser.add_argument("--stability-auto-adaptive-history-min-samples", type=int, default=3, help="Minimum historical adaptive risk scores required before history-based threshold calibration is applied.")
    parser.add_argument("--stability-auto-adaptive-history-min-run-count", type=int, default=0, help="Minimum stability run_count required before applying adaptive history calibration.")
    parser.add_argument("--stability-auto-adaptive-history-min-pair-count", type=int, default=0, help="Minimum stability pair_count required before applying adaptive history calibration.")
    parser.add_argument("--stability-auto-adaptive-history-warmup-runs", type=int, default=10, help="Historical sample count at which adaptive threshold calibration reaches full weight.")
    parser.add_argument("--stability-auto-adaptive-history-robust-quantile", type=float, default=0.8, help="History quantile [0,1] used as robust threshold target during adaptive calibration.")
    parser.add_argument("--stability-auto-adaptive-history-balanced-quantile", type=float, default=0.5, help="History quantile [0,1] used as balanced threshold target during adaptive calibration.")
    parser.add_argument("--stability-auto-adaptive-history-outlier-iqr-multiplier", type=float, default=0.0, help="When >0 and enough history exists, apply IQR outlier filtering to adaptive history risk scores before threshold calibration.")
    parser.add_argument("--stability-auto-adaptive-history-recency-half-life", type=float, default=0.0, help="When >0, apply exponential recency weighting (half-life in sample count) when aggregating adaptive history thresholds.")
    parser.add_argument("--stability-auto-adaptive-history-dispersion-scale", type=float, default=0.0, help="When >0, damp adaptive history calibration weight using IQR-based dispersion confidence (smaller values damp more).")
    parser.add_argument("--stability-auto-adaptive-history-drift-scale", type=float, default=0.0, help="When >0, damp adaptive history calibration weight when old-vs-recent history median drift is large.")
    parser.add_argument("--stability-auto-adaptive-history-drift-min-segment", type=int, default=2, help="Minimum segment size used when searching best old-vs-recent drift split in adaptive history.")
    parser.add_argument("--stability-auto-adaptive-history-reliability-strength", type=float, default=0.0, help="History reliability weighting strength in [0,1]; higher values give more influence to history summaries with stronger run/pair evidence.")
    parser.add_argument("--stability-auto-adaptive-history-confidence-calibration", choices=["none", "auto"], default="none", help="Auto-calibrate confidence scales (dispersion/drift) from observed history risk distribution when manual scales are not set.")
    return parser


def main(argv: list[str] | None = None) -> dict:
    parser = build_parser()
    args = parser.parse_args(argv)
    profile_overrides = _load_profile_overrides(args.profile_json)
    _apply_profile_overrides(args, profile_overrides)
    exclude_sparse_measurement_dominant = bool(args.hard_exclude_sparse_measurement_dominant)
    if bool(args.include_sparse_measurement_dominant):
        exclude_sparse_measurement_dominant = False
    return run_mimic_experiment(
        args.input,
        args.out_dir,
        limit=args.limit,
        episode_sampling_seed=args.episode_sampling_seed,
        episode_sampling_unique_stays=bool(args.episode_sampling_unique_stays),
        filter_anchor_kind=args.filter_anchor_kind,
        filter_window_hours=args.filter_window_hours,
        balance_episode_types=not args.disable_episode_type_balance,
        min_core_observed_channels=args.min_core_observed_channels,
        min_core_observed_points=args.min_core_observed_points,
        min_episode_timesteps=args.min_episode_timesteps,
        min_rule_positive_count=args.min_rule_positive_count,
        min_rule_control_count=args.min_rule_control_count,
        allow_zero_control_candidates=args.allow_zero_control_candidates,
        enrichment_rate_prior_count=args.enrichment_rate_prior_count,
        zero_control_rule_penalty=args.zero_control_rule_penalty,
        miner_top_candidates=args.miner_top_candidates,
        miner_top_m1=args.miner_top_m1,
        miner_top_m2=args.miner_top_m2,
        motif_verify_top_k=args.motif_verify_top_k,
        motif_min_support_ratio=args.motif_min_support_ratio,
        rule_top_k=args.rule_top_k,
        exclude_sparse_measurement_dominant=exclude_sparse_measurement_dominant,
        sparse_measurement_penalty=args.sparse_measurement_penalty,
        sparse_measurement_penalty_exponent=args.sparse_measurement_penalty_exponent,
        rule_anchor_saturation_penalty_weight=args.rule_anchor_saturation_penalty_weight,
        rule_anchor_saturation_penalty_exponent=args.rule_anchor_saturation_penalty_exponent,
        rule_mechanism_saturation_penalty_weight=args.rule_mechanism_saturation_penalty_weight,
        rule_mechanism_saturation_penalty_exponent=args.rule_mechanism_saturation_penalty_exponent,
        rule_leadtime_quality_weight=args.rule_leadtime_quality_weight,
        rule_support_reliability_weight=args.rule_support_reliability_weight,
        rule_support_reliability_exponent=args.rule_support_reliability_exponent,
        rule_consensus_boost_weight=args.rule_consensus_boost_weight,
        rule_consensus_top_k=args.rule_consensus_top_k,
        rule_consensus_rerank_epsilon=args.rule_consensus_rerank_epsilon,
        rule_consensus_quality_floor_ratio=args.rule_consensus_quality_floor_ratio,
        rule_anchor_diversity_weight=args.rule_anchor_diversity_weight,
        rule_anchor_diversity_top_k=args.rule_anchor_diversity_top_k,
        rule_anchor_diversity_rerank_epsilon=args.rule_anchor_diversity_rerank_epsilon,
        rule_anchor_diversity_quality_floor_ratio=args.rule_anchor_diversity_quality_floor_ratio,
        rule_mechanism_diversity_weight=args.rule_mechanism_diversity_weight,
        rule_mechanism_diversity_top_k=args.rule_mechanism_diversity_top_k,
        rule_mechanism_diversity_rerank_epsilon=args.rule_mechanism_diversity_rerank_epsilon,
        rule_mechanism_diversity_quality_floor_ratio=args.rule_mechanism_diversity_quality_floor_ratio,
        rule_mechanism_diversity_target=args.rule_mechanism_diversity_target,
        rule_mechanism_diversity_two_stage_swap_budget=args.rule_mechanism_diversity_two_stage_swap_budget,
        rule_mechanism_diversity_stage1_min_support_ratio=args.rule_mechanism_diversity_stage1_min_support_ratio,
        rule_mechanism_diversity_confidence_gate_enabled=args.rule_mechanism_diversity_confidence_gate_enabled,
        rule_mechanism_diversity_confidence_min_support_ratio=args.rule_mechanism_diversity_confidence_min_support_ratio,
        rule_mechanism_diversity_confidence_min_reliable_rules=args.rule_mechanism_diversity_confidence_min_reliable_rules,
        rule_mechanism_diversity_confidence_min_expected_gain=args.rule_mechanism_diversity_confidence_min_expected_gain,
        rule_mechanism_diversity_confidence_domain_thresholds_path=args.rule_mechanism_diversity_confidence_domain_thresholds_path,
        rule_low_coverage_recovery_enabled=args.rule_low_coverage_recovery_enabled,
        rule_low_coverage_recovery_trigger_top_k=args.rule_low_coverage_recovery_trigger_top_k,
        rule_low_coverage_recovery_min_unique_groups=args.rule_low_coverage_recovery_min_unique_groups,
        rule_low_coverage_recovery_min_control_count=args.rule_low_coverage_recovery_min_control_count,
        rule_low_coverage_recovery_allow_zero_control_candidates=args.rule_low_coverage_recovery_allow_zero_control_candidates,
        rule_low_coverage_recovery_zero_control_penalty=args.rule_low_coverage_recovery_zero_control_penalty,
        rule_low_coverage_recovery_support_reliability_weight=args.rule_low_coverage_recovery_support_reliability_weight,
        rule_low_coverage_recovery_support_reliability_exponent=args.rule_low_coverage_recovery_support_reliability_exponent,
        rule_low_coverage_recovery_significance_max_q_value=args.rule_low_coverage_recovery_significance_max_q_value,
        min_rule_contrastive_score=args.min_rule_contrastive_score,
        rule_anchor_candidate_top_n=args.rule_anchor_candidate_top_n,
        rule_mechanism_candidate_top_n=args.rule_mechanism_candidate_top_n,
        rule_candidate_quality_floor_ratio=args.rule_candidate_quality_floor_ratio,
        rule_care_process_max_ratio=args.rule_care_process_max_ratio,
        rule_care_process_penalty=args.rule_care_process_penalty,
        rule_anchor_min_unique=args.rule_anchor_min_unique,
        rule_anchor_max_share=args.rule_anchor_max_share,
        rule_mechanism_min_unique=args.rule_mechanism_min_unique,
        rule_significance_weight=args.rule_significance_weight,
        rule_significance_prior_count=args.rule_significance_prior_count,
        rule_significance_bootstrap_samples=args.rule_significance_bootstrap_samples,
        rule_significance_random_state=args.rule_significance_random_state,
        rule_significance_max_q_value=args.rule_significance_max_q_value,
        statistical_null_permutations=args.statistical_null_permutations,
        statistical_null_random_seed=args.statistical_null_random_seed,
        statistical_null_top_k=args.statistical_null_top_k,
        actionability_top_k=args.actionability_top_k,
        actionability_min_lead_hours=args.actionability_min_lead_hours,
        program_leadtime_diversity_weight=args.program_leadtime_diversity_weight,
        program_leadtime_diversity_gap_hours=args.program_leadtime_diversity_gap_hours,
        program_mechanism_diversity_weight=args.program_mechanism_diversity_weight,
        program_mechanism_max_same_count=args.program_mechanism_max_same_count,
        program_care_process_penalty=args.program_care_process_penalty,
        program_redundancy_penalty_weight=args.program_redundancy_penalty_weight,
        program_coverage_target=args.program_coverage_target,
        program_coverage_target_weight=args.program_coverage_target_weight,
        program_anchor_diversity_weight=args.program_anchor_diversity_weight,
        program_anchor_candidate_top_n=args.program_anchor_candidate_top_n,
        program_anchor_candidate_quality_ratio=args.program_anchor_candidate_quality_ratio,
        program_mechanism_candidate_top_n=args.program_mechanism_candidate_top_n,
        program_mechanism_min_unique=args.program_mechanism_min_unique,
        program_mechanism_group_mode=args.program_mechanism_group_mode,
        program_mechanism_feature_score_floor_ratio=args.program_mechanism_feature_score_floor_ratio,
        program_score_calibration_mode=args.program_score_calibration_mode,
        program_score_calibration_quantile_low=args.program_score_calibration_quantile_low,
        program_score_calibration_quantile_high=args.program_score_calibration_quantile_high,
        program_score_calibration_target_low=args.program_score_calibration_target_low,
        program_score_calibration_target_high=args.program_score_calibration_target_high,
        program_score_calibration_blend_weight=args.program_score_calibration_blend_weight,
        program_support_confidence_weight=args.program_support_confidence_weight,
        program_support_confidence_z_score=args.program_support_confidence_z_score,
        program_support_confidence_floor=args.program_support_confidence_floor,
        program_support_confidence_exponent=args.program_support_confidence_exponent,
        program_anchor_max_same_count=args.program_anchor_max_same_count,
        program_anchor_novelty_target=args.program_anchor_novelty_target,
        program_anchor_novelty_weight=args.program_anchor_novelty_weight,
        program_anchor_novelty_rerank_epsilon=args.program_anchor_novelty_rerank_epsilon,
        program_diversity_quality_floor_ratio=args.program_diversity_quality_floor_ratio,
        program_topk_care_process_max_count=args.program_topk_care_process_max_count,
        program_topk_care_process_window=args.program_topk_care_process_window,
        min_program_representative_score=args.min_program_representative_score,
        program_top5_unique_groups_target=args.program_top5_unique_groups_target,
        program_top5_fixed_prefix=args.program_top5_fixed_prefix,
        program_top_window_unique_groups_target=args.program_top_window_unique_groups_target,
        program_top_window_size=args.program_top_window_size,
        program_top_window_fixed_prefix=args.program_top_window_fixed_prefix,
        program_constrained_candidate_pool_size=args.program_constrained_candidate_pool_size,
        program_constrained_quality_floor_ratio=args.program_constrained_quality_floor_ratio,
        program_constrained_two_stage_swap_budget=args.program_constrained_two_stage_swap_budget,
        program_constraint_autotune_steps=args.program_constraint_autotune_steps,
        program_constraint_autotune_group_step=args.program_constraint_autotune_group_step,
        program_constraint_autotune_pool_step=args.program_constraint_autotune_pool_step,
        program_preselect_coverage_min_unique_groups=args.program_preselect_coverage_min_unique_groups,
        program_preselect_coverage_window=args.program_preselect_coverage_window,
        program_preselect_quality_floor_ratio=args.program_preselect_quality_floor_ratio,
        transfer_baseline_top5_unique_groups=args.transfer_baseline_top5_unique_groups,
        transfer_baseline_top10_unique_groups=args.transfer_baseline_top10_unique_groups,
        transfer_baseline_top5_score_min=args.transfer_baseline_top5_score_min,
        transfer_max_drop_top5_unique_groups=args.transfer_max_drop_top5_unique_groups,
        transfer_max_drop_top10_unique_groups=args.transfer_max_drop_top10_unique_groups,
        transfer_max_drop_top5_score_min=args.transfer_max_drop_top5_score_min,
        shadow_rule_blend_weight=args.shadow_rule_blend_weight,
        shadow_rule_max_adjustment=args.shadow_rule_max_adjustment,
        shadow_rule_quality_floor_ratio=args.shadow_rule_quality_floor_ratio,
        shadow_rule_exclude_care_process=not bool(args.shadow_rule_allow_care_process_boost),
        shadow_rule_apply_adjusted_score=bool(args.shadow_rule_apply_adjusted_score),
        shadow_predictive_blend_weight=args.shadow_predictive_blend_weight,
        shadow_predictive_max_adjustment=args.shadow_predictive_max_adjustment,
        shadow_predictive_quality_floor_ratio=args.shadow_predictive_quality_floor_ratio,
        shadow_predictive_exclude_care_process=not bool(args.shadow_predictive_allow_care_process_boost),
        shadow_predictive_training_limit=args.shadow_predictive_training_limit,
        shadow_predictive_seed=args.shadow_predictive_seed,
        shadow_predictive_split_manifest=args.shadow_predictive_split_manifest,
        algorithm_ready_gate_profile=args.algorithm_ready_gate_profile,
        algorithm_ready_baseline_rules_count=args.algorithm_ready_baseline_rules_count,
        algorithm_ready_baseline_programs_count=args.algorithm_ready_baseline_programs_count,
        algorithm_ready_rule_count_min_ratio=args.algorithm_ready_rule_count_min_ratio,
        algorithm_ready_program_count_min_ratio=args.algorithm_ready_program_count_min_ratio,
        algorithm_ready_rule_count_min_abs=args.algorithm_ready_rule_count_min_abs,
        algorithm_ready_program_count_min_abs=args.algorithm_ready_program_count_min_abs,
        algorithm_ready_top5_care_process_max=args.algorithm_ready_top5_care_process_max,
        algorithm_ready_top5_max_same_anchor_max=args.algorithm_ready_top5_max_same_anchor_max,
        algorithm_ready_top5_unique_anchor_min=args.algorithm_ready_top5_unique_anchor_min,
        algorithm_ready_top5_unique_groups_min=args.algorithm_ready_top5_unique_groups_min,
        algorithm_ready_top10_unique_groups_min=args.algorithm_ready_top10_unique_groups_min,
        algorithm_ready_top5_score_min_threshold=args.algorithm_ready_top5_score_min_threshold,
        cache_contrastive_inputs=not bool(args.no_cache_contrastive_inputs),
        stop_after_contrastive_cache=bool(args.stop_after_contrastive_cache),
        reuse_contrastive_inputs=args.reuse_contrastive_inputs,
        stability_cache_glob=args.stability_cache_glob,
        stability_top_k=args.stability_top_k,
        stability_rule_jaccard_min=args.stability_rule_jaccard_min,
        stability_anchor_jaccard_min=args.stability_anchor_jaccard_min,
        stability_mechanism_jaccard_min=args.stability_mechanism_jaccard_min,
        stability_tuning_profile=args.stability_tuning_profile,
        stability_auto_preset=args.stability_auto_preset,
        stability_auto_robust_rule_jaccard_max=args.stability_auto_robust_rule_jaccard_max,
        stability_auto_robust_anchor_jaccard_max=args.stability_auto_robust_anchor_jaccard_max,
        stability_auto_robust_mechanism_jaccard_max=args.stability_auto_robust_mechanism_jaccard_max,
        stability_auto_balanced_rule_jaccard_max=args.stability_auto_balanced_rule_jaccard_max,
        stability_auto_balanced_anchor_jaccard_max=args.stability_auto_balanced_anchor_jaccard_max,
        stability_auto_balanced_mechanism_jaccard_max=args.stability_auto_balanced_mechanism_jaccard_max,
        stability_auto_robust_sensitivity=args.stability_auto_robust_sensitivity,
        stability_auto_balanced_sensitivity=args.stability_auto_balanced_sensitivity,
        stability_auto_adaptive_history_glob=args.stability_auto_adaptive_history_glob,
        stability_auto_adaptive_history_domain=args.stability_auto_adaptive_history_domain,
        stability_auto_adaptive_history_domain_auto=not args.disable_stability_auto_adaptive_history_domain_auto,
        stability_auto_adaptive_history_min_samples=args.stability_auto_adaptive_history_min_samples,
        stability_auto_adaptive_history_min_run_count=args.stability_auto_adaptive_history_min_run_count,
        stability_auto_adaptive_history_min_pair_count=args.stability_auto_adaptive_history_min_pair_count,
        stability_auto_adaptive_history_warmup_runs=args.stability_auto_adaptive_history_warmup_runs,
        stability_auto_adaptive_history_robust_quantile=args.stability_auto_adaptive_history_robust_quantile,
        stability_auto_adaptive_history_balanced_quantile=args.stability_auto_adaptive_history_balanced_quantile,
        stability_auto_adaptive_history_outlier_iqr_multiplier=args.stability_auto_adaptive_history_outlier_iqr_multiplier,
        stability_auto_adaptive_history_recency_half_life=args.stability_auto_adaptive_history_recency_half_life,
        stability_auto_adaptive_history_dispersion_scale=args.stability_auto_adaptive_history_dispersion_scale,
        stability_auto_adaptive_history_drift_scale=args.stability_auto_adaptive_history_drift_scale,
        stability_auto_adaptive_history_drift_min_segment=args.stability_auto_adaptive_history_drift_min_segment,
        stability_auto_adaptive_history_reliability_strength=args.stability_auto_adaptive_history_reliability_strength,
        stability_auto_adaptive_history_confidence_calibration=args.stability_auto_adaptive_history_confidence_calibration,
    )


if __name__ == "__main__":
    main()
