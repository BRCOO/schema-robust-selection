from __future__ import annotations

from itertools import combinations

from .shock_contrastive_miner import MECHANISM_GROUP_BY_FEATURE


def _jaccard(left: set[str], right: set[str]) -> float:
    union = left.union(right)
    if not union:
        return 1.0
    return float(len(left.intersection(right))) / float(len(union))


def _pairwise_jaccard_mean(values: list[set[str]]) -> float:
    if len(values) <= 1:
        return 1.0
    scores = [_jaccard(values[i], values[j]) for i, j in combinations(range(len(values)), 2)]
    if not scores:
        return 1.0
    return float(sum(scores) / len(scores))


def _topk_rule_ids(run_summary: dict, top_k: int) -> set[str]:
    rules = list(run_summary.get("contrastive_rules", []))[: max(0, int(top_k))]
    return {str(row.get("rule_id")) for row in rules if str(row.get("rule_id"))}


def _topk_anchor_features(run_summary: dict, top_k: int) -> set[str]:
    rules = list(run_summary.get("contrastive_rules", []))[: max(0, int(top_k))]
    anchors: set[str] = set()
    for rule in rules:
        for feature in list(rule.get("antecedent_features", [])):
            base = str(feature).split(":", 1)[0]
            if base:
                anchors.add(base)
    return anchors


def _topk_mechanism_groups(run_summary: dict, top_k: int) -> set[str]:
    rules = list(run_summary.get("contrastive_rules", []))[: max(0, int(top_k))]
    groups: set[str] = set()
    for rule in rules:
        for feature in list(rule.get("consequent_features", [])):
            base = str(feature).split(":", 1)[0]
            if not base:
                continue
            groups.add(MECHANISM_GROUP_BY_FEATURE.get(base, f"other:{base}"))
    return groups


def compute_rule_stability(run_summaries: list[dict], top_k: int = 10) -> dict:
    normalized_runs = [row for row in run_summaries if isinstance(row, dict)]
    k = max(1, int(top_k))

    rule_sets = [_topk_rule_ids(run, k) for run in normalized_runs]
    anchor_sets = [_topk_anchor_features(run, k) for run in normalized_runs]
    mechanism_sets = [_topk_mechanism_groups(run, k) for run in normalized_runs]

    pair_count = 0 if len(normalized_runs) <= 1 else int(len(normalized_runs) * (len(normalized_runs) - 1) / 2)
    return {
        "run_count": int(len(normalized_runs)),
        "top_k": int(k),
        "pair_count": int(pair_count),
        "topk_rule_jaccard_mean": _pairwise_jaccard_mean(rule_sets),
        "topk_anchor_jaccard_mean": _pairwise_jaccard_mean(anchor_sets),
        "topk_mechanism_jaccard_mean": _pairwise_jaccard_mean(mechanism_sets),
    }


def summarize_robustness_matrix(run_summaries: list[dict], top_k: int = 10) -> dict:
    normalized_runs = [row for row in run_summaries if isinstance(row, dict)]
    window_groups: dict[str, list[dict]] = {}
    anchor_groups: dict[str, list[dict]] = {}

    for run in normalized_runs:
        config = dict(run.get("config") or {})
        window_value = config.get("window_hours")
        anchor_value = config.get("anchor_kind")
        if window_value is None or anchor_value is None:
            runs_payload = list(run.get("runs") or [])
            if runs_payload:
                first_run = dict(runs_payload[0] or {})
                metadata = dict(first_run.get("metadata") or {})
                if window_value is None:
                    shape = list(first_run.get("shape") or [])
                    resample_minutes = metadata.get("resample_minutes", 60)
                    if shape:
                        try:
                            window_value = int(round((int(shape[0]) - 1) * float(resample_minutes) / 60.0))
                        except (TypeError, ValueError):
                            window_value = None
                if anchor_value is None:
                    anchor_value = metadata.get("anchor_kind")
        if window_value is not None:
            window_groups.setdefault(str(window_value), []).append(run)
        if anchor_value is not None:
            anchor_groups.setdefault(str(anchor_value), []).append(run)

    return {
        "run_count": int(len(normalized_runs)),
        "top_k": int(max(1, int(top_k))),
        "window_sensitivity": {
            key: compute_rule_stability(group, top_k=top_k) for key, group in sorted(window_groups.items())
        },
        "anchor_sensitivity": {
            key: compute_rule_stability(group, top_k=top_k) for key, group in sorted(anchor_groups.items())
        },
    }


def evaluate_stability_gates(metrics: dict, gates: dict) -> dict:
    checks: dict[str, dict] = {}
    for gate_name, threshold_raw in dict(gates or {}).items():
        gate_key = str(gate_name)
        if not gate_key.endswith("_min"):
            continue
        metric_name = gate_key[: -len("_min")]
        threshold = float(threshold_raw)
        value = float(metrics.get(metric_name, 0.0))
        checks[metric_name] = {
            "value": value,
            "threshold_min": threshold,
            "pass": bool(value >= threshold),
        }

    all_pass = all(bool(row.get("pass", False)) for row in checks.values()) if checks else True
    return {
        "all_pass": bool(all_pass),
        "checks": checks,
    }
