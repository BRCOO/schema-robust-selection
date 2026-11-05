"""RuleGrowth-style sequential rule baseline.

This wrapper mines short antecedent-to-consequent rules from ordered event
sequences, then ranks them by support, confidence, and positive-versus-control
contrast under the shared discovery contract.
"""

from __future__ import annotations

from .mining_contract import MiningContract
from .subsequence_utils import contains_subsequence, rule_fires


def mine_rulegrowth_rules(
    episodes: list[dict],
    support_grid: list[float],
    confidence_grid: list[float],
    candidate_budget: int,
) -> list[dict]:
    """Mine short sequential rules and score them contrastively."""
    positives = [ep for ep in episodes if int(ep.get("label", 0)) == 1]
    controls = [ep for ep in episodes if int(ep.get("label", 0)) == 0]
    if not positives:
        return []

    candidate_rules: dict[tuple[tuple[str, ...], tuple[str, ...]], None] = {}
    max_events_per_episode = 40
    for episode in positives[: min(len(positives), 40)]:
        seq = [str(tok) for tok in episode.get("event_sequence", [])[:max_events_per_episode]]
        if len(seq) < 2:
            continue
        for idx in range(0, len(seq) - 1):
            antecedent = (seq[idx],)
            consequent = (seq[idx + 1],)
            candidate_rules[(antecedent, consequent)] = None
            if idx + 2 < len(seq):
                antecedent2 = (seq[idx], seq[idx + 1])
                consequent2 = (seq[idx + 2],)
                candidate_rules[(antecedent2, consequent2)] = None
            if len(candidate_rules) >= max(candidate_budget * 2, 120):
                break
        if len(candidate_rules) >= max(candidate_budget * 2, 120):
            break

    rules: list[dict] = []
    pos_count = float(len(positives))
    ctrl_count = float(len(controls)) if controls else 1.0
    for antecedent, consequent in candidate_rules.keys():
        antecedent_support = sum(
            1 for ep in positives
            if contains_subsequence(ep.get("event_sequence", []), list(antecedent))
        ) / max(pos_count, 1.0)
        pos_support = sum(
            1 for ep in positives
            if rule_fires(ep.get("event_sequence", []), list(antecedent), list(consequent))
        ) / max(pos_count, 1.0)
        ctrl_support = sum(
            1 for ep in controls
            if rule_fires(ep.get("event_sequence", []), list(antecedent), list(consequent))
        ) / max(ctrl_count, 1.0)
        confidence = pos_support / max(antecedent_support, 1e-8)
        contrast = max(0.0, pos_support - ctrl_support)

        for min_support in support_grid:
            if pos_support + 1e-12 < float(min_support):
                continue
            for min_conf in confidence_grid:
                if confidence + 1e-12 < float(min_conf):
                    continue
                score = contrast * confidence * (1.0 + pos_support)
                if score <= 0.0:
                    continue
                rules.append(
                    {
                        "rule_id": f"rulegrowth_{len(rules)}",
                        "antecedent": list(antecedent),
                        "consequent": list(consequent),
                        "support": float(pos_support),
                        "control_support": float(ctrl_support),
                        "confidence": float(confidence),
                        "score": float(score),
                        "min_support": float(min_support),
                        "min_confidence": float(min_conf),
                    }
                )

    rules.sort(
        key=lambda row: (
            -float(row["score"]),
            -float(row["confidence"]),
            -float(row["support"]),
            row["antecedent"],
            row["consequent"],
        )
    )
    return rules[:candidate_budget]


def normalize_mining_output(
    mined: list[dict],
    top_k: int = 10,
) -> dict:
    """Normalize raw mining output into top_k_programs schema."""
    sorted_mined = sorted(mined, key=lambda x: -x.get("score", 0))[:top_k]
    return {
        "programs": {
            r["rule_id"]: {
                "score": float(r.get("score", 0)),
                "antecedent": r.get("antecedent", []),
                "consequent": r.get("consequent", []),
                "support": float(r.get("support", 0)),
                "control_support": float(r.get("control_support", 0)),
                "confidence": float(r.get("confidence", 0)),
                "min_support": float(r.get("min_support", 0)),
                "min_confidence": float(r.get("min_confidence", 0)),
            }
            for r in sorted_mined
        },
        "program_count": len(sorted_mined),
        "top_k": top_k,
    }


def run_rulegrowth_baseline(
    episodes: list[dict],
    contract: MiningContract,
) -> dict:
    """Run RuleGrowth and normalize output into top_k_programs schema."""
    mined = mine_rulegrowth_rules(
        episodes,
        support_grid=contract.support_grid,
        confidence_grid=contract.confidence_grid,
        candidate_budget=contract.candidate_budget,
    )
    return normalize_mining_output(mined=mined, top_k=contract.top_k)
