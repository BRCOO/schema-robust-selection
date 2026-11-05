"""PrefixSpan-style sequential pattern baseline (ICDE 2001).

This is a lightweight but real sequence-mining approximation: it mines
frequent ordered event subsequences from positive episodes, scores them by
positive-versus-control support contrast, and exports the top-k patterns
through the shared discovery contract.
"""

from __future__ import annotations

from .mining_contract import MiningContract
from .subsequence_utils import contains_subsequence


def mine_prefixspan_patterns(
    episodes: list[dict],
    support_grid: list[float],
    candidate_budget: int,
) -> list[dict]:
    """Mine short ordered event subsequences and score them contrastively."""
    positives = [ep for ep in episodes if int(ep.get("label", 0)) == 1]
    controls = [ep for ep in episodes if int(ep.get("label", 0)) == 0]
    if not positives:
        return []

    candidate_patterns: dict[tuple[str, ...], dict] = {}
    max_pattern_len = 3
    max_events_per_episode = 40

    for episode in positives[: min(len(positives), 40)]:
        seq = [str(tok) for tok in episode.get("event_sequence", [])[:max_events_per_episode]]
        seen_in_episode: set[tuple[str, ...]] = set()
        for plen in (2, 3):
            if plen > max_pattern_len or len(seq) < plen:
                continue
            for start in range(0, len(seq) - plen + 1):
                pattern = tuple(seq[start : start + plen])
                if pattern in seen_in_episode:
                    continue
                seen_in_episode.add(pattern)
                candidate_patterns.setdefault(
                    pattern,
                    {
                        "events": list(pattern),
                    },
                )
                if len(candidate_patterns) >= max(candidate_budget * 2, 100):
                    break
            if len(candidate_patterns) >= max(candidate_budget * 2, 100):
                break
        if len(candidate_patterns) >= max(candidate_budget * 2, 100):
            break

    scored: list[dict] = []
    pos_count = float(len(positives))
    ctrl_count = float(len(controls)) if controls else 1.0
    for pattern, payload in candidate_patterns.items():
        pos_support = sum(
            1 for ep in positives
            if contains_subsequence(ep.get("event_sequence", []), list(pattern))
        ) / max(pos_count, 1.0)
        ctrl_support = sum(
            1 for ep in controls
            if contains_subsequence(ep.get("event_sequence", []), list(pattern))
        ) / max(ctrl_count, 1.0)
        for min_support in support_grid:
            if pos_support + 1e-12 < float(min_support):
                continue
            contrast = max(0.0, pos_support - ctrl_support)
            score = contrast * (1.0 + 0.2 * len(pattern)) * (1.0 + pos_support)
            if score <= 0.0:
                continue
            scored.append(
                {
                    "pattern_id": f"prefixspan_{len(scored)}",
                    "events": list(pattern),
                    "support": float(pos_support),
                    "control_support": float(ctrl_support),
                    "score": float(score),
                    "min_support": float(min_support),
                }
            )

    scored.sort(key=lambda row: (-float(row["score"]), -float(row["support"]), row["events"]))
    return scored[:candidate_budget]


def normalize_mining_output(
    mined: list[dict],
    top_k: int = 10,
) -> dict:
    """Normalize raw mining output into top_k_programs schema."""
    sorted_mined = sorted(mined, key=lambda x: -x.get("score", 0))[:top_k]
    return {
        "programs": {
            p["pattern_id"]: {
                "score": float(p.get("score", 0)),
                "events": p.get("events", []),
                "support": float(p.get("support", 0)),
                "control_support": float(p.get("control_support", 0)),
                "min_support": float(p.get("min_support", 0)),
            }
            for p in sorted_mined
        },
        "program_count": len(sorted_mined),
        "top_k": top_k,
    }


def run_prefixspan_baseline(
    episodes: list[dict],
    contract: MiningContract,
) -> dict:
    """Run PrefixSpan and normalize output into top_k_programs schema."""
    mined = mine_prefixspan_patterns(
        episodes,
        support_grid=contract.support_grid,
        candidate_budget=contract.candidate_budget,
    )
    return normalize_mining_output(mined=mined, top_k=contract.top_k)
