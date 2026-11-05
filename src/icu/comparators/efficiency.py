"""Efficiency and scalability instrumentation for comparator baselines."""

from __future__ import annotations


def build_efficiency_summary(
    runtime_seconds: float,
    peak_memory_mb: float,
    candidate_count: int,
    pruned_candidate_count: int,
) -> dict:
    """Build an efficiency summary dict with pruning ratio."""
    pruned = max(int(pruned_candidate_count), 0)
    total = max(int(candidate_count), 1)
    return {
        "runtime_seconds": float(runtime_seconds),
        "peak_memory_mb": float(peak_memory_mb),
        "candidate_count": int(candidate_count),
        "pruned_candidate_count": pruned,
        "pruning_ratio": float(pruned / total),
    }
