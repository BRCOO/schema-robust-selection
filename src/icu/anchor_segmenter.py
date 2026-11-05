from __future__ import annotations

from collections import Counter


def segment_episode_relative_to_anchor(
    lead_hours: list[float],
    instability_score: list[float],
    min_phase_len: int = 2,
) -> list[dict]:
    series_length = min(len(lead_hours), len(instability_score))
    if series_length <= 0:
        return []

    minimum = max(1, int(min_phase_len))
    if series_length < (minimum * 3):
        return [
            {
                "phase": "pre_anchor_instability",
                "start_idx": 0,
                "end_idx": series_length - 1,
                "lead_hours_start": float(lead_hours[0]),
                "lead_hours_end": float(lead_hours[series_length - 1]),
            }
        ]

    ordered_scores = [float(value) for value in instability_score[:series_length]]
    best_transition = None
    best_instability = None
    for split_idx in range(minimum, series_length - minimum + 1):
        left_mean = sum(ordered_scores[:split_idx]) / float(split_idx)
        right_count = series_length - split_idx
        right_mean = sum(ordered_scores[split_idx:]) / float(max(1, right_count))
        lift = right_mean - left_mean
        if best_instability is None or lift > best_instability:
            best_instability = lift
            best_transition = split_idx

    pre_anchor_start = max(minimum * 2, int(best_transition or (series_length - minimum)))
    pre_anchor_start = min(pre_anchor_start, series_length - minimum)
    baseline_end = max(minimum - 1, pre_anchor_start - minimum - 1)
    pre_deterioration_start = baseline_end + 1

    return [
        {
            "phase": "baseline",
            "start_idx": 0,
            "end_idx": baseline_end,
            "lead_hours_start": float(lead_hours[0]),
            "lead_hours_end": float(lead_hours[baseline_end]),
        },
        {
            "phase": "pre_deterioration",
            "start_idx": pre_deterioration_start,
            "end_idx": pre_anchor_start - 1,
            "lead_hours_start": float(lead_hours[pre_deterioration_start]),
            "lead_hours_end": float(lead_hours[pre_anchor_start - 1]),
        },
        {
            "phase": "pre_anchor_instability",
            "start_idx": pre_anchor_start,
            "end_idx": series_length - 1,
            "lead_hours_start": float(lead_hours[pre_anchor_start]),
            "lead_hours_end": float(lead_hours[series_length - 1]),
        },
    ]


def summarize_anchor_segments(anchor_segments: list[dict]) -> dict:
    if not anchor_segments:
        return {
            "episode_count": 0,
            "phase_counts": {},
        }

    phase_counter = Counter()
    for episode in anchor_segments:
        for segment in episode.get("segments", []):
            phase = str(segment.get("phase", "")).strip()
            if phase:
                phase_counter[phase] += 1
    return {
        "episode_count": int(len(anchor_segments)),
        "phase_counts": dict(sorted(phase_counter.items())),
    }
