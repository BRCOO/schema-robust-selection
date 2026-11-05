from __future__ import annotations

from pathlib import Path
import json
import math

import numpy as np

from .comparators.statistics import aggregate_seed_metrics, multiple_comparison_correction


METRICS = (
    "overall_pass",
    "top5_unique_groups",
    "top10_unique_groups",
    "top5_score_min",
    "actionability_green",
    "lead_time_mean_hours",
    "lead_time_above_threshold_ratio",
    "topk_max_anchor_share",
    "topk_unique_anchor_count",
)


def _safe_float(value: object) -> float:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return x


def _extract_metrics(summary: dict) -> dict[str, float]:
    gates = dict(summary.get("algorithm_ready_gates", {}))
    div = dict(summary.get("contrastive_qc", {}).get("mechanism_diversity", {}))
    actionability = dict(summary.get("actionability_bridge", {}))
    lead_time = dict(actionability.get("lead_time_hours", {}))
    anchor_consistency = dict(actionability.get("anchor_consistency", {}))

    status = str(actionability.get("actionability_status", "")).strip().lower()
    return {
        "overall_pass": 1.0 if bool(gates.get("overall_pass", False)) else 0.0,
        "top5_unique_groups": _safe_float(div.get("top5_unique_groups")),
        "top10_unique_groups": _safe_float(div.get("top10_unique_groups")),
        "top5_score_min": _safe_float(div.get("top5_score_min")),
        "actionability_green": 1.0 if status == "green" else 0.0,
        "lead_time_mean_hours": _safe_float(lead_time.get("mean")),
        "lead_time_above_threshold_ratio": _safe_float(lead_time.get("above_threshold_ratio")),
        "topk_max_anchor_share": _safe_float(anchor_consistency.get("topk_max_anchor_share")),
        "topk_unique_anchor_count": _safe_float(anchor_consistency.get("topk_unique_anchor_count")),
    }


def load_seed_profile_run(summary_path: Path | str) -> dict:
    path = Path(summary_path)
    if not path.exists():
        raise FileNotFoundError(f"summary not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    metrics = _extract_metrics(payload)
    profile = str(payload.get("seed_sweep_profile", "")).strip()
    seed_raw = payload.get("seed_sweep_seed")
    if not profile or seed_raw is None:
        raise ValueError(f"summary missing seed sweep metadata: {path}")
    return {
        "summary_path": str(path),
        "profile": profile,
        "seed": int(seed_raw),
        "metrics": metrics,
    }


def collect_rows(summary_paths: list[Path | str]) -> list[dict]:
    rows = [load_seed_profile_run(path) for path in summary_paths]
    rows.sort(key=lambda row: (str(row["profile"]), int(row["seed"])))
    return rows


def build_metric_summary(rows: list[dict]) -> dict:
    profiles = sorted({str(row["profile"]) for row in rows})
    out: dict[str, dict] = {}
    for profile in profiles:
        profile_rows = [row for row in rows if str(row["profile"]) == profile]
        out[profile] = {}
        for metric in METRICS:
            metric_rows = []
            for row in profile_rows:
                metric_rows.append({metric: row.get("metrics", {}).get(metric)})
            out[profile][metric] = aggregate_seed_metrics(metric_rows, metric_key=metric, n_bootstrap=2000, seed=42)
    return out


def _paired_metric_values(rows: list[dict], profile_a: str, profile_b: str, metric: str) -> tuple[np.ndarray, np.ndarray]:
    by_profile_seed: dict[tuple[str, int], float] = {}
    for row in rows:
        key = (str(row["profile"]), int(row["seed"]))
        value = _safe_float(row.get("metrics", {}).get(metric))
        if math.isfinite(value):
            by_profile_seed[key] = value

    paired_a = []
    paired_b = []
    seeds = sorted({seed for profile, seed in by_profile_seed.keys() if profile in {profile_a, profile_b}})
    for seed in seeds:
        key_a = (profile_a, seed)
        key_b = (profile_b, seed)
        if key_a not in by_profile_seed or key_b not in by_profile_seed:
            continue
        paired_a.append(float(by_profile_seed[key_a]))
        paired_b.append(float(by_profile_seed[key_b]))
    return np.asarray(paired_a, dtype=float), np.asarray(paired_b, dtype=float)


def _paired_bootstrap_mean_delta(
    values_a: np.ndarray,
    values_b: np.ndarray,
    n_bootstrap: int = 4000,
    seed: int = 42,
) -> dict:
    if values_a.shape[0] == 0 or values_b.shape[0] == 0 or values_a.shape[0] != values_b.shape[0]:
        return {
            "n_pairs": 0,
            "mean_delta": float("nan"),
            "ci_low": float("nan"),
            "ci_high": float("nan"),
            "p_value": float("nan"),
        }

    deltas = values_a - values_b
    observed = float(np.mean(deltas))
    rng = np.random.default_rng(int(seed))
    n = int(deltas.shape[0])
    boot = np.zeros((max(200, int(n_bootstrap)),), dtype=float)
    for idx in range(boot.shape[0]):
        sample_idx = rng.integers(0, n, size=n)
        boot[idx] = float(np.mean(deltas[sample_idx]))

    p_low = float((1.0 + float(np.sum(boot <= 0.0))) / (float(boot.shape[0]) + 1.0))
    p_high = float((1.0 + float(np.sum(boot >= 0.0))) / (float(boot.shape[0]) + 1.0))
    p_two_sided = float(min(1.0, 2.0 * min(p_low, p_high)))
    return {
        "n_pairs": int(n),
        "mean_delta": observed,
        "ci_low": float(np.percentile(boot, 2.5)),
        "ci_high": float(np.percentile(boot, 97.5)),
        "p_value": p_two_sided,
    }


def build_pairwise_significance(rows: list[dict], profile_a: str, profile_b: str) -> dict:
    key = f"{profile_a}_vs_{profile_b}"
    out = {key: {}}

    raw_p_values: list[float] = []
    valid_metrics: list[str] = []

    for metric in METRICS:
        values_a, values_b = _paired_metric_values(rows, profile_a=profile_a, profile_b=profile_b, metric=metric)
        result = _paired_bootstrap_mean_delta(values_a, values_b, n_bootstrap=3000, seed=42)
        out[key][metric] = result
        p_value = float(result.get("p_value", float("nan")))
        if math.isfinite(p_value):
            raw_p_values.append(p_value)
            valid_metrics.append(metric)

    holm = multiple_comparison_correction(raw_p_values, method="holm")
    fdr = multiple_comparison_correction(raw_p_values, method="fdr_bh")
    for idx, metric in enumerate(valid_metrics):
        out[key][metric]["p_value_holm"] = float(holm[idx])
        out[key][metric]["p_value_fdr_bh"] = float(fdr[idx])

    return out


def write_significance_report(report: dict, out_json: Path | str, out_md: Path | str) -> None:
    json_path = Path(out_json)
    md_path = Path(out_md)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.parent.mkdir(parents=True, exist_ok=True)

    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    lines = []
    lines.append("# AegisGate-ICU Seed Significance Report")
    lines.append("")
    lines.append("## Run Set")
    lines.append(f"- run_count: {int(report.get('run_count', 0))}")
    lines.append(f"- seed_count: {int(report.get('seed_count', 0))}")
    lines.append(f"- profiles: {', '.join(report.get('profiles', []))}")
    lines.append("")
    lines.append("## Metric Summary (mean ± CI)")
    lines.append("| Profile | Metric | n | mean | std | ci_low | ci_high |")
    lines.append("| --- | --- | ---: | ---: | ---: | ---: | ---: |")
    for profile, metric_map in report.get("metric_summary", {}).items():
        for metric in METRICS:
            row = dict(metric_map.get(metric, {}))
            lines.append(
                f"| {profile} | {metric} | {int(row.get('n', 0))} | {float(row.get('mean', float('nan'))):.6f} | {float(row.get('std', float('nan'))):.6f} | {float(row.get('ci_low', float('nan'))):.6f} | {float(row.get('ci_high', float('nan'))):.6f} |"
            )

    lines.append("")
    lines.append("## Pairwise Significance")
    lines.append("| Pair | Metric | n_pairs | mean_delta | ci_low | ci_high | p_raw | p_holm | p_fdr_bh |")
    lines.append("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for pair, metric_map in report.get("pairwise_significance", {}).items():
        for metric in METRICS:
            row = dict(metric_map.get(metric, {}))
            lines.append(
                f"| {pair} | {metric} | {int(row.get('n_pairs', 0))} | {float(row.get('mean_delta', float('nan'))):.6f} | {float(row.get('ci_low', float('nan'))):.6f} | {float(row.get('ci_high', float('nan'))):.6f} | {float(row.get('p_value', float('nan'))):.6f} | {float(row.get('p_value_holm', float('nan'))):.6f} | {float(row.get('p_value_fdr_bh', float('nan'))):.6f} |"
            )

    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
