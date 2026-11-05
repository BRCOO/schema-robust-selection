from __future__ import annotations

from pathlib import Path
import json
import math
import hashlib

from .statistics import (
    aggregate_seed_metrics,
    bootstrap_confidence_interval,
    multiple_comparison_correction,
    paired_bootstrap_pvalue,
)


METRICS = ("auroc", "auprc", "brier")


def _safe_float(value):
    try:
        x = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return x


def _split_signature(payload: dict) -> str:
    split = payload.get("split", {}).get("split", {})
    train = list(split.get("train", []))
    val = list(split.get("val", []))
    test = list(split.get("test", []))
    key = "|".join(train) + "#" + "|".join(val) + "#" + "|".join(test)
    return hashlib.md5(key.encode("utf-8")).hexdigest()


def load_comparator_run(run_dir: Path | str) -> dict:
    base = Path(run_dir)
    summary_path = base / "comparator_summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"comparator summary not found: {summary_path}")
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    payload["_run_dir"] = str(base)
    payload["_split_signature"] = _split_signature(payload)
    return payload


def collect_seed_rows(runs: list[dict], split_name: str = "test") -> list[dict]:
    rows = []
    for run in runs:
        run_seed = int(run.get("split", {}).get("split", {}).get("seed", -1))
        models = run.get("models", {})
        for model_name, model_info in models.items():
            metric_block = model_info.get("metrics", {}).get(split_name, {})
            row = {
                "seed": run_seed,
                "run_dir": run.get("_run_dir", ""),
                "model": model_name,
            }
            for metric in METRICS:
                row[metric] = _safe_float(metric_block.get(metric))
            rows.append(row)
    return rows


def build_metric_summary(rows: list[dict]) -> dict:
    models = sorted({row["model"] for row in rows})
    out = {}
    for model in models:
        model_rows = [row for row in rows if row["model"] == model]
        out[model] = {}
        for metric in METRICS:
            summary = aggregate_seed_metrics(model_rows, metric_key=metric, n_bootstrap=2000, seed=42)
            out[model][metric] = summary
    return out


def build_pairwise_significance(
    runs: list[dict],
    reference_model: str = "lr_stats",
    split_name: str = "test",
) -> dict:
    comparisons = {}
    for run in runs:
        seed = int(run.get("split", {}).get("split", {}).get("seed", -1))
        models = run.get("models", {})
        ref = models.get(reference_model, {})
        ref_pred = ref.get("predictions", {}).get(split_name, {})
        y_true = ref_pred.get("y_true", [])
        y_prob_ref = ref_pred.get("y_prob", [])
        if not y_true or not y_prob_ref:
            continue

        for model_name, model_info in models.items():
            if model_name == reference_model:
                continue
            model_pred = model_info.get("predictions", {}).get(split_name, {})
            y_prob_cmp = model_pred.get("y_prob", [])
            cmp_true = model_pred.get("y_true", [])
            if not y_prob_cmp or not cmp_true:
                continue
            if len(cmp_true) != len(y_true):
                continue
            key = f"{reference_model}_vs_{model_name}"
            comparisons.setdefault(key, {metric: [] for metric in METRICS})
            for metric in METRICS:
                result = paired_bootstrap_pvalue(
                    y_true=y_true,
                    y_prob_a=y_prob_ref,
                    y_prob_b=y_prob_cmp,
                    metric=metric,
                    n_bootstrap=2000,
                    seed=max(0, seed),
                )
                result["seed"] = seed
                comparisons[key][metric].append(result)

    merged = {}
    raw_p_by_metric: dict[str, list[tuple[str, float]]] = {metric: [] for metric in METRICS}
    for key, metric_map in comparisons.items():
        merged[key] = {}
        for metric, runs_metric in metric_map.items():
            finite_ps = [float(row["p_value"]) for row in runs_metric if math.isfinite(float(row.get("p_value", float("nan"))))]
            finite_delta = [float(row["observed_delta"]) for row in runs_metric if math.isfinite(float(row.get("observed_delta", float("nan"))))]
            mean_delta = float(sum(finite_delta) / len(finite_delta)) if finite_delta else float("nan")
            median_p = float(sorted(finite_ps)[len(finite_ps) // 2]) if finite_ps else float("nan")
            delta_ci = bootstrap_confidence_interval(finite_delta, n_bootstrap=1000, alpha=0.05, seed=42)
            merged[key][metric] = {
                "n_runs": len(runs_metric),
                "mean_observed_delta": mean_delta,
                "median_p_value": median_p,
                "delta_ci_low": delta_ci["low"],
                "delta_ci_high": delta_ci["high"],
            }
            raw_p_by_metric[metric].append((key, median_p))

    for metric in METRICS:
        keyed = raw_p_by_metric.get(metric, [])
        if not keyed:
            continue
        keys = [pair_key for pair_key, _ in keyed]
        values = [p for _, p in keyed]
        holm = multiple_comparison_correction(values, method="holm")
        fdr = multiple_comparison_correction(values, method="fdr_bh")
        for idx, pair_key in enumerate(keys):
            if pair_key not in merged:
                continue
            if metric not in merged[pair_key]:
                continue
            merged[pair_key][metric]["p_value_holm"] = holm[idx]
            merged[pair_key][metric]["p_value_fdr_bh"] = fdr[idx]
    return merged


def write_significance_report(report: dict, out_json: Path | str, out_md: Path | str) -> None:
    json_path = Path(out_json)
    md_path = Path(out_md)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.parent.mkdir(parents=True, exist_ok=True)

    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    lines = []
    lines.append("# Comparator Seed Significance Report")
    lines.append("")
    lines.append("## Run Set")
    lines.append(f"- run_count: {len(report.get('runs', []))}")
    lines.append(f"- split: {report.get('split_name', 'test')}")
    lines.append(f"- unique_split_signatures: {report.get('unique_split_signatures', 0)}")
    if int(report.get("unique_split_signatures", 0)) <= 1:
        lines.append("- warning: split signatures are identical; seed sweep does not reflect split variability")
    lines.append("")
    lines.append("## Metric Summary (mean ± CI)")
    lines.append("| Model | Metric | n | mean | std | ci_low | ci_high |")
    lines.append("| --- | --- | ---: | ---: | ---: | ---: | ---: |")
    for model_name, metric_map in report.get("metric_summary", {}).items():
        for metric in METRICS:
            row = metric_map.get(metric, {})
            lines.append(
                f"| {model_name} | {metric} | {row.get('n', 0)} | {row.get('mean', float('nan')):.6f} | {row.get('std', float('nan')):.6f} | {row.get('ci_low', float('nan')):.6f} | {row.get('ci_high', float('nan')):.6f} |"
            )

    lines.append("")
    lines.append("## Pairwise Significance (reference-first)")
    lines.append("| Pair | Metric | n_runs | mean_delta | delta_ci_low | delta_ci_high | p_raw | p_holm | p_fdr_bh |")
    lines.append("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for pair, metric_map in report.get("pairwise_significance", {}).items():
        for metric in METRICS:
            row = metric_map.get(metric, {})
            lines.append(
                f"| {pair} | {metric} | {row.get('n_runs', 0)} | {row.get('mean_observed_delta', float('nan')):.6f} | {row.get('delta_ci_low', float('nan')):.6f} | {row.get('delta_ci_high', float('nan')):.6f} | {row.get('median_p_value', float('nan')):.6f} | {row.get('p_value_holm', float('nan')):.6f} | {row.get('p_value_fdr_bh', float('nan')):.6f} |"
            )

    result_layers = report.get("result_layers", [])
    if result_layers:
        lines.append("")
        lines.append("## Result Layers")
        lines.append(f"- layers: {', '.join(result_layers)}")
        for layer in result_layers:
            layer_data = report.get(layer, {})
            paper_rows = layer_data.get("paper_rows", {})
            if paper_rows:
                lines.append(f"")
                lines.append(f"### {layer}")
                lines.append("| Method | Key Metrics |")
                lines.append("| --- | --- |")
                for method, data in paper_rows.items():
                    lines.append(f"| {method} | {json.dumps(data)} |")

    external_transfer = report.get("external_transfer", {})
    if external_transfer:
        lines.append("")
        lines.append("## External Transfer")
        cohorts = external_transfer.get("cohorts", {})
        for cohort_name, cohort_data in cohorts.items():
            drop = cohort_data.get("internal_to_external_drop", "N/A")
            ratio = cohort_data.get("retained_dimension_ratio", "N/A")
            lines.append(f"- {cohort_name}: drop={drop}, retained_dim={ratio}")

    stress = report.get("stress_matrix", {})
    if stress:
        lines.append("")
        lines.append("## Stress Matrix")
        rows = stress.get("rows", [])
        for row in rows:
            scenario = row.get("scenario", "unknown")
            status = row.get("status", "unknown")
            lines.append(f"- {scenario}: {status}")

    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
