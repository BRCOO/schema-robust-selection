#!/usr/bin/env python3
"""Multiple-comparison control for the SRS-C-V comparisons over all E3 runs (paper table).

Collects every paired bootstrap of ``srs_c_val`` against an established selector -- internal test
AUPRC and mean target AUPRC (composite deployment) -- from the E3 runs (``run_union_selection.py``)
and their add-on runs (``run_union_extra.py``: JMI, CMIM, L1 with schema dropout), and, separately,
of ``srs_c_unknown_val`` (no target schema) against the selectors that use no target schema.

Two-sided p-values from each bootstrap: ``percentile`` = 2 min(k + 1, B - k + 1) / (B + 1) with k
the replicates <= 0 (resolution 2 / (B + 1)); ``normal`` = from z = diff / SE with SE the 95%
percentile interval width / 3.92. Holm (family = one run's tests of one reference) and
Benjamini-Hochberg (all tests of one reference) adjustments at alpha = 0.05.

  PYTHONDONTWRITEBYTECODE=1 /d/BRCoo/miniconda/python scripts/v2/summarize_significance.py
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RUNS = {"union_k5": "K=5", "union": "K=10", "union_k20": "K=20", "union_large_k10": "large K=10", "union_large_k20": "large K=20"}
LABELS = ("gap1h", "gap2h", "gap4h", "aki_gap6h")
ESTABLISHED = ("mrmr", "mrmr_filter_known", "mmrfs", "jmi", "cmim", "l1_logistic", "l1_dropout", "mmr", "dpp", "mms_xin")
NO_TARGET = ("fs_cv", "mrmr", "jmi", "cmim", "mmrfs", "l1_logistic", "l1_dropout", "mmr", "dpp", "mms_xin")


def p_values(row: dict, n_boot: int) -> tuple[float, float]:
    k = round(float(row["frac_le_zero"]) * n_boot)
    percentile = min(1.0, 2.0 * min(k + 1, n_boot - k + 1) / (n_boot + 1))
    se = (float(row["high"]) - float(row["low"])) / 3.919928
    z = abs(float(row["point_diff"])) / se if se > 0 else math.inf
    normal = math.erfc(z / math.sqrt(2.0))
    return percentile, normal


def holm(ps: list[float]) -> list[float]:
    order = sorted(range(len(ps)), key=lambda i: ps[i])
    adjusted, running = [0.0] * len(ps), 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(ps) - rank) * ps[i]))
        adjusted[i] = running
    return adjusted


def bh(ps: list[float]) -> list[float]:
    order = sorted(range(len(ps)), key=lambda i: ps[i], reverse=True)
    adjusted, running = [0.0] * len(ps), 1.0
    for rank, i in enumerate(order):
        running = min(running, ps[i] * len(ps) / (len(ps) - rank))
        adjusted[i] = running
    return adjusted


def read(path: Path) -> list[dict]:
    return list(csv.DictReader(open(path, encoding="utf-8"))) if path.exists() else []


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out-dir", default="artifacts/v2/significance")
    parser.add_argument("--n-boot", type=int, default=1000)
    args = parser.parse_args(argv)
    tests = []
    for root, budget in RUNS.items():
        for label in LABELS:
            e3, extra = ROOT / "artifacts/v2" / root / f"f_mimic_iv_{label}", ROOT / "artifacts/v2/union_extra" / root / f"f_mimic_iv_{label}"
            if not (e3 / "manifest.json").exists() or "srs_c_val" not in json.loads((e3 / "manifest.json").read_text(encoding="utf-8"))["meta"]:
                continue
            run = f"{budget} {label}"
            for source in (e3, extra):
                for r in read(source / "paired_internal.csv"):
                    if (r["a"] == "srs_c_val" and r["b"] in ESTABLISHED) or (r["a"] == "srs_c_unknown_val" and r["b"] in NO_TARGET):
                        tests.append({"run": run, "reference": r["a"], "opponent": r["b"], "metric": "internal", **{k: r[k] for k in ("point_diff", "low", "high", "frac_le_zero")},
                                      "n_boot": int(float(r.get("n_boot_used") or args.n_boot))})
                for r in read(source / "paired_targets.csv"):
                    a, b = r["a"].split(":"), r["b"].split(":")
                    if r["target"] != "mean_over_targets" or a[1] != "composite" or b[1] != "composite":
                        continue
                    if (a[0] == "srs_c_val" and b[0] in ESTABLISHED) or (a[0] == "srs_c_unknown_val" and b[0] in NO_TARGET):
                        tests.append({"run": run, "reference": a[0], "opponent": b[0], "metric": "mean_target", **{k: r[k] for k in ("point_diff", "low", "high", "frac_le_zero")},
                                      "n_boot": args.n_boot})
    for t in tests:
        t["p_percentile"], t["p_normal"] = p_values(t, t["n_boot"])
    for kind in ("percentile", "normal"):
        families = defaultdict(list)
        for i, t in enumerate(tests):
            families[(t["reference"], t["run"])].append(i)
        for members in families.values():
            for i, p in zip(members, holm([tests[i][f"p_{kind}"] for i in members])):
                tests[i][f"holm_{kind}"] = p
        for reference in {t["reference"] for t in tests}:
            members = [i for i, t in enumerate(tests) if t["reference"] == reference]
            for i, p in zip(members, bh([tests[i][f"p_{kind}"] for i in members])):
                tests[i][f"bh_{kind}"] = p

    def verdict(t, key=None):
        if key is None:
            return 0 if float(t["low"]) > 0 else (2 if float(t["high"]) < 0 else 1)
        return 1 if t[key] >= 0.05 else (0 if float(t["point_diff"]) > 0 else 2)

    summary = []
    for (reference, opponent, metric) in sorted({(t["reference"], t["opponent"], t["metric"]) for t in tests}):
        group = [t for t in tests if (t["reference"], t["opponent"], t["metric"]) == (reference, opponent, metric)]
        row = {"reference": reference, "opponent": opponent, "metric": metric, "n_runs": len(group)}
        for name, key in (("ci", None), ("holm_percentile", "holm_percentile"), ("holm_normal", "holm_normal"), ("bh_percentile", "bh_percentile"), ("bh_normal", "bh_normal")):
            counts = [0, 0, 0]
            for t in group:
                counts[verdict(t, key)] += 1
            row[name] = "/".join(map(str, counts))
        diffs = sorted(float(t["point_diff"]) for t in group)
        row.update(mean_diff=sum(diffs) / len(diffs), median_diff=diffs[len(diffs) // 2] if len(diffs) % 2 else (diffs[len(diffs) // 2 - 1] + diffs[len(diffs) // 2]) / 2,
                   min_diff=diffs[0], max_diff=diffs[-1])
        summary.append(row)
    out = ROOT / args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    for name, records in (("significance_tests.csv", tests), ("significance_summary.csv", summary)):
        with open(out / name, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)
    print(f"{len(tests)} tests; better/n.s./worse")
    print(f"{'reference':18s} {'opponent':18s} {'metric':12s} {'n':>3s} {'CI':>8s} {'Holm(pct)':>10s} {'Holm(norm)':>10s} {'BH(pct)':>8s} {'BH(norm)':>9s} {'mean':>7s} {'min':>7s}")
    for r in summary:
        print(f"{r['reference']:18s} {r['opponent']:18s} {r['metric']:12s} {r['n_runs']:3d} {r['ci']:>8s} {r['holm_percentile']:>10s} {r['holm_normal']:>10s} "
              f"{r['bh_percentile']:>8s} {r['bh_normal']:>9s} {r['mean_diff']:+.3f} {r['min_diff']:+.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
