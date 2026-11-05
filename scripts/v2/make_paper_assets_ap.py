"""Paper assets of the revised method (SRS-AP) and of the reviewer-driven experiments (2026-11).

Called by ``make_paper_assets.py`` (``builds(m)`` with that module, whose helpers are reused), so
one command still builds every table and figure. Every builder skips silently when its inputs
are missing. Naming: the paper calls SRS-AP simply "SRS" (source schema only) and "SRS+T" (with
the target schemas); the likelihood-decision versions are ablations.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

STANDARD = ("f_mimic_iv_gap1h", "f_mimic_iv_gap2h", "f_mimic_iv_gap4h", "f_mimic_iv_aki_gap6h")
ICU_RUNS = tuple(f"{root}/{label}" for root in ("union_k5", "union", "union_k20") for label in STANDARD) + \
    tuple(f"{root}/{label}" for root in ("union_large_k10", "union_large_k20") for label in ("f_mimic_iv_gap1h", "f_mimic_iv_aki_gap6h"))
DRIVE_RUNS = tuple(f"drive_k{k}/f_backblaze_seagate_gap2d" for k in (5, 10, 20))
LABEL = {
    "srs_ap_t": "SRS+T", "srs_ap": "SRS", "srs_c_val": "SRS+T, likelihood decision", "srs_c_unknown_val": "SRS, likelihood decision",
    "srs_ap_unaware": "schema-unaware SRS (no scenarios)", "srs_ap_t_nosur": "SRS+T, no surrogates in the objective", "fs_cv": "schema-unaware SRS, likelihood decision",
    "mrmr": "mRMR", "nmifs": "NMIFS", "mrmr_filter_known": "mRMR + filter (known targets)", "mmrfs": "MMRFS", "jmi": "JMI", "cmim": "CMIM",
    "l1_logistic": "L1 logistic", "l1_dropout": "L1 + schema-dropout training", "mmr_ig": "MMR (IG relevance)", "mms_xin_ig": "MMS (IG relevance)",
    "dpp_ig": "DPP (IG relevance)", "mmr_rank": "MMR (rank relevance)", "mms_xin_rank": "MMS (rank relevance)", "dpp_rank": "DPP (rank relevance)",
    "mmr": "MMR (generator scores)", "mms_xin": "MMS (generator scores)", "dpp": "DPP (generator scores)", "mrmr_stratified": "mRMR, stratified by generator",
    "best_single_srs": "best single generator",
}
OPPONENTS = ("mrmr", "nmifs", "mrmr_filter_known", "mmrfs", "jmi", "cmim", "l1_logistic", "l1_dropout", "mmr_ig", "mms_xin_ig", "dpp_ig",
             "mrmr_stratified", "best_single_srs")
TARGET_ORDER = ("eicu", "hirid", "drop_labs", "drop_ventilation", "drop_bp", "drop_urine", "vitals_only")
TARGET_TEXT = {"eicu": "eICU", "hirid": "HiRID", "drop_labs": "no labs", "drop_ventilation": "no ventilation", "drop_bp": "no BP",
               "drop_urine": "no urine", "vitals_only": "vitals only", "hgst": "HGST", "toshiba": "Toshiba", "wdc": "WDC"}


def _diff(r: dict) -> float:
    for key in ("point_diff", "diff"):
        if r.get(key) not in (None, ""):
            return float(r[key])
    return float("nan")


def _tally(rows: list[dict]) -> str:
    if not rows:
        return "--"
    counts = [0, 0, 0]
    for r in rows:
        counts[0 if float(r["low"]) > 0 else (2 if float(r["high"]) < 0 else 1)] += 1
    return f"{'/'.join(map(str, counts))} ({np.mean([_diff(r) for r in rows]):+.3f})"


def _metric(r: dict) -> str:
    return "internal" if r.get("metric") in ("internal", "auprc") and r.get("target", "internal") in ("internal", "") else "target"


def builds(m) -> list:
    A = m.ROOT / "artifacts/v2"

    def paired_ap(runs=ICU_RUNS):
        rows, inputs = [], []
        for run in runs:
            path = A / "union_ap" / run / "paired.csv"
            if path.exists():
                inputs.append(path)
                rows += m.read_csv(path)
            ab = A / "union_ap" / run / "ablation_paired.csv"
            if ab.exists():
                inputs.append(ab)
                for r in m.read_csv(ab):
                    if r["metric"] == "internal" or r.get("target") == "mean_over_targets":
                        rows.append({**r, "metric": "internal" if r["metric"] == "internal" else "mean_target"})
        return rows, inputs

    def ap_main_comparison():
        rows, inputs = paired_ap()
        if not rows:
            return
        cell = defaultdict(list)
        for r in rows:
            cell[(r["a"], r["b"], "internal" if r["metric"] == "internal" else "target")].append(r)
        table = []
        for b in (*OPPONENTS, "srs_ap_unaware"):
            table.append([LABEL[b], _tally(cell[("srs_ap", b, "internal")]), _tally(cell[("srs_ap", b, "target")]),
                          _tally(cell[("srs_ap_t", b, "internal")]), _tally(cell[("srs_ap_t", b, "target")])])
        table.append([LABEL["srs_c_val"].replace("SRS+T, ", "") + " (ablation)", _tally(cell[("srs_ap", "srs_c_unknown_val", "internal")]),
                      _tally(cell[("srs_ap", "srs_c_unknown_val", "target")]), _tally(cell[("srs_ap_t", "srs_c_val", "internal")]),
                      _tally(cell[("srs_ap_t", "srs_c_val", "target")])])
        m.write_table("ap_main_comparison", ["Opponent", "SRS: internal", "SRS: mean target", "SRS+T: internal", "SRS+T: mean target"], table,
                      "ICU union pools, 16 runs. Cells: runs in which SRS is significantly better / not significantly different / significantly worse "
                      "(95% patient-clustered paired bootstrap), and the mean AUPRC difference. SRS uses only the source schema, SRS+T also the eICU and "
                      "HiRID schemas. Every registry is deployed with surrogates. MMR, MMS and DPP rank by TRAIN information gain (Section 5). The "
                      "single-generator comparison exists for 10 runs. The last two rows are ablations: no schema scenarios, and the likelihood decision.",
                      *sorted(set(inputs)))

    def ap_significance():
        rows, inputs = paired_ap()
        if not rows:
            return
        tests = []
        for r in rows:
            if r["b"] not in OPPONENTS or r["b"] == "best_single_srs":
                continue
            n_boot = int(float(r.get("n_boot") or 1000))
            k = round(float(r["frac_le_zero"]) * n_boot)
            p = min(1.0, 2.0 * min(k + 1, n_boot - k + 1) / (n_boot + 1))
            tests.append({**r, "p": p, "family": (r["run"], r["a"], "internal" if r["metric"] == "internal" else "target")})
        by_family = defaultdict(list)
        for t in tests:
            by_family[t["family"]].append(t)
        for fam in by_family.values():
            order = sorted(range(len(fam)), key=lambda i: fam[i]["p"])
            running = 0.0
            for rank, i in enumerate(order):
                running = max(running, min(1.0, (len(fam) - rank) * fam[i]["p"]))
                fam[i]["holm"] = running
        for ref in ("srs_ap", "srs_ap_t"):
            for metric in ("internal", "target"):
                sel = [t for t in tests if t["a"] == ref and t["family"][2] == metric]
                order = sorted(range(len(sel)), key=lambda i: sel[i]["p"], reverse=True)
                running = 1.0
                for rank, i in enumerate(order):
                    running = min(running, sel[i]["p"] * len(sel) / (len(sel) - rank))
                    sel[i]["bh"] = running
        table = []
        for ref in ("srs_ap_t", "srs_ap"):
            for b in OPPONENTS[:-1]:
                for metric in ("internal", "target"):
                    sel = [t for t in tests if t["a"] == ref and t["b"] == b and t["family"][2] == metric]
                    if not sel:
                        continue

                    def count(key):
                        better = sum(1 for t in sel if t[key] < 0.05 and _diff(t) > 0)
                        worse = sum(1 for t in sel if t[key] < 0.05 and _diff(t) < 0)
                        return f"{better}/{len(sel) - better - worse}/{worse}"
                    ci = _tally(sel).split(" ")[0]
                    table.append([LABEL[ref], LABEL[b], metric, len(sel), ci, count("holm"), count("bh")])
        m.write_table("ap_significance", ["Method", "Opponent", "Metric", "Runs", "95% CI: b/ns/w", "Holm: b/ns/w", "BH: b/ns/w"], table,
                      "Multiple-comparison control of the main comparison: Holm within each run (one family per run, method and metric), "
                      "Benjamini-Hochberg over all runs; two-sided bootstrap p-values (resolution 2/(B+1), B = 1000).", *sorted(set(inputs)))

    def ap_ablation_by_target():
        rows, inputs = [], []
        share = defaultdict(list)
        for run in ICU_RUNS:
            path = A / "union_ap" / run / "ablation_paired.csv"
            if not path.exists():
                continue
            inputs.append(path)
            rows += [r for r in m.read_csv(path) if r["a"] == "srs_ap_t" and r["b"] == "srs_ap_unaware"]
            man = A / "union_ap" / run / "ablation_manifest.json"
            if man.exists():
                for t, v in json.loads(man.read_text(encoding="utf-8"))["executable_share_unaware"].items():
                    share[t].append(v)
        if not rows:
            return
        table = []
        for t in TARGET_ORDER:
            sel = [r for r in rows if r.get("target") == t]
            if sel:
                tally = _tally(sel)
                table.append([TARGET_TEXT[t], f"{np.mean(share[t]):.2f}" if share[t] else "--", tally.split(" ")[0], tally.split(" ")[1].strip("()")])
        m.write_table("ap_ablation_by_target", ["Target", "Executable", "Better / n.s. / worse", "Mean diff."], table,
                      "SRS+T minus its schema-unaware ablation (no target or dropout scenarios; both deployed with surrogates), per target over the 16 "
                      "ICU runs; executable = share of the ablation's programs that run on the target without a surrogate.", *inputs)

    def ap_drive_table():
        cells, inputs = defaultdict(dict), []
        for run in DRIVE_RUNS:
            path = A / "union_ap" / run / "paired.csv"
            if not path.exists():
                continue
            inputs.append(path)
            k = run.split("/")[0].replace("drive_k", "")
            for r in m.read_csv(path):
                if r["a"] != "srs_ap_t":
                    continue
                star = "*" if float(r["low"]) > 0 or float(r["high"]) < 0 else ""
                key = "internal" if r["metric"] == "internal" else "target"
                cells[r["b"]][(k, key)] = f"{_diff(r):+.3f}{star}"
            ab = A / "union_ap" / run / "ablation_paired.csv"
            if ab.exists():
                inputs.append(ab)
                for r in m.read_csv(ab):
                    if r["a"] == "srs_ap_t" and (r["metric"] == "internal" or r.get("target") == "mean_over_targets"):
                        star = "*" if float(r["low"]) > 0 or float(r["high"]) < 0 else ""
                        cells[r["b"]][(k, "internal" if r["metric"] == "internal" else "target")] = f"{_diff(r):+.3f}{star}"
        if not cells:
            return
        order = ("mrmr", "nmifs", "mmrfs", "cmim", "jmi", "l1_logistic", "l1_dropout", "mrmr_filter_known", "srs_ap_unaware", "srs_ap_t_nosur", "srs_c_val", "srs_ap")
        table = [[LABEL.get(b, b)] + [f"{cells[b].get((k, 'internal'), '--')} / {cells[b].get((k, 'target'), '--')}" for k in ("5", "10", "20")]
                 for b in order if b in cells]
        m.write_table("ap_drive_paired", ["SRS+T minus", "K = 5: internal / mean target", "K = 10: internal / mean target", "K = 20: internal / mean target"], table,
                      "Second domain (pre-specified): hard-drive failure, source Seagate, targets HGST, Toshiba and WDC. SRS+T minus each registry; "
                      "* = 95% CI excludes 0 (drive-clustered paired bootstrap). K = 10 is the pre-specified primary budget.", *inputs)

    def ap_stress():
        series = (("srs_ap_t", "SRS+T"), ("srs_ap", "SRS"), ("nmifs", "NMIFS"), ("mrmr", "mRMR"), ("l1_dropout", "L1 + schema-dropout training"))
        panels, inputs = [], []
        for run in ("union/f_mimic_iv_gap1h", "union/f_mimic_iv_aki_gap6h"):
            path = A / "stress_ap" / run / "stress.csv"
            if path.exists():
                inputs.append(path)
                data = defaultdict(list)
                for r in m.read_csv(path):
                    data[(r["family"], float(r["level"]), r["registry"])].append(float(r["auprc"]))
                panels.append((run, data))
        if panels:
            families = (("feature", "features dropped"), ("group", "feature groups dropped"))
            fig, axes = m.plt.subplots(1, 2 * len(panels), figsize=(2.55 * 2 * len(panels), 2.35), squeeze=False)
            for p_index, (run, data) in enumerate(panels):
                title = "1 h" if "gap1h" in run else "AKI, 6 h"
                for f_index, (family, text) in enumerate(families):
                    ax = axes[0, 2 * p_index + f_index]
                    m.style_axes(ax)
                    for i, (name, label) in enumerate(series):
                        xs = sorted({lv for (fam, lv, reg) in data if fam == family and reg == name})
                        if xs:
                            ax.plot(xs, [float(np.mean(data[(family, lv, name)])) for lv in xs], color=m.LINE_COLORS[i], linewidth=2 if i == 0 else 1.3,
                                    marker=m.LINE_MARKERS[i], markersize=3, linestyle=m.LINE_STYLES[i], label=label)
                    ax.set_title(f"{title}: {text}", fontsize=8, color=m.INK)
                    ax.set_xlabel("schema loss l", fontsize=8, color=m.INK_2)
                    if p_index == 0 and f_index == 0:
                        ax.set_ylabel("internal test AUPRC", fontsize=8, color=m.INK_2)
            handles, labels = axes[0, 0].get_legend_handles_labels()
            fig.legend(handles, labels, loc="lower center", ncol=len(series), fontsize=7.5, frameon=False, bbox_to_anchor=(0.5, -0.07))
            fig.tight_layout(rect=(0, 0.06, 1, 1))
            m.save(fig, "ap_stress_curves", *inputs)
        means, wins, inputs = defaultdict(list), defaultdict(lambda: [0, 0, 0]), []
        for run in ICU_RUNS:
            path = A / "stress_ap" / run / "stress.csv"
            if not path.exists():
                continue
            inputs.append(path)
            draw = defaultdict(dict)
            for r in m.read_csv(path):
                means[(r["family"], r["level"], r["registry"])].append(float(r["auprc"]))
                draw[(r["family"], r["level"], r["draw"])][r["registry"]] = float(r["auprc"])
            for (family, level, _), v in draw.items():
                for opp in ("mrmr", "nmifs"):
                    if "srs_ap_t" in v and opp in v:
                        d = v["srs_ap_t"] - v[opp]
                        wins[(family, level, opp)][0 if d > 1e-12 else (2 if d < -1e-12 else 1)] += 1
        if not means:
            return
        regs = ("srs_ap_t", "srs_ap", "nmifs", "mrmr", "mrmr_filter_known", "l1_dropout", "srs_c_val")
        table = []
        for family in ("feature", "group"):
            for level in sorted({k[1] for k in means if k[0] == family}, key=float):
                table.append([family, level] + [f"{np.mean(means[(family, level, r)]):.3f}" if means[(family, level, r)] else "--" for r in regs]
                             + ["/".join(map(str, wins[(family, level, "mrmr")])), "/".join(map(str, wins[(family, level, "nmifs")]))])
        m.write_table("ap_stress_summary", ["Loss unit", "Loss l", *(LABEL[r] for r in regs), "SRS+T vs mRMR: w/t/l", "SRS+T vs NMIFS: w/t/l"], table,
                      "Schema-loss stress test over the 16 ICU runs (internal TEST, 30 random schemas per level, the draws of the original analysis): mean "
                      "AUPRC of every registry deployed with surrogates, and paired draws won / tied / lost by SRS+T.", *inputs)

    def ap_resplit():
        rows, inputs = [], []
        for task, text in (("gap1h", "1 h"), ("aki_gap6h", "AKI, 6 h")):
            values, verdicts = defaultdict(list), defaultdict(lambda: {"internal": [], "target": []})
            n = 0
            for i in range(1, 6):
                base = A / "union_ap/union_resplit" / f"r{i}_mimic_iv_{task}"
                if not (base / "rows.csv").exists():
                    continue
                n += 1
                inputs += [base / "rows.csv", base / "paired.csv"]
                for r in m.read_csv(base / "rows.csv"):
                    values[r["registry"]].append((float(r["internal_auprc"]), float(r["mean_target_auprc"])))
                for r in m.read_csv(base / "paired.csv"):
                    if r["a"] == "srs_ap_t":
                        verdicts[r["b"]]["internal" if r["metric"] == "internal" else "target"].append(r)
            for name in ("srs_ap_t", "srs_ap", "nmifs", "mrmr", "mrmr_filter_known", "mmrfs", "l1_dropout", "srs_c_val"):
                if not values[name]:
                    continue
                a = np.array(values[name])
                rows.append([text, n, LABEL[name], f"{a[:, 0].mean():.3f} +- {a[:, 0].std():.3f}", f"{a[:, 1].mean():.3f} +- {a[:, 1].std():.3f}",
                             "--" if name == "srs_ap_t" else _tally(verdicts[name]["internal"]).split(" ")[0],
                             "--" if name == "srs_ap_t" else _tally(verdicts[name]["target"]).split(" ")[0]])
        if rows:
            m.write_table("ap_resplit", ["Run", "Splits", "Selection", "Internal AUPRC (mean +- SD)", "Mean target AUPRC (mean +- SD)",
                                         "SRS+T minus: internal b/ns/w", "SRS+T minus: target b/ns/w"], rows,
                          "Robustness to the patient-level split: five new TRAIN/VAL/TEST splits of the internal cohort, pools re-mined on each new TRAIN "
                          "with the four non-FMG generators, union pool, K = 10; paired bootstrap per split.", *inputs)

    def ap_robustness():
        rows, inputs = [], []
        for label in STANDARD:
            base = A / "union_robustness_ap" / label
            if not (base / "manifest.json").exists():
                continue
            inputs += [base / "manifest.json", base / "decoder_paired.csv", base / "subgroup_paired.csv"]
            rank = json.loads((base / "manifest.json").read_text(encoding="utf-8"))["rank_agreement"]
            paired = m.read_csv(base / "decoder_paired.csv")

            def verdict(b, decoder, target):
                sel = [r for r in paired if r["b"] == b and r["decoder"] == decoder and r.get("target") == target]
                if not sel:
                    return "--"
                r = sel[0]
                star = "*" if float(r["low"]) > 0 or float(r["high"]) < 0 else ""
                return f"{_diff(r):+.3f}{star}"
            sub = m.read_csv(base / "subgroup_paired.csv")
            sg = {b: (sum(1 for r in sub if r["b"] == b and float(r["low"]) > 0), sum(1 for r in sub if r["b"] == b and float(r["high"]) < 0),
                      sum(1 for r in sub if r["b"] == b)) for b in ("mrmr", "nmifs")}
            rows.append([label.replace("f_mimic_iv_", ""), f"{rank['internal_auprc']['kendall_tau']:.2f} / {rank['mean_target_auprc']['kendall_tau']:.2f}",
                         f"{verdict('mrmr', 'gbdt', 'internal')} / {verdict('mrmr', 'gbdt', 'mean_over_targets')}",
                         f"{verdict('nmifs', 'gbdt', 'internal')} / {verdict('nmifs', 'gbdt', 'mean_over_targets')}",
                         f"{sg['mrmr'][0]}/{sg['mrmr'][2] - sg['mrmr'][0] - sg['mrmr'][1]}/{sg['mrmr'][1]}",
                         f"{sg['nmifs'][0]}/{sg['nmifs'][2] - sg['nmifs'][0] - sg['nmifs'][1]}/{sg['nmifs'][1]}"])
        if rows:
            m.write_table("ap_robustness", ["Run", "Kendall tau, logistic vs trees (internal / target)", "SRS+T minus mRMR, trees", "SRS+T minus NMIFS, trees",
                                            "Subgroups vs mRMR: b/ns/w", "Subgroups vs NMIFS: b/ns/w"], rows,
                          "Decoder and subgroups (K = 10): gradient-boosted trees replace the logistic probe on the same coordinates; subgroups by sex and "
                          "age on MIMIC-IV TEST, eICU and HiRID. * = 95% CI excludes 0.", *inputs)

    def ap_sensitivity():
        rows, inputs = [], []
        settings = []
        for label in STANDARD:
            path = A / "srs_sensitivity_ap" / label / "sensitivity.csv"
            if path.exists():
                inputs.append(path)
                settings.append((label, {(r["parameter"], r["value"]): r for r in m.read_csv(path)}))
        if not settings:
            return
        keys = list(dict.fromkeys(k for _, s in settings for k in s))
        for key in keys:
            cells = []
            for _, s in settings:
                r = s.get(key)
                cells.append("--" if r is None else f"{float(r['internal_auprc']):.3f} / {float(r['mean_target_auprc']):.3f}")
            overlap = np.mean([float(s[key]["overlap_with_default"]) for _, s in settings if key in s])
            rows.append([key[0] if key[0] == "default" else f"{key[0]} = {key[1]}", *cells, f"{overlap:.2f}"])
        m.write_table("ap_sensitivity", ["Setting", *(f"{l.replace('f_mimic_iv_', '')}: internal / mean target" for l, _ in settings), "Mean overlap with default"], rows,
                      "One hyper-parameter (or the cross-validation seed) changed at a time, SRS+T, K = 10.", *inputs)

    def deployment_modes_table():
        rows_in, inputs = [], []
        for path in sorted((A / "deployment_modes_ap").rglob("paired.csv")):
            inputs.append(path)
            rows_in += m.read_csv(path)
        if not rows_in:
            return
        level = defaultdict(list)
        for path in sorted((A / "deployment_modes_ap").rglob("rows.csv")):
            inputs.append(path)
            for r in m.read_csv(path):
                if not r["run"].startswith("drive") and not r["mode"].startswith("internal"):
                    level[(r["registry"], r["mode"])].append(float(r["auprc"]))
        modes = (("surrogate", "frozen probe, surrogates (shipped)"), ("zero", "frozen probe, zeroed"), ("soft", "frozen probe, learned soft surrogates"),
                 ("robust", "probe trained with schema dropout"), ("reduced", "probe re-fit per target schema"), ("reduced_surrogate", "re-fit per schema, with surrogates"))
        table = []
        for mode, text in modes:
            icu = [r for r in rows_in if r["mode"] == mode and not r["run"].startswith("drive")]
            table.append([text, f"{np.mean(level[('srs_ap_t', mode)]):.3f}" if level[("srs_ap_t", mode)] else "--",
                          f"{np.mean(level[('mrmr', mode)]):.3f}" if level[("mrmr", mode)] else "--",
                          _tally([r for r in icu if r["a"] == "srs_ap_t" and r["b"] == "mrmr"]),
                          _tally([r for r in icu if r["a"] == "srs_ap_t" and r["b"] == "nmifs"]),
                          _tally([r for r in icu if r["a"] == "srs_ap" and r["b"] == "mrmr"])])
        m.write_table("ap_deployment_modes", ["Deployment", "SRS+T: mean target", "mRMR: mean target", "SRS+T minus mRMR", "SRS+T minus NMIFS", "SRS minus mRMR"], table,
                      "The same registries deployed six ways (16 ICU runs, mean target AUPRC; tallies b/ns/w of the paired bootstrap and mean difference). "
                      "Soft surrogates predict a missing program's firing from the executable ones (label-free, fit on source TRAIN).", *inputs)

    def hospital_table():
        base = A / "hospital_schemas_ap/cov0.05_min1"
        paired, inputs, share = [], [], []
        for run in ICU_RUNS:
            p = base / run / "paired.csv"
            if p.exists():
                inputs.append(p)
                paired += m.read_csv(p)
                man = json.loads((base / run / "manifest.json").read_text(encoding="utf-8"))["summary"]
                share.append((man["n_affected"], man["n_episodes"], man["n_hospitals"], len(man["patterns"])))
        if not paired:
            return
        table = []
        for a in ("srs_ap_t", "srs_ap"):
            for b in ("mrmr", "nmifs", "mrmr_filter_known", "mmrfs", "cmim", "l1_dropout", "fs_cv"):
                sel = {s: [r for r in paired if r["a"] == a and r["b"] == b and r["mode"] == "surrogate" and r["subset"] == s] for s in ("all", "affected")}
                if sel["all"]:
                    table.append([LABEL[a], LABEL[b], _tally(sel["all"]), _tally(sel["affected"])])
        aff = np.array([s[0] / s[1] for s in share])
        m.write_table("ap_hospital_schemas", ["Method", "Opponent", "eICU, all episodes", "eICU, hospitals with a reduced schema"], table,
                      f"Real heterogeneous targets: every eICU hospital with its own schema (a measured feature belongs to it if observed in at least 5% "
                      f"of its windows; sedation if ever on). {aff.min():.0%}-{aff.max():.0%} of the eICU episodes are at hospitals with a reduced schema "
                      f"({min(s[3] for s in share)}-{max(s[3] for s in share)} distinct patterns). 16 ICU runs; tallies b/ns/w and mean AUPRC difference.", *inputs)

    def sql_table():
        rows, inputs = [], []
        for path in sorted((A / "sql_execution_exact").rglob("exactness.csv")):
            inputs.append(path)
            for r in m.read_csv(path):
                pool = "large" if "large" in r["run"] else "standard"
                rows.append([r["run"].split("/")[-1].replace("f_mimic_iv_", ""), pool, r["site"], r["n_programs"], r["n_episodes"], r["n_mismatched_programs"],
                             f"{float(r['seconds_tokens']):.2f}", f"{float(r['seconds_registry']):.2f}"])
        if rows:
            m.write_table("sql_exactness", ["Task", "Pool", "Database", "Programs", "Episodes", "Mismatched programs", "Tokens (s)", "Registry (s)"], rows,
                          "Every compilable program executed as SQL in DuckDB against the Python reference engine (all generators but FMG's legacy rules); "
                          "Registry = the shipped SRS+T registry, every coordinate resolved against the database's schema; 6 threads, idle workstation.", *inputs)

    def sql_scale_table():
        rows, inputs = [], []
        for label, text in (("f_mimic_iv_gap1h", "vasopressor 1 h"), ("f_mimic_iv_aki_gap6h", "AKI 6 h")):
            path = A / "sql_execution/union" / label / "scale.csv"
            if not path.exists():
                continue
            inputs.append(path)
            for r in m.read_csv(path):
                rows.append([text, f"x{r['replicate']}", f"{int(r['n_episodes']):,}", f"{int(r['n_obs_rows']) / 1e6:.1f}M",
                             f"{float(r['seconds_tokens']):.1f}", f"{float(r['seconds_registry']):.1f}"])
        if rows:
            diag = A / "sql_execution/memory_diagnostic.txt"
            m.write_table("sql_scale", ["Task", "eICU replicated", "Episodes", "Observation rows", "Tokens (s)", "Registry (s)"], rows,
                          "SQL scale-up: the eICU episodes replicated 10 and 30 times, in-memory DuckDB, 6 threads, idle workstation (32 GB), shipped "
                          "SRS+T registry. At AKI x30 the database peaks at 21 GB and the machine runs out of memory (memory_diagnostic.txt).",
                          *inputs, *([diag] if diag.exists() else []))

    def ap_timing_table():
        path = A / "selection_scalability/selection_scalability.csv"
        if not path.exists():
            return
        ok = [r for r in m.read_csv(path) if r["axis"] == "size" and r["status"] == "ok" and r["method"] in ("srs_c_v", "srs_ap_v")]
        by = {(r["method"], int(r["n_candidates"])): float(r["seconds"]) for r in ok}
        rows = []
        for n in sorted({n for (meth, n) in by if meth == "srs_ap_v"}):
            if ("srs_c_v", n) in by:
                a, b = by[("srs_c_v", n)], by[("srs_ap_v", n)]
                rows.append([f"{n:,}", f"{a:.1f}", f"{b:.1f}", f"{b - a:+.1f}", f"{100 * (b - a) / a:+.1f}%"])
        if rows:
            m.write_table("ap_timing", ["Candidates", "Likelihood decision (s)", "AUPRC decision (s)", "Difference (s)", "Relative"], rows,
                          "Selection time of SRS+T (surrogate chains plus fast selection with q = 0 and q = 0.25, K = 10) with the likelihood and the "
                          "AUPRC decision; idle workstation, 6 BLAS threads on the performance cores.", path)

    def exact_table():
        rows, inputs = [], []
        for path in sorted((A / "exact_small_pools").glob("rows_*.csv")):
            inputs.append(path)
            rows += m.read_csv(path)
        if not rows:
            return
        table = []
        for obj, text in (("deployed_q0", "deployed objective, q = 0"), ("deployed_q25", "deployed objective, q = 0.25"), ("analyzable", "analyzable variant (Theorem 1)")):
            for k in ("3", "4"):
                sel = [r for r in rows if r["objective"] == obj and r["k"] == k]
                v = np.array([float(r["ratio"]) for r in sel])
                table.append([text, k, len(sel), f"{v.mean():.3f}", f"{v.min():.3f}", f"{sum(r['greedy_is_optimal'] == 'True' for r in sel)}/{len(sel)}"])
        m.write_table("exact_small_pools", ["Objective", "K", "Instances", "Mean J(greedy)/J(opt)", "Min", "Greedy optimal"], table,
                      "Greedy against the exact optimum on sub-pools of 20 real candidates (vasopressor 1 h and AKI; half drawn from the 60 most informative "
                      "candidates, half from the whole pool), every K-subset enumerated.", *inputs)

    def restricted_table():
        table, inputs = [], []
        for restrict, text in (("care", "no care-process features"), ("care_obs", "no care-process or measurement tokens"),
                               ("generator_erminer", "ERMiner alone (large pool)"), ("generator_prefixspan", "PrefixSpan alone (large pool)")):
            base = A / "restricted_pools" / restrict
            for path in sorted(base.rglob("rows.csv")):
                inputs.append(path)
                rows = {r["registry"]: r for r in m.read_csv(path)}
                paired = m.read_csv(path.parent / "paired.csv")
                run = rows["srs_ap_t"]["run"].split(":")[-1].replace("f_mimic_iv_", "")

                def v(b, metric):
                    sel = [r for r in paired if r["b"] == b and (r["metric"] == metric)]
                    return _tally(sel).split(" ")[0] if sel else "--"
                table.append([text, run, rows["srs_ap_t"]["n_candidates"], f"{float(rows['srs_ap_t']['internal_auprc']):.3f} / {float(rows['srs_ap_t']['mean_target_auprc']):.3f}",
                              f"{float(rows['nmifs']['internal_auprc']):.3f} / {float(rows['nmifs']['mean_target_auprc']):.3f}",
                              f"{float(rows['mrmr']['internal_auprc']):.3f} / {float(rows['mrmr']['mean_target_auprc']):.3f}",
                              f"{v('nmifs', 'internal')} | {v('nmifs', 'mean_target')}", f"{v('mrmr', 'internal')} | {v('mrmr', 'mean_target')}"])
        if table:
            m.write_table("restricted_pools", ["Pool", "Task", "Candidates", "SRS+T: internal / target", "NMIFS", "mRMR", "SRS+T vs NMIFS (int. | target)",
                                               "SRS+T vs mRMR (int. | target)"], table,
                          "Restricted candidate pools (K = 10): without care-process programs, and single generators of the large pools (about 1,000 "
                          "candidates, twice the standard union pool). Verdicts b/ns/w of the paired bootstrap.", *inputs)

    def proxy_table():
        table, inputs = [], []
        for path in sorted((A / "proxy_correlation").glob("heldout_*.csv")):
            inputs.append(path)
            agg = defaultdict(list)
            for r in m.read_csv(path):
                if int(r["step"]) == 0 or r["spearman"] in ("", "nan") or math.isnan(float(r["spearman"])):
                    continue
                agg[(r["proxy"], r["truth"], "union" if r["scope"] == "union" else "within")].append(float(r["spearman"]))
            mean = lambda key: f"{np.mean(agg[key]):.2f}" if agg[key] else "--"  # noqa: E731
            table.append([path.stem.replace("heldout_f_mimic_iv_", ""),
                          f"{mean(('mrmr', 'cv_loglik', 'within'))} / {mean(('mrmr', 'cv_loglik', 'union'))}",
                          f"{mean(('mrmr', 'val_auprc', 'within'))} / {mean(('mrmr', 'val_auprc', 'union'))}",
                          f"{mean(('nmifs', 'val_auprc', 'within'))} / {mean(('nmifs', 'val_auprc', 'union'))}",
                          f"{mean(('objective', 'val_auprc', 'within'))} / {mean(('objective', 'val_auprc', 'union'))}"])
        if table:
            m.write_table("proxy_heldout", ["Task", "mRMR vs CV log-lik. gain (within / union)", "mRMR vs VAL AUPRC gain", "NMIFS vs VAL AUPRC gain",
                                            "CV objective vs VAL AUPRC gain"], table,
                          "Rank correlation (Spearman, steps 1-3 of forward selection) of each score with a candidate's true marginal value, within each "
                          "generator's candidates and on the union. The VAL AUPRC gain is a held-out truth independent of the selection objective.", *inputs)

    def q_rule_table():
        path = A / "q_rule/q_rule.csv"
        if not path.exists():
            return
        rows = m.read_csv(path)
        table = []
        for dom, keep in (("ICU (16 runs)", lambda r: not r["run"].startswith("drive")), ("drives (3 budgets)", lambda r: r["run"].startswith("drive"))):
            for method in ("SRS", "SRS+T"):
                for rule, text in (("val_auprc", "VAL AUPRC (used)"), ("masked_auprc", "masked VAL AUPRC"), ("masked_ll", "masked VAL log-likelihood")):
                    sel = [r for r in rows if keep(r) and r["method"] == method and r["rule"] == rule]
                    if sel:
                        table.append([dom, method, text, sum(float(r["chosen_q"]) > 0 for r in sel), f"{np.mean([float(r['internal_auprc']) for r in sel]):.4f}",
                                      f"{np.mean([float(r['mean_target_auprc']) for r in sel]):.4f}"])
        m.write_table("q_rule", ["Domain", "Method", "Rule for q", "q = 0.25 chosen", "Internal AUPRC", "Mean target AUPRC"], table,
                      "Choosing the dropout rate (likelihood-decision SRS): the clean VAL AUPRC used by SRS against VAL objectives averaged over the dropout "
                      "scenarios.", path)

    def ap_case_study():
        for name in ("gap1h_k10", "aki_gap6h_k10"):
            path = A / "case_study_ap" / name / "case_study.csv"
            if not path.exists():
                continue
            rows = m.read_csv(path)
            table = [[r["position"], r["generator"], r["program"], f"{float(r['train_firing_rate']):.3f}", f"{float(r['train_precision']):.2f}",
                      f"{float(r['probe_coef']):+.2f}", r.get("branch_eicu", "--"), r.get("branch_vitals_only", "--")] for r in rows]
            m.write_table(f"ap_case_study_{name}", ["#", "Generator", "Program", "TRAIN firing", "Precision", "Probe weight", "eICU", "vitals only"], table,
                          f"The SRS+T registry ({name.replace('_k10', '').replace('gap', 'gap ')}, K = 10), program by program, with its resolution at eICU and "
                          f"under vitals only.", path)

    def ap_main_numbers():
        """Absolute AUPRC per target at K = 10 (paper text and the per-target table)."""
        table, inputs = [], []
        for label in STANDARD:
            path = A / "union_ap/union" / label / "rows.csv"
            if not path.exists():
                continue
            inputs.append(path)
            rows = {r["registry"]: r for r in m.read_csv(path)}
            for reg in ("srs_ap_t", "srs_ap", "nmifs", "mrmr", "l1_dropout", "mrmr_filter_known"):
                if reg not in rows:
                    continue
                r = rows[reg]
                controlled = np.mean([float(r[f"{t}_auprc"]) for t in TARGET_ORDER[2:] if f"{t}_auprc" in r])
                table.append([label.replace("f_mimic_iv_", ""), LABEL[reg], f"{float(r['internal_auprc']):.3f}", f"{float(r['eicu_auprc']):.3f}",
                              f"{float(r['hirid_auprc']):.3f}", f"{controlled:.3f}", f"{float(r['mean_target_auprc']):.3f}",
                              f"{float(r['mean_target_auprc']) - float(r['internal_auprc']):+.3f}"])
        if table:
            m.write_table("ap_absolute_k10", ["Task", "Selection", "Internal", "eICU", "HiRID", "Controlled (mean of 5)", "Mean target", "Target - internal"], table,
                          "Absolute AUPRC at K = 10, every registry deployed with surrogates.", *inputs)

    return [ap_main_comparison, ap_significance, ap_ablation_by_target, ap_drive_table, ap_stress, ap_resplit, ap_robustness, ap_sensitivity,
            deployment_modes_table, hospital_table, sql_table, sql_scale_table, ap_timing_table, exact_table, restricted_table, proxy_table, q_rule_table, ap_case_study, ap_main_numbers]
