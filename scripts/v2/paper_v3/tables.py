"""LaTeX tables of the PVLDB v3 paper: best value in bold, second best underlined (per column, among the selectors). Every
number is computed from artifacts/v2 (SRS-AP results after the exact-tie fix; extra metrics, the NMIFS+AP hybrid, the
reference registries, seed replicates and extra pairs of 2026-11-05); the CSV twin of each table is written for the
source-data map.

  PYTHONDONTWRITEBYTECODE=1 /d/BRCoo/miniconda/python scripts/v2/paper_v3/tables.py
"""
from __future__ import annotations

import glob
import json
import os

import numpy as np
import pandas as pd

import data
from style import ART, LABEL, OUT_SRC, OUT_TAB

LABEL = dict(LABEL)
LABEL.update({"fs_cv": "Wrapper-lik", "srs_ap_unaware": "Wrapper-AP", "nmifs_ap_t": "NMIFS+AP", "filter_nmifs": "Filter+NMIFS",
              "srs_c_val": r"\srst-lik", "srs_c_unknown_val": r"\srs-lik"})
MAIN_ORDER = [("Ours", ["srs_ap_t", "srs_ap", "srs_c_val"]),
              ("Informed", ["filter_nmifs", "nmifs_ap_t", "srs_ap_unaware", "fs_cv"]),
              ("Filters", ["nmifs", "mrmr", "mrmr_stratified", "jmi", "cmim", "mrmr_filter_known"]),
              ("Pattern sets", ["mmrfs", "mmr_ig", "mms_xin_ig", "dpp_ig"]),
              ("Embedded", ["l1_logistic", "l1_dropout"])]
CONTROLLED = ("drop_labs", "drop_ventilation", "drop_bp", "drop_urine", "vitals_only")
#: runs per ICU setting among the 16 (K = 5, 10, 20 and the large pools at K = 10, 20)
RUNS_PER_LABEL = {"f_mimic_iv_gap1h": 5, "f_mimic_iv_gap2h": 3, "f_mimic_iv_gap4h": 3, "f_mimic_iv_aki_gap6h": 5}


def mark(values: list, higher: bool = True, fmt: str = "{:.3f}", skip: set[int] | None = None, digits: int = 3) -> list[str]:
    """Bold the best and underline the second best value of a column (ties share the mark); ``skip`` rows are unmarked."""
    skip = skip or set()
    pool = [(v, i) for i, v in enumerate(values) if i not in skip and v is not None and not np.isnan(v)]
    order = sorted({round(v, digits) for v, _ in pool}, reverse=higher)
    best, second = (order + [None, None])[:2]
    out = []
    for i, v in enumerate(values):
        if v is None or (isinstance(v, float) and np.isnan(v)):
            out.append("--")
            continue
        s = fmt.format(v)
        if i in skip:
            out.append(r"\textit{" + s + "}")
        elif round(v, digits) == best:
            out.append(r"\textbf{" + s + "}")
        elif second is not None and round(v, digits) == second:
            out.append(r"\underline{" + s + "}")
        else:
            out.append(s)
    return out


def write(name: str, latex: str, frame: pd.DataFrame) -> None:
    (OUT_TAB / f"{name}.tex").write_text(latex, encoding="utf-8")
    frame.to_csv(OUT_SRC / f"{name}.csv", index=False)
    print(f"wrote {name}")


def tally(sub: pd.DataFrame) -> str:
    if sub is None or len(sub) == 0:
        return "--"
    v = np.where(sub["low"] > 0, 1, np.where(sub["high"] < 0, -1, 0))
    return f"{(v == 1).sum()}/{(v == 0).sum()}/{(v == -1).sum()}"


def read_all(pattern: str) -> pd.DataFrame:
    return pd.concat([pd.read_csv(f) for f in glob.glob(str(ART / pattern))], ignore_index=True)


def rows_all() -> pd.DataFrame:
    return read_all("extra_metrics/*/*/rows_all.csv")


def pair_frames() -> pd.DataFrame:
    """Every paired result against the main configurations, one frame: columns a, b, metric (internal / mean_target), low,
    high. A pair computed by several drivers is taken from the first source in this order: union_ap, ablations, NMIFS+AP,
    references, extra pairs (same registries; the later drivers re-ran the bootstrap)."""
    p = data.paired_all()[["a", "b", "metric", "point_diff", "low", "high"]]
    a = data.ablation_all()
    a = a[a.target.isin(["internal", "mean_over_targets"])].copy()
    a["metric"] = a.target.map({"internal": "internal", "mean_over_targets": "mean_target"})
    sources = [p, a[["a", "b", "metric", "point_diff", "low", "high"]]]
    for pattern in ("nmifs_ap/union*/*/paired.csv", "references/union*/*/paired.csv", "paired_extra/union*/*/paired.csv"):
        df = read_all(pattern)
        if "run" in df.columns:  # the 16 ICU runs only (the glob also matches the re-splits)
            df = df[~df.run.astype(str).str.contains("resplit")]
        sources.append(df[df.metric.isin(["internal", "mean_target"])][["a", "b", "metric", "point_diff", "low", "high"]])
    out, seen = [], set()
    for rank, df in enumerate(sources):
        for key, g in df.groupby(["a", "b", "metric"]):
            if key in seen:
                continue
            seen.add(key)
            out.append(g.assign(source=rank))
    return pd.concat(out, ignore_index=True)


def main_table() -> None:
    ra = rows_all()
    refs = read_all("references/union*/*/rows.csv")
    fn = refs[refs.registry == "filter_nmifs"].copy()
    fn["mean_target_auroc"] = np.nan
    ra = pd.concat([ra, fn], ignore_index=True)
    P = pair_frames()
    methods = [m for _, ms in MAIN_ORDER for m in ms]
    piv = ra[ra.registry.isin(methods)].copy()
    piv["controlled"] = piv[[f"{t}_auprc" for t in CONTROLLED]].mean(axis=1)
    ranks = piv.pivot_table(index="run", columns="registry", values="mean_target_auprc").rank(axis=1, ascending=False)
    assert ranks.shape == (16, 18), ranks.shape
    stats = {}
    for m in methods:
        s = piv[piv.registry == m]
        assert s.run.nunique() == 16, m
        stats[m] = {"internal": s.internal_auprc.mean(), "eicu": s.eicu_auprc.mean(), "hirid": s.hirid_auprc.mean(),
                    "controlled": s.controlled.mean(), "target": s.mean_target_auprc.mean(),
                    "auroc": s.mean_target_auroc.mean() if s.mean_target_auroc.notna().all() else np.nan, "rank": ranks[m].mean()}

        def t(a, metric):
            sub = P[(P.a == a) & (P.b == m) & (P.metric == metric)]
            return tally(sub) if len(sub) == 16 else ("--" if len(sub) == 0 else f"?{len(sub)}")

        stats[m]["t_int"] = "--" if m in ("srs_ap_t",) else t("srs_ap_t", "internal")
        stats[m]["t_tgt"] = "--" if m in ("srs_ap_t",) else t("srs_ap_t", "mean_target")
        stats[m]["s_tgt"] = "--" if m in ("srs_ap_t", "srs_ap") else t("srs_ap", "mean_target")
    # reference rows (not selectors): random K, all programs, supervised GBDT on full windows
    rk = refs[refs.registry == "random_k"].copy()
    rk["controlled"] = rk[[f"{t}_auprc" for t in CONTROLLED]].mean(axis=1)
    rk = rk.groupby("run")[["internal_auprc", "eicu_auprc", "hirid_auprc", "controlled", "mean_target_auprc"]].mean()
    al = refs[refs.registry == "all_programs"].copy()
    al["controlled"] = al[[f"{t}_auprc" for t in CONTROLLED]].mean(axis=1)
    sup = pd.read_csv(ART / "paper_v3_cache/supervised_refs.csv")
    sup = sup[sup.model == "gbdt_stats"].groupby(["label", "cohort"]).auprc.mean().unstack()
    w = pd.Series(RUNS_PER_LABEL)
    refs_rows = {"random_k": {"internal": rk.internal_auprc.mean(), "eicu": rk.eicu_auprc.mean(), "hirid": rk.hirid_auprc.mean(),
                              "controlled": rk.controlled.mean(), "target": rk.mean_target_auprc.mean()},
                 "all_programs": {"internal": al.internal_auprc.mean(), "eicu": al.eicu_auprc.mean(), "hirid": al.hirid_auprc.mean(),
                                  "controlled": al.controlled.mean(), "target": al.mean_target_auprc.mean()},
                 "supervised": {"internal": float((sup.internal_test * w).sum() / w.sum()), "eicu": float((sup.eicu * w).sum() / w.sum()),
                                "hirid": float((sup.hirid * w).sum() / w.sum()), "controlled": np.nan, "target": np.nan}}
    ref_names = {"random_k": r"Random $K$ (20 draws)", "all_programs": r"All programs ($|\C|$)", "supervised": "Supervised trees, full windows"}
    keys = ("internal", "eicu", "hirid", "controlled", "target", "auroc", "rank")
    allrows = [stats[m] for m in methods] + [{**refs_rows[k], "auroc": np.nan, "rank": np.nan} for k in refs_rows]
    skip = set(range(len(methods), len(allrows)))
    cols = {c: mark([r[c] for r in allrows], higher=(c != "rank"), fmt="{:.2f}" if c == "rank" else "{:.3f}", digits=2 if c == "rank" else 3,
                    skip=skip) for c in keys}
    lines = [r"\begin{tabular}{@{}llccccccccccc@{}}", r"\toprule",
             r" & & \multicolumn{6}{c}{Mean over the 16 runs} & & \multicolumn{2}{c}{\srst: b/n.s./w} & \srs: b/n.s./w \\",
             r"\cmidrule(lr){3-8}\cmidrule(lr){10-11}\cmidrule(l){12-12}",
             r"Family & Method & Internal & eICU & HiRID & Controlled & Targets & AUROC & Rank & Internal & Targets & Targets \\",
             r"\midrule"]
    i = 0
    out = []
    for g, (family, ms) in enumerate(MAIN_ORDER):
        for j, m in enumerate(ms):
            fam = (r"\multirow{%d}{*}{%s}" % (len(ms), family)) if (j == 0 and len(ms) > 1) else (family if len(ms) == 1 else "")
            name = {"srs_ap_t": r"\srst", "srs_ap": r"\srs"}.get(m, LABEL[m])
            lines.append(f"{fam} & {name} & " + " & ".join(cols[c][i] for c in keys) + f" & {stats[m]['t_int']} & {stats[m]['t_tgt']} & {stats[m]['s_tgt']} \\\\")
            out.append({"family": family, "method": LABEL.get(m, m), **stats[m]})
            i += 1
        lines.append(r"\midrule")
    for j, k in enumerate(refs_rows):
        fam = r"\multirow{3}{*}{\textit{Reference}}" if j == 0 else ""
        lines.append(f"{fam} & \\textit{{{ref_names[k]}}} & " + " & ".join(cols[c][i] for c in keys) + r" & -- & -- & -- \\")
        out.append({"family": "Reference", "method": k, **refs_rows[k]})
        i += 1
    lines += [r"\bottomrule", r"\end{tabular}"]
    write("tab_main", "% generated by scripts/v2/paper_v3/tables.py -- do not edit\n" + "\n".join(lines) + "\n", pd.DataFrame(out))


SEEDED = ("srs_ap_t", "srs_ap", "nmifs_ap_t", "srs_ap_unaware", "srs_c_val", "fs_cv")


def main_table_v2() -> None:
    """Table 2 after the second review: real targets as the primary endpoint, CV-based selectors averaged over five CV
    seeds, pooled seed-aware differences to SRS+T with 95% intervals and seed-aware tallies (round2.py seeds_pooled)."""
    import json as _json
    T = pd.read_csv(OUT_SRC / "q2_seeds_long.csv")
    R = pd.read_csv(OUT_SRC / "q3_joint.csv")
    R = R[R.a == "srs_ap_t"]
    methods = [m for _, ms in MAIN_ORDER for m in ms]
    cols = ("internal", "eicu", "hirid", "real", "controlled")
    stats = {}
    for m in methods:
        d = T[T.registry == m]
        if m not in SEEDED:
            d = d[d.seed == 0]
        per_run = d.groupby("run")[list(cols)].mean()
        assert len(per_run) == 16, (m, len(per_run))
        stats[m] = {c: float(per_run[c].mean()) for c in cols}
        stats[m]["n_seeds"] = int(d.groupby("run").seed.nunique().min())
        if os.environ.get("ALLOW_PARTIAL_SEEDS") != "1":
            assert m not in SEEDED or stats[m]["n_seeds"] == 5, (m, stats[m]["n_seeds"])
        if m == "srs_ap_t":
            stats[m].update({"delta": "--", "t_int": "--", "t_real": "--"})
            continue
        r_real = R[(R.b == m) & (R.endpoint == "real")].iloc[0]
        r_int = R[(R.b == m) & (R.endpoint == "internal")].iloc[0]
        if os.environ.get("ALLOW_PARTIAL_SEEDS") != "1":
            assert r_real.n_seeds_b == (5 if m in SEEDED else 1), (m, r_real.n_seeds_b)
        sg = lambda x: (f"{x:+.2f}" if abs(x) < 0.05 else f"{x:+.1f}").replace("-", "$-$")  # noqa: E731
        stats[m]["delta"] = f"{sg(100 * r_real.point)} [{sg(100 * r_real.re_low)}, {sg(100 * r_real.re_high)}]"
        stats[m]["t_int"], stats[m]["t_real"] = r_int.tally, r_real.tally
    # references: random K, all programs (reference registries), K window statistics, LR and trees on all statistics
    refs = read_all("references/union*/*/rows.csv")
    refs["controlled"] = refs[[f"{t}_auprc" for t in CONTROLLED]].mean(axis=1)
    refs["real"] = refs[["eicu_auprc", "hirid_auprc"]].mean(axis=1)

    def ref_row(frame):
        g = frame.groupby("run")[["internal_auprc", "eicu_auprc", "hirid_auprc", "real", "controlled"]].mean()
        return {"internal": g.internal_auprc.mean(), "eicu": g.eicu_auprc.mean(), "hirid": g.hirid_auprc.mean(), "real": g.real.mean(),
                "controlled": g.controlled.mean()}

    ref_rows = {"random_k": ref_row(refs[refs.registry == "random_k"]), "all_programs": ref_row(refs[refs.registry == "all_programs"])}
    srows = []
    for root, label in data.ICU_RUNS:
        k = data.K_OF[root]
        d = {r["model"]: r for r in _json.loads((ART / "stat_baselines" / label / "rows.json").read_text(encoding="utf-8"))["rows"]}
        for model, name in ((f"stats_l1_k{k}", "stats_l1"), ("lr_stats", "lr_stats"), ("gbdt_stats", "gbdt_stats")):
            r = d[model]
            srows.append({"run": f"{root}/{label}", "registry": name, "internal": r["internal_auprc"], "eicu": r["eicu_auprc"], "hirid": r["hirid_auprc"],
                          "real": r["real_auprc"], "controlled": float(np.mean([r[f"{t}_auprc"] for t in CONTROLLED]))})
    sr = pd.DataFrame(srows)
    for name in ("stats_l1", "lr_stats", "gbdt_stats"):
        g = sr[sr.registry == name]
        ref_rows[name] = {c: float(g[c].mean()) for c in cols}
    ref_names = {"random_k": r"Random $K$ programs", "all_programs": r"All programs of the pool", "stats_l1": r"$K$ window statistics, L1",
                 "gbdt_stats": r"Trees, all statistics"}
    allrows = [stats[m] for m in methods] + [ref_rows[k] for k in ref_names]
    skip = set(range(len(methods), len(allrows)))
    marked = {c: mark([r[c] for r in allrows], skip=skip) for c in cols}
    lines = [r"\begin{tabular}{@{}llccccccccc@{}}", r"\toprule",
             r" & & \multicolumn{5}{c}{AUPRC, mean over the 16 runs} & \multicolumn{3}{c}{\srst $-$ method} \\",
             r"\cmidrule(lr){3-7}\cmidrule(l){8-10}",
             r"Family & Method & Internal & eICU & HiRID & Real & Controlled & $\Delta$ Real ($\times 100$) & b/n.s./w Int. & b/n.s./w Real \\",
             r"\midrule"]
    i = 0
    out = []
    for family, ms in MAIN_ORDER:
        for j, m in enumerate(ms):
            fam = (r"\multirow{%d}{*}{%s}" % (len(ms), family)) if (j == 0 and len(ms) > 1) else (family if len(ms) == 1 else "")
            name = {"srs_ap_t": r"\srst", "srs_ap": r"\srs"}.get(m, LABEL[m])
            lines.append(f"{fam} & {name} & " + " & ".join(marked[c][i] for c in cols) +
                         f" & {stats[m]['delta']} & {stats[m]['t_int']} & {stats[m]['t_real']} \\\\")
            out.append({"family": family, "method": LABEL.get(m, m), **stats[m]})
            i += 1
        lines.append(r"\midrule")
    for j, k in enumerate(ref_names):
        fam = r"\multirow{4}{*}{\textit{Reference}}" if j == 0 else ""
        lines.append(f"{fam} & \\textit{{{ref_names[k]}}} & " + " & ".join(marked[c][i] for c in cols) + r" & -- & -- & -- \\")
        out.append({"family": "Reference", "method": k, **ref_rows[k]})
        i += 1
    lines += [r"\bottomrule", r"\end{tabular}"]
    write("tab_main", "% generated by scripts/v2/paper_v3/tables.py (main_table_v2) -- do not edit\n" + "\n".join(lines) + "\n", pd.DataFrame(out))


def per_run_table() -> None:
    """Extended version: AUPRC per run of the shipped registries (CV seed 0), internal and real targets."""
    T = pd.read_csv(OUT_SRC / "q2_seeds_long.csv")
    T = T[T.seed == 0]
    meths = [("srs_ap_t", r"\srst"), ("srs_ap", r"\srs"), ("nmifs_ap_t", "NMIFS+AP"), ("srs_ap_unaware", "Wrapper-AP"), ("nmifs", "NMIFS"),
             ("mrmr", "mRMR")]
    lines = [r"\begin{tabular}{@{}l" + "cc" * len(meths) + "@{}}", r"\toprule",
             " & " + " & ".join(r"\multicolumn{2}{c}{" + n + "}" for _, n in meths) + r" \\",
             "".join(r"\cmidrule(lr){%d-%d}" % (2 + 2 * i, 3 + 2 * i) for i in range(len(meths))),
             "Run & " + " & ".join(["Int. & Real"] * len(meths)) + r" \\", r"\midrule"]
    out = []
    for root, label in data.ICU_RUNS:
        run = f"{root}/{label}"
        d = T[T.run == run].set_index("registry")
        vals = []
        for m, _ in meths:
            vals += [d.loc[m, "internal"], d.loc[m, "real"]]
        marked_int = mark([vals[2 * i] for i in range(len(meths))])
        marked_real = mark([vals[2 * i + 1] for i in range(len(meths))])
        cells = [x for pair in zip(marked_int, marked_real) for x in pair]
        lines.append(data.run_name(root, label) + " & " + " & ".join(cells) + r" \\")
        out.append({"run": run, **{f"{m}_{e}": vals[2 * i + j] for i, (m, _) in enumerate(meths) for j, e in enumerate(("internal", "real"))}})
    lines += [r"\bottomrule", r"\end{tabular}"]
    write("tab_per_run", "% generated by scripts/v2/paper_v3/tables.py (per_run_table) -- do not edit\n" + "\n".join(lines) + "\n", pd.DataFrame(out))


def significance_table() -> None:
    """Extended version: SRS+T against each established selector, runs better / n.s. / worse internally and over all
    targets, seed-aware like Table 2 (per-run difference of CV-seed means with the run's joint-bootstrap and seed
    variance, q3_joint_runs.csv): with plain 95% intervals, Holm's correction within each run over the twelve selectors,
    and Benjamini-Hochberg over all runs and selectors."""
    from scipy.stats import norm

    d = pd.read_csv(OUT_SRC / "q3_joint_runs.csv")
    order = ["nmifs", "mrmr", "mrmr_stratified", "jmi", "cmim", "mmrfs", "mmr_ig", "mms_xin_ig", "dpp_ig", "l1_logistic", "l1_dropout", "mrmr_filter_known"]
    d = d[(d.a == "srs_ap_t") & d.b.isin(order) & d.endpoint.isin(["internal", "targets"])].copy()
    d["p"] = 2 * (1 - norm.cdf((d["diff"] / d["se"]).abs()))
    d["sign"] = np.sign(d["diff"])
    verdicts = []
    for e, g in d.groupby("endpoint"):
        g = g.copy()
        g["ci"] = np.where(g["diff"] - 1.96 * g["se"] > 0, 1, np.where(g["diff"] + 1.96 * g["se"] < 0, -1, 0))
        holm = []
        for run, h in g.groupby("run"):  # Holm within each run over the twelve comparisons
            h = h.sort_values("p")
            m = len(h)
            adj, running = [], 0.0
            for i, pv in enumerate(h.p):
                running = max(running, min(1.0, (m - i) * pv))
                adj.append(running)
            holm.append(pd.Series(adj, index=h.index))
        g["p_holm"] = pd.concat(holm)
        srt = g.sort_values("p")  # Benjamini-Hochberg over all runs and comparisons
        m = len(srt)
        bh = (srt.p * m / np.arange(1, m + 1))[::-1].cummin()[::-1].clip(upper=1.0)
        g["p_bh"] = bh.reindex(g.index)
        g["holm"] = np.where(g.p_holm < 0.05, g.sign, 0)
        g["bh"] = np.where(g.p_bh < 0.05, g.sign, 0)
        verdicts.append(g)
    V = pd.concat(verdicts)

    def tal(s):
        return f"{int((s == 1).sum())}/{int((s == 0).sum())}/{int((s == -1).sum())}"

    lines = [r"\begin{tabular}{@{}lcccccc@{}}", r"\toprule",
             r" & \multicolumn{3}{c}{Internal} & \multicolumn{3}{c}{All targets} \\", r"\cmidrule(lr){2-4}\cmidrule(l){5-7}",
             r"Comparator & CI & Holm & BH & CI & Holm & BH \\", r"\midrule"]
    out = []
    for b in order:
        gi, gt = V[(V.b == b) & (V.endpoint == "internal")], V[(V.b == b) & (V.endpoint == "targets")]
        cells = [tal(gi.ci), tal(gi.holm), tal(gi.bh), tal(gt.ci), tal(gt.holm), tal(gt.bh)]
        lines.append(LABEL.get(b, b) + " & " + " & ".join(cells) + r" \\")
        out.append({"comparator": b, "internal_ci": cells[0], "internal_holm": cells[1], "internal_bh": cells[2], "targets_ci": cells[3],
                    "targets_holm": cells[4], "targets_bh": cells[5]})
    lines += [r"\bottomrule", r"\end{tabular}"]
    write("tab_significance", "% generated by scripts/v2/paper_v3/tables.py (significance_table, seed-aware) -- do not edit\n" + "\n".join(lines) + "\n",
          pd.DataFrame(out))


def ablation_table() -> None:
    """Extended version: per-target means of SRS+T and its variants over the 16 runs, CV-seed means (the variant without
    surrogates in its objective was run with seed 0 only), with seed-aware tallies over all targets (q3_joint.csv)."""
    T = pd.read_csv(OUT_SRC / "q2_seeds_long.csv")
    J = pd.read_csv(OUT_SRC / "q3_joint.csv")
    targets = ["internal", "targets"] + list(data.TARGETS)
    variants = [("srs_ap_t", r"\srst"), ("srs_c_val", r"\enspace lik.\ decision"), ("srs_ap_unaware", r"\enspace no scenarios$^{a}$"),
                ("srs_ap_t_nosur", r"\enspace no surr.\ in obj.$^{d}$"), ("nmifs_ap_t", r"\enspace NMIFS screen$^{b}$"),
                ("srs_ap", r"\enspace no target schemas$^{c}$")]
    a = data.ablation_all()
    r = data.rows_all()
    vals, tal = {}, {}
    for v, _ in variants:
        out = {}
        if v == "srs_ap_t_nosur":
            s = a[(a.a == "srs_ap_t") & (a.b == v)]
            ref = r[r.registry == "srs_ap_t"].set_index("name")
            col = {"internal": "internal_auprc", "mean_over_targets": "mean_target_auprc", **{t: f"{t}_auprc" for t in data.TARGETS}}
            for t, key in [("internal", "internal"), ("mean_over_targets", "targets")] + [(t, t) for t in data.TARGETS]:
                sub = s[s.target == t].set_index("name")
                out[key] = float((ref.loc[sub.index, col[t]] - sub["point_diff"]).mean())
            sub = pair_frames()
            sub = sub[(sub.a == "srs_ap_t") & (sub.b == v) & (sub.metric == "mean_target")]
            tal[v] = tally(sub)
        else:
            g = T[T.registry == v].groupby("run")[targets].mean()
            assert len(g) == 16, (v, len(g))
            out = {t: float(g[t].mean()) for t in targets}
            tal[v] = "--" if v == "srs_ap_t" else J[(J.a == "srs_ap_t") & (J.b == v) & (J.endpoint == "targets")].iloc[0].tally
        vals[v] = out
    cols = {t: mark([vals[v][t] for v, _ in variants]) for t in targets}
    short = {"eicu": "eICU", "hirid": "HiRID", "drop_labs": "labs", "drop_ventilation": "vent.", "drop_bp": "BP", "drop_urine": "urine",
             "vitals_only": "vitals"}
    head = " & ".join([r"Int.", r"All"] + [short[t] for t in data.TARGETS])
    lines = [r"\begin{tabular}{@{}l" + "c" * len(targets) + "c@{}}", r"\toprule",
             r" & & & \multicolumn{2}{c}{Real} & \multicolumn{5}{c}{Controlled: no \ldots{} / only vitals} & \\",
             r"\cmidrule(lr){4-5}\cmidrule(lr){6-10}", "Variant & " + head + r" & b/n.s./w \\", r"\midrule"]
    rows = []
    for i, (v, text) in enumerate(variants):
        lines.append(text + " & " + " & ".join(cols[t][i] for t in targets) + f" & {tal[v]}" + r" \\")
        rows.append({"variant": v, **vals[v], "tally_targets": tal[v]})
    lines += [r"\bottomrule", r"\end{tabular}"]
    write("tab_ablation", "% generated by scripts/v2/paper_v3/tables.py -- do not edit\n" + "\n".join(lines) + "\n", pd.DataFrame(rows))


def drive_table() -> None:
    rows, paired, _ = data.drives()

    def with_k(pattern):
        return pd.concat([pd.read_csv(f).assign(k=int(f.split("drive_k")[1].split("\\")[0].split("/")[0])) for f in glob.glob(str(ART / pattern))],
                         ignore_index=True)

    hyb_rows, hyb_pair = with_k("nmifs_ap/drive_k*/*/rows.csv"), with_k("nmifs_ap/drive_k*/*/paired.csv")
    ref_rows, ref_pair = with_k("references/drive_k*/*/rows.csv"), with_k("references/drive_k*/*/paired.csv")
    rows = pd.concat([rows, hyb_rows[hyb_rows.registry == "nmifs_ap_t"], ref_rows[ref_rows.registry == "filter_nmifs"]], ignore_index=True)
    # SRS+T against SRS, post hoc (scripts/v2/paper_v3/drive_srs_vs_srst.py, same deployment and bootstrap as the ablations)
    srs_pair = with_k("union_ap/drive_k*/*/srs_pair.csv")
    srs_pair = srs_pair[srs_pair.target.isin(["internal", "mean_over_targets"])].copy()
    srs_pair["metric"] = np.where(srs_pair.target == "internal", "internal", "mean_target")
    extra = pd.concat([hyb_pair[(hyb_pair.a == "srs_ap_t") & (hyb_pair.b == "nmifs_ap_t")],
                       ref_pair[(ref_pair.a == "srs_ap_t") & (ref_pair.metric.isin(["internal", "mean_target"]))], srs_pair], ignore_index=True)
    extra["verdict"] = np.where(extra.low > 0, 1, np.where(extra.high < 0, -1, 0))
    paired = pd.concat([paired, extra], ignore_index=True)
    methods = ["srs_ap_t", "srs_ap", "srs_c_val", "srs_c_unknown_val", "nmifs_ap_t", "nmifs", "mrmr", "mmrfs", "jmi", "cmim", "l1_logistic",
               "l1_dropout", "mrmr_filter_known", "filter_nmifs"]
    vals, marks = {}, {}
    for m in methods:
        for k in (5, 10, 20):
            s = rows[(rows.registry == m) & (rows.k == k)]
            vals[(m, k, "internal")] = float(s.internal_auprc.iloc[0])
            vals[(m, k, "target")] = float(s.mean_target_auprc.iloc[0])
            for metric, key in (("internal", "internal"), ("mean_target", "target")):
                pp = paired[(paired.a == "srs_ap_t") & (paired.b == m) & (paired.k == k) & (paired.metric == metric)]
                marks[(m, k, key)] = int(pp.verdict.iloc[0]) if len(pp) else None
    cells = {}
    for k in (5, 10, 20):
        for key in ("internal", "target"):
            col = mark([vals[(m, k, key)] for m in methods])
            for m, c in zip(methods, col):
                v = marks[(m, k, key)]
                cells[(m, k, key)] = c + (r"$^{\ast}$" if v == 1 else (r"$^{\circ}$" if v == -1 else ""))
    lines = [r"\begin{tabular}{@{}lcccccc@{}}", r"\toprule",
             r" & \multicolumn{2}{c}{$K = 5$} & \multicolumn{2}{c}{$K = 10$} & \multicolumn{2}{c}{$K = 20$} \\",
             r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(l){6-7}",
             r"Method & Int. & Tgt. & Int. & Tgt. & Int. & Tgt. \\",
             r"\midrule"]
    out = []
    for m in methods:
        name = {"srs_ap_t": r"\srst", "srs_ap": r"\srs"}.get(m, LABEL[m])
        lines.append(name + " & " + " & ".join(cells[(m, k, key)] for k in (5, 10, 20) for key in ("internal", "target")) + r" \\")
        out.append({"method": m, **{f"k{k}_{key}": vals[(m, k, key)] for k in (5, 10, 20) for key in ("internal", "target")},
                    **{f"k{k}_{key}_verdict": marks[(m, k, key)] for k in (5, 10, 20) for key in ("internal", "target")}})
        if m in ("srs_ap", "srs_c_unknown_val", "nmifs_ap_t"):
            lines.append(r"\midrule")
    lines += [r"\bottomrule", r"\end{tabular}"]
    write("tab_drives", "% generated by scripts/v2/paper_v3/tables.py -- do not edit\n" + "\n".join(lines) + "\n", pd.DataFrame(out))


def dataset_table() -> None:
    rows = [("Vasopressor", "1, 2, 4 h", "MIMIC-IV", "source", "3,670 / 4,013", 3670 / 7683, "17", "--"),
            ("", "", "eICU", "target", "3,390 / 3,475", 3390 / 6865, "16", "vent."),
            ("", "", "HiRID", "target", "1,236 / 1,499", 1236 / 2735, "17", "--"),
            ("AKI onset", "6 h", "MIMIC-IV", "source", "12,054 / 11,803", 12054 / 23857, "18", "--"),
            ("", "", "eICU", "target", "8,974 / 8,824", 8974 / 17798, "17", "vent."),
            ("", "", "HiRID", "target", "1,906 / 1,888", 1906 / 3794, "18", "--"),
            ("Drive failure", "2 d", "Seagate", "source", "1,898 / 1,898", 0.5, "20", "--"),
            ("", "", "3 targets", "target", "319--1,092 ea.", 0.5, "13--14", "6--7 attr.")]
    lines = [r"\begin{tabular}{@{}llllrrcl@{}}", r"\toprule",
             r"Task & Gap & Database & Role & Pos.\ / ctrl. & Prev. & $|F|$ & Absent \\", r"\midrule"]
    for i, row in enumerate(rows):
        lines.append(" & ".join(list(row[:5]) + [f"{row[5]:.2f}", row[6], row[7]]) + r" \\")
        if i in (2, 5):
            lines.append(r"\midrule")
    lines += [r"\bottomrule", r"\end{tabular}"]
    frame = pd.DataFrame(rows, columns=["task", "gap", "database", "role", "pos_controls", "prevalence", "n_features", "absent"])
    write("tab_datasets", "% generated by scripts/v2/paper_v3/tables.py (counts from paper/v2_assets/tables/data_overview.csv)\n" + "\n".join(lines) + "\n", frame)


def hybrid_table() -> None:
    """Extended version: window-statistic models with and without the K = 10 programs of a registry, AUPRC averaged over
    the four standard settings (run_hybrid_programs_stats.py); the last column is the contrast with the row it extends,
    over all targets, with a joint patient bootstrap interval (x 100)."""
    M = pd.read_csv(OUT_SRC / "q4_hybrid_means.csv")
    C = pd.read_csv(OUT_SRC / "q4_hybrid_contrasts.csv")
    rows = [("stats_lr", r"All statistics, logistic", None), ("stats_lr+srs", r"\enspace + \srst programs", "stats_lr"),
            ("stats_lr+nmifs", r"\enspace + NMIFS programs", "stats_lr"), ("stats_gbdt", r"All statistics, trees", None),
            ("stats_gbdt+srs", r"\enspace + \srst programs", "stats_gbdt"), ("kstats", r"$K$ statistics", None),
            ("kstats+srs", r"\enspace + \srst programs", "kstats"), ("kstats2", r"$2K$ statistics", None),
            ("kstats+srs", r"$K$ statistics + \srst programs", "kstats2"), ("kstats+nmifs", r"$K$ statistics + NMIFS programs", "kstats2"),
            ("prog_srs", r"\srst programs alone", None), ("prog_nmifs", r"NMIFS programs alone", None)]
    ends = ["internal", "real", "controlled", "targets"]
    val = {(r.model, r.endpoint): r["mean"] for _, r in M.iterrows()}
    lines = [r"\begin{tabular}{@{}lccccc@{}}", r"\toprule",
             r"Model & Int. & Real & Controlled & All & $\Delta$ All ($\times 100$) \\", r"\midrule"]
    out = []
    for i, (m, text, ref) in enumerate(rows):
        cells = [f"{val[(m, e)]:.3f}" for e in ends]
        if ref is None:
            delta = "--"
        else:
            c = C[(C.a == m) & (C.b == ref) & (C.endpoint == "targets")].iloc[0]
            delta = f"{100 * c['diff']:+.2f} [{100 * c.low:+.2f}, {100 * c.high:+.2f}]".replace("-", "$-$")
        lines.append(text + " & " + " & ".join(cells) + f" & {delta}" + r" \\")
        if i in (2, 4, 6, 9):
            lines.append(r"\midrule")
        out.append({"model": m, "label": text, "reference": ref, **{e: val[(m, e)] for e in ends}, "delta_all": delta})
    lines += [r"\bottomrule", r"\end{tabular}"]
    write("tab_hybrid", "% generated by scripts/v2/paper_v3/tables.py (hybrid_table) -- do not edit\n" + "\n".join(lines) + "\n", pd.DataFrame(out))


def confirm_table() -> None:
    """Extended version: the confirmatory study, means over its 6 runs (2 tasks x K in {5, 10, 20}) per endpoint, and SRS+T
    minus each registry at the hospitals that lack features with a 95% joint patient bootstrap interval (x 100)."""
    M = pd.read_csv(OUT_SRC / "q4_confirmatory_means.csv")
    S = pd.read_csv(OUT_SRC / "q4_confirmatory_summary.csv")
    regs = ["srs_ap_t", "srs_ap", "srs_c_val", "nmifs", "cmim", "mrmr", "dpp_ig", "l1_dropout", "mmrfs", "mms_xin_ig", "mrmr_filter_known",
            "l1_logistic", "mmr_ig", "mrmr_stratified", "jmi"]
    names = {"srs_ap_t": r"\srst", "srs_ap": r"\srs", "srs_c_val": r"\srst-lik"}
    ends = [("sites", "Sites"), ("databases", "MIMIC, HiRID"), ("controlled", "Controlled"), ("internal", "Internal")]
    val = {(r.registry, r.endpoint): r["mean"] for _, r in M.iterrows()}
    cols = {e: mark([val[(m, e)] for m in regs]) for e, _ in ends}
    lines = [r"\begin{tabular}{@{}lccccc@{}}", r"\toprule",
             "Registry & " + " & ".join(t for _, t in ends) + r" & $\Delta$ sites ($\times 100$) \\", r"\midrule"]
    out = []
    for i, m in enumerate(regs):
        if m == "srs_ap_t":
            delta = "--"
        else:
            r = S[(S.a == "srs_ap_t") & (S.b == m) & (S.endpoint == "sites")].iloc[0]
            delta = f"{100 * r['diff']:+.2f} [{100 * r.low:+.2f}, {100 * r.high:+.2f}]".replace("-", "$-$")
        lines.append(names.get(m, LABEL.get(m, m)) + " & " + " & ".join(cols[e][i] for e, _ in ends) + f" & {delta}" + r" \\")
        if m == "srs_c_val":
            lines.append(r"\midrule")
        out.append({"registry": m, **{e: val[(m, e)] for e, _ in ends}, "delta_sites": delta})
    lines += [r"\bottomrule", r"\end{tabular}"]
    write("tab_confirm", "% generated by scripts/v2/paper_v3/tables.py (confirm_table) -- do not edit\n" + "\n".join(lines) + "\n", pd.DataFrame(out))


def confirm2_table() -> None:
    """Extended version: the second confirmatory study (pools with window-statistic predicates), means over its 6 runs per
    endpoint (CV-based selectors averaged over CV seeds 0-4), and SRS+T minus each registry at the hospitals that lack features
    with the protocol's 95% interval (joint patient bootstrap plus CV-seed variance; x 100)."""
    M = pd.read_csv(OUT_SRC / "q5_confirmatory2_means.csv")
    S = pd.read_csv(OUT_SRC / "q5_confirmatory2_summary.csv")
    regs = ["srs_ap_t", "srs_ap", "srs_c_val", "srs_ap_t_five", "nmifs", "cmim", "mrmr", "l1_logistic", "l1_dropout", "mrmr_filter_known",
            "dpp_ig", "mmrfs", "mms_xin_ig", "mmr_ig", "jmi", "mrmr_stratified", "kstats", "lr_stats", "gbdt_stats"]
    names = {"srs_ap_t": r"\srst", "srs_ap": r"\srs", "srs_c_val": r"\srst-lik", "srs_ap_t_five": r"\srst, no pred.",
             "kstats": r"$K$ statistics", "lr_stats": "All stat., LR", "gbdt_stats": "All stat., trees"}
    ends = [("sites", "Sites"), ("databases", "MIMIC, HiRID"), ("controlled", "Controlled"), ("internal", "Internal")]
    val = {(r.registry, r.endpoint): r["mean"] for _, r in M.iterrows()}
    selectors = [m for m in regs if m not in ("kstats", "lr_stats", "gbdt_stats", "srs_ap_t_five")]
    cols = {e: dict(zip(selectors, mark([val[(m, e)] for m in selectors]))) for e, _ in ends}
    lines = [r"\begin{tabular}{@{}lccccc@{}}", r"\toprule",
             "Registry & " + " & ".join(t for _, t in ends) + r" & $\Delta$ sites ($\times 100$) \\", r"\midrule"]
    out = []
    for m in regs:
        if m == "srs_ap_t":
            delta = "--"
        else:
            r = S[(S.a == "srs_ap_t") & (S.b == m) & (S.endpoint == "sites")].iloc[0]
            delta = f"{100 * r['diff']:+.2f} [{100 * r.low:+.2f}, {100 * r.high:+.2f}]".replace("-", "$-$")
        cells = [cols[e].get(m, f"{val[(m, e)]:.3f}") for e, _ in ends]
        lines.append(names.get(m, LABEL.get(m, m)) + " & " + " & ".join(cells) + f" & {delta}" + r" \\")
        if m in ("srs_ap_t_five", "mrmr_stratified"):
            lines.append(r"\midrule")
        out.append({"registry": m, **{e: val[(m, e)] for e, _ in ends}, "delta_sites": delta})
    lines += [r"\bottomrule", r"\end{tabular}"]
    write("tab_confirm2", "% generated by scripts/v2/paper_v3/tables.py (confirm2_table) -- do not edit\n" + "\n".join(lines) + "\n", pd.DataFrame(out))


def engine_table() -> None:
    """Extended version: the shipped AKI registry (ten predicates at eICU) and the K-statistics model of the same run, both
    executed and scored entirely in SQL on eICU's AKI cohort, x1 and x10; medians of the fresh-process repeats
    (artifacts/v2/sql_engines/timing.json, sql_engine_timing.py). Load includes the row stores' index."""
    path = ART / "sql_engines/timing.json"
    if not path.exists():
        return
    rep = pd.DataFrame(json.loads(path.read_text(encoding="utf-8"))["rows"])
    rep["seconds_load_index"] = rep.seconds_load.fillna(0) + rep.seconds_index.fillna(0) if "seconds_index" in rep else rep.seconds_load
    engines = [("duckdb_memory", "DuckDB, memory"), ("duckdb_disk", "DuckDB, disk"), ("postgres", "PostgreSQL"), ("sqlite", "SQLite"),
               ("python", "Python engine")]
    cols = [("seconds_load_index", "Load"), ("seconds_tokens", "Tokens"), ("seconds_registry", "Registry"),
            ("seconds_score_registry", "Score reg."), ("seconds_score_kstats", "Score $K$-st.")]
    lines = [r"\begin{tabular}{@{}l" + "r" * (2 * len(cols)) + "@{}}", r"\toprule",
             r" & \multicolumn{5}{c}{$\times 1$ (17,798 episodes)} & \multicolumn{5}{c}{$\times 10$} \\",
             r"\cmidrule(lr){2-6}\cmidrule(l){7-11}", "Engine & " + " & ".join(t_ for _, t_ in cols * 2) + r" \\",
             r"\midrule"]
    out = []

    def fmt(v):
        return "--" if v is None or not np.isfinite(v) else (f"{v:.2f}" if v < 10 else f"{v:.0f}")

    for eng, name in engines:
        cells, rec = [], {"engine": name}
        for r_ in (1, 10):
            for c, ctext in cols:
                s = rep[(rep.engine == eng) & (rep.replicate == r_)]
                v = float(s[c].median()) if c in s and s[c].notna().any() and not (eng == "python" and c == "seconds_load_index") else None
                cells.append(fmt(v))
                rec[f"x{r_}_{c}"] = v
        lines.append(name + " & " + " & ".join(cells) + r" \\")
        out.append(rec)
    lines += [r"\bottomrule", r"\end{tabular}"]
    write("tab_engines", "% generated by scripts/v2/paper_v3/tables.py (engine_table) -- do not edit\n" + "\n".join(lines) + "\n", pd.DataFrame(out))


if __name__ == "__main__":
    engine_table()
    main_table_v2()
    per_run_table()
    significance_table()
    ablation_table()
    drive_table()
    dataset_table()
    hybrid_table()
    confirm_table()
    confirm2_table()
