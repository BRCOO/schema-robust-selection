"""Experiment figures of the PVLDB v3 paper, fourth design (after the second review of the experiments, 2026-11-05):
every text element at 7 pt or more, real targets as the primary endpoint, pooled effects with intervals, the wrapper
decomposition, extended proxy fidelity with shortlist recall, and random loss as mean differences.

Workflow of the cs-experiment-figure-studio skill: a figure contract in each docstring, the source data of every panel in
paper/v3_assets/source_data/ (prefix r1..r6), PDF/SVG/PNG exports, Nature teal-green palette (style.py). Numbers come from
round2.py (q2_*.csv) and the artifacts it reads.

  PYTHONDONTWRITEBYTECODE=1 /d/BRCoo/miniconda/python scripts/v2/paper_v3/figs_v5.py [r1 r2 r3 r4 r6]
"""
from __future__ import annotations

import glob
import sys

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

import data
import style as S
from figs_b import src
from figs_v4 import HYB, LABEL, per_target_verdicts, per_variant_tallies, read_all, style_of

S.setup()
FS = 7.0          # every text element
FS_TITLE = 7.5
mpl.rcParams.update({"font.size": FS, "axes.labelsize": FS, "xtick.labelsize": FS, "ytick.labelsize": FS, "legend.fontsize": FS,
                     "axes.titlesize": FS_TITLE})
LABEL = dict(LABEL)
LABEL.update({"srs_ap_unaware": "Wrapper-AP", "fs_cv": "Wrapper-lik", "nmifs_ap_t": "NMIFS+AP", "srs_ap_t_nosur": "no surr. in obj.",
              "srs_c_val": "SRS+T-lik", "srs_c_unknown_val": "SRS-lik", "filter_nmifs": "Filter+NMIFS"})
SRC = S.OUT_SRC


def title(ax, text: str, y: float = 1.04) -> None:
    ax.set_title(text, loc="left", fontsize=FS_TITLE, fontweight="bold", y=y, pad=0)


def letter(ax, text: str, dx: float, dy: float = 1.04) -> None:
    ax.text(dx, dy, text, transform=ax.transAxes, fontsize=9, fontweight="bold", va="bottom", ha="right", color=S.INK)


# ================================================================================================= R1
R1_HYBRID = ["srs_ap", "srs_c_val", "srs_ap_unaware", "nmifs_ap_t", "fs_cv", "filter_nmifs"]
R1_ESTABLISHED = ["nmifs", "mrmr", "mrmr_filter_known", "l1_dropout", "cmim", "mrmr_stratified", "dpp_ig", "mmrfs", "mms_xin_ig", "mmr_ig", "jmi",
                  "l1_logistic"]
R1_ORDER = R1_HYBRID + R1_ESTABLISHED
TASK_COLOR = {"Vaso 1h": S.SKY, "Vaso 2h": S.BLUE, "Vaso 4h": "#7E8FBF", "AKI": S.TEAL}


def stat_refs() -> pd.DataFrame:
    """K-matched window-statistic references per ICU run (the large pools reuse their setting's statistics)."""
    import json as _json
    rows = []
    for root, label in data.ICU_RUNS:
        k = data.K_OF[root]
        d = _json.loads((S.ART / "stat_baselines" / label / "rows.json").read_text(encoding="utf-8"))["rows"]
        by = {r["model"]: r for r in d}
        for model, name in ((f"stats_l1_k{k}", "stats_l1"), ("lr_stats", "lr_stats"), ("gbdt_stats", "gbdt_stats")):
            r = by[model]
            rows.append({"run": f"{root}/{label}", "registry": name, "k": k, "internal": r["internal_auprc"], "real": r["real_auprc"],
                         "targets": r["mean_target_auprc"]})
    return pd.DataFrame(rows)


def fig_r1() -> None:
    """Contract. Claim (RQ1): over the 16 runs, SRS+T is ahead of every established selector at the real target databases
    and internally; against the hybrids that reuse its parts its lead is below one point; the lead over NMIFS varies with
    the setting and the CV seed; registries of K programs trail K window statistics; and SRS+T's registries are more
    accurate at the source and equally portable. Evidence: (a) forest of the differences of the 16-run means with
    random-effects intervals whose sampling covariance comes from a joint patient bootstrap (dot-whisker), (b) real-target
    AUPRC against K with references (lines), (c) per-run CV-seed replicates of the lead over NMIFS (strip), (d) internal
    against real-target AUPRC per selector (scatter with fit). Data: q3_joint.csv (run_joint_bootstrap.py), q2_seeds_long.csv,
    q2_pooled_seeds.csv (per-run tallies), references, stat_baselines. Higher = better."""
    J = pd.read_csv(SRC / "q3_joint.csv")
    J = J[J.a == "srs_ap_t"]
    R = pd.read_csv(SRC / "q2_pooled_seeds.csv")
    T = pd.read_csv(SRC / "q2_seeds_long.csv")
    seed_mean = T.groupby(["run", "registry"])[["internal", "real", "targets"]].mean().reset_index()
    fig = plt.figure(figsize=(S.FULL_W, 3.0))
    top, bottom = 0.885, 0.235
    h = top - bottom

    # (a) forest: hybrids and informed comparators, then the established selectors, each sorted by the real-target difference
    ax = fig.add_axes([0.112, bottom, 0.15, h])
    real = J[J.endpoint == "real"].set_index("b")
    groups = [[b for b in R1_HYBRID if b in real.index], [b for b in R1_ESTABLISHED if b in real.index]]
    order = []
    for g in groups:
        order += sorted(g, key=lambda b: real.loc[b, "point"])
    out, ypos, y = [], {}, 0.0
    for gi, g in enumerate(groups):
        for b in sorted(g, key=lambda b: real.loc[b, "point"]):
            ypos[b] = y
            y += 1
        y += 0.6
    for b in order:
        for e, color, filled, off in (("real", S.TEAL, True, -0.17), ("internal", S.GREY, False, 0.17)):
            r = J[(J.b == b) & (J.endpoint == e)].iloc[0]
            ax.hlines(ypos[b] + off, 100 * r.re_low, 100 * r.re_high, color=color, lw=1.0)
            ax.scatter(100 * r.point, ypos[b] + off, s=13, marker="o", facecolor=color if filled else "white", edgecolor=color, lw=0.9, zorder=3)
            out.append({"comparator": LABEL.get(b, b), "endpoint": e, "point": r.point, "low": r.re_low, "high": r.re_high})
        tr = R[(R.b == b) & (R.endpoint == "real")]
        ax.text(1.03, ypos[b], tr.iloc[0].tally if len(tr) else "--", transform=ax.get_yaxis_transform(), fontsize=FS, va="center", ha="left",
                color=S.INK2)
    ax.text(1.03, -1.0, "b/n.s./w", transform=ax.get_yaxis_transform(), fontsize=FS, va="center", ha="left", color=S.INK2, fontweight="bold")
    sep = len(groups[0]) + 0.6 / 2 - 0.5
    ax.axhline(sep, color=S.GREY_LIGHT, lw=0.6, ls=(0, (2, 1.5)))
    ax.axvline(0, color=S.INK2, lw=0.6)
    ax.set_yticks([ypos[b] for b in order], [LABEL.get(b, b) for b in order])
    ax.set_ylim(y - 0.6 - 0.4, -0.7)
    ax.set_xlabel("SRS+T $-$ comparator ($\\times$100)")
    ax.legend(handles=[Line2D([], [], marker="o", color=S.TEAL, lw=1.0, label="eICU + HiRID"),
                       Line2D([], [], marker="o", color=S.GREY, mfc="white", lw=1.0, label="internal")],
              loc="upper center", bbox_to_anchor=(0.35, -0.17), ncol=2, handlelength=1.2, columnspacing=0.6, handletextpad=0.3)
    ax.grid(axis="x")
    title(ax, "Difference of means")
    letter(ax, "a", dx=-0.66)
    src("r1a_pooled", pd.DataFrame(out))

    # (b) budget at the real targets (four standard settings; CV-based selectors averaged over their seeds), with references
    ax = fig.add_axes([0.43, bottom, 0.12, h])
    std = seed_mean[~seed_mean.run.str.startswith("union_large")].copy()
    std["k"] = std.run.str.split("/").str[0].map(data.K_OF)
    out = []
    for m in ("srs_ap_t", "nmifs", "mrmr", "l1_dropout"):
        g = std[std.registry == m].groupby("k").real.mean()
        sty = style_of(m)
        ax.plot(g.index, g.values, color=sty["color"], ls=sty["ls"], marker=sty["marker"], ms=2.8, mfc=sty["mfc"], mec=sty["color"], lw=sty["lw"],
                label=LABEL[m], zorder=3)
        out.append(pd.DataFrame({"method": LABEL[m], "k": g.index, "real": g.values}))
    refs = read_all("references/union*/*/rows.csv")
    refs = refs[~refs.run.str.startswith("union_large")].copy()
    refs["k"] = refs.run.str.split("/").str[0].map(data.K_OF)
    refs["real"] = refs[["eicu_auprc", "hirid_auprc"]].mean(axis=1)
    rk = refs[refs.registry == "random_k"].groupby("k").real.mean()
    allp = refs[refs.registry == "all_programs"].real.mean()
    st = stat_refs()
    st = st[~st.run.str.startswith("union_large")]
    sk = st[st.registry == "stats_l1"].groupby("k").real.mean()
    ax.plot(rk.index, rk.values, color=S.GREY_LIGHT, ls=(0, (1, 1.2)), lw=1.0, marker="o", ms=2, zorder=2)
    ax.text(5, rk.values[0] - 0.007, "random $K$", fontsize=FS, color=S.GREY, ha="left", va="top")
    ax.axhline(allp, color=S.GREY, lw=0.7, ls=(0, (3, 2)), zorder=1)
    ax.text(20, allp - 0.004, "all programs", fontsize=FS, color=S.GREY, ha="right", va="top")
    ax.plot(sk.index, sk.values, color=S.INK, ls=(0, (4, 1.5)), lw=1.0, marker="X", ms=3.2, zorder=2)
    ax.text(20, sk.values[-1] + 0.006, "$K$ statistics", fontsize=FS, color=S.INK2, ha="right", va="bottom")
    out += [pd.DataFrame({"method": "random K", "k": rk.index, "real": rk.values}), pd.DataFrame({"method": "all programs", "k": [0], "real": [allp]}),
            pd.DataFrame({"method": "K statistics (L1)", "k": sk.index, "real": sk.values})]
    ax.set_xscale("log", base=2)
    ax.set_xticks([5, 10, 20], ["5", "10", "20"])
    ax.minorticks_off()
    ax.set_xlim(4.3, 23)
    ax.set_ylim(0.56, max(sk.values) + 0.03)
    ax.set_xlabel("budget $K$")
    ax.set_ylabel("AUPRC, eICU + HiRID")
    ax.legend(loc="upper center", bbox_to_anchor=(0.4, -0.17), ncol=2, handlelength=1.5, columnspacing=0.4, handletextpad=0.3, labelspacing=0.15)
    ax.grid(axis="y")
    title(ax, "Budget")
    letter(ax, "b", dx=-0.5)
    src("r1b_budget", pd.concat(out))

    # (c) five CV seeds per run: SRS+T minus NMIFS at the real targets (identical values offset vertically)
    ax = fig.add_axes([0.665, bottom, 0.1, h])
    runs = [f"{r}/{l}" for r, l in data.ICU_RUNS]
    nm = T[(T.registry == "nmifs") & (T.seed == 0)].set_index("run").real
    out = []
    for i, run in enumerate(runs):
        root, label = run.split("/")
        d = T[(T.run == run) & (T.registry == "srs_ap_t")].sort_values("seed").real.values - nm[run]
        color = TASK_COLOR[data.TASK[label]]
        offs = np.linspace(-0.26, 0.26, len(d))
        ax.scatter(100 * d, i + offs, s=6, color=color, lw=0, alpha=0.9, zorder=3)
        ax.vlines(100 * d.mean(), i - 0.36, i + 0.36, color=color, lw=1.6, zorder=4)
        out.append({"run": run, "n_seeds": len(d), "mean": d.mean(), **{f"seed_{j}": x for j, x in enumerate(d)}})
    ax.axvline(0, color=S.BLUE, lw=0.8, ls=(0, (2, 1.5)))
    names = [data.run_name(*r.split("/")).replace("Vaso ", "V").replace(" L ", " L") for r in runs]
    ax.set_yticks(range(len(runs)), names)
    ax.set_ylim(len(runs) - 0.5, -0.7)
    ax.set_xlabel("SRS+T $-$ NMIFS ($\\times$100)")
    ax.legend(handles=[Line2D([], [], marker="o", ls="", color=c, ms=3, label=t.replace("Vaso ", "V")) for t, c in TASK_COLOR.items()],
              loc="upper center", bbox_to_anchor=(0.3, -0.17), ncol=2, handletextpad=0.05, columnspacing=0.5, labelspacing=0.15)
    ax.grid(axis="x")
    title(ax, "CV seeds")
    letter(ax, "c", dx=-0.85)
    src("r1c_seeds", pd.DataFrame(out))

    # (d) accuracy and portability, one point per selector (16-run means; CV-based selectors averaged over their seeds)
    ax = fig.add_axes([0.855, bottom, 0.14, h])
    s0 = seed_mean.groupby("registry")[["internal", "real"]].mean()
    sel = ["srs_ap_t"] + [b for b in R1_HYBRID + R1_ESTABLISHED if b in s0.index and b != "srs_c_val"]  # the 18 selectors
    s0 = s0.loc[sel]
    coef = np.polyfit(s0.internal.values, s0.real.values, 1)
    xs = np.linspace(s0.internal.min() - 0.005, s0.internal.max() + 0.005, 10)
    ax.plot(xs, np.polyval(coef, xs), color=S.GREY_LIGHT, lw=0.9, ls=(0, (3, 1.5)), zorder=1)
    colors = {"srs_ap_t": S.TEAL, "srs_ap": S.TEAL_MID, "srs_c_val": S.TEAL_DARK, "nmifs": S.BLUE, "nmifs_ap_t": HYB, "srs_ap_unaware": "#5DBAA6",
              "mrmr": S.SKY}
    for m, r in s0.iterrows():
        ax.scatter(r.internal, r.real, s=20 if m == "srs_ap_t" else 13, color=colors.get(m, S.GREY_LIGHT), zorder=3, edgecolor="white", lw=0.4)
    for m, text, dx, dy, ha in (("srs_ap_t", "SRS+T", -4, 3, "right"), ("nmifs", "NMIFS", -5, -4, "right"), ("mrmr", "mRMR", 4, -3, "left"),
                                ("l1_logistic", "L1", 4, 0, "left"), ("jmi", "JMI", 4, -2, "left"), ("mmr_ig", "MMR", 4, 4, "left")):
        if m in s0.index:
            ax.annotate(text, (s0.loc[m, "internal"], s0.loc[m, "real"]), xytext=(dx, dy), textcoords="offset points", ha=ha, va="center", fontsize=FS,
                        color=S.INK2)
    ax.set_xticks([0.70, 0.75])
    ax.set_xlabel("internal AUPRC")
    ax.set_ylabel("AUPRC, eICU + HiRID")
    ax.text(0.97, 0.05, f"slope {coef[0]:.2f}", transform=ax.transAxes, ha="right", va="bottom", fontsize=FS, color=S.INK2)
    ax.grid()
    title(ax, "Source vs targets")
    letter(ax, "d", dx=-0.47)
    src("r1d_portability", s0.reset_index().rename(columns={"index": "registry"}))
    S.save_exact(fig, "fig_r1_effectiveness")


# ================================================================================================= R2
def fig_r2() -> None:
    """Contract. Claim (RQ2): the gain over NMIFS comes mostly from deciding the shortlist by held-out AUPRC; the likelihood
    wrapper alone ties NMIFS, scenarios and surrogates add little on average and help where schemas are lost, and target
    schemas add a little more. The estimated value ranks candidates better than proxy scores only for the first picks, yet
    its shortlist holds the VAL-best candidate more often early on. Evidence: (a) pooled step effects of the decomposition
    with 95% intervals (horizontal bars), (b) Spearman fidelity across picks 1-20 (lines with ranges), (c) shortlist recall
    by pick range (dot plot), (d) per-target tallies of SRS+T against each variant (heat map). Data: q2_decomposition.csv,
    proxy_correlation/ext_corr_*.csv, q2_recall.csv, ablation_paired + paired_extra. Higher = better."""
    fig = plt.figure(figsize=(S.FULL_W, 2.7))
    top, bottom = 0.87, 0.27
    h = top - bottom

    # (a) decomposition: step effects, pooled over pools
    ax = fig.add_axes([0.16, bottom, 0.15, h])
    D = pd.read_csv(SRC / "q3_joint.csv")
    steps = [("fs_cv", "nmifs", "Wrapper-lik"), ("srs_ap_unaware", "fs_cv", "+ AUPRC decision"), ("srs_ap", "srs_ap_unaware", "+ scenarios, surr."),
             ("srs_ap_t", "srs_ap", "+ target schemas"), ("srs_ap_t", "nmifs", "total: SRS+T"), ("srs_ap_t", "srs_c_val", "AUPRC added last")]
    ends = [("internal", "internal", S.GREY), ("real", "eICU + HiRID", S.TEAL), ("targets", "all targets", S.BLUE)]
    hgt = 0.25
    out = []
    for i, (a, b, text) in enumerate(steps):
        for k, (e, ename, color) in enumerate(ends):
            r = D[(D.a == a) & (D.b == b) & (D.endpoint == e)].iloc[0]
            y = i + (k - 1) * hgt
            ax.barh(y, 100 * r.point, height=hgt * 0.9, color=color, edgecolor="white", lw=0.4, label=ename if i == 0 else None)
            ax.errorbar(100 * r.point, y, xerr=[[100 * (r.point - r.re_low)], [100 * (r.re_high - r.point)]], fmt="none", ecolor=S.INK,
                        elinewidth=0.6, capsize=1.2)
            out.append({"step": text, "endpoint": e, "point": r.point, "low": r.re_low, "high": r.re_high})
    ax.axvline(0, color=S.INK2, lw=0.6)
    ax.axhline(3.5, color=S.GREY_LIGHT, lw=0.6, ls=(0, (2, 1.5)))
    ax.axhline(4.5, color=S.GREY_LIGHT, lw=0.6, ls=(0, (2, 1.5)))
    ax.set_yticks(range(len(steps)), [s[2] for s in steps])
    ax.set_ylim(len(steps) - 0.5, -0.5)
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel("AUPRC gain ($\\times$100)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.12, -0.2), ncol=3, handlelength=0.8, columnspacing=0.5, handletextpad=0.3)
    ax.grid(axis="x")
    title(ax, "Decomposition")
    letter(ax, "a", dx=-0.85)
    src("r2a_decomposition", pd.DataFrame(out))

    # (b) fidelity across picks 1-20, union pools
    ax = fig.add_axes([0.395, bottom, 0.155, h])
    C = pd.concat([pd.read_csv(p) for p in glob.glob(str(S.ART / "proxy_correlation/ext_corr_*.csv"))], ignore_index=True)
    U = C[C.scope == "union"]
    PROXY = {"objective": ("estimate", S.TEAL, "o"), "nmifs": ("NMIFS", S.BLUE, "s"), "mrmr": ("mRMR", S.SKY, "^")}
    out = []
    for proxy, (name, color, marker) in PROXY.items():
        g = U[U.proxy == proxy].groupby("pick").spearman
        mean, lo, hi = g.mean(), g.min(), g.max()
        if proxy != "mrmr":  # mRMR's range is drawn as a line only: its band would hide the others
            ax.fill_between(mean.index, lo, hi, color=color, alpha=0.1, lw=0)
        ax.plot(mean.index, mean.values, color=color, marker=marker, ms=2.2, lw=1.1, label=name)
        out.append(pd.DataFrame({"proxy": proxy, "pick": mean.index, "mean": mean.values, "min": lo.values, "max": hi.values}))
    ax.axhline(0, color=S.INK2, lw=0.5)
    ax.set_xticks([1, 5, 10, 15, 20])
    ax.set_xlim(0.5, 20.5)
    ax.set_xlabel("pick")
    ax.set_ylabel("Spearman $\\rho$ with VAL gain")
    ax.legend(loc="upper right", handlelength=1.3, borderpad=0.1, labelspacing=0.2)
    ax.grid(axis="y")
    title(ax, "Proxy fidelity")
    letter(ax, "b", dx=-0.3)
    src("r2b_fidelity_picks", pd.concat(out))

    # (c) shortlist recall by pick range
    ax = fig.add_axes([0.625, bottom, 0.1, h])
    R = pd.read_csv(SRC / "q2_recall.csv")
    order = ["1", "2-4", "5-7", "8-11", "12-20"]
    out = []
    for y in range(len(order)):
        ax.axhspan(y - 0.45, y + 0.45, color="#F4F4F4" if y % 2 else "white", lw=0, zorder=0)
    for proxy, (name, color, marker) in PROXY.items():
        r = R[R.proxy == proxy].set_index("picks").reindex(order)
        ys = np.arange(len(order)) + {"objective": -0.2, "nmifs": 0.0, "mrmr": 0.2}[proxy]
        ax.scatter(100 * r.recall.values, ys, color=color, marker=marker, s=15, zorder=3, label=name)
        out.append(r.reset_index().assign(proxy=proxy))
    ax.set_yticks(range(len(order)), order)
    ax.set_ylabel("picks")
    ax.set_ylim(len(order) - 0.5, -0.5)
    ax.set_xlim(-8, 108)
    ax.set_xticks([0, 50, 100])
    ax.set_xlabel("VAL-best in top 10 (%)")
    ax.text(14, 0, "same\nranking", fontsize=FS, color=S.INK2, va="center", ha="left")
    ax.grid(axis="x")
    title(ax, "Shortlist recall")
    letter(ax, "c", dx=-0.52)
    src("r2c_shortlist_recall", pd.concat(out))

    # (d) SRS+T against each variant, per target (targets as rows)
    ax = fig.add_axes([0.815, bottom, 0.14, h])
    V = per_variant_tallies()
    tg = [("internal", "internal"), ("eicu", "eICU"), ("hirid", "HiRID"), ("drop_labs", "no labs"), ("drop_ventilation", "no vent."),
          ("drop_bp", "no BP"), ("drop_urine", "no urine"), ("vitals_only", "vitals only")]
    variants = [("srs_c_val", "lik. decision"), ("srs_ap_unaware", "no scenarios"), ("srs_ap_t_nosur", "no surr. obj."),
                ("nmifs_ap_t", "NMIFS screen"), ("srs_ap", "no targets")]
    net = np.zeros((len(tg), len(variants)))
    text = [["" for _ in variants] for _ in tg]
    out = []
    for j, (v, vname) in enumerate(variants):
        for i, (t, tname) in enumerate(tg):
            s = V[(V.b == v) & (V.target == t)]
            assert len(s) == 16, (v, t, len(s))
            b, w = int((s.verdict == 1).sum()), int((s.verdict == -1).sum())
            net[i, j] = (b - w) / 16
            text[i][j] = f"{b}/{w}"
            out.append({"variant": vname, "target": tname, "better": b, "ns": 16 - b - w, "worse": w})
    im = ax.imshow(net, cmap=S.DIV, vmin=-1, vmax=1, aspect="auto")
    for i in range(net.shape[0]):
        for j in range(net.shape[1]):
            ax.text(j, i, text[i][j], ha="center", va="center", fontsize=FS, color="white" if abs(net[i, j]) > 0.6 else S.INK)
    cax = fig.add_axes([0.962, bottom + 0.25 * h, 0.007, 0.5 * h])
    cb = fig.colorbar(im, cax=cax, ticks=[-1, 0, 1])
    cb.ax.set_yticklabels(["$-$1", "0", "1"])
    cb.ax.tick_params(labelsize=FS, length=1.5, pad=1)
    cb.outline.set_visible(False)
    ax.axhline(0.5, color="white", lw=1.8)
    ax.axhline(2.5, color="white", lw=1.8)
    ax.set_yticks(range(len(tg)), [t for _, t in tg])
    ax.set_xticks(range(len(variants)), [n for _, n in variants], rotation=40, ha="right", rotation_mode="anchor")
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    title(ax, "SRS+T vs variant")
    letter(ax, "d", dx=-0.5)
    src("r2d_variant_tallies", pd.DataFrame(out))
    S.save_exact(fig, "fig_r2_mechanism")


# ================================================================================================= R3
TGT_ROWS = [("hospitals", "eICU hospitals"), ("eicu", "eICU"), ("hirid", "HiRID"), ("drop_labs", "no labs"),
            ("drop_ventilation", "no vent."), ("drop_bp", "no BP"), ("drop_urine", "no urine"), ("vitals_only", "vitals only")]
KNOWN_TO_SRST = ("eicu", "hirid", "drop_ventilation")  # schemas SRS+T selects for; shown in italics


def fig_r3() -> None:
    """Contract. Claim (RQ3): at the real databases and at eICU hospitals with their own schemas SRS+T is ahead of NMIFS
    in most runs and never behind; at controlled targets it leads where surrogates or scenarios cover the loss and trails
    with vital signs only; surrogate chains add accuracy for every registry; under random loss the mean lead over NMIFS is
    positive until most programs are lost. Evidence: (a) UpSet of absent features at eICU hospitals, (b) diverging verdict
    bars per target, known schemas marked, (c) dot plot over six deployment rules, (d) mean paired difference against the
    loss level with 95% run-bootstrap bands. Data: hospital_schemas_ap, extra_metrics paired_targets, deployment_modes_ap,
    q2_stress_wtl.csv. Higher = better."""
    hr, hp, hm = data.hospital_all()
    fig = plt.figure(figsize=(S.FULL_W, 2.95))
    top, bottom = 0.87, 0.33
    h = top - bottom

    # (a) UpSet of absent features at eICU hospitals (vasopressor 1 h)
    summ = hm[("union", "f_mimic_iv_gap1h")]["summary"]
    items = sorted([(k, v) for k, v in summ["patterns"].items() if k != "-"], key=lambda kv: -kv[1])[:6]
    feats = ["Lactate", "pH", "sedation_on", "Bilirubin_Total", "urine_output"]
    fname = {"Lactate": "lactate", "pH": "pH", "sedation_on": "sedation", "Bilirubin_Total": "bilirubin", "urine_output": "urine"}
    axb = fig.add_axes([0.085, bottom + 0.54 * h, 0.12, 0.46 * h])
    axm = fig.add_axes([0.085, bottom, 0.12, 0.5 * h], sharex=axb)
    xs = np.arange(len(items))
    axb.bar(xs, [v for _, v in items], color=S.TEAL, width=0.62)
    for x, (_, v) in zip(xs, items):
        axb.text(x, v + 15, str(v), ha="center", va="bottom", fontsize=FS, color=S.INK2)
    axb.set_ylabel("episodes")
    axb.spines["bottom"].set_visible(False)
    axb.tick_params(axis="x", length=0, labelbottom=False)
    axb.set_ylim(0, max(v for _, v in items) * 1.35)
    axb.text(0.72, 0.97, f"{100 * summ['n_affected'] / summ['n_episodes']:.0f}% of\nepisodes", transform=axb.transAxes, ha="center", va="top",
             fontsize=FS, color=S.INK2)
    for j, f in enumerate(feats):
        axm.axhspan(j - 0.5, j + 0.5, color="#F5F5F5" if j % 2 else "white", zorder=0, lw=0)
        for x, (k, _) in zip(xs, items):
            axm.scatter(x, j, s=12, color=S.TEAL_DARK if f in k.split(",") else S.GREY_PALE, zorder=2, lw=0)
    for x, (k, _) in zip(xs, items):
        on = [j for j, f in enumerate(feats) if f in k.split(",")]
        if len(on) > 1:
            axm.plot([x, x], [min(on), max(on)], color=S.TEAL_DARK, lw=0.9, zorder=1)
    axm.set_yticks(range(len(feats)), [fname[f] for f in feats])
    axm.set_ylim(len(feats) - 0.5, -0.5)
    axm.set_xticks([])
    for spine in ("bottom", "left"):
        axm.spines[spine].set_visible(False)
    axm.tick_params(axis="y", length=0)
    title(axb, "eICU hospitals")
    letter(axb, "a", dx=-0.55)
    rows = [{"run": f"{r}/{l}", "pattern": k, "episodes": v, "n_affected": m["summary"]["n_affected"], "n_episodes": m["summary"]["n_episodes"],
             "n_hospitals": m["summary"]["n_hospitals"]} for (r, l), m in hm.items() for k, v in m["summary"]["patterns"].items()]
    src("r3a_hospital_patterns", pd.DataFrame(rows))

    # (b) diverging verdict bars, three pairs; eICU and HiRID from the real-target bootstraps (same draws as Table 2)
    V = per_target_verdicts()
    rt = pd.concat([pd.read_csv(f) for f in glob.glob(str(S.ART / "real_targets/union*/*/paired.csv"))], ignore_index=True)
    rt = rt[rt.target.isin(["eicu", "hirid"])].copy()
    rt["verdict"] = np.where(rt.low > 0, 1, np.where(rt.high < 0, -1, 0))
    V = pd.concat([V[~V.target.isin(["eicu", "hirid"])], rt[["run", "target", "a", "b", "point_diff", "low", "high", "verdict"]]], ignore_index=True)
    out = []
    for ci, (a, b, head) in enumerate((("srs_ap_t", "nmifs", "SRS+T vs\nNMIFS"), ("srs_ap", "nmifs", "SRS vs\nNMIFS"), ("srs_ap_t", "mrmr", "SRS+T vs\nmRMR"))):
        ax = fig.add_axes([0.325 + ci * 0.088, bottom, 0.082, h])
        for i, (t, text) in enumerate(TGT_ROWS):
            s = V[(V.a == a) & (V.b == b) & (V.target == t)]
            assert len(s) == 16, (a, b, t, len(s))
            w, n, bt = int((s.verdict == -1).sum()), int((s.verdict == 0).sum()), int((s.verdict == 1).sum())
            for left, width, color, ink in ((-(w + n / 2), w, S.WARM, "white"), (-n / 2, n, S.GREY_PALE, S.INK2), (n / 2, bt, S.TEAL, "white")):
                if width:
                    ax.barh(i, width, left=left, height=0.74, color=color, edgecolor="white", lw=0.5)
                    if width >= (6 if width >= 10 else 4):
                        ax.text(left + width / 2, i, str(width), ha="center", va="center", fontsize=FS, color=ink)
            out.append({"pair": head.replace("\n", " "), "target": text, "better": bt, "ns": n, "worse": w})
        ax.axvline(0, color=S.INK2, lw=0.5, zorder=0)
        ax.axhline(2.5, color=S.INK2, lw=0.5, ls=(0, (1, 1.5)))
        ax.set_xlim(-16, 17)
        ax.set_xticks([-10, 0, 10], ["10", "0", "10"])
        ax.set_ylim(len(TGT_ROWS) - 0.5, -0.5)
        ax.set_xlabel(head)
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="y", length=0)
        if ci == 0:
            ax.set_yticks(range(len(TGT_ROWS)), [t for _, t in TGT_ROWS])
            for lbl, (t, _) in zip(ax.get_yticklabels(), TGT_ROWS):
                if t in KNOWN_TO_SRST:
                    lbl.set_fontstyle("italic")
            title(ax, "Verdicts over 16 runs")
            letter(ax, "b", dx=-1.12)
        else:
            ax.set_yticks([])
        if ci == 1:
            ax.legend(handles=[Patch(color=S.TEAL, label="first better"), Patch(color=S.GREY_PALE, label="n.s."), Patch(color=S.WARM, label="first worse")],
                      loc="upper center", bbox_to_anchor=(0.5, -0.36), ncol=3, handlelength=0.9, handletextpad=0.3, columnspacing=0.7)
    src("r3b_target_verdicts", pd.DataFrame(out))

    # (c) deployment rules: dot plot
    dr, _ = data.deploy_all()
    modes = [("zero", "zeroed"), ("reduced", "re-fit"), ("robust", "dropout probe"), ("surrogate", "surrogates"),
             ("reduced_surrogate", "re-fit + surr."), ("soft", "soft surr.")]
    tg = dr[~dr.target.isin(["internal_test"]) & dr["mode"].isin([m for m, _ in modes])]
    mt = tg.groupby(["registry", "mode", "name"]).auprc.mean().groupby(["registry", "mode"]).mean().unstack()
    ax = fig.add_axes([0.695, bottom, 0.115, h])
    meths = ("srs_ap_t", "srs_ap", "nmifs", "mrmr", "l1_dropout")
    dodge = dict(zip(meths, np.linspace(-0.28, 0.28, len(meths))))
    for i, (md, text) in enumerate(modes):
        vals = [mt.loc[m, md] for m in meths]
        ax.axhspan(i - 0.42, i + 0.42, color="#F6F6F6" if i % 2 else "white", lw=0, zorder=0)
        for m in meths:
            st = style_of(m)
            ax.scatter(mt.loc[m, md], i + dodge[m], marker=st["marker"], s=14, facecolor=st["mfc"], edgecolor=st["color"], lw=0.9, zorder=3,
                       label=LABEL[m] if i == 0 else None)
    ax.set_yticks(range(len(modes)), [t for _, t in modes])
    ax.set_ylim(len(modes) - 0.5, -0.6)
    ax.set_xlabel("mean target AUPRC")
    ax.legend(loc="upper center", bbox_to_anchor=(0.12, -0.24), ncol=2, handletextpad=0.1, columnspacing=0.5, labelspacing=0.25)
    ax.grid(axis="x")
    title(ax, "Deployment")
    letter(ax, "c", dx=-0.75)
    src("r3c_deployment_modes", mt.reset_index())

    # (d) random loss: mean paired difference with run-bootstrap bands
    ax = fig.add_axes([0.875, bottom, 0.12, h])
    W = pd.read_csv(SRC / "q2_stress_wtl.csv")
    out = []
    for b, color, marker in (("nmifs", S.BLUE, "s"), ("mrmr", S.SKY, "^")):
        for fam, ls in (("feature", "-"), ("group", (0, (3, 1.5)))):
            s = W[(W.b == b) & (W.family == fam)].sort_values("level")
            ax.fill_between(s.level, 100 * s.low, 100 * s.high, color=color, alpha=0.12, lw=0)
            ax.plot(s.level, 100 * s.mean_diff, color=color, ls=ls, marker=marker, ms=2.6, lw=1.1, mfc=color if fam == "feature" else "white")
            out.append(s.assign(b=b))
    ax.axhline(0, color=S.INK2, lw=0.6)
    ax.set_xticks([0.1, 0.5, 0.9], ["0.1", "0.5", "0.9"])
    ax.set_xlabel("loss probability $l$")
    ax.set_ylabel("SRS+T $-$ comparator ($\\times 100$)")
    ax.legend(handles=[Line2D([], [], color=S.BLUE, marker="s", ms=3, label="vs NMIFS"), Line2D([], [], color=S.SKY, marker="^", ms=3, label="vs mRMR"),
                       Line2D([], [], color=S.INK2, ls="-", label="features"), Line2D([], [], color=S.INK2, ls=(0, (3, 1.5)), label="groups")],
              loc="upper center", bbox_to_anchor=(0.32, -0.24), ncol=1, handlelength=1.6, handletextpad=0.3, labelspacing=0.25)
    ax.grid(axis="y")
    title(ax, "Random loss")
    letter(ax, "d", dx=-0.45)
    src("r3d_random_loss", pd.concat(out))
    S.save_exact(fig, "fig_r3_schema_shift")




# ================================================================================================= R4, R6 (figs_v4 layout, fonts raised)
def bump_fonts(fig, minimum: float = FS) -> None:
    for t in fig.findobj(mpl.text.Text):
        if t.get_text() and t.get_fontsize() < minimum:
            t.set_fontsize(minimum)


def _with_bumped_fonts(builder) -> None:
    import figs_v4
    original = S.save

    def save(fig, name):
        bump_fonts(fig)
        return original(fig, name)

    figs_v4.S.save = save
    try:
        builder()
    finally:
        figs_v4.S.save = original


def fig_r5() -> None:
    """Contract (extended version). Claim (RQ5): conclusions survive new patient splits, a tree decoder, demographic
    subgroups and one-at-a-time changes of every setting incl. the shortlist size; greedy reaches at least 93% of the exact
    optimum of the likelihood estimate; the AUPRC decision re-ranks the top of the screen rather than reaching deep into it.
    Evidence: (a) strip plot over five re-splits, (b) forest of decoder differences, (c) categorical heat map of subgroup
    verdicts, (d) range bars of setting changes with VAL marks, (e) ECDF of greedy/optimum, (f) histogram of the screen rank
    of the chosen candidate. Data: union_resplit (+ nmifs_ap, ablation_paired), union_robustness_ap, srs_sensitivity_ap,
    decision_diagnostics, exact_small_pools."""
    from matplotlib.colors import ListedColormap
    from matplotlib.transforms import blended_transform_factory
    rng = np.random.default_rng(2)
    fig = plt.figure(figsize=(S.FULL_W, 5.4))
    gs = GridSpec(2, 3, wspace=0.62, hspace=0.72, left=0.095, right=0.985, top=0.95, bottom=0.12)

    # (a) re-splits
    ax = fig.add_subplot(gs[0, 0])
    rr, _ = data.resplits()
    extra = []
    for task in ("gap1h", "aki_gap6h"):
        for i in range(1, 6):
            run = f"union_resplit/r{i}_mimic_iv_{task}"
            base = rr[(rr.task == ("Vaso 1h" if task == "gap1h" else "AKI")) & (rr.split == i) & (rr.registry == "srs_ap_t")].mean_target_auprc.iloc[0]
            h = pd.read_csv(S.ART / "nmifs_ap" / run / "rows.csv")
            extra.append({"task": "Vaso 1h" if task == "gap1h" else "AKI", "split": i, "registry": "nmifs_ap_t",
                          "mean_target_auprc": float(h[h.registry == "nmifs_ap_t"].mean_target_auprc.iloc[0])})
            a = pd.read_csv(S.ART / "union_ap" / run / "ablation_paired.csv")
            d = a[(a.a == "srs_ap_t") & (a.b == "srs_ap_unaware") & (a.target == "mean_over_targets")].point_diff.iloc[0]
            extra.append({"task": "Vaso 1h" if task == "gap1h" else "AKI", "split": i, "registry": "srs_ap_unaware", "mean_target_auprc": base - d})
    rr = pd.concat([rr, pd.DataFrame(extra)], ignore_index=True)
    meths = ["srs_ap_t", "nmifs_ap_t", "srs_ap_unaware", "nmifs", "mrmr", "l1_dropout"]
    colors = {"srs_ap_t": S.TEAL, "nmifs_ap_t": HYB, "srs_ap_unaware": "#5DBAA6", "nmifs": S.BLUE, "mrmr": S.SKY, "l1_dropout": S.GREY}
    rows = []
    for ti, task in enumerate(("Vaso 1h", "AKI")):
        for mi, m in enumerate(meths):
            v = rr[(rr.task == task) & (rr.registry == m)].mean_target_auprc.values
            x = ti * (len(meths) + 1) + mi
            ax.scatter(np.full(len(v), x) + rng.uniform(-0.18, 0.18, len(v)), v, s=8, color=colors[m], lw=0, alpha=0.9)
            ax.hlines(v.mean(), x - 0.32, x + 0.32, color=S.INK, lw=0.9)
            rows.append({"task": task, "method": LABEL.get(m, m), **{f"split{i + 1}": vv for i, vv in enumerate(v)}})
    ax.set_xticks([2.5, 9.5], ["Vaso 1h", "AKI"])
    ax.set_ylabel("AUPRC, mean of all targets")
    ax.legend(handles=[Line2D([], [], marker="o", ls="", color=colors[m], label=LABEL.get(m, m)) for m in meths],
              loc="upper center", bbox_to_anchor=(0.45, -0.12), ncol=2, handletextpad=0.05, columnspacing=0.4, labelspacing=0.15)
    ax.grid(axis="y")
    title(ax, "New patient splits")
    letter(ax, "a", dx=-0.3)
    src("r5a_resplits", pd.DataFrame(rows))

    # (b) tree decoder
    ax = fig.add_subplot(gs[0, 1])
    rb = data.robustness()
    y, ticks, rows = 0, [], []
    for label, task in zip(data.STANDARD, ("V1h", "V2h", "V4h", "AKI")):
        dp = rb[label]["decoder_paired"]
        for b, color, marker in (("mrmr", S.SKY, "^"), ("nmifs", S.BLUE, "s")):
            s_ = dp[(dp.a == "srs_ap_t") & (dp.b == b) & (dp.decoder == "gbdt") & (dp.target == "mean_over_targets")]
            ax.hlines(y, 100 * s_.low, 100 * s_.high, color=color, lw=1.0)
            ax.scatter(100 * s_.point_diff, y, color=color, marker=marker, s=13, zorder=3)
            rows.append({"setting": task, "b": b, "diff": float(s_.point_diff.iloc[0]), "low": float(s_.low.iloc[0]), "high": float(s_.high.iloc[0])})
            y += 1
        ticks.append((y - 1.5, task))
        y += 0.6
    ax.set_yticks([t[0] for t in ticks], [t[1] for t in ticks])
    ax.set_ylim(y - 0.4, -0.9)
    ax.axvline(0, color=S.INK2, lw=0.6, ls=(0, (2, 2)))
    ax.set_xlabel("SRS+T $-$ comparator ($\\times 100$)")
    ax.legend(handles=[Line2D([], [], marker="^", ls="", color=S.SKY, label="vs mRMR"), Line2D([], [], marker="s", ls="", color=S.BLUE, label="vs NMIFS")],
              loc="upper center", bbox_to_anchor=(0.45, -0.2), ncol=2, handletextpad=0.05, columnspacing=0.6)
    title(ax, "Tree decoder")
    letter(ax, "b", dx=-0.25)
    src("r5b_decoder", pd.DataFrame(rows))

    # (c) subgroups
    ax = fig.add_subplot(gs[0, 2])
    first = rb[data.STANDARD[0]]["subgroup_paired"]
    sub_keys = [tuple(x) for x in first[(first.a == "srs_ap_t") & (first.b == "mrmr")][["cohort", "subgroup"]].drop_duplicates().values]
    sub_keys = [k for k in sub_keys if k[1] != "all"]
    mat = np.zeros((len(sub_keys), 8))
    for bi, b in enumerate(("mrmr", "nmifs")):
        for li, label in enumerate(data.STANDARD):
            sp = rb[label]["subgroup_paired"]
            for si, (cohort, sg) in enumerate(sub_keys):
                s_ = sp[(sp.a == "srs_ap_t") & (sp.b == b) & (sp.cohort == cohort) & (sp.subgroup == sg)]
                mat[si, bi * 4 + li] = 1 if float(s_.low.iloc[0]) > 0 else (-1 if float(s_.high.iloc[0]) < 0 else 0)
    ax.imshow(mat, cmap=ListedColormap([S.WARM, "#EFEFEF", S.TEAL]), vmin=-1, vmax=1, aspect="auto")
    ax.axvline(3.5, color="white", lw=2.0)
    for x in range(8):
        ax.axvline(x + 0.5, color="white", lw=0.4)
    for yy in range(len(sub_keys)):
        ax.axhline(yy + 0.5, color="white", lw=0.4)
    cohort_name = {"mimic_iv_test": "MIMIC", "eicu": "eICU", "hirid": "HiRID"}
    ax.set_yticks(range(len(sub_keys)), [f"{cohort_name.get(c, c)} {g}".replace(">=", "\u2265") for c, g in sub_keys])
    ax.set_xticks(range(8), ["1h", "2h", "4h", "AKI"] * 2, rotation=90)
    nb = (mat[:, :4] == 1).sum(), (mat[:, :4] == -1).sum(), (mat[:, 4:] == 1).sum(), (mat[:, 4:] == -1).sum()
    tr = blended_transform_factory(ax.transData, ax.transAxes)
    ax.text(1.5, -0.16, f"vs mRMR\n{nb[0]} b, {nb[1]} w", transform=tr, ha="center", va="top", fontsize=FS, color=S.INK2)
    ax.text(5.5, -0.16, f"vs NMIFS\n{nb[2]} b, {nb[3]} w", transform=tr, ha="center", va="top", fontsize=FS, color=S.INK2)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    title(ax, "Subgroups")
    letter(ax, "c", dx=-0.55)
    src("r5c_subgroups", pd.DataFrame(mat, index=[f"{c} {g}" for c, g in sub_keys],
                                      columns=[f"{b}_{l}" for b in ("mrmr", "nmifs") for l in data.STANDARD]).reset_index())

    # (d) one-at-a-time setting changes, incl. the shortlist size
    ax = fig.add_subplot(gs[1, 0])
    sens = data.sensitivity()
    default = sens[sens.parameter == "default"].set_index("label")
    names = {"q": "$\\omega$", "max_surrogates": "$M$", "n_folds": "$V$", "C": "$\\eta$", "n_dropout": "$D$", "seed": "seed"}
    rows = []
    keep = sens[(sens.parameter != "default") & ~((sens.parameter == "q") & (sens.value.astype(str) == "0.25"))]
    for (param, value), g in keep.groupby(["parameter", "value"], sort=False):
        g = g.set_index("label")
        d = (g.mean_target_auprc - default.mean_target_auprc).values
        dv = (g.validation_score - default.validation_score).values
        rows.append({"variant": f"{names.get(param, param)} = {float(value):g}", "min": d.min(), "max": d.max(), "mean": d.mean(), "val_mean": dv.mean()})
    sl = pd.read_csv(S.ART / "decision_diagnostics/shortlist.csv")
    base = {l: pd.read_csv(S.ART / "union_ap/union" / l / "rows.csv").set_index("registry").loc["srs_ap_t"] for l in data.STANDARD}
    import json as _json
    bval = {l: _json.loads((S.ART / "union_ap/union" / l / "select.json").read_text(encoding="utf-8"))["srs_ap_t"]["val_auprc"] for l in data.STANDARD}
    for mval, g in sl.groupby("m"):
        d = np.array([g[g.run == f"union/{l}"].mean_target_auprc.iloc[0] - base[l].mean_target_auprc for l in data.STANDARD])
        dv = np.array([g[g.run == f"union/{l}"].val_auprc.iloc[0] - bval[l] for l in data.STANDARD])
        rows.append({"variant": f"$m$ = {int(mval)}", "min": d.min(), "max": d.max(), "mean": d.mean(), "val_mean": dv.mean()})
    tor = pd.DataFrame(rows).sort_values("min").reset_index(drop=True)
    for k, row in tor.iterrows():
        ax.hlines(k, 100 * row["min"], 100 * row["max"], color=S.TEAL_LIGHT, lw=3.2)
        ax.scatter(100 * row["mean"], k, s=9, color=S.TEAL_DARK, zorder=3)
        ax.scatter(100 * row["val_mean"], k, s=12, marker="|", color=S.WARM, zorder=4, lw=1.1)
    ax.axvline(0, color=S.INK2, lw=0.6)
    ax.set_yticks(range(len(tor)), tor.variant)
    ax.set_ylim(len(tor) - 0.4, -0.6)
    ax.set_xlabel("$\\Delta$ AUPRC vs default ($\\times 100$)")
    ax.legend(handles=[Line2D([], [], color=S.TEAL_LIGHT, lw=3, label="targets: range"), Line2D([], [], marker="o", ls="", color=S.TEAL_DARK, ms=3, label="targets: mean"),
                       Line2D([], [], marker="|", ls="", color=S.WARM, ms=5, label="VAL: mean")],
              loc="upper center", bbox_to_anchor=(0.45, -0.16), ncol=3, handletextpad=0.3, columnspacing=0.5, handlelength=1.2)
    title(ax, "Settings, one at a time")
    letter(ax, "d", dx=-0.3)
    src("r5d_sensitivity", tor)

    # (e) greedy against the exact optimum
    ax = fig.add_subplot(gs[1, 1])
    ex = pd.concat([pd.read_csv(S.ART / f"exact_small_pools/rows_{t}.csv") for t in ("gap1h", "aki_gap6h")])
    rows = []
    for obj, color, ls, text in (("deployed_q0", S.TEAL, "-", "$\\omega = 0$"), ("deployed_q25", S.BLUE, (0, (3, 1.5)), "$\\omega = 0.25$"),
                                 ("analyzable", S.SAND, (0, (1, 1)), "relaxation")):
        v = np.sort(ex[ex.objective == obj].ratio.values)
        ax.step(v, np.arange(1, len(v) + 1) / len(v), where="post", color=color, ls=ls, lw=1.2, label=text)
        rows.append(pd.DataFrame({"objective": obj, "ratio": v}))
    ax.set_xlim(0.93, 1.004)
    ax.set_xticks([0.94, 0.97, 1.0])
    ax.set_xlabel("greedy / optimum (likelihood)")
    ax.set_ylabel("share of 80 sub-pools")
    ax.legend(loc="upper left", handlelength=1.5, borderpad=0.1)
    ax.grid()
    title(ax, "Near-optimal")
    letter(ax, "e", dx=-0.3)
    src("r5e_exact_ratio", pd.concat(rows))

    # (f) screen rank of the candidate the AUPRC decision chose
    ax = fig.add_subplot(gs[1, 2])
    tr_ = pd.read_csv(S.ART / "decision_diagnostics/trace.csv")
    tr_ = tr_[~tr_.stopped]
    counts = tr_.chosen_screen_rank.value_counts().reindex(range(1, 11), fill_value=0)
    same = (tr_.chosen == tr_.loglik_best).mean()
    ax.bar(counts.index, counts.values, color=S.TEAL, width=0.7)
    ax.set_xticks(range(1, 11))
    ax.set_xlabel("screen rank of the added candidate")
    ax.set_ylabel("greedy steps (16 runs)")
    ax.text(0.97, 0.95, f"{len(tr_)} steps\n{100 * (1 - same):.0f}% differ from\nthe likelihood choice", transform=ax.transAxes, ha="right", va="top",
            fontsize=FS, color=S.INK2)
    ax.grid(axis="y")
    title(ax, "AUPRC decision")
    letter(ax, "f", dx=-0.28)
    src("r5f_decision_rank", counts.rename_axis("rank").reset_index(name="steps"))
    S.save_exact(fig, "fig_r5_robustness")


def scalability_repeats() -> pd.DataFrame:
    """Size-axis rows of every selection-timing repeat (the first run, the two repeats on an idle machine, exact greedy at
    10^4 and 2*10^4 for the first run, and the AUPRC-decision run at 5*10^4 and 10^5), one row per repeat, method and size."""
    frames = []
    first = pd.read_csv(S.ART / "selection_scalability/selection_scalability.csv").assign(repeat="r1")
    frames.append(first)
    base = S.ART / "selection_scalability_repeats"
    for name, rep in (("r2", "r2"), ("r3", "r3"), ("r1_exact", "r1"), ("ap_large", "r1")):
        f = base / name / "selection_scalability.csv"
        if f.exists():
            frames.append(pd.read_csv(f).assign(repeat=rep))
    d = pd.concat(frames, ignore_index=True)
    d = d[(d.axis == "size") & (d.status == "ok")]
    return d.drop_duplicates(["repeat", "method", "n_candidates"], keep="last")


def fig_r4() -> None:
    """Contract. Claim (RQ4): selection is linear in |C| and costs seconds to hours offline, screening does the work of
    exact greedy several times faster, filters are three to four orders of magnitude cheaper but less accurate, and the
    compiled registry runs in DuckDB in seconds, faster than the Python reference engine, also beyond memory with a
    disk-backed database. Evidence: (a) log-log runtime against |C|, median of the repeats with their range (lines),
    (b) share of time per phase (100% stacked bars), (c) time at 500 candidates against AUPRC at the real targets over the 16
    runs (scatter with frontier; two different experiments), (d) SQL and Python timings, median and IQR of five
    fresh-process repeats (dot-range). Data: selection_scalability (+ _repeats), q2_seeds_long, sql_execution timing_repeats
    + memory diagnostic. Time: lower = better."""
    import json as _json
    reps = scalability_repeats()
    med = reps.groupby(["method", "n_candidates"]).agg(seconds=("seconds", "median"), lo=("seconds", "min"), hi=("seconds", "max"),
                                                       n=("seconds", "size"), phase_chains=("phase_chains", "median"),
                                                       phase_select_q0=("phase_select_q0", "median"),
                                                       phase_select_q25=("phase_select_q25", "median")).reset_index()
    fq = reps[reps.method == "srs_c_v"].copy()
    fq["seconds"] = fq.phase_chains + fq.phase_select_q0
    fq = fq.groupby("n_candidates").seconds.agg(["median", "min", "max", "size"]).reset_index()
    fig = plt.figure(figsize=(S.FULL_W, 2.9))
    top, bottom = 0.87, 0.36
    h = top - bottom

    def sec_ticks(ax, values):
        ax.set_yticks(values, [f"{v:g}" if v < 1000 else f"{v:,.0f}" for v in values])
        ax.minorticks_off()

    # (a) runtime: median of the repeats, range as a band
    ax = fig.add_axes([0.092, bottom, 0.165, h])
    lines = [("srs_ap_v", "SRS+T", S.TEAL, "-", "o", 1.6), ("srs_c_v", "SRS+T-lik", S.TEAL_DARK, (0, (2, 1.2)), "o", 1.0),
             ("fast_q0", "screening, $\\omega$ = 0", "#7CC7B6", "-", "^", 1.0), ("srs_c_exact", "exact, $\\omega$ = 0", S.SAND, (0, (4, 1.5)), "s", 1.2),
             ("nmifs", "NMIFS, mRMR", S.BLUE, "-", "s", 1.0)]
    rows = []
    for m, text, color, ls, marker, lw in lines:
        if m == "fast_q0":
            d = fq[fq.n_candidates <= 20000].rename(columns={"median": "seconds", "min": "lo", "max": "hi", "size": "n"})
        else:
            d = med[med.method == m].sort_values("n_candidates")
        ax.fill_between(d.n_candidates, d.lo, d.hi, color=color, alpha=0.18, lw=0)
        ax.plot(d.n_candidates, d.seconds, color=color, ls=ls, marker=marker, ms=2.4, lw=lw, label=text, mfc=color if m != "srs_c_exact" else "white")
        rows.append(d.assign(method=m)[["method", "n_candidates", "seconds", "lo", "hi", "n"]])
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xticks([1e3, 1e4, 1e5], ["1k", "10k", "100k"])
    sec_ticks(ax, [0.01, 1, 100, 10000])
    ax.set_xlabel("candidates $|\\mathcal{C}|$")
    ax.set_ylabel("selection time (s)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.42, -0.22), handlelength=1.6, ncol=2, columnspacing=0.5, handletextpad=0.3, labelspacing=0.2)
    ax.grid(which="major")
    title(ax, "Selection time")
    letter(ax, "a", dx=-0.3)
    src("r4a_runtime", pd.concat(rows))

    # (b) phases, totals printed in the bars (medians)
    ax = fig.add_axes([0.33, bottom, 0.13, h])
    v = med.set_index(["method", "n_candidates"])
    big = "srs_ap_v" if ("srs_ap_v", 100000) in v.index else "srs_c_v"
    bars = [("srs_ap_v", 500, "500"), ("srs_ap_v", 2274, "2.3k"), ("srs_ap_v", 10000, "10k"), (big, 100000, "100k")]
    phases = [("phase_chains", "surrogate chains", S.SAND), ("phase_select_q0", "selection, $\\omega$ = 0", S.TEAL_LIGHT),
              ("phase_select_q25", "selection, $\\omega$ = 0.25", S.TEAL)]
    base = np.zeros(len(bars))
    out = []
    for col, text, color in phases:
        share = np.array([v.loc[(m, n), col] / v.loc[(m, n), "seconds"] for m, n, _ in bars])
        ax.bar(range(len(bars)), share, bottom=base, color=color, width=0.72, edgecolor="white", lw=0.4, label=text)
        base += share
    for k, (m, n, lab) in enumerate(bars):
        t = v.loc[(m, n), "seconds"]
        total = f"{t:.0f} s" if t < 1000 else (f"{t / 60:.0f} min" if t < 3600 else f"{t / 3600:.1f} h")
        ax.text(k, 0.97, total + (" (lik.)" if m == "srs_c_v" else ""), rotation=90, ha="center", va="top", fontsize=FS, color="white")
        out.append({"method": m, "n_candidates": n, **{c: v.loc[(m, n), c] for c, _, _ in phases}, "seconds": t})
    ax.set_xticks(range(len(bars)), [b[2] for b in bars])
    ax.set_ylim(0, 1.0)
    ax.set_yticks([0, 0.5, 1.0], ["0", "50%", "100%"])
    ax.set_xlabel("candidates $|\\mathcal{C}|$")
    ax.legend(loc="upper center", bbox_to_anchor=(0.45, -0.22), ncol=1, handlelength=0.8, labelspacing=0.2)
    title(ax, "Time per phase")
    letter(ax, "b", dx=-0.42)
    src("r4b_phases", pd.DataFrame(out))

    # (c) cost and quality (two experiments: time from the benchmark, quality from the 16 runs, seed means)
    ax = fig.add_axes([0.55, bottom, 0.165, h])
    T = pd.read_csv(SRC / "q2_seeds_long.csv")
    quality = T.groupby(["run", "registry"]).real.mean().groupby("registry").mean()
    pts = [("srs_ap_v", "srs_ap_t", "SRS+T", S.TEAL), ("srs_c_v", "srs_c_val", "SRS+T-lik", S.TEAL_DARK),
           ("fs_cv", "fs_cv", "Wrapper-lik", S.GREY_LIGHT), ("dpp", "dpp_ig", "DPP", S.GREY), ("mmrfs", "mmrfs", "MMRFS", S.GREY),
           ("l1_logistic", "l1_logistic", "L1", S.GREY), ("cmim", "cmim", "CMIM", S.GREY), ("jmi", "jmi", "JMI", S.GREY),
           ("nmifs", "nmifs", "NMIFS", S.BLUE), ("mrmr", "mrmr", "mRMR", S.SKY)]
    t500 = med[med.n_candidates == 500].set_index("method").seconds
    frame = pd.DataFrame([{"method": text, "seconds": float(t500[tm]), "auprc": float(quality[qm]), "color": color}
                          for tm, qm, text, color in pts if tm in t500.index and qm in quality.index])
    frame = frame.sort_values("seconds").reset_index(drop=True)
    best, front = -1.0, []
    for _, row in frame.iterrows():
        if row.auprc > best:
            front.append(row)
            best = row.auprc
    fx, fy = [r.seconds for r in front], [r.auprc for r in front]
    ax.step(fx + [400], fy + [fy[-1]], where="post", color=S.TEAL_LIGHT, lw=1.0, zorder=1)
    for _, row in frame.iterrows():
        ax.scatter(row.seconds, row.auprc, s=18, color=row.color, zorder=3, edgecolor="white", lw=0.4)
    offsets = {"SRS+T": (0, 5, "center", "bottom"), "SRS+T-lik": (-5, -2, "right", "top"), "Wrapper-lik": (0, -5, "center", "top"),
               "DPP": (0, 5, "center", "bottom"), "MMRFS": (0, -5, "center", "top"), "L1": (5, 0, "left", "center"),
               "CMIM": (5, 0, "left", "center"), "JMI": (5, 0, "left", "center"), "NMIFS": (5, 3, "left", "bottom"), "mRMR": (5, -2, "left", "top")}
    for _, row in frame.iterrows():
        dx, dy, ha, va = offsets[row.method]
        ax.annotate(row.method, (row.seconds, row.auprc), xytext=(dx, dy), textcoords="offset points", ha=ha, va=va, fontsize=FS, color=S.INK2)
    ax.set_xscale("log")
    ax.set_xlim(2e-3, 400)
    ax.set_xticks([1e-2, 1, 1e2], ["0.01", "1", "100"])
    ax.minorticks_off()
    ax.set_ylim(frame.auprc.min() - 0.008, frame.auprc.max() + 0.012)
    ax.set_xlabel("time at 500 candidates (s)")
    ax.set_ylabel("AUPRC, eICU + HiRID")
    ax.grid(which="major")
    title(ax, "Cost and quality")
    letter(ax, "c", dx=-0.36)
    src("r4c_cost_quality", frame.drop(columns="color"))

    # (d) SQL and Python engine timings
    ax = fig.add_axes([0.83, bottom, 0.165, h])
    rep = pd.DataFrame(_json.loads((S.ART / "sql_execution/timing_repeats_f_mimic_iv_aki_gap6h.json").read_text(encoding="utf-8"))["rows"])
    mem30 = _json.loads((S.ART / "sql_execution/disk_scale_f_mimic_iv_aki_gap6h_memory.json").read_text(encoding="utf-8"))
    mem30 = [r for r in mem30 if r["replicate"] == 30][0]
    out, yl = [], []
    y = 0
    engines = [("duckdb_memory", "DuckDB, memory", S.TEAL, "o", True), ("duckdb_disk", "DuckDB, disk", S.TEAL, "o", False),
               ("python", "Python engine", S.SAND, "D", True)]
    for r_ in (1, 10, 30):
        for col, ptext in (("seconds_tokens", "tokens"), ("seconds_registry", "registry")):
            for off, (eng, etext, color, marker, filled) in zip((-0.24, 0.0, 0.24), engines):
                d = rep[(rep.engine == eng) & (rep.replicate == r_)][col]
                if d.empty:
                    if eng == "duckdb_memory" and r_ == 30:
                        val = mem30[col]
                        ax.scatter(val, y + off, s=16, marker="o", facecolor="white", edgecolor=S.TEAL_DARK, lw=0.9, zorder=3)
                        ax.scatter(val, y + off, s=4, marker="o", color=S.TEAL_DARK, zorder=4)
                        out.append({"replicate": r_, "phase": ptext, "engine": etext + " (one run)", "median_s": val, "q1": val, "q3": val, "n": 1})
                    continue
                med_, q1, q3 = d.median(), d.quantile(0.25), d.quantile(0.75)
                ax.hlines(y + off, q1, q3, color=color, lw=1.6)
                ax.scatter(med_, y + off, s=15, marker=marker, facecolor=color if filled else "white", edgecolor=color, lw=0.9, zorder=3)
                out.append({"replicate": r_, "phase": ptext, "engine": etext, "median_s": med_, "q1": q1, "q3": q3, "n": len(d)})
            yl.append((y, f"×{r_} {ptext}"))
            y += 1
        y += 0.45
    ax.set_yticks([t[0] for t in yl], [t[1] for t in yl])
    ax.set_ylim(y - 0.25, -0.7)
    ax.set_xscale("log")
    ax.set_xticks([0.1, 1, 10, 100], ["0.1", "1", "10", "100"])
    ax.minorticks_off()
    ax.set_xlabel("seconds")
    ax.legend(handles=[Line2D([], [], marker=m, ls="", mfc=c if f else "white", mec=c, label=t) for _, t, c, m, f in engines] +
              [Line2D([], [], marker="o", ls="", mfc="white", mec=S.TEAL_DARK, label="memory, one run")],
              loc="upper center", bbox_to_anchor=(0.05, -0.22), ncol=2, handletextpad=0.1, columnspacing=0.4, labelspacing=0.2)
    ax.grid(axis="x", which="major")
    title(ax, "Registry as SQL")
    letter(ax, "d", dx=-0.62)
    src("r4d_sql_timing", pd.DataFrame(out))
    S.save_exact(fig, "fig_r4_efficiency")


def fig_r6() -> None:
    """Contract. Claim (RQ6): in the pre-specified drive domain the pre-registered test replicated partially; the post hoc
    AUPRC decision helped only at the pre-specified K = 10, where SRS+T and L1+dropout tie; surrogates decide this domain.
    Evidence: (a) heat map of paired differences per budget with significance glyphs, (b) stacked bars of how the K = 10
    programs run, (c) target AUPRC against K incl. the pre-specified likelihood version (lines), (d) forest of the
    pre-registered tests, pre-specified and post hoc. Data: union_ap/drive_k*, nmifs_ap/drive_k*, references/drive_k*,
    paper_v3_cache/drive_executability.csv, drive/union_k10, drive/union_extra_k10."""
    from figs_v4 import verdicts as _verdicts, ZERO as _ZERO
    rows, paired, abl = data.drives()
    with_k = lambda pattern: pd.concat([pd.read_csv(f).assign(k=int(f.split("drive_k")[1].split("\\")[0].split("/")[0]))  # noqa: E731
                                        for f in glob.glob(str(S.ART / pattern))], ignore_index=True)
    hyb_rows, hyb_pair, ref_pair = with_k("nmifs_ap/drive_k*/*/rows.csv"), with_k("nmifs_ap/drive_k*/*/paired.csv"), with_k("references/drive_k*/*/paired.csv")
    rows = pd.concat([rows, hyb_rows[hyb_rows.registry == "nmifs_ap_t"]], ignore_index=True)
    hp = hyb_pair[(hyb_pair.a == "srs_ap_t") & (hyb_pair.b == "nmifs_ap_t")].copy()
    rp = ref_pair[(ref_pair.a == "srs_ap_t") & (ref_pair.metric.isin(["internal", "mean_target"]))].copy()
    paired = pd.concat([paired, _verdicts(hp), _verdicts(rp)], ignore_index=True)
    fig = plt.figure(figsize=(S.FULL_W, 2.75))
    top, bottom = 0.87, 0.3
    h = top - bottom

    # (a) heat map with significance glyphs
    ax = fig.add_axes([0.13, bottom, 0.14, h])
    base = ["nmifs_ap_t", "nmifs", "filter_nmifs", "mrmr", "mmrfs", "cmim", "jmi", "l1_logistic", "l1_dropout", "mrmr_filter_known"]
    mat = np.zeros((len(base), 6))
    ver = np.zeros_like(mat)
    for i, b in enumerate(base):
        for ki, k in enumerate((5, 10, 20)):
            for mi, metric in enumerate(("internal", "mean_target")):
                d = paired[(paired.a == "srs_ap_t") & (paired.b == b) & (paired.k == k) & (paired.metric == metric)]
                mat[i, ki * 2 + mi], ver[i, ki * 2 + mi] = d.point_diff.iloc[0], d.verdict.iloc[0]
    lim = 0.1
    im = ax.imshow(np.clip(mat, -lim, lim), cmap=S.DIV, vmin=-lim, vmax=lim, aspect="auto")
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            if ver[i, j]:
                ax.text(j, i, "\u25b2" if ver[i, j] == 1 else "\u25bc", ha="center", va="center", fontsize=FS,
                        color="white" if abs(mat[i, j]) > 0.05 else S.INK)
    for x in (1.5, 3.5):
        ax.axvline(x, color="white", lw=2.0)
    ax.set_xticks(range(6), ["int.", "tgt."] * 3, rotation=90)
    for ki, k in enumerate((5, 10, 20)):
        ax.text(ki * 2 + 0.5, -0.75, f"$K$={k}" + ("*" if k == 10 else ""), ha="center", va="bottom", fontsize=FS, color=S.INK2)
    ax.set_yticks(range(len(base)), [LABEL[b] for b in base])
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    cax = ax.inset_axes([0.0, -0.33, 1.0, 0.04])
    cb = fig.colorbar(im, cax=cax, orientation="horizontal", ticks=[-0.1, 0, 0.1])
    cb.ax.set_xticklabels(["$-$.1", "0", "+.1"])
    cb.ax.tick_params(labelsize=FS, length=1.5, pad=1)
    cb.outline.set_visible(False)
    ax.set_title("SRS+T $-$ comparator", loc="left", fontsize=FS_TITLE, fontweight="bold", y=1.1, pad=0)
    letter(ax, "a", dx=-0.8, dy=1.1)
    src("r6a_drive_heatmap", pd.DataFrame(mat, index=[LABEL[b] for b in base],
                                          columns=[f"K{k}_{m}" for k in (5, 10, 20) for m in ("internal", "target")]).reset_index())

    # (b) how the K = 10 programs run at the three targets
    ax = fig.add_axes([0.39, bottom, 0.135, h])
    ex = pd.read_csv(S.ART / "paper_v3_cache/drive_executability.csv")
    ex = ex[ex.k == 10]
    meths = ["srs_ap_t", "srs_ap", "nmifs_ap_t", "nmifs", "mrmr", "l1_dropout", "fs_cv"]
    tot = ex.groupby("registry")[["primary", "surrogate", "zeroed"]].sum().loc[meths]
    left = np.zeros(len(meths))
    for col, text, color in (("primary", "selected", S.TEAL), ("surrogate", "surrogate", S.TEAL_LIGHT), ("zeroed", "zeroed", _ZERO)):
        ax.barh(range(len(meths)), tot[col].values, left=left, color=color, height=0.7, edgecolor="white", lw=0.5, label=text)
        for k_, v_ in enumerate(tot[col].values):
            if v_ >= 4:
                ax.text(left[k_] + v_ / 2, k_, str(int(v_)), ha="center", va="center", fontsize=FS, color="white" if col == "primary" else S.INK)
        left += tot[col].values
    ax.set_yticks(range(len(meths)), [LABEL[m] for m in meths])
    ax.set_ylim(len(meths) - 0.5, -0.6)
    ax.set_xlim(0, 30)
    ax.set_xticks([0, 10, 20, 30])
    ax.set_xlabel("program\u2013target pairs")
    ax.legend(ncol=1, loc="upper center", bbox_to_anchor=(0.45, -0.2), handlelength=0.8, labelspacing=0.2)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    title(ax, "How programs run")
    letter(ax, "b", dx=-0.72)
    src("r6b_executability", ex)

    # (c) budget
    ax = fig.add_axes([0.6, bottom, 0.12, h])
    ref = rows[rows.registry == "srs_ap_t"].set_index("k").mean_target_auprc.loc[[5, 10, 20]]
    out = []
    series = [("srs_ap_t", "SRS+T"), ("srs_c_val", "lik. (pre-spec.)"), ("nmifs_ap_t", "NMIFS+AP"), ("nmifs", "NMIFS"), ("l1_dropout", "L1+dropout")]
    for m, text in series:
        vals = rows[rows.registry == m].set_index("k").mean_target_auprc.loc[[5, 10, 20]].values
        sty = style_of(m) if m != "srs_c_val" else dict(color=S.TEAL_DARK, ls=(0, (4, 1.5)), marker="o", lw=1.0, mfc="white")
        ax.plot([5, 10, 20], vals, color=sty["color"], ls=sty["ls"], marker=sty["marker"], ms=3, mfc=sty["mfc"], mec=sty["color"], lw=sty["lw"], label=text)
        out.append({"variant": m, "k5": vals[0], "k10": vals[1], "k20": vals[2]})
    d = abl[(abl.a == "srs_ap_t") & (abl.b == "srs_ap_t_nosur") & (abl.target == "mean_over_targets")].set_index("k").point_diff.loc[[5, 10, 20]]
    vals = ref.values - d.values
    ax.plot([5, 10, 20], vals, color=S.WARM, ls=(0, (3, 1.5)), marker="o", ms=3, mfc="white", mec=S.WARM, lw=1.0, label="no surr. obj.")
    out.append({"variant": "srs_ap_t_nosur", "k5": vals[0], "k10": vals[1], "k20": vals[2]})
    ax.set_xscale("log", base=2)
    ax.set_xticks([5, 10, 20], ["5", "10", "20"])
    ax.minorticks_off()
    ax.set_xlabel("budget $K$")
    ax.set_ylabel("mean target AUPRC")
    ax.legend(loc="upper center", bbox_to_anchor=(0.45, -0.2), ncol=2, handlelength=1.5, columnspacing=0.4, labelspacing=0.15, handletextpad=0.3)
    ax.grid(axis="y")
    title(ax, "Budget")
    letter(ax, "c", dx=-0.5)
    src("r6c_variants", pd.DataFrame(out))

    # (d) pre-registered tests at K = 10
    ax = fig.add_axes([0.85, bottom, 0.145, h])
    pre_dir = S.ART / "drive/union_k10/f_backblaze_seagate_gap2d"
    pi = pd.read_csv(pre_dir / "paired_internal.csv")
    pt = pd.read_csv(pre_dir / "paired_targets.csv")
    e = pd.read_csv(S.ART / "drive/union_extra_k10/f_backblaze_seagate_gap2d/paired_targets.csv")
    for df in (pi, pt, e):
        df["a"] = df["a"].str.replace(":composite", "")
        df["b"] = df["b"].str.replace(":composite", "")

    def ci(d):
        d = d.iloc[0]
        return float(d.point_diff), float(d.low), float(d.high)

    def post(a, b, metric):
        return ci(paired[(paired.a == a) & (paired.b == b) & (paired.k == 10) & (paired.metric == metric)])

    tests = [("H1 mRMR, int.", ci(pi[(pi.a == "srs_c_val") & (pi.b == "mrmr")]), post("srs_ap_t", "mrmr", "internal")),
             ("H1 mRMR, tgt.", ci(pt[(pt.a == "srs_c_val") & (pt.b == "mrmr") & (pt.target == "mean_over_targets")]), post("srs_ap_t", "mrmr", "mean_target")),
             ("H2 MMRFS, int.", ci(pi[(pi.a == "srs_c_val") & (pi.b == "mmrfs")]), post("srs_ap_t", "mmrfs", "internal")),
             ("H2 MMRFS, tgt.", ci(pt[(pt.a == "srs_c_val") & (pt.b == "mmrfs") & (pt.target == "mean_over_targets")]), post("srs_ap_t", "mmrfs", "mean_target")),
             ("H2 L1, int.", ci(pi[(pi.a == "srs_c_val") & (pi.b == "l1_logistic")]), post("srs_ap_t", "l1_logistic", "internal")),
             ("H2 L1, tgt.", ci(pt[(pt.a == "srs_c_val") & (pt.b == "l1_logistic") & (pt.target == "mean_over_targets")]), post("srs_ap_t", "l1_logistic", "mean_target")),
             ("H3 SRS, tgt.", ci(e[(e.a == "srs_c_unknown_val") & (e.b == "mrmr") & (e.target == "mean_over_targets")]), post("srs_ap", "mrmr", "mean_target"))]
    xmax = 0.07
    out = []
    for i, (name, a_pre, a_post) in enumerate(tests):
        for off, (vals, filled) in zip((-0.2, 0.2), ((a_pre, False), (a_post, True))):
            color = S.TEAL if filled else S.GREY
            lo, hi, pt_ = vals[1], min(vals[2], xmax), vals[0]
            if pt_ > xmax:
                ax.scatter(xmax - 0.003, i + off, marker=">", s=14, color=color, zorder=3)
                ax.text(xmax - 0.007, i + off, f"{pt_:+.2f}", ha="right", va="center", fontsize=FS, color=color)
                continue
            ax.hlines(i + off, lo, hi, color=color, lw=1.0)
            ax.scatter(pt_, i + off, s=14, marker="o", facecolor=color if filled else "white", edgecolor=color, lw=0.9, zorder=3)
        out.append({"test": name, "pre_diff": a_pre[0], "pre_low": a_pre[1], "pre_high": a_pre[2], "post_diff": a_post[0], "post_low": a_post[1],
                    "post_high": a_post[2]})
    ax.axvline(0, color=S.INK2, lw=0.6, ls=(0, (2, 2)))
    ax.set_xlim(-0.02, xmax)
    ax.set_xticks([0, 0.05], ["0", ".05"])
    ax.set_yticks(range(len(tests)), [t[0] for t in tests])
    ax.set_ylim(len(tests) - 0.4, -0.6)
    ax.set_xlabel("AUPRC difference")
    ax.legend(handles=[Line2D([], [], marker="o", ls="-", lw=0.9, mfc="white", mec=S.GREY, color=S.GREY, label="pre-specified"),
                       Line2D([], [], marker="o", ls="-", lw=0.9, color=S.TEAL, label="post hoc")],
              loc="upper center", bbox_to_anchor=(0.2, -0.2), ncol=1, handletextpad=0.3, labelspacing=0.2, handlelength=1.4)
    ax.grid(axis="x")
    title(ax, "Pre-registered")
    letter(ax, "d", dx=-0.85)
    src("r6d_hypotheses", pd.DataFrame(out))
    S.save_exact(fig, "fig_r6_drives")


if __name__ == "__main__":
    pd.set_option("display.width", 200)
    figs = {"r1": fig_r1, "r2": fig_r2, "r3": fig_r3, "r4": fig_r4, "r5": fig_r5, "r6": fig_r6}
    for name in sys.argv[1:] or list(figs):
        figs[name]()
        print("wrote", name)
