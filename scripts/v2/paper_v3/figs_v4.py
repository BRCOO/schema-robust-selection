"""Experiment figures of the PVLDB v3 paper, third design (after the second review of 2026-11-05): six experiments, one
multi-panel figure each, every panel a different chart type, text at 6-7.5 pt.

Workflow of the cs-experiment-figure-studio skill: a figure contract in each docstring, source data of every panel in
paper/v3_assets/source_data/ (prefix r1..r6), PDF/SVG/PNG exports, Nature teal-green palette (style.py).

  PYTHONDONTWRITEBYTECODE=1 /d/BRCoo/miniconda/python scripts/v2/paper_v3/figs_v4.py [r1 ... r6]
"""
from __future__ import annotations

import glob
import json
import sys

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.transforms import blended_transform_factory

import data
import style as S
from figs_b import method_style, src
from figs_v3 import ART, LABEL, all_paired, name_of, rows_all, verdict

S.setup()
mpl.rcParams.update({"font.size": 7, "axes.labelsize": 7, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "legend.fontsize": 6.3})
LABEL = dict(LABEL)
LABEL.update({"filter_nmifs": "Filter+NMIFS", "srs_c_val": "SRS+T, lik.", "all_programs": "all programs", "random_k": "random K"})
HYB = "#7E8FBF"
ZERO = "#F1C9BE"
RUN_ORDER = [data.run_name(r, l) for r, l in data.ICU_RUNS]
TASK_SHORT = {"Vaso 1h": "1", "Vaso 2h": "2", "Vaso 4h": "4", "AKI": "A"}
TGT8 = [("internal", "Int."), ("eicu", "eICU"), ("hirid", "HiRID"), ("drop_labs", "labs"), ("drop_ventilation", "vent."),
        ("drop_bp", "BP"), ("drop_urine", "urine"), ("vitals_only", "vitals"), ("mean", "Tgt.")]
RNG = np.random.default_rng(7)


def title(ax, text: str, y: float | None = None) -> None:
    if y is None:
        ax.set_title(text, loc="left", fontsize=7.5, fontweight="bold", pad=3)
    else:
        ax.set_title(text, loc="left", fontsize=7.5, fontweight="bold", y=y)


def letter(ax, text: str, dx: float, dy: float = 1.02) -> None:
    ax.text(dx, dy, text, transform=ax.transAxes, fontsize=9, fontweight="bold", va="bottom", ha="right", color=S.INK)


def style_of(m: str) -> dict:
    extra = {"nmifs_ap_t": dict(color=HYB, ls=(0, (3, 1.2)), marker="s", lw=1.1, mfc="white"),
             "srs_ap_unaware": dict(color="#5DBAA6", ls=(0, (1, 1)), marker="^", lw=1.0, mfc="white")}
    return extra[m] if m in extra else method_style(m)


def verdicts(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["verdict"] = verdict(df)
    return df


def read_all(pattern: str) -> pd.DataFrame:
    return pd.concat([pd.read_csv(f) for f in glob.glob(str(ART / pattern))], ignore_index=True)


def paired_srs_t() -> pd.DataFrame:
    """SRS+T against every comparator incl. Filter+NMIFS, per run, internal and mean target."""
    P = all_paired()
    r = read_all("references/union*/*/paired.csv")
    r = r[(r.a == "srs_ap_t") & (r.metric.isin(["internal", "mean_target"]))].copy()
    r["name"] = r.run.map(name_of)
    out = pd.concat([P, verdicts(r)[["name", "b", "metric", "point_diff", "low", "high", "verdict"]]], ignore_index=True)
    return out


def tally_text(v: pd.Series) -> str:
    return f"{(v == 1).sum()}/{(v == 0).sum()}/{(v == -1).sum()}"


# ================================================================================================= R1
def fig_r1() -> None:
    """Contract. Claim (RQ1): SRS+T is significantly ahead of the classic selectors in nearly all runs, its lead grows
    from K = 5 to K = 10 over mRMR but not over NMIFS, and across CV seeds it is ahead of NMIFS and NMIFS+AP on AKI and at
    K = 20 but not on the short vasopressor gaps at K = 10. Evidence: (a) per-run target differences with significance,
    (b) target AUPRC against K with the random-K floor and the all-programs probe, (c) CV-seed replicates of the three
    CV-based selectors relative to NMIFS, (d) rank frequencies among 18 selectors. Higher = better."""
    P = paired_srs_t()
    ra = rows_all()
    refs = read_all("references/union*/*/rows.csv")
    refs["name"] = refs.run.map(name_of)
    fig = plt.figure(figsize=(S.FULL_W, 2.55))
    gs = GridSpec(1, 4, width_ratios=[1.55, 0.95, 1.05, 1.05], wspace=0.78, left=0.1, right=0.985, top=0.84, bottom=0.25)

    # (a) per-run heatmap
    ax = fig.add_subplot(gs[0])
    order_b = ["nmifs_ap_t", "srs_ap_unaware", "nmifs", "filter_nmifs", "l1_dropout", "mrmr_filter_known", "mrmr", "cmim", "mmrfs"]
    mat = np.full((len(order_b), len(RUN_ORDER)), np.nan)
    ver = np.zeros_like(mat)
    tallies = []
    for i, b in enumerate(order_b):
        s = P[(P.b == b) & (P.metric == "mean_target")].drop_duplicates("name").set_index("name")
        for j, n in enumerate(RUN_ORDER):
            mat[i, j], ver[i, j] = s.loc[n, "point_diff"], s.loc[n, "verdict"]
        tallies.append(tally_text(s.verdict))
    lim = 0.05
    im = ax.imshow(np.clip(mat, -lim, lim), cmap=S.DIV, vmin=-lim, vmax=lim, aspect="auto")
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            if ver[i, j] == 0:
                ax.scatter(j, i, s=9, facecolor="none", edgecolor=S.INK, lw=0.55)
            elif ver[i, j] == -1:
                ax.scatter(j, i, s=12, marker="x", color=S.INK, lw=0.9)
    for a0, a1, text in [(0, 4, "K5"), (4, 8, "K10"), (8, 12, "K20"), (12, 14, "L10"), (14, 16, "L20")]:
        ax.text((a0 + a1 - 1) / 2, -0.8, text, ha="center", va="bottom", fontsize=6.3, color=S.INK2)
        if a0:
            ax.axvline(a0 - 0.5, color="white", lw=1.8)
    ax.axhline(2.5, color="white", lw=1.8)
    ax.set_xticks(range(len(RUN_ORDER)), [TASK_SHORT[n.split(" L")[0].rsplit(" K", 1)[0]] for n in RUN_ORDER], fontsize=6)
    ax.set_yticks(range(len(order_b)), [LABEL[b] for b in order_b], fontsize=6.5)
    for i, t in enumerate(tallies):
        ax.text(len(RUN_ORDER) - 0.25, i, t, fontsize=6.2, va="center", ha="left", color=S.INK2)
    ax.text(len(RUN_ORDER) - 0.25, -0.8, "b/n.s./w", fontsize=6.2, va="bottom", ha="left", color=S.INK2, fontweight="bold")
    cax = ax.inset_axes([0.0, -0.2, 0.32, 0.045])
    cb = fig.colorbar(im, cax=cax, orientation="horizontal", ticks=[-0.05, 0, 0.05])
    cb.ax.tick_params(labelsize=6, length=1.5, pad=1)
    cb.outline.set_visible(False)
    ax.text(0.38, -0.18, "○ n.s.   × SRS+T worse", transform=ax.transAxes, fontsize=6.2, va="center", color=S.INK2)
    title(ax, "SRS+T $-$ comparator, targets", y=1.1)
    letter(ax, "a", dx=-0.27, dy=1.1)
    src("r1a_per_run", pd.DataFrame(mat, index=[LABEL[b] for b in order_b], columns=RUN_ORDER).reset_index())

    # (b) budget with references
    ax = fig.add_subplot(gs[1])
    std = ra[~ra.run.str.startswith("union_large")].copy()
    std["k"] = std.run.str.split("/").str[0].map(data.K_OF)
    rows = []
    for m in ("srs_ap_t", "srs_ap", "nmifs_ap_t", "nmifs", "mrmr", "l1_dropout"):
        g = std[std.registry == m].groupby("k").mean_target_auprc.mean()
        sty = style_of(m)
        ax.plot(g.index, g.values, color=sty["color"], ls=sty["ls"], marker=sty["marker"], ms=3, mfc=sty["mfc"], mec=sty["color"],
                lw=sty["lw"], label=LABEL[m], zorder=3)
        rows.append(pd.DataFrame({"method": LABEL[m], "k": g.index, "mean_target": g.values}))
    rstd = refs[~refs.run.str.startswith("union_large")].copy()
    rstd["k"] = rstd.run.str.split("/").str[0].map(data.K_OF)
    rk = rstd[rstd.registry == "random_k"].groupby("k").mean_target_auprc.mean()
    allp = rstd[rstd.registry == "all_programs"].mean_target_auprc.mean()
    ax.plot(rk.index, rk.values, color=S.GREY_LIGHT, ls=(0, (1, 1.2)), lw=1.0, marker="o", ms=2.2, zorder=2)
    ax.text(20, rk.values[-1] - 0.004, "random", fontsize=6, color=S.GREY, ha="right", va="top")
    ax.axhline(allp, color=S.GREY, lw=0.7, ls=(0, (3, 2)), zorder=1)
    ax.text(5, allp + 0.002, f"all programs ({allp:.3f})", fontsize=6, color=S.GREY, va="bottom")
    rows.append(pd.DataFrame({"method": "random K", "k": rk.index, "mean_target": rk.values}))
    rows.append(pd.DataFrame({"method": "all programs", "k": [0], "mean_target": [allp]}))
    ax.set_xscale("log", base=2)
    ax.set_xticks([5, 10, 20], ["5", "10", "20"])
    ax.minorticks_off()
    ax.set_xlabel("budget $K$")
    ax.set_ylabel("mean target AUPRC (4 settings)")
    ax.legend(fontsize=6, loc="upper center", bbox_to_anchor=(0.42, -0.22), ncol=2, handlelength=1.7, columnspacing=0.6, handletextpad=0.3,
              labelspacing=0.2)
    ax.grid(axis="y")
    title(ax, "Budget", y=1.1)
    letter(ax, "b", dx=-0.36, dy=1.1)
    src("r1b_budget", pd.concat(rows))

    # (c) CV-seed replicates relative to NMIFS
    ax = fig.add_subplot(gs[2])
    seeds = read_all("seed_replicates/*/rows.csv") if glob.glob(str(ART / "seed_replicates/f_*/rows.csv")) else pd.DataFrame()
    seeds20 = read_all("seed_replicates/union_k20/*/rows.csv") if glob.glob(str(ART / "seed_replicates/union_k20/*/rows.csv")) else pd.DataFrame()
    sens = data.sensitivity()
    hyb_rows = read_all("nmifs_ap/union*/*/rows.csv")
    out, ylabels = [], []
    y = 0
    for root, kname in (("union", "K10"), ("union_k20", "K20")):
        for label, task in (("f_mimic_iv_gap1h", "V1h"), ("f_mimic_iv_gap2h", "V2h"), ("f_mimic_iv_gap4h", "V4h"), ("f_mimic_iv_aki_gap6h", "AKI")):
            run = f"{root}/{label}"
            base = ra[ra.run == run].set_index("registry").mean_target_auprc
            nm = float(base["nmifs"])
            groups = {}
            if root == "union":
                s = sens[(sens.label == label) & (sens.parameter.isin(["default", "seed"]))].mean_target_auprc.values
                groups["srs_ap_t"] = list(s)
            else:
                extra = seeds20[(seeds20.run == run) & (seeds20.registry == "srs_ap_t")].mean_target_auprc.tolist() if len(seeds20) else []
                groups["srs_ap_t"] = [float(base["srs_ap_t"])] + extra
            src_rows = seeds if root == "union" else seeds20
            for m in ("nmifs_ap_t", "srs_ap_unaware"):
                extra = src_rows[(src_rows.run == run) & (src_rows.registry == m)].mean_target_auprc.tolist() if len(src_rows) else []
                seed0 = float(base[m]) if m in base else float(hyb_rows[(hyb_rows.run == run) & (hyb_rows.registry == m)].mean_target_auprc.iloc[0])
                groups[m] = [seed0] + extra
            for off, (m, color, marker) in zip((-0.24, 0.0, 0.24), (("srs_ap_t", S.TEAL, "o"), ("nmifs_ap_t", HYB, "s"), ("srs_ap_unaware", "#5DBAA6", "^"))):
                v = np.array(groups[m]) - nm
                ax.scatter(v, np.full(len(v), y + off), s=9, marker=marker, color=color, lw=0, alpha=0.9, zorder=3)
                if len(v) > 1:
                    ax.hlines(y + off, v.min(), v.max(), color=color, lw=0.8, alpha=0.6, zorder=2)
                out.append({"run": run, "method": LABEL[m], "n_seeds": len(v), **{f"seed{i}": x for i, x in enumerate(v)}})
            ylabels.append((y, f"{task} {kname}"))
            y += 1
        y += 0.4
    ax.axvline(0, color=S.BLUE, lw=0.8, ls=(0, (2, 1.5)))
    ax.text(0.002, y - 0.35, "NMIFS", color=S.BLUE, fontsize=6, va="bottom")
    ax.set_yticks([t[0] for t in ylabels], [t[1] for t in ylabels], fontsize=6.2)
    ax.set_ylim(y - 0.2, -0.9)
    ax.set_xlabel("target AUPRC $-$ NMIFS")
    ax.legend(handles=[Line2D([], [], marker="o", ls="", color=S.TEAL, label="SRS+T"), Line2D([], [], marker="s", ls="", color=HYB, label="NMIFS+AP"),
                       Line2D([], [], marker="^", ls="", color="#5DBAA6", label="Wrapper, AUPRC")],
              fontsize=6, loc="upper center", bbox_to_anchor=(0.4, -0.22), ncol=2, handletextpad=0.1, columnspacing=0.5)
    ax.grid(axis="x")
    title(ax, "Five CV seeds", y=1.1)
    letter(ax, "c", dx=-0.33, dy=1.1)
    src("r1c_seeds", pd.DataFrame(out))

    # (d) ranks among 18 selectors
    ax = fig.add_subplot(gs[3])
    fn = refs[refs.registry == "filter_nmifs"][["run", "registry", "mean_target_auprc"]]
    methods = ["srs_ap_t", "srs_ap", "srs_ap_unaware", "fs_cv", "nmifs_ap_t"] + list(S.BASELINES)
    piv = pd.concat([ra[ra.registry.isin(methods)][["run", "registry", "mean_target_auprc"]], fn]).pivot_table(index="run", columns="registry",
                                                                                                                values="mean_target_auprc")
    assert piv.shape == (16, 18), piv.shape
    ranks = piv.rank(axis=1, ascending=False, method="min")
    show = ["srs_ap_t", "srs_ap", "srs_ap_unaware", "nmifs_ap_t", "nmifs", "filter_nmifs", "l1_dropout", "mrmr"]
    bins = [(1, 1, "1st"), (2, 2, "2nd"), (3, 3, "3rd"), (4, 5, "4-5th"), (6, 99, "6th+")]
    colors = [S.TEAL_DARK, S.TEAL, "#5DBAA6", S.TEAL_LIGHT, S.GREY_PALE]
    left = np.zeros(len(show))
    table = []
    for (a0, a1, text), color in zip(bins, colors):
        vals = np.array([((ranks[m] >= a0) & (ranks[m] <= a1)).sum() for m in show])
        ax.barh(range(len(show)), vals, left=left, color=color, height=0.68, edgecolor="white", lw=0.5, label=text)
        for k, v in enumerate(vals):
            if v >= 2:
                ax.text(left[k] + v / 2, k, str(v), ha="center", va="center", fontsize=6, color="white" if a0 <= 2 else S.INK)
        left += vals
        table.append(pd.Series(vals, index=[LABEL.get(m, m) for m in show], name=text))
    ax.set_yticks(range(len(show)), [LABEL.get(m, m) for m in show], fontsize=6.5)
    ax.set_ylim(len(show) - 0.5, -0.6)
    ax.set_xlim(0, 16)
    ax.set_xticks([0, 4, 8, 12, 16])
    ax.set_xlabel("runs (of 16)")
    ax.legend(ncol=5, loc="upper center", bbox_to_anchor=(0.3, -0.22), fontsize=6, handlelength=0.7, columnspacing=0.4, handletextpad=0.2)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    title(ax, "Rank among 18 selectors", y=1.1)
    letter(ax, "d", dx=-0.55, dy=1.1)
    src("r1d_ranks", pd.concat(table, axis=1).reset_index().rename(columns={"index": "method"}))
    S.save(fig, "fig_r1_effectiveness")


# ================================================================================================= R2
def per_variant_tallies() -> pd.DataFrame:
    """SRS+T against each component variant, per run and target: better / n.s. / worse counts."""
    a = data.ablation_all()
    a = a[(a.a == "srs_ap_t") & (a.b.isin(["srs_ap_unaware", "srs_ap_t_nosur"]))][["name", "b", "target", "point_diff", "low", "high"]]
    a["target"] = a.target.replace({"mean_over_targets": "mean"})
    e = read_all("paired_extra/union*/*/paired.csv")
    e = e[(e.a == "srs_ap_t") & (e.b.isin(["srs_c_val", "nmifs_ap_t", "srs_ap"]))].copy()
    e["name"] = e.run.map(name_of)
    e["target"] = e.target.replace({"mean_over_targets": "mean"})
    out = pd.concat([a, e[["name", "b", "target", "point_diff", "low", "high"]]], ignore_index=True)
    return verdicts(out)


def fig_r2() -> None:
    """Contract. Claim (RQ2): on heterogeneous pools the estimated deployed value ranks the next program far better than
    relevance-redundancy scores for the first picks, an advantage that fades by the eighth pick; unions of generators beat
    larger single-generator pools; and each component helps where its mechanism applies and costs where it does not.
    Evidence: (a) dumbbell of score fidelity within generators vs on the union, (b) fidelity by pick 1-11, (c) union vs
    single-generator pools, (d) tally heatmap of SRS+T against each variant per target. Data: proxy_correlation heldout11,
    restricted_pools, ablation_paired, paired_extra."""
    fig = plt.figure(figsize=(S.FULL_W, 2.45))
    gs = GridSpec(1, 4, width_ratios=[1.0, 0.95, 1.0, 1.35], wspace=0.6, left=0.075, right=0.985, top=0.86, bottom=0.24)
    rows = []
    for label, task in (("f_mimic_iv_gap1h", "Vaso 1h"), ("f_mimic_iv_gap2h", "Vaso 2h"), ("f_mimic_iv_gap4h", "Vaso 4h"), ("f_mimic_iv_aki_gap6h", "AKI")):
        d = pd.read_csv(ART / "proxy_correlation" / f"heldout11_{label}.csv")
        d = d[d.truth == "val_auprc"].dropna(subset=["spearman"]).copy()
        d["task"] = task
        rows.append(d)
    pc = pd.concat(rows, ignore_index=True)
    PROXY = {"objective": ("SRS estimate", S.TEAL), "nmifs": ("NMIFS score", S.BLUE), "mrmr": ("mRMR score", S.SKY)}

    # (a) dumbbell, picks 2-4
    ax = fig.add_subplot(gs[0])
    early = pc[pc.step.between(1, 3)].copy()
    early["where"] = np.where(early.scope == "union", "union", "within")
    agg = early.groupby(["proxy", "task", "where"]).spearman.mean().unstack().reset_index()
    labels, pos = [], []
    y = 0
    for proxy in ("objective", "nmifs", "mrmr"):
        name, color = PROXY[proxy]
        y += 1.35
        ax.text(0.28, y - 0.78, name, fontsize=6.3, color=color, fontweight="bold", va="bottom")
        for task in ("Vaso 1h", "Vaso 2h", "Vaso 4h", "AKI"):
            row = agg[(agg.proxy == proxy) & (agg.task == task)].iloc[0]
            ax.plot([row["within"], row["union"]], [y, y], color=color, lw=1.0, alpha=0.6, zorder=1)
            ax.scatter(row["within"], y, s=13, facecolor="white", edgecolor=color, lw=0.9, zorder=2)
            ax.scatter(row["union"], y, s=15, color=color, zorder=3)
            labels.append(task)
            pos.append(y)
            y += 1
    ax.set_yticks(pos, labels, fontsize=6.2)
    ax.set_ylim(y, 0.0)
    ax.set_xlim(0.28, 0.84)
    ax.set_xlabel("Spearman $\\rho$, picks 2-4")
    ax.legend(handles=[Line2D([], [], marker="o", ls="", mfc="white", mec=S.INK2, label="one generator"),
                       Line2D([], [], marker="o", ls="", color=S.INK2, label="union")],
              loc="lower right", fontsize=6, handletextpad=0.1, borderpad=0.1)
    ax.grid(axis="x")
    title(ax, "Score fidelity")
    letter(ax, "a", dx=-0.34)
    src("r2a_fidelity", agg)

    # (b) fidelity by pick
    ax = fig.add_subplot(gs[1])
    u = pc[pc.scope == "union"].groupby(["proxy", "step", "task"]).spearman.mean().reset_index()
    out = []
    for proxy in ("objective", "nmifs", "mrmr"):
        name, color = PROXY[proxy]
        g = u[u.proxy == proxy].groupby("step").spearman
        mean, lo, hi = g.mean(), g.min(), g.max()
        x = mean.index + 1
        ax.fill_between(x, lo, hi, color=color, alpha=0.13, lw=0)
        ax.plot(x, mean.values, color=color, marker="o", ms=2.4, lw=1.2, label=name.split()[0])
        out.append(pd.DataFrame({"proxy": proxy, "pick": x, "mean": mean.values, "min": lo.values, "max": hi.values}))
    ax.axhline(0, color=S.INK2, lw=0.5)
    ax.set_xticks([1, 3, 5, 7, 9, 11])
    ax.set_xlabel("pick (program number)")
    ax.set_ylabel("$\\rho$ on the union pool")
    ax.legend(fontsize=6, loc="lower left", handlelength=1.4, borderpad=0.1, labelspacing=0.2)
    ax.grid(axis="y")
    title(ax, "Across picks")
    letter(ax, "b", dx=-0.36)
    src("r2b_picks", pd.concat(out))

    # (c) union vs single-generator pools
    ax = fig.add_subplot(gs[2])
    rp = data.restricted()
    r0 = data.rows_all()
    union = r0[(r0.root == "union")].set_index(["label", "registry"])
    rows = []
    ylabels = []
    y = 0
    for label, task in (("f_mimic_iv_gap1h", "V1h"), ("f_mimic_iv_aki_gap6h", "AKI")):
        for metric, mtext in (("internal_auprc", "int."), ("mean_target_auprc", "tgt.")):
            vals = {"union (~510)": float(union.loc[(label, "srs_ap_t"), metric])}
            for gen, gtext in (("generator_erminer", "ERMiner (1,000)"), ("generator_prefixspan", "PrefixSpan (1,000)")):
                s = rp[(rp.restrict_dir == gen) & (rp.registry == "srs_ap_t") & (rp.run.str.contains(label))]
                vals[gtext] = float(s[metric].iloc[0])
            xs_ = list(vals.values())
            ax.plot([min(xs_), max(xs_)], [y, y], color=S.GREY_PALE, lw=2.6, zorder=1, solid_capstyle="round")
            for (name, v), (color, marker) in zip(vals.items(), ((S.TEAL, "o"), (S.TEAL_DARK, "s"), (S.SAND, "^"))):
                ax.scatter(v, y, s=17, color=color, marker=marker, zorder=3, label=name if y == 0 else None, lw=0)
            rows.append({"task": task, "metric": mtext, **vals})
            ylabels.append(f"{task}, {mtext}")
            y += 1
    ax.set_yticks(range(len(ylabels)), ylabels, fontsize=6.3)
    ax.set_ylim(len(ylabels) - 0.4, -0.6)
    ax.set_xlabel("AUPRC of SRS+T ($K = 10$)")
    ax.legend(fontsize=6, loc="upper center", bbox_to_anchor=(0.3, -0.22), ncol=2, handletextpad=0.1, columnspacing=0.6)
    ax.grid(axis="x")
    title(ax, "Pool composition")
    letter(ax, "c", dx=-0.5)
    src("r2c_single_generator", pd.DataFrame(rows))

    # (d) tallies of SRS+T against each variant, per target
    ax = fig.add_subplot(gs[3])
    V = per_variant_tallies()
    variants = [("srs_c_val", "lik. decision"), ("srs_ap_unaware", "no scenarios"), ("srs_ap_t_nosur", "no surr. in obj."),
                ("nmifs_ap_t", "NMIFS screen"), ("srs_ap", "no target schemas")]
    net = np.zeros((len(variants), len(TGT8)))
    text = [["" for _ in TGT8] for _ in variants]
    out = []
    for i, (v, vname) in enumerate(variants):
        for j, (t, tname) in enumerate(TGT8):
            s = V[(V.b == v) & (V.target == t)]
            assert len(s) == 16, (v, t, len(s))
            b, w = int((s.verdict == 1).sum()), int((s.verdict == -1).sum())
            net[i, j] = (b - w) / 16
            text[i][j] = f"{b}/{w}"
            out.append({"variant": vname, "target": tname, "better": b, "ns": 16 - b - w, "worse": w})
    ax.imshow(net, cmap=S.DIV, vmin=-1, vmax=1, aspect="auto")
    for i in range(net.shape[0]):
        for j in range(net.shape[1]):
            ax.text(j, i, text[i][j], ha="center", va="center", fontsize=6, color="white" if abs(net[i, j]) > 0.55 else S.INK)
    ax.axvline(len(TGT8) - 1.5, color="white", lw=1.8)
    ax.axvline(0.5, color="white", lw=1.8)
    ax.set_xticks(range(len(TGT8)), [t for _, t in TGT8], fontsize=6.2, rotation=40, ha="right", rotation_mode="anchor")
    ax.set_yticks(range(len(variants)), [n for _, n in variants], fontsize=6.4)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    title(ax, "SRS+T vs variant: b/w runs")
    letter(ax, "d", dx=-0.44)
    src("r2d_variant_tallies", pd.DataFrame(out))
    S.save(fig, "fig_r2_mechanism")


# ================================================================================================= R3
TGT_ROWS = [("hospitals", "eICU hospitals"), ("eicu", "eICU"), ("hirid", "HiRID"), ("drop_labs", "no labs"),
            ("drop_ventilation", "no vent."), ("drop_bp", "no BP"), ("drop_urine", "no urine"), ("vitals_only", "vitals only")]


def per_target_verdicts() -> pd.DataFrame:
    pt = read_all("extra_metrics/*/*/paired_targets.csv")
    _, hp, _ = data.hospital_all()
    h = hp[(hp.subset == "affected") & (hp["mode"] == "surrogate")].copy()
    h["run"] = h.root + "/" + h.label
    h["target"] = "hospitals"
    cols = ["run", "target", "a", "b", "point_diff", "low", "high"]
    return verdicts(pd.concat([pt[cols], h[cols]], ignore_index=True))


def fig_r3() -> None:
    """Contract. Claim (RQ3): under real per-hospital schemas, the real databases and controlled removals, SRS+T is ahead
    of NMIFS on most targets and of mRMR on almost all, while SRS without target schemas ties NMIFS; both trail NMIFS
    with vital signs only. The order of the three best registries holds under every deployment rule; under random loss SRS+T
    wins most draws until registries lose most programs. Evidence: (a) UpSet of absent features at eICU hospitals,
    (b) diverging verdict bars per target for three pairs, (c) dot plot over six deployment modes, (d) heatmap of strict
    win shares. Data: hospital_schemas_ap, extra_metrics paired_targets, deployment_modes_ap, stress_ap."""
    hr, hp, hm = data.hospital_all()
    fig = plt.figure(figsize=(S.FULL_W, 2.6))
    outer = GridSpec(1, 4, width_ratios=[0.9, 1.7, 1.12, 1.2], wspace=0.6, left=0.065, right=0.99, top=0.87, bottom=0.23)

    # (a) UpSet
    inner = GridSpecFromSubplotSpec(2, 1, subplot_spec=outer[0], height_ratios=[0.85, 1.0], hspace=0.08)
    summ = hm[("union", "f_mimic_iv_gap1h")]["summary"]
    items = sorted([(k, v) for k, v in summ["patterns"].items() if k != "-"], key=lambda kv: -kv[1])[:7]
    feats = ["Lactate", "pH", "sedation_on", "Bilirubin_Total", "urine_output"]
    fname = {"Lactate": "lactate", "pH": "pH", "sedation_on": "sedation", "Bilirubin_Total": "bilirubin", "urine_output": "urine"}
    axb = fig.add_subplot(inner[0])
    axm = fig.add_subplot(inner[1], sharex=axb)
    xs = np.arange(len(items))
    axb.bar(xs, [v for _, v in items], color=S.TEAL, width=0.62)
    for x, (_, v) in zip(xs, items):
        axb.text(x, v + 18, str(v), ha="center", va="bottom", fontsize=5.8, color=S.INK2)
    axb.set_ylabel("episodes", fontsize=6.5)
    axb.spines["bottom"].set_visible(False)
    axb.tick_params(axis="x", length=0, labelbottom=False)
    axb.tick_params(axis="y", labelsize=6)
    axb.set_ylim(0, max(v for _, v in items) * 1.3)
    axb.text(0.7, 0.6, f"{100 * summ['n_affected'] / summ['n_episodes']:.0f}% of\nepisodes", transform=axb.transAxes, ha="center", va="center",
             fontsize=6.2, color=S.INK2)
    for j, f in enumerate(feats):
        axm.axhspan(j - 0.5, j + 0.5, color="#F5F5F5" if j % 2 else "white", zorder=0, lw=0)
        for x, (k, _) in zip(xs, items):
            axm.scatter(x, j, s=11, color=S.TEAL_DARK if f in k.split(",") else S.GREY_PALE, zorder=2, lw=0)
    for x, (k, _) in zip(xs, items):
        on = [j for j, f in enumerate(feats) if f in k.split(",")]
        if len(on) > 1:
            axm.plot([x, x], [min(on), max(on)], color=S.TEAL_DARK, lw=0.9, zorder=1)
    axm.set_yticks(range(len(feats)), [fname[f] for f in feats], fontsize=6.3)
    axm.set_ylim(len(feats) - 0.5, -0.5)
    axm.set_xticks([])
    for spine in ("bottom", "left"):
        axm.spines[spine].set_visible(False)
    axm.tick_params(axis="y", length=0)
    title(axb, "Hospitals")
    letter(axb, "a", dx=-0.42)
    rows = [{"run": f"{r}/{l}", "pattern": k, "episodes": v, "n_affected": m["summary"]["n_affected"], "n_episodes": m["summary"]["n_episodes"],
             "n_hospitals": m["summary"]["n_hospitals"]} for (r, l), m in hm.items() for k, v in m["summary"]["patterns"].items()]
    src("r3a_hospital_patterns", pd.DataFrame(rows))

    # (b) diverging verdict bars, three pairs
    V = per_target_verdicts()
    inner = GridSpecFromSubplotSpec(1, 3, subplot_spec=outer[1], wspace=0.1)
    out = []
    for ci, (a, b, head) in enumerate((("srs_ap_t", "nmifs", "SRS+T\nvs NMIFS"), ("srs_ap", "nmifs", "SRS\nvs NMIFS"), ("srs_ap_t", "mrmr", "SRS+T\nvs mRMR"))):
        ax = fig.add_subplot(inner[ci])
        for i, (t, text) in enumerate(TGT_ROWS):
            s = V[(V.a == a) & (V.b == b) & (V.target == t)]
            assert len(s) == 16, (a, b, t, len(s))
            w, n, bt = int((s.verdict == -1).sum()), int((s.verdict == 0).sum()), int((s.verdict == 1).sum())
            for left, width, color, ink in ((-(w + n / 2), w, S.WARM, "white"), (-n / 2, n, S.GREY_PALE, S.INK2), (n / 2, bt, S.TEAL, "white")):
                if width:
                    ax.barh(i, width, left=left, height=0.72, color=color, edgecolor="white", lw=0.5)
                    if width >= (3 if color == S.GREY_PALE else 2):
                        ax.text(left + width / 2, i, str(width), ha="center", va="center", fontsize=6, color=ink)
            out.append({"pair": head.replace("\n", " "), "target": text, "better": bt, "ns": n, "worse": w})
        ax.axvline(0, color=S.INK2, lw=0.5, zorder=0)
        ax.axhline(2.5, color=S.INK2, lw=0.4, ls=(0, (1, 1.5)))
        ax.set_xlim(-15, 17)
        ax.set_xticks([-10, 0, 10], ["10", "0", "10"], fontsize=6)
        ax.set_ylim(len(TGT_ROWS) - 0.5, -0.5)
        ax.set_xlabel(head, fontsize=6.4)
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="y", length=0)
        if ci == 0:
            ax.set_yticks(range(len(TGT_ROWS)), [t for _, t in TGT_ROWS], fontsize=6.4)
            title(ax, "Verdicts per target")
            letter(ax, "b", dx=-0.85)
        else:
            ax.set_yticks([])
        if ci == 1:
            ax.legend(handles=[Patch(color=S.TEAL, label="first better"), Patch(color=S.GREY_PALE, label="n.s."), Patch(color=S.WARM, label="first worse")],
                      loc="upper center", bbox_to_anchor=(0.5, -0.25), ncol=3, fontsize=6, handlelength=0.9, handletextpad=0.3, columnspacing=0.7)
    src("r3b_target_verdicts", pd.DataFrame(out))

    # (c) deployment modes: dot plot
    dr, _ = data.deploy_all()
    modes = [("zero", "zeroed"), ("reduced", "re-fit"), ("robust", "dropout probe"), ("surrogate", "surrogates"),
             ("reduced_surrogate", "re-fit + surr."), ("soft", "soft surr.")]
    tg = dr[~dr.target.isin(["internal_test"]) & dr["mode"].isin([m for m, _ in modes])]
    mt = tg.groupby(["registry", "mode", "name"]).auprc.mean().groupby(["registry", "mode"]).mean().unstack()
    ax = fig.add_subplot(outer[2])
    meths = ("srs_ap_t", "srs_ap", "nmifs", "mrmr", "l1_dropout")
    for i, (md, text) in enumerate(modes):
        vals = [mt.loc[m, md] for m in meths]
        ax.hlines(i, min(vals), max(vals), color=S.GREY_PALE, lw=2.4, zorder=1)
        for m in meths:
            st = style_of(m)
            ax.scatter(mt.loc[m, md], i, marker=st["marker"], s=17, facecolor=st["mfc"] if st["mfc"] != "white" else "white",
                       edgecolor=st["color"], lw=0.9, zorder=3, label=LABEL[m] if i == 0 else None)
    ax.set_yticks(range(len(modes)), [t for _, t in modes], fontsize=6.3)
    ax.set_ylim(len(modes) - 0.5, -0.6)
    ax.set_xlabel("mean target AUPRC (16 runs)")
    ax.legend(fontsize=6, loc="upper center", bbox_to_anchor=(0.3, -0.2), ncol=3, handletextpad=0.1, columnspacing=0.5)
    ax.grid(axis="x")
    title(ax, "Six ways to deploy")
    letter(ax, "c", dx=-0.5)
    src("r3c_deployment_modes", mt.reset_index())

    # (d) strict wins at five levels
    ax = fig.add_subplot(outer[3])
    st_ = data.stress_all()
    st_ = st_[st_.variant == "composite"]
    piv = st_.pivot_table(index=["family", "level", "draw", "name"], columns="registry", values="auprc")
    opps = ["nmifs", "mrmr", "l1_dropout", "mrmr_filter_known", "cmim"]
    levels = [0.1, 0.3, 0.5, 0.7, 0.9]
    win = np.zeros((len(opps), 2 * len(levels)))
    tie = np.zeros_like(win)
    for i, o in enumerate(opps):
        for fi, fam in enumerate(("feature", "group")):
            for li, lv in enumerate(levels):
                d = piv.loc[(fam, lv)]["srs_ap_t"] - piv.loc[(fam, lv)][o]
                win[i, fi * len(levels) + li] = float((d > 1e-12).mean())
                tie[i, fi * len(levels) + li] = float((d.abs() <= 1e-12).mean())
    ax.imshow(win, cmap=S.SEQ, vmin=0.2, vmax=1.0, aspect="auto")
    for i in range(win.shape[0]):
        for j in range(win.shape[1]):
            ax.text(j, i, f"{100 * win[i, j]:.0f}", ha="center", va="center", fontsize=6, color="white" if win[i, j] > 0.7 else S.INK)
    ax.axvline(len(levels) - 0.5, color="white", lw=2.0)
    ax.set_xticks(range(2 * len(levels)), [f"{lv:.1f}".lstrip("0") for lv in levels] * 2, fontsize=6)
    ax.set_yticks(range(len(opps)), [LABEL[o] for o in opps], fontsize=6.4)
    tr = blended_transform_factory(ax.transData, ax.transAxes)
    ax.text(len(levels) / 2 - 0.5, -0.11, "features, $l$", transform=tr, ha="center", va="top", fontsize=6.4)
    ax.text(1.5 * len(levels) - 0.5, -0.11, "groups, $l$", transform=tr, ha="center", va="top", fontsize=6.4)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    title(ax, "Draws won strictly (%)")
    letter(ax, "d", dx=-0.4)
    cols = [f"{fam}_{lv}" for fam in ("feature", "group") for lv in levels]
    src("r3d_strict_wins", pd.concat([pd.DataFrame(win, index=[LABEL[o] for o in opps], columns=cols).assign(kind="strict win"),
                                      pd.DataFrame(tie, index=[LABEL[o] for o in opps], columns=cols).assign(kind="tie")]).reset_index())
    S.save(fig, "fig_r3_schema_shift")


# ================================================================================================= R4
def fig_r4() -> None:
    """Contract. Claim (RQ4): selection is linear in |C| and costs seconds to hours offline; fast screening does the work
    of exact greedy 6-9x faster; filters are three to four orders of magnitude cheaper but less accurate; registries run
    in DuckDB in seconds, faster than the Python reference engine, and beyond memory with a disk-backed database.
    Evidence: (a) log-log runtime vs |C|, (b) 100% stacked phases, (c) cost-quality scatter with frontier, (d) dot-range
    plot of SQL and Python times (median, IQR of repeats). Data: selection_scalability, extra_metrics rows_all,
    sql_execution/timing_repeats_*.json. Time: lower = better."""
    sc = pd.read_csv(ART / "selection_scalability/selection_scalability.csv")
    sz = sc[(sc.axis == "size") & (sc.status == "ok")]
    fig = plt.figure(figsize=(S.FULL_W, 2.5))
    gs = GridSpec(1, 4, width_ratios=[1.2, 0.95, 1.1, 1.1], wspace=0.55, left=0.06, right=0.975, top=0.86, bottom=0.27)

    # (a) runtime
    ax = fig.add_subplot(gs[0])
    lines = [("srs_ap_v", "SRS+T", S.TEAL, "-", "o", 1.5), ("srs_c_v", "SRS+T, lik.", S.TEAL_DARK, (0, (2, 1.2)), "o", 1.0),
             ("fast_q0", "screening, $q{=}0$", "#7CC7B6", "-", "o", 1.0), ("srs_c_exact", "exact, $q{=}0$", "#7CC7B6", (0, (3, 1.5)), "s", 1.0),
             ("fs_cv", "Wrapper, lik.", S.GREY_LIGHT, "-", "D", 1.0), ("dpp", "DPP", S.SAND, "-", "x", 1.0),
             ("l1_logistic", "L1", S.GREY, "-", ">", 1.0), ("cmim", "CMIM, JMI", HYB, "-", "v", 1.0), ("nmifs", "NMIFS, mRMR", S.BLUE, "-", "s", 1.0)]
    rows = []
    for m, text, color, ls, marker, lw in lines:
        if m == "fast_q0":
            s = sz[sz.method == "srs_c_v"].sort_values("n_candidates").copy()
            s = s[s.n_candidates <= 5000]
            s["seconds"] = s.phase_chains + s.phase_select_q0
            s["method"] = "fast_q0"
        else:
            s = sz[sz.method == m].sort_values("n_candidates")
        ax.plot(s.n_candidates, s.seconds, color=color, ls=ls, marker=marker, ms=2.3, lw=lw, label=text, mfc=color)
        rows.append(s[["method", "n_candidates", "seconds", "peak_mb"]])
    xs = np.array([500, 1e5])
    ax.plot(xs, 30 * xs / 500 * 2.2, color=S.INK2, lw=0.5, ls=(0, (2, 2)))
    ax.text(2.4e4, 30 * 2.4e4 / 500 * 2.2 * 1.6, r"$\propto |C|$", fontsize=6.3, color=S.INK2, rotation=24, ha="center")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("candidates $|C|$")
    ax.set_ylabel("selection time (s)")
    ax.legend(fontsize=5.9, loc="upper center", bbox_to_anchor=(0.55, -0.22), handlelength=1.5, ncol=3, columnspacing=0.4, handletextpad=0.25,
              labelspacing=0.2)
    ax.grid(which="major")
    title(ax, "Selection time")
    letter(ax, "a", dx=-0.24)
    src("r4a_runtime", pd.concat(rows))

    # (b) phases
    ax = fig.add_subplot(gs[1])
    v = sz.set_index(["method", "n_candidates"])
    bars = [("srs_ap_v", 500, "500"), ("srs_ap_v", 2274, "2,274"), ("srs_ap_v", 10000, "$10^4$"), ("srs_c_v", 100000, "$10^5\\dagger$")]
    phases = [("phase_chains", "surrogate chains", S.SAND), ("phase_select_q0", "selection, $q=0$", S.TEAL_LIGHT),
              ("phase_select_q25", "selection, $q=0.25$", S.TEAL)]
    bottom = np.zeros(len(bars))
    out = []
    for col, text, color in phases:
        share = np.array([v.loc[(m, n), col] / v.loc[(m, n), "seconds"] for m, n, _ in bars])
        ax.bar(range(len(bars)), share, bottom=bottom, color=color, width=0.66, edgecolor="white", lw=0.4, label=text)
        bottom += share
    ticks = []
    for k, (m, n, lab) in enumerate(bars):
        t = v.loc[(m, n), "seconds"]
        total = f"{t:.0f} s" if t < 1000 else (f"{t / 60:.0f} min" if t < 3600 else f"{t / 3600:.2f} h")
        ticks.append(f"{lab}\n{total}")
        out.append({"method": m, "n_candidates": n, **{c: v.loc[(m, n), c] for c, _, _ in phases}, "seconds": t})
    ax.set_xticks(range(len(bars)), ticks, fontsize=5.9)
    ax.set_ylim(0, 1.0)
    ax.set_yticks([0, 0.5, 1.0], ["0", "50%", "100%"])
    ax.set_xlabel("candidates $|C|$")
    ax.legend(fontsize=6, loc="upper center", bbox_to_anchor=(0.42, -0.3), ncol=1, handlelength=0.8, labelspacing=0.2)
    title(ax, "Where the time goes")
    letter(ax, "b", dx=-0.32)
    src("r4b_phases", pd.DataFrame(out))

    # (c) cost-quality
    ax = fig.add_subplot(gs[2])
    ra = rows_all()
    quality = ra.groupby("registry").mean_target_auprc.mean()
    pts = [("srs_ap_v", "srs_ap_t", "SRS+T", S.TEAL), ("srs_c_v", "srs_c_val", "SRS+T, lik.", S.TEAL_DARK),
           ("fs_cv", "fs_cv", "Wrapper, lik.", S.GREY_LIGHT), ("dpp", "dpp_ig", "DPP", S.SAND), ("mmrfs", "mmrfs", "MMRFS", "#C9B48F"),
           ("l1_logistic", "l1_logistic", "L1", S.GREY), ("cmim", "cmim", "CMIM", HYB), ("jmi", "jmi", "JMI", HYB),
           ("nmifs", "nmifs", "NMIFS", S.BLUE), ("mrmr", "mrmr", "mRMR", S.SKY)]
    t500 = sz[sz.n_candidates == 500].set_index("method").seconds
    frame = pd.DataFrame([{"method": text, "seconds": float(t500[tm]), "auprc": float(quality[qm]), "color": color} for tm, qm, text, color in pts])
    frame = frame.sort_values("seconds").reset_index(drop=True)
    best, front = -1.0, []
    for _, row in frame.iterrows():
        if row.auprc > best:
            front.append(row)
            best = row.auprc
    fx, fy = [r.seconds for r in front], [r.auprc for r in front]
    ax.step(fx + [200], fy + [fy[-1]], where="post", color=S.TEAL_LIGHT, lw=1.0, zorder=1)
    for _, row in frame.iterrows():
        ax.scatter(row.seconds, row.auprc, s=18, color=row.color, zorder=3, edgecolor="white", lw=0.4)
    offsets = {"SRS+T": (-3, 1, "right", "bottom"), "SRS+T, lik.": (-3, 3, "right", "bottom"), "Wrapper, lik.": (0, -4, "center", "top"),
               "DPP": (3, 0, "left", "center"), "MMRFS": (0, 3, "center", "bottom"), "L1": (4, 0, "left", "center"),
               "CMIM": (3, 0, "left", "center"), "JMI": (3, -1, "left", "center"), "NMIFS": (3, 1, "left", "bottom"), "mRMR": (3, -1, "left", "top")}
    for _, row in frame.iterrows():
        dx, dy, ha, va = offsets[row.method]
        ax.annotate(row.method, (row.seconds, row.auprc), xytext=(dx, dy), textcoords="offset points", ha=ha, va=va, fontsize=6, color=S.INK2)
    ax.set_xscale("log")
    ax.set_xlim(3e-3, 150)
    ax.set_ylim(0.62, 0.716)
    ax.set_xlabel("time at $|C| = 500$ (s)")
    ax.set_ylabel("mean target AUPRC (16 runs)")
    ax.grid(which="major")
    title(ax, "Cost and quality")
    letter(ax, "c", dx=-0.32)
    src("r4c_cost_quality", frame.drop(columns="color"))

    # (d) SQL and Python engine timings
    ax = fig.add_subplot(gs[3])
    path = ART / "sql_execution/timing_repeats_f_mimic_iv_aki_gap6h.json"
    rep = pd.DataFrame(json.loads(path.read_text(encoding="utf-8"))["rows"]) if path.exists() else None
    out = []
    yl = []
    y = 0
    engines = [("duckdb_memory", "DuckDB, in memory", S.TEAL, "o", True), ("duckdb_disk", "DuckDB, disk, 8 GB", S.TEAL, "o", False),
               ("python", "Python engine", S.SAND, "D", True)]
    for r_ in (1, 10, 30):
        for col, ptext in (("seconds_tokens", "tokens"), ("seconds_registry", "registry")):
            for off, (eng, etext, color, marker, filled) in zip((-0.22, 0.0, 0.22), engines):
                if rep is None:
                    continue
                s = rep[(rep.engine == eng) & (rep.replicate == r_)][col]
                if s.empty:
                    continue
                med, q1, q3 = s.median(), s.quantile(0.25), s.quantile(0.75)
                ax.hlines(y + off, q1, q3, color=color, lw=1.0)
                ax.scatter(med, y + off, s=14, marker=marker, facecolor=color if filled else "white", edgecolor=color, lw=0.9, zorder=3)
                out.append({"replicate": r_, "phase": ptext, "engine": etext, "median_s": med, "q1": q1, "q3": q3, "n": len(s)})
            yl.append((y, f"×{r_} {ptext}"))
            y += 1
        if rep is not None:
            pk = rep[rep.replicate == r_].groupby("engine").peak_rss_gb.median()
            txt = " / ".join(f"{pk[e]:.1f}" for e in ("duckdb_memory", "duckdb_disk") if e in pk)
            ax.text(1.02, (y - 1.5 + 0.5) / 1, "", transform=ax.transData)
            ax.annotate(f"{txt} GB", xy=(1.0, y - 1.5), xycoords=("axes fraction", "data"), xytext=(2, 0), textcoords="offset points",
                        fontsize=5.9, color=S.INK2, va="center", annotation_clip=False)
        y += 0.5
    ax.set_yticks([t[0] for t in yl], [t[1] for t in yl], fontsize=6.2)
    ax.set_ylim(y - 0.3, -0.7)
    ax.set_xscale("log")
    ax.set_xlabel("seconds (eICU AKI ×1 to ×30)")
    ax.legend(handles=[Line2D([], [], marker=m, ls="", mfc=c if f else "white", mec=c, label=t) for _, t, c, m, f in engines],
              fontsize=6, loc="upper center", bbox_to_anchor=(0.35, -0.22), ncol=2, handletextpad=0.1, columnspacing=0.5)
    ax.grid(axis="x", which="major")
    title(ax, "Registries as SQL")
    letter(ax, "d", dx=-0.42)
    src("r4d_sql_timing", pd.DataFrame(out))
    S.save(fig, "fig_r4_efficiency")


# ================================================================================================= R5
def fig_r5() -> None:
    """Contract. Claim (RQ5): conclusions survive new patient splits, a tree decoder, demographic subgroups and changes of
    every hyper-parameter (on TEST and on VAL), and greedy reaches at least 93% of the exact optimum of its objective; the
    recurring exception is NMIFS on vasopressor 1 h. Evidence: (a) strip plot over five re-splits, (b) forest of decoder
    differences, (c) categorical heatmap of subgroup verdicts, (d) range bars of hyper-parameter changes with VAL marks,
    (e) ECDF of greedy/optimum. Data: union_resplit, union_robustness_ap, srs_sensitivity_ap, exact_small_pools."""
    rng = np.random.default_rng(2)
    fig = plt.figure(figsize=(S.FULL_W, 2.55))
    gs = GridSpec(1, 5, width_ratios=[1.0, 0.95, 1.1, 1.05, 0.85], wspace=0.62, left=0.055, right=0.985, top=0.86, bottom=0.26)

    ax = fig.add_subplot(gs[0])
    rr, _ = data.resplits()
    meths = ["srs_ap_t", "nmifs", "mrmr", "l1_dropout", "mrmr_filter_known"]
    rows = []
    for ti, task in enumerate(("Vaso 1h", "AKI")):
        for mi, m in enumerate(meths):
            v = rr[(rr.task == task) & (rr.registry == m)].mean_target_auprc.values
            x = ti * (len(meths) + 1) + mi
            ax.scatter(np.full(len(v), x) + rng.uniform(-0.18, 0.18, len(v)), v, s=8, color=S.METHOD_COLOR[m], lw=0, alpha=0.9)
            ax.hlines(v.mean(), x - 0.32, x + 0.32, color=S.INK, lw=0.9)
            rows.append({"task": task, "method": LABEL[m], **{f"split{i + 1}": vv for i, vv in enumerate(v)}})
    ax.set_xticks([2, 8], ["Vaso 1h", "AKI"])
    ax.set_ylabel("mean target AUPRC")
    ax.legend(handles=[Line2D([], [], marker="o", ls="", color=S.METHOD_COLOR[m], label=LABEL[m]) for m in meths], fontsize=6,
              loc="upper center", bbox_to_anchor=(0.45, -0.12), ncol=2, handletextpad=0.05, columnspacing=0.4, labelspacing=0.15)
    ax.grid(axis="y")
    title(ax, "New patient splits")
    letter(ax, "a", dx=-0.3)
    src("r5a_resplits", pd.DataFrame(rows))

    ax = fig.add_subplot(gs[1])
    rb = data.robustness()
    y, ticks, rows = 0, [], []
    for label, task in zip(data.STANDARD, ("V1h", "V2h", "V4h", "AKI")):
        dp = rb[label]["decoder_paired"]
        for b, color, marker in (("mrmr", S.SKY, "^"), ("nmifs", S.BLUE, "s")):
            s_ = dp[(dp.a == "srs_ap_t") & (dp.b == b) & (dp.decoder == "gbdt") & (dp.target == "mean_over_targets")]
            ax.hlines(y, s_.low, s_.high, color=color, lw=1.0)
            ax.scatter(s_.point_diff, y, color=color, marker=marker, s=12, zorder=3)
            rows.append({"setting": task, "b": b, "diff": float(s_.point_diff.iloc[0]), "low": float(s_.low.iloc[0]), "high": float(s_.high.iloc[0])})
            y += 1
        ticks.append((y - 1.5, task))
        y += 0.6
    ax.set_yticks([t[0] for t in ticks], [t[1] for t in ticks], fontsize=6.4)
    ax.set_ylim(y - 0.4, -0.9)
    ax.axvline(0, color=S.INK2, lw=0.6, ls=(0, (2, 2)))
    ax.set_xlabel("SRS+T $-$ comp., trees")
    ax.legend(handles=[Line2D([], [], marker="^", ls="", color=S.SKY, label="vs mRMR"), Line2D([], [], marker="s", ls="", color=S.BLUE, label="vs NMIFS")],
              fontsize=6, loc="upper center", bbox_to_anchor=(0.45, -0.22), ncol=1, handletextpad=0.05, labelspacing=0.2)
    title(ax, "Tree decoder")
    letter(ax, "b", dx=-0.3)
    src("r5b_decoder", pd.DataFrame(rows))

    ax = fig.add_subplot(gs[2])
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
    ax.set_yticks(range(len(sub_keys)), [f"{cohort_name.get(c, c)} {g}" for c, g in sub_keys], fontsize=6)
    ax.set_xticks(range(8), ["1", "2", "4", "A"] * 2, fontsize=6)
    nb = (mat[:, :4] == 1).sum(), (mat[:, :4] == -1).sum(), (mat[:, 4:] == 1).sum(), (mat[:, 4:] == -1).sum()
    tr = blended_transform_factory(ax.transData, ax.transAxes)
    ax.text(1.5, -0.1, f"vs mRMR\n{nb[0]} b, {nb[1]} w", transform=tr, ha="center", va="top", fontsize=6.1, color=S.INK2)
    ax.text(5.5, -0.1, f"vs NMIFS\n{nb[2]} b, {nb[3]} w", transform=tr, ha="center", va="top", fontsize=6.1, color=S.INK2)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    title(ax, "Subgroups")
    letter(ax, "c", dx=-0.5)
    src("r5c_subgroups", pd.DataFrame(mat, index=[f"{c} {g}" for c, g in sub_keys],
                                      columns=[f"{b}_{l}" for b in ("mrmr", "nmifs") for l in data.STANDARD]).reset_index())

    ax = fig.add_subplot(gs[3])
    sens = data.sensitivity()
    default = sens[sens.parameter == "default"].set_index("label")
    names = {"q": "q", "max_surrogates": "M", "n_folds": "V", "C": "1/η", "n_dropout": "D", "seed": "seed"}
    rows = []
    keep = sens[(sens.parameter != "default") & ~((sens.parameter == "q") & (sens.value.astype(str) == "0.25"))]
    for (param, value), g in keep.groupby(["parameter", "value"], sort=False):
        g = g.set_index("label")
        d = (g.mean_target_auprc - default.mean_target_auprc).values
        dv = (g.validation_score - default.validation_score).values
        rows.append({"variant": f"{names.get(param, param)} = {float(value):g}", "min": d.min(), "max": d.max(), "mean": d.mean(), "val_mean": dv.mean()})
    tor = pd.DataFrame(rows).sort_values("min").reset_index(drop=True)
    for k, row in tor.iterrows():
        ax.hlines(k, row["min"], row["max"], color=S.TEAL_LIGHT, lw=3.2)
        ax.scatter(row["mean"], k, s=8, color=S.TEAL_DARK, zorder=3)
        ax.scatter(row["val_mean"], k, s=10, marker="|", color=S.WARM, zorder=4, lw=1.1)
    ax.axvline(0, color=S.INK2, lw=0.6)
    ax.set_yticks(range(len(tor)), tor.variant, fontsize=5.9)
    ax.set_ylim(len(tor) - 0.4, -0.6)
    ax.set_xlabel("Δ AUPRC vs default")
    ax.legend(handles=[Line2D([], [], color=S.TEAL_LIGHT, lw=3, label="targets: range"), Line2D([], [], marker="o", ls="", color=S.TEAL_DARK, ms=3, label="targets: mean"),
                       Line2D([], [], marker="|", ls="", color=S.WARM, ms=5, label="VAL: mean")],
              fontsize=6, loc="upper center", bbox_to_anchor=(0.35, -0.2), ncol=1, handletextpad=0.3, labelspacing=0.15)
    title(ax, "Hyper-parameters")
    letter(ax, "d", dx=-0.36)
    src("r5d_sensitivity", tor)

    ax = fig.add_subplot(gs[4])
    ex = pd.concat([pd.read_csv(ART / f"exact_small_pools/rows_{t}.csv") for t in ("gap1h", "aki_gap6h")])
    rows = []
    for obj, color, ls, text in (("deployed_q0", S.TEAL, "-", "$q = 0$"), ("deployed_q25", S.BLUE, (0, (3, 1.5)), "$q = 0.25$"),
                                 ("analyzable", S.SAND, (0, (1, 1)), "analyzable")):
        v = np.sort(ex[ex.objective == obj].ratio.values)
        ax.step(v, np.arange(1, len(v) + 1) / len(v), where="post", color=color, ls=ls, lw=1.2, label=text)
        rows.append(pd.DataFrame({"objective": obj, "ratio": v}))
    ax.set_xlim(0.93, 1.004)
    ax.set_xticks([0.94, 0.97, 1.0])
    ax.set_xlabel("greedy / optimum")
    ax.set_ylabel("share of 80 sub-pools")
    ax.legend(fontsize=6, loc="upper left", handlelength=1.5, borderpad=0.1)
    ax.grid()
    title(ax, "Near-optimal")
    letter(ax, "e", dx=-0.38)
    src("r5e_exact_ratio", pd.concat(rows))
    S.save(fig, "fig_r5_robustness")


# ================================================================================================= R6
def fig_r6() -> None:
    """Contract. Claim (RQ6): in a pre-specified second domain the pre-registered test replicated only partially; with the
    AUPRC decision SRS+T leads at K >= 10, ties L1 with schema dropout and loses to NMIFS+AP at K = 5; surrogates decide this
    domain. Evidence: (a) heatmap of paired differences per budget, (b) stacked bars of how the K = 10 programs run, (c) target
    AUPRC against K incl. the pre-specified likelihood variant, (d) forest of the pre-registered tests. Data: union_ap/drive_k*,
    nmifs_ap/drive_k*, references/drive_k*, paper_v3_cache/drive_executability.csv, drive/union_k10."""
    rows, paired, abl = data.drives()
    hyb_rows = pd.concat([pd.read_csv(f).assign(k=int(f.split("drive_k")[1].split("\\")[0].split("/")[0]))
                          for f in glob.glob(str(ART / "nmifs_ap/drive_k*/*/rows.csv"))], ignore_index=True)
    hyb_pair = pd.concat([pd.read_csv(f).assign(k=int(f.split("drive_k")[1].split("\\")[0].split("/")[0]))
                          for f in glob.glob(str(ART / "nmifs_ap/drive_k*/*/paired.csv"))], ignore_index=True)
    ref_pair = pd.concat([pd.read_csv(f).assign(k=int(f.split("drive_k")[1].split("\\")[0].split("/")[0]))
                          for f in glob.glob(str(ART / "references/drive_k*/*/paired.csv"))], ignore_index=True)
    rows = pd.concat([rows, hyb_rows[hyb_rows.registry == "nmifs_ap_t"]], ignore_index=True)
    hp = hyb_pair[(hyb_pair.a == "srs_ap_t") & (hyb_pair.b == "nmifs_ap_t")].copy()
    rp = ref_pair[(ref_pair.a == "srs_ap_t") & (ref_pair.metric.isin(["internal", "mean_target"]))].copy()
    paired = pd.concat([paired, verdicts(hp), verdicts(rp)], ignore_index=True)
    fig = plt.figure(figsize=(S.FULL_W, 2.55))
    gs = GridSpec(1, 4, width_ratios=[1.3, 1.05, 0.95, 1.0], wspace=0.66, left=0.085, right=0.985, top=0.86, bottom=0.26)

    ax = fig.add_subplot(gs[0])
    base = ["nmifs_ap_t", "nmifs", "filter_nmifs", "mrmr", "mmrfs", "cmim", "jmi", "l1_logistic", "l1_dropout", "mrmr_filter_known"]
    mat = np.zeros((len(base), 6))
    ver = np.zeros_like(mat)
    for i, b in enumerate(base):
        for ki, k in enumerate((5, 10, 20)):
            for mi, metric in enumerate(("internal", "mean_target")):
                s = paired[(paired.a == "srs_ap_t") & (paired.b == b) & (paired.k == k) & (paired.metric == metric)]
                mat[i, ki * 2 + mi], ver[i, ki * 2 + mi] = s.point_diff.iloc[0], s.verdict.iloc[0]
    lim = 0.12
    ax.imshow(np.clip(mat, -lim, lim), cmap=S.DIV, vmin=-lim, vmax=lim, aspect="auto")
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            mark = "*" if ver[i, j] == 1 else ("°" if ver[i, j] == -1 else "")
            v100 = 100 * mat[i, j]
            txt = f"{v100:+.0f}" if abs(v100) >= 9.95 else f"{v100:+.1f}"
            ax.text(j, i, txt + mark, ha="center", va="center", fontsize=5.9, color="white" if abs(mat[i, j]) > 0.07 else S.INK)
    for x in (1.5, 3.5):
        ax.axvline(x, color="white", lw=2.0)
    ax.set_xticks(range(6), ["int.", "tgt."] * 3, fontsize=6.2)
    for ki, k in enumerate((5, 10, 20)):
        ax.text(ki * 2 + 0.5, len(base) + 0.6, f"K = {k}" + ("\n(pre-spec.)" if k == 10 else ""), ha="center", va="top", fontsize=6.2, color=S.INK2)
    ax.set_yticks(range(len(base)), [LABEL[b] for b in base], fontsize=6.4)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    title(ax, "SRS+T $-$ comparator (×100)")
    letter(ax, "a", dx=-0.4)
    src("r6a_drive_heatmap", pd.DataFrame(mat, index=[LABEL[b] for b in base],
                                          columns=[f"K{k}_{m}" for k in (5, 10, 20) for m in ("internal", "target")]).reset_index())

    ax = fig.add_subplot(gs[1])
    ex = pd.read_csv(ART / "paper_v3_cache/drive_executability.csv")
    ex = ex[ex.k == 10]
    meths = ["srs_ap_t", "srs_ap", "nmifs_ap_t", "nmifs", "mrmr", "l1_dropout", "fs_cv"]
    tot = ex.groupby("registry")[["primary", "surrogate", "zeroed"]].sum().loc[meths]
    left = np.zeros(len(meths))
    for col, text, color in (("primary", "as selected", S.TEAL), ("surrogate", "via surrogate", S.TEAL_LIGHT), ("zeroed", "zeroed", ZERO)):
        ax.barh(range(len(meths)), tot[col].values, left=left, color=color, height=0.68, edgecolor="white", lw=0.5, label=text)
        for k_, v_ in enumerate(tot[col].values):
            if v_ >= 2:
                ax.text(left[k_] + v_ / 2, k_, str(int(v_)), ha="center", va="center", fontsize=6, color="white" if col == "primary" else S.INK)
        left += tot[col].values
    ax.set_yticks(range(len(meths)), [LABEL[m] for m in meths], fontsize=6.4)
    ax.set_ylim(len(meths) - 0.5, -0.6)
    ax.set_xlim(0, 30)
    ax.set_xticks([0, 10, 20, 30])
    ax.set_xlabel("program-target pairs ($K = 10$)")
    ax.legend(ncol=3, fontsize=6, loc="upper center", bbox_to_anchor=(0.32, -0.2), handlelength=0.8, columnspacing=0.5, handletextpad=0.3)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    title(ax, "How the programs run")
    letter(ax, "b", dx=-0.45)
    src("r6b_executability", ex)

    ax = fig.add_subplot(gs[2])
    ref = rows[rows.registry == "srs_ap_t"].set_index("k").mean_target_auprc.loc[[5, 10, 20]]
    out = []
    for m in ("srs_ap_t", "srs_c_val", "srs_ap", "nmifs_ap_t", "nmifs", "l1_dropout"):
        vals = rows[rows.registry == m].set_index("k").mean_target_auprc.loc[[5, 10, 20]].values
        sty = style_of(m) if m != "srs_c_val" else dict(color=S.TEAL_DARK, ls=(0, (4, 1.5)), marker="o", lw=1.0, mfc="white")
        ax.plot([5, 10, 20], vals, color=sty["color"], ls=sty["ls"], marker=sty["marker"], ms=3, mfc=sty["mfc"], mec=sty["color"], lw=sty["lw"],
                label=LABEL[m] + (" (pre-spec.)" if m == "srs_c_val" else ""))
        out.append({"variant": m, "k5": vals[0], "k10": vals[1], "k20": vals[2]})
    s = abl[(abl.a == "srs_ap_t") & (abl.b == "srs_ap_t_nosur") & (abl.target == "mean_over_targets")].set_index("k").point_diff.loc[[5, 10, 20]]
    vals = ref.values - s.values
    ax.plot([5, 10, 20], vals, color=S.WARM, ls=(0, (3, 1.5)), marker="o", ms=3, mfc="white", mec=S.WARM, lw=1.0, label="no surr. in obj.")
    out.append({"variant": "srs_ap_t_nosur", "k5": vals[0], "k10": vals[1], "k20": vals[2]})
    ax.set_xscale("log", base=2)
    ax.set_xticks([5, 10, 20], ["5", "10", "20"])
    ax.minorticks_off()
    ax.set_xlabel("budget $K$")
    ax.set_ylabel("mean target AUPRC")
    ax.legend(fontsize=5.9, loc="upper center", bbox_to_anchor=(0.4, -0.2), ncol=2, handlelength=1.6, columnspacing=0.4, labelspacing=0.15)
    ax.grid(axis="y")
    title(ax, "Budget")
    letter(ax, "c", dx=-0.42)
    src("r6c_variants", pd.DataFrame(out))

    ax = fig.add_subplot(gs[3])
    pre_dir = ART / "drive/union_k10/f_backblaze_seagate_gap2d"
    pi = pd.read_csv(pre_dir / "paired_internal.csv")
    pt = pd.read_csv(pre_dir / "paired_targets.csv")
    e = pd.read_csv(ART / "drive/union_extra_k10/f_backblaze_seagate_gap2d/paired_targets.csv")
    for df in (pi, pt, e):
        df["a"] = df["a"].str.replace(":composite", "")
        df["b"] = df["b"].str.replace(":composite", "")

    def ci(s):
        s = s.iloc[0]
        return float(s.point_diff), float(s.low), float(s.high)

    def post(a, b, metric):
        return ci(paired[(paired.a == a) & (paired.b == b) & (paired.k == 10) & (paired.metric == metric)])

    tests = [("H1 mRMR int.", ci(pi[(pi.a == "srs_c_val") & (pi.b == "mrmr")]), post("srs_ap_t", "mrmr", "internal")),
             ("H1 mRMR tgt.", ci(pt[(pt.a == "srs_c_val") & (pt.b == "mrmr") & (pt.target == "mean_over_targets")]), post("srs_ap_t", "mrmr", "mean_target")),
             ("H2 MMRFS int.", ci(pi[(pi.a == "srs_c_val") & (pi.b == "mmrfs")]), post("srs_ap_t", "mmrfs", "internal")),
             ("H2 MMRFS tgt.", ci(pt[(pt.a == "srs_c_val") & (pt.b == "mmrfs") & (pt.target == "mean_over_targets")]), post("srs_ap_t", "mmrfs", "mean_target")),
             ("H2 L1 int.", ci(pi[(pi.a == "srs_c_val") & (pi.b == "l1_logistic")]), post("srs_ap_t", "l1_logistic", "internal")),
             ("H2 L1 tgt.", ci(pt[(pt.a == "srs_c_val") & (pt.b == "l1_logistic") & (pt.target == "mean_over_targets")]), post("srs_ap_t", "l1_logistic", "mean_target")),
             ("H3 SRS tgt.", ci(e[(e.a == "srs_c_unknown_val") & (e.b == "mrmr") & (e.target == "mean_over_targets")]), post("srs_ap", "mrmr", "mean_target"))]
    xmax = 0.065
    out = []
    for i, (name, a_pre, a_post) in enumerate(tests):
        for off, (vals, filled) in zip((-0.17, 0.17), ((a_pre, False), (a_post, True))):
            color = S.TEAL if filled else S.GREY
            lo, hi, pt_ = vals[1], min(vals[2], xmax), vals[0]
            if pt_ > xmax:
                ax.annotate(f"{pt_:+.2f}", xy=(xmax, i + off), xytext=(-2, 0), textcoords="offset points", ha="right", va="center", fontsize=5.8, color=color)
                ax.scatter(xmax, i + off, marker=">", s=14, color=color, zorder=3)
                continue
            ax.hlines(i + off, lo, hi, color=color, lw=1.0)
            ax.scatter(pt_, i + off, s=14, marker="o", facecolor=color if filled else "white", edgecolor=color, lw=0.9, zorder=3)
        out.append({"test": name, "pre_diff": a_pre[0], "pre_low": a_pre[1], "pre_high": a_pre[2], "post_diff": a_post[0], "post_low": a_post[1],
                    "post_high": a_post[2]})
    ax.axvline(0, color=S.INK2, lw=0.6, ls=(0, (2, 2)))
    ax.set_xlim(-0.02, xmax)
    ax.set_yticks(range(len(tests)), [t[0] for t in tests], fontsize=6.2)
    ax.set_ylim(len(tests) - 0.4, -0.6)
    ax.set_xlabel("AUPRC difference")
    ax.legend(handles=[Line2D([], [], marker="o", ls="-", lw=0.9, mfc="white", mec=S.GREY, color=S.GREY, label="pre-specified (lik.)"),
                       Line2D([], [], marker="o", ls="-", lw=0.9, color=S.TEAL, label="post hoc (AUPRC)")],
              fontsize=6, loc="upper center", bbox_to_anchor=(0.25, -0.2), ncol=1, handletextpad=0.3, labelspacing=0.2, handlelength=1.4)
    ax.grid(axis="x")
    title(ax, "Pre-registered tests")
    letter(ax, "d", dx=-0.72)
    src("r6d_hypotheses", pd.DataFrame(out))
    S.save(fig, "fig_r6_drives")


if __name__ == "__main__":
    for name in sys.argv[1:] or ["r1", "r2", "r3", "r4", "r5", "r6"]:
        globals()[f"fig_{name}"]()
        print("built", name)
