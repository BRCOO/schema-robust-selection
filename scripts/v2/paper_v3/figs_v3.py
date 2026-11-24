"""Experiment figures of the PVLDB v3 paper, second design (after the review of 2026-11-05): seven experiments, one
multi-panel figure each, every panel a different chart type and a different claim than the tables.

Workflow of the cs-experiment-figure-studio skill: a figure contract in each docstring, source data of every panel in
paper/v3_assets/source_data/, PDF/SVG/PNG exports, Nature teal-green palette (style.py).

  PYTHONDONTWRITEBYTECODE=1 python scripts/v2/paper_v3/figs_v3.py [x1 ... x7]
"""
from __future__ import annotations

import glob
import json
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D

import data
import style as S
from figs_b import GEN_COLOR, GEN_ORDER, PROXY, TASKS, method_style, src
from style import LABEL

S.setup()
RNG = np.random.default_rng(3)
LABEL = dict(LABEL)
LABEL.update({"fs_cv": "Wrapper, lik.", "srs_ap_unaware": "Wrapper, AUPRC", "nmifs_ap_t": "NMIFS+AP"})
S.METHOD_COLOR.update({"nmifs_ap_t": "#7E8FBF", "srs_ap_unaware": "#8FD1C3", "fs_cv": S.GREY_LIGHT})
RUN_ORDER = [data.run_name(r, l) for r, l in data.ICU_RUNS]
SHORT = {"Vaso 1h": "1", "Vaso 2h": "2", "Vaso 4h": "4", "AKI": "A"}
ART = S.ART


def name_of(run: str) -> str:
    root, label = run.split("/")
    return data.run_name(root, label)


def rows_all() -> pd.DataFrame:
    d = pd.concat([pd.read_csv(f) for f in glob.glob(str(ART / "extra_metrics/*/*/rows_all.csv"))], ignore_index=True)
    d["name"] = d.run.map(name_of)
    return d


def verdict(df: pd.DataFrame) -> np.ndarray:
    return np.where(df["low"] > 0, 1, np.where(df["high"] < 0, -1, 0))


def all_paired() -> pd.DataFrame:
    """SRS+T against every comparator of Table 2, per run, internal and mean target."""
    p = data.paired_all()
    p = p[p.a == "srs_ap_t"][["name", "b", "metric", "point_diff", "low", "high"]]
    a = data.ablation_all()
    a = a[(a.a == "srs_ap_t") & (a.b == "srs_ap_unaware") & (a.target.isin(["internal", "mean_over_targets"]))].copy()
    a["metric"] = a.target.map({"internal": "internal", "mean_over_targets": "mean_target"})
    h = pd.concat([pd.read_csv(f) for f in glob.glob(str(ART / "nmifs_ap/union*/*/paired.csv"))], ignore_index=True)
    h = h[(h.a == "srs_ap_t") & (h.b == "nmifs_ap_t")].copy()
    h["name"] = h.run.map(name_of)
    out = pd.concat([p, a[["name", "b", "metric", "point_diff", "low", "high"]], h[["name", "b", "metric", "point_diff", "low", "high"]]],
                    ignore_index=True)
    out["verdict"] = verdict(out)
    return out


def heat_cells(ax, mat, lim, cmap, fmt=None, fontsize=5.4, thresh=None):
    im = ax.imshow(np.clip(mat, -lim, lim) if lim else mat, cmap=cmap, vmin=-lim if lim else None, vmax=lim if lim else None, aspect="auto")
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    return im


# ================================================================================================= X1
def fig_x1() -> None:
    """Contract. Claim (RQ1): SRS+T leads the classic selectors in every run, while its lead over the strongest ones
    (NMIFS, NMIFS+AP, the AUPRC wrapper) is small, run-dependent and of the size of CV-seed variance. Evidence:
    (a) per-run target differences against 15 comparators with significance, (b) internal vs target differences against
    NMIFS and NMIFS+AP, (c) five CV seeds against NMIFS and NMIFS+AP at K = 10, (d) rank frequencies among 17 selectors.
    Data: union_ap paired/ablation, nmifs_ap paired, extra_metrics rows_all, srs_sensitivity_ap. Higher = better."""
    P = all_paired()
    order_b = ["nmifs_ap_t", "srs_ap_unaware", "nmifs", "fs_cv", "l1_dropout", "mrmr_filter_known", "mrmr", "cmim", "dpp_ig",
               "mrmr_stratified", "mmrfs", "mms_xin_ig", "mmr_ig", "jmi", "l1_logistic"]
    fig = plt.figure(figsize=(S.FULL_W, 2.45))
    gs = GridSpec(1, 4, width_ratios=[1.7, 0.95, 0.95, 1.0], wspace=0.78, left=0.09, right=0.985, top=0.8, bottom=0.21)

    ax = fig.add_subplot(gs[0])
    mat = np.full((len(order_b), len(RUN_ORDER)), np.nan)
    ver = np.zeros_like(mat)
    tallies = []
    for i, b in enumerate(order_b):
        s = P[(P.b == b) & (P.metric == "mean_target")].set_index("name")
        for j, n in enumerate(RUN_ORDER):
            if n in s.index:
                mat[i, j], ver[i, j] = s.loc[n, "point_diff"], s.loc[n, "verdict"]
        tallies.append(f"{(s.verdict == 1).sum()}/{(s.verdict == 0).sum()}/{(s.verdict == -1).sum()}")
    lim = 0.06
    im = heat_cells(ax, mat, lim, S.DIV)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            if ver[i, j] == 0:
                ax.scatter(j, i, s=7, facecolor="none", edgecolor=S.INK, lw=0.5)
            elif ver[i, j] == -1:
                ax.scatter(j, i, s=9, marker="x", color=S.INK, lw=0.8)
    for a0, a1, text in [(0, 4, "K5"), (4, 8, "K10"), (8, 12, "K20"), (12, 14, "L10"), (14, 16, "L20")]:
        ax.text((a0 + a1 - 1) / 2, -0.75, text, ha="center", va="bottom", fontsize=5.2, color=S.INK2)
        if a0:
            ax.axvline(a0 - 0.5, color="white", lw=1.6)
    ax.axhline(2.5, color="white", lw=1.6)
    ax.set_xticks(range(len(RUN_ORDER)), [SHORT[n.split(" L")[0].rsplit(" K", 1)[0]] for n in RUN_ORDER], fontsize=5.2)
    ax.set_yticks(range(len(order_b)), [LABEL[b] for b in order_b], fontsize=5.4)
    for i, t in enumerate(tallies):
        ax.text(len(RUN_ORDER) - 0.3, i, t, fontsize=5.1, va="center", ha="left", color=S.INK2)
    ax.text(len(RUN_ORDER) - 0.3, -0.75, "b/ns/w", fontsize=5.1, va="bottom", ha="left", color=S.INK2, fontweight="bold")
    cax = ax.inset_axes([0.0, -0.19, 0.35, 0.04])
    cb = fig.colorbar(im, cax=cax, orientation="horizontal", ticks=[-0.06, 0, 0.06])
    cb.ax.tick_params(labelsize=5.0, length=1.5, pad=1)
    cb.outline.set_visible(False)
    ax.text(0.4, -0.17, "○ n.s.,  × SRS+T worse", transform=ax.transAxes, fontsize=5.0, va="center", color=S.INK2)
    ax.set_title("Per-run difference on the targets", loc="left", fontsize=6.8, fontweight="bold", y=1.11)
    S.panel_label(ax, "a", dx=-0.16, dy=1.11)
    src("x1a_per_run", pd.DataFrame(mat, index=[LABEL[b] for b in order_b], columns=RUN_ORDER).reset_index())

    ax = fig.add_subplot(gs[1])
    rows = []
    for b, marker, color in (("nmifs", "o", S.BLUE), ("nmifs_ap_t", "s", "#7E8FBF")):
        ti = P[(P.b == b) & (P.metric == "internal")].set_index("name").point_diff.loc[RUN_ORDER]
        tt = P[(P.b == b) & (P.metric == "mean_target")].set_index("name").point_diff.loc[RUN_ORDER]
        ax.scatter(ti, tt, marker=marker, s=12, facecolor=color, edgecolor="white", lw=0.4, label=f"vs {LABEL[b]}", zorder=3)
        rows.append(pd.DataFrame({"run": RUN_ORDER, "comparator": LABEL[b], "internal": ti.values, "target": tt.values}))
    allv = pd.concat(rows)
    lo, hi = min(-0.015, allv[["internal", "target"]].min().min() - 0.004), allv[["internal", "target"]].max().max() + 0.006
    ax.add_patch(plt.Rectangle((0, 0), hi, hi, color=S.TEAL_PALE, zorder=0, lw=0))
    ax.axhline(0, color=S.INK2, lw=0.6)
    ax.axvline(0, color=S.INK2, lw=0.6)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    n1 = int(((rows[0].internal >= 0) & (rows[0].target >= 0)).sum())
    n2 = int(((rows[1].internal >= 0) & (rows[1].target >= 0)).sum())
    ax.text(0.97, 0.04, f"ahead on both\n{n1}/16 NMIFS\n{n2}/16 NMIFS+AP", transform=ax.transAxes, ha="right", va="bottom", fontsize=4.8, color=S.TEAL_DARK)
    ax.set_xlabel("internal difference")
    ax.set_ylabel("target difference")
    ax.legend(loc="upper center", bbox_to_anchor=(0.45, -0.22), ncol=2, fontsize=4.8, handletextpad=0.1, columnspacing=0.6)
    S.title(ax, "Strongest comparators")
    S.panel_label(ax, "b", dx=-0.32)
    src("x1b_quadrant", allv)

    ax = fig.add_subplot(gs[2])
    sens = data.sensitivity()
    ra = rows_all()
    rows = []
    for k, (label, task) in enumerate(TASKS):
        s = sens[(sens.label == label) & (sens.parameter.isin(["default", "seed"]))].mean_target_auprc.values
        nm = float(ra[(ra.run == f"union/{label}") & (ra.registry == "nmifs")].mean_target_auprc.iloc[0])
        hy = float(ra[(ra.run == f"union/{label}") & (ra.registry == "nmifs_ap_t")].mean_target_auprc.iloc[0])
        ax.hlines(k, s.min(), s.max(), color=S.TEAL_LIGHT, lw=3.0, zorder=1)
        ax.scatter(s, np.full(len(s), k) + RNG.uniform(-0.12, 0.12, len(s)), s=8, color=S.TEAL, zorder=3, lw=0)
        ax.scatter(nm, k, marker="D", s=14, facecolor="white", edgecolor=S.BLUE, lw=0.9, zorder=4)
        ax.scatter(hy, k, marker="s", s=12, facecolor="white", edgecolor="#7E8FBF", lw=0.9, zorder=4)
        rows.append({"setting": task, **{f"seed{i}": v for i, v in enumerate(s)}, "nmifs": nm, "nmifs_ap": hy})
    ax.set_yticks(range(4), [t for _, t in TASKS], fontsize=5.6)
    ax.set_ylim(3.6, -0.6)
    ax.set_xlabel("mean target AUPRC ($K = 10$)")
    ax.legend(handles=[Line2D([], [], marker="o", ls="", color=S.TEAL, label="SRS+T, 5 CV seeds"),
                       Line2D([], [], marker="D", ls="", mfc="white", mec=S.BLUE, label="NMIFS"),
                       Line2D([], [], marker="s", ls="", mfc="white", mec="#7E8FBF", label="NMIFS+AP")],
              fontsize=4.8, loc="upper center", bbox_to_anchor=(0.45, -0.22), ncol=2, handletextpad=0.1, columnspacing=0.5)
    ax.grid(axis="x")
    S.title(ax, "Selection variance")
    S.panel_label(ax, "c", dx=-0.32)
    src("x1c_seed_variance", pd.DataFrame(rows))

    ax = fig.add_subplot(gs[3])
    methods = ["srs_ap_t", "srs_ap"] + order_b
    piv = ra[ra.registry.isin(methods)].pivot_table(index="run", columns="registry", values="mean_target_auprc")
    ranks = piv.rank(axis=1, ascending=False, method="min")
    show = ["srs_ap_t", "srs_ap", "srs_ap_unaware", "nmifs_ap_t", "nmifs", "l1_dropout", "mrmr"]
    bins = [(1, 1, "1st"), (2, 2, "2nd"), (3, 3, "3rd"), (4, 5, "4-5th"), (6, 99, "6th+")]
    colors = [S.TEAL_DARK, S.TEAL, "#5DBAA6", S.TEAL_LIGHT, S.GREY_PALE]
    left = np.zeros(len(show))
    table = []
    for (a0, a1, text), color in zip(bins, colors):
        vals = np.array([((ranks[m] >= a0) & (ranks[m] <= a1)).sum() for m in show])
        ax.barh(range(len(show)), vals, left=left, color=color, height=0.66, edgecolor="white", lw=0.4, label=text)
        for k, v in enumerate(vals):
            if v >= 3:
                ax.text(left[k] + v / 2, k, str(v), ha="center", va="center", fontsize=5.0, color="white" if a0 <= 2 else S.INK)
        left += vals
        table.append(pd.Series(vals, index=[LABEL.get(m, m) for m in show], name=text))
    ax.set_yticks(range(len(show)), [LABEL.get(m, m) for m in show], fontsize=5.5)
    ax.set_ylim(len(show) - 0.5, -0.6)
    ax.set_xlim(0, 16)
    ax.set_xticks([0, 4, 8, 12, 16])
    ax.set_xlabel("runs (of 16)")
    ax.legend(ncol=5, loc="upper center", bbox_to_anchor=(0.35, -0.22), fontsize=4.8, handlelength=0.7, columnspacing=0.4, handletextpad=0.2)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    S.title(ax, "Rank among 17 selectors")
    S.panel_label(ax, "d", dx=-0.5)
    src("x1d_ranks", pd.concat(table, axis=1).reset_index().rename(columns={"index": "method"}))
    S.save(fig, "fig_x1_effectiveness")


# ================================================================================================= X2
def fig_x2() -> None:
    """Contract. Claim (RQ2): on heterogeneous union pools the estimated deployed value ranks candidates by their
    held-out value far better than relevance-redundancy scores; but its lead over NMIFS needs both the AUPRC decision and
    the screening by deployed value, which contribute separately. Evidence: (a) dumbbell of score fidelity within one
    generator vs on the union, (b) fidelity by selection step, (c) two component paths from the schema-unaware
    likelihood wrapper with reference levels, (d) per-run decomposition of SRS+T - NMIFS into decision and screening.
    Data: proxy_correlation/heldout_*.csv, extra_metrics rows_all."""
    rows = []
    for label, task in TASKS:
        d = pd.read_csv(ART / "proxy_correlation" / f"heldout_{label}.csv")
        d = d[(d.step > 0) & (d.truth == "val_auprc")].dropna(subset=["spearman"])
        d["task"] = task
        rows.append(d)
    pc = pd.concat(rows, ignore_index=True)
    pc["where"] = np.where(pc.scope == "union", "union", "within")
    agg = pc.groupby(["proxy", "task", "where"]).spearman.mean().unstack().reset_index()
    fig = plt.figure(figsize=(S.FULL_W, 2.3))
    gs = GridSpec(1, 4, width_ratios=[1.1, 0.9, 1.05, 1.05], wspace=0.55, left=0.075, right=0.965, top=0.84, bottom=0.22)

    ax = fig.add_subplot(gs[0])
    labels, pos = [], []
    y = 0
    for gi, proxy in enumerate(("objective", "nmifs", "mrmr")):
        name, color = PROXY[proxy]
        y += 1.3
        ax.text(0.305, y - 0.72, name, fontsize=5.6, color=color, fontweight="bold", va="bottom")
        for _, task in TASKS:
            row = agg[(agg.proxy == proxy) & (agg.task == task)].iloc[0]
            ax.plot([row["within"], row["union"]], [y, y], color=color, lw=1.0, alpha=0.6, zorder=1)
            ax.scatter(row["within"], y, s=12, facecolor="white", edgecolor=color, lw=0.9, zorder=2)
            ax.scatter(row["union"], y, s=14, color=color, zorder=3)
            labels.append(task)
            pos.append(y)
            y += 1
    ax.set_yticks(pos, labels, fontsize=5.3)
    ax.set_ylim(y, 0.0)
    ax.set_xlim(0.3, 0.82)
    ax.set_xlabel("Spearman ρ with held-out gain")
    ax.legend(handles=[Line2D([], [], marker="o", ls="", mfc="white", mec=S.INK2, label="within a generator"),
                       Line2D([], [], marker="o", ls="", color=S.INK2, label="union pool")],
              loc="lower right", fontsize=5.0, handletextpad=0.1, borderpad=0.1)
    ax.grid(axis="x")
    S.title(ax, "Score fidelity")
    S.panel_label(ax, "a", dx=-0.3)
    src("x2a_fidelity", agg)

    ax = fig.add_subplot(gs[1])
    u = pc[pc.scope == "union"].groupby(["proxy", "step", "task"]).spearman.mean().reset_index()
    out = []
    for proxy in ("objective", "nmifs", "mrmr"):
        name, color = PROXY[proxy]
        g = u[u.proxy == proxy].groupby("step").spearman
        mean, lo, hi = g.mean(), g.min(), g.max()
        ax.fill_between(mean.index, lo, hi, color=color, alpha=0.15, lw=0)
        ax.plot(mean.index, mean.values, color=color, marker="o", ms=2.8, lw=1.3)
        ax.text(3.08, mean.values[-1] + {"objective": 0.0, "nmifs": 0.02, "mrmr": -0.02}[proxy], name.split()[0], fontsize=5.1, color=color, va="center")
        out.append(pd.DataFrame({"proxy": proxy, "step": mean.index, "mean": mean.values, "min": lo.values, "max": hi.values}))
    ax.set_xticks([1, 2, 3])
    ax.set_xlim(0.85, 3.75)
    ax.set_xlabel("selection step")
    ax.set_ylabel("ρ on the union pool")
    ax.grid(axis="y")
    S.title(ax, "Across steps")
    S.panel_label(ax, "b", dx=-0.38)
    src("x2b_steps", pd.concat(out))

    ax = fig.add_subplot(gs[2])
    ra = rows_all()
    m = ra.groupby("registry").mean_target_auprc.mean()
    start = m["fs_cv"]
    paths = [("AUPRC decision first", [start, m["srs_ap_unaware"], m["srs_ap_t"]], S.TEAL, "o", None),
             ("scenarios first", [start, m["srs_c_val"], m["srs_ap_t"]], S.BLUE, "s", None)]
    for name, vals, color, marker, steps in paths:
        ax.plot([0, 1, 2], vals, color=color, marker=marker, ms=3.4, lw=1.2, label=name, zorder=3)
    for ref, text, color in (("nmifs", "NMIFS", S.BLUE), ("nmifs_ap_t", "NMIFS+AP", "#7E8FBF"), ("mrmr", "mRMR", S.SKY)):
        ax.axhline(m[ref], color=color, lw=0.7, ls=(0, (2, 2)), zorder=1)
        ax.text(-0.05, m[ref] + 0.0005, text, fontsize=4.8, color=color, va="bottom", ha="left")
    ax.set_xticks([0, 1, 2], ["wrapper\n(lik.)", "+ one\npart", "+ both\n(SRS+T)"], fontsize=5.2)
    ax.set_xlim(-0.1, 2.3)
    ax.set_ylabel("mean target AUPRC (16 runs)")
    ax.legend(fontsize=4.8, loc="lower right", bbox_to_anchor=(1.0, 0.1), handletextpad=0.3, borderpad=0.1, handlelength=1.6)
    ax.grid(axis="y")
    S.title(ax, "Component paths")
    S.panel_label(ax, "c", dx=-0.3)
    src("x2c_paths", pd.DataFrame({"registry": m.index, "mean_target_auprc": m.values}))

    ax = fig.add_subplot(gs[3])
    piv_t = ra.pivot_table(index="run", columns="registry", values="mean_target_auprc")
    piv_i = ra.pivot_table(index="run", columns="registry", values="internal_auprc")
    out = []
    for piv, filled, text in ((piv_t, True, "targets"), (piv_i, False, "internal")):
        x = piv.nmifs_ap_t - piv.nmifs
        yv = piv.srs_ap_t - piv.nmifs_ap_t
        ax.scatter(x, yv, s=12, marker="o", facecolor=S.TEAL if filled else "white", edgecolor=S.TEAL, lw=0.8, label=text, zorder=3)
        out.append(pd.DataFrame({"run": piv.index, "metric": text, "decision": x.values, "screening": yv.values}))
    allv = pd.concat(out)
    lim = max(allv[["decision", "screening"]].abs().max()) + 0.004
    ax.axhline(0, color=S.INK2, lw=0.6)
    ax.axvline(0, color=S.INK2, lw=0.6)
    ax.plot([-lim, lim], [lim, -lim], color=S.GREY_LIGHT, lw=0.6, ls=(0, (2, 2)))
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    p0, p1 = ax.transData.transform((-lim, lim)), ax.transData.transform((lim, -lim))
    angle = np.degrees(np.arctan2(p1[1] - p0[1], p1[0] - p0[0]))
    ax.text(0.74 * lim, -0.74 * lim + 0.05 * lim, "SRS+T = NMIFS", fontsize=4.6, color=S.GREY, ha="center", va="bottom",
            rotation=angle, rotation_mode="anchor")
    ax.set_xlabel("decision gain")
    ax.set_ylabel("screening gain")
    ax.legend(fontsize=4.9, loc="lower left", handletextpad=0.1, borderpad=0.2)
    S.title(ax, "Decision vs. screening")
    S.panel_label(ax, "d", dx=-0.32)
    src("x2d_decomposition", allv)
    S.save(fig, "fig_x2_mechanism")


if __name__ == "__main__":
    for name in sys.argv[1:] or ["x1", "x2"]:
        globals()[f"fig_{name}"]()
        print("built", name)
