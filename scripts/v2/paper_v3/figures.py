"""Experiment figures of the PVLDB v3 paper (one multi-panel figure per experiment; every panel a different chart type).

Workflow of the cs-experiment-figure-studio skill: figure contract (claim, evidence chain, data map, metric direction) in
the docstring of each function; source data of every panel written to paper/v3_assets/source_data/; PDF/SVG/PNG exports.

  PYTHONDONTWRITEBYTECODE=1 /d/BRCoo/miniconda/python scripts/v2/paper_v3/figures.py [e1 e2 ...]
"""
from __future__ import annotations

import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

import data
import style as S
from style import LABEL

S.setup()
RNG = np.random.default_rng(0)
RUN_ORDER = [data.run_name(r, l) for r, l in data.ICU_RUNS]
SHORT = {"Vaso 1h": "1", "Vaso 2h": "2", "Vaso 4h": "4", "AKI": "A"}


def src(name: str, frame: pd.DataFrame) -> None:
    frame.to_csv(S.OUT_SRC / f"{name}.csv", index=False)


def ci_mean(x: np.ndarray) -> tuple[float, float]:
    x = np.asarray(x, dtype=float)
    if len(x) < 2:
        return float("nan"), float("nan")
    half = 2.131 * x.std(ddof=1) / np.sqrt(len(x)) if len(x) == 16 else 1.96 * x.std(ddof=1) / np.sqrt(len(x))
    return x.mean() - half, x.mean() + half


# ================================================================================================= E1
def fig_e1() -> None:
    """Contract. Claim: SRS+T is better than every established selector at the source and at the targets in nearly all
    16 runs; NMIFS is the closest. Evidence: (a) forest of mean paired differences, (b) per-run x per-baseline heatmap with
    significance, (c) quadrant of internal vs target differences against the two strongest filters, (d) rank frequencies.
    Data: union_ap/<run>/paired.csv (a, b, metric, point_diff, low, high) and rows.csv (mean_target_auprc). Higher = better."""
    p = data.paired_all()
    r = data.rows_all()
    base = list(S.BASELINES) + ["best_single_srs"]
    ref = p[p.a == "srs_ap_t"]
    stats = []
    for b in base:
        t = ref[(ref.b == b) & (ref.metric == "mean_target")]
        i = ref[(ref.b == b) & (ref.metric == "internal")]
        lo, hi = ci_mean(t.point_diff.values)
        stats.append({"b": b, "t_mean": t.point_diff.mean(), "t_lo": lo, "t_hi": hi, "i_mean": i.point_diff.mean(),
                      "tally": f"{(t.verdict == 1).sum()}/{(t.verdict == 0).sum()}/{(t.verdict == -1).sum()}", "n": len(t)})
    st = pd.DataFrame(stats).sort_values("t_mean", ascending=False).reset_index(drop=True)
    order = list(st.b)
    src("e1a_forest", st)

    fig = plt.figure(figsize=(S.FULL_W, 2.45))
    gs = GridSpec(1, 4, width_ratios=[1.1, 1.5, 1.0, 1.0], wspace=0.66, left=0.085, right=0.99, top=0.8, bottom=0.2)

    # (a) forest ---------------------------------------------------------------------------------
    ax = fig.add_subplot(gs[0])
    y = np.arange(len(order))
    for k, b in enumerate(order):
        d = ref[(ref.b == b) & (ref.metric == "mean_target")].point_diff.values
        ax.scatter(d, np.full(len(d), k) + RNG.uniform(-0.18, 0.18, len(d)), s=3, color=S.TEAL_LIGHT, lw=0, zorder=1)
    ax.hlines(y, st.t_lo, st.t_hi, color=S.TEAL_DARK, lw=1.0, zorder=2)
    ax.scatter(st.t_mean, y, marker="D", s=14, color=S.TEAL, edgecolor="white", lw=0.4, zorder=3, label="targets")
    ax.scatter(st.i_mean, y, marker="o", s=12, facecolor="white", edgecolor=S.BLUE, lw=0.8, zorder=3, label="internal")
    ax.axvline(0, color=S.INK2, lw=0.6, ls=(0, (2, 2)))
    ax.set_yticks(y, [LABEL[b] + ("$^{\dagger}$" if b == "best_single_srs" else "") for b in order])
    ax.set_ylim(len(order) - 0.5, -0.5)
    ax.set_xlabel("SRS+T $-$ baseline (AUPRC)")
    ax.legend(loc="lower right", handletextpad=0.2, borderpad=0.1, fontsize=5.6)
    ax.grid(axis="x")
    S.title(ax, "Mean paired difference")
    S.panel_label(ax, "a", dx=-0.45)

    # (b) heatmap --------------------------------------------------------------------------------
    ax = fig.add_subplot(gs[1])
    mat = np.full((len(order), len(RUN_ORDER)), np.nan)
    ver = np.zeros_like(mat)
    for i, b in enumerate(order):
        for j, name in enumerate(RUN_ORDER):
            row = ref[(ref.b == b) & (ref.metric == "mean_target") & (ref.name == name)]
            if len(row):
                mat[i, j], ver[i, j] = row.point_diff.iloc[0], row.verdict.iloc[0]
    lim = 0.08
    im = ax.imshow(np.clip(mat, -lim, lim), cmap=S.DIV, vmin=-lim, vmax=lim, aspect="auto")
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            if np.isnan(mat[i, j]):
                ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, color=S.GREY_PALE, lw=0))
            elif ver[i, j] == 0:
                ax.scatter(j, i, s=8, facecolor="none", edgecolor=S.INK, lw=0.5)
            elif ver[i, j] == -1:
                ax.scatter(j, i, s=10, marker="x", color=S.WARM, lw=0.9)
    groups = [(0, 4, "K5"), (4, 8, "K10"), (8, 12, "K20"), (12, 14, "L10"), (14, 16, "L20")]
    for a0, a1, text in groups:
        ax.text((a0 + a1 - 1) / 2, -0.7, text, ha="center", va="bottom", fontsize=5.2, color=S.INK2)
        if a0:
            ax.axvline(a0 - 0.5, color="white", lw=1.6)
    ax.set_xticks(range(len(RUN_ORDER)), [SHORT[n.split(" L")[0].rsplit(" K", 1)[0]] for n in RUN_ORDER], fontsize=5.2)
    ax.set_yticks([])
    ax.set_ylim(len(order) - 0.5, -0.5)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    for k, row in st.iterrows():
        ax.text(len(RUN_ORDER) - 0.2, k, row.tally, fontsize=5.3, va="center", ha="left", color=S.INK2)
    ax.text(len(RUN_ORDER) - 0.2, -0.62, "b/ns/w", fontsize=5.2, va="bottom", ha="left", color=S.INK2, fontweight="bold")
    cax = ax.inset_axes([0.0, -0.3, 0.42, 0.045])
    cb = fig.colorbar(im, cax=cax, orientation="horizontal", ticks=[-0.08, 0, 0.08])
    cb.ax.tick_params(labelsize=5.0, length=1.5, pad=1)
    cb.outline.set_visible(False)
    ax.text(0.47, -0.27, "target diff.;  ○ n.s.,  × worse;  1/2/4 = vasopressor gap (h), A = AKI", transform=ax.transAxes, fontsize=5.0,
            va="center", ha="left", color=S.INK2)
    ax.set_title("Target difference per run", loc="left", fontsize=6.8, fontweight="bold", color=S.INK, y=1.13)
    S.panel_label(ax, "b", dx=-0.03, dy=1.1)
    src("e1b_heatmap", pd.DataFrame(mat, index=[LABEL[b] for b in order], columns=RUN_ORDER).reset_index())

    # (c) quadrant scatter -------------------------------------------------------------------------
    ax = fig.add_subplot(gs[2])
    quad = []
    for b, marker, color in (("nmifs", "o", S.BLUE), ("mrmr", "^", S.SKY)):
        ti = ref[(ref.b == b) & (ref.metric == "internal")].set_index("name").point_diff
        tt = ref[(ref.b == b) & (ref.metric == "mean_target")].set_index("name").point_diff
        xs, ys = ti.loc[RUN_ORDER].values, tt.loc[RUN_ORDER].values
        ax.scatter(xs, ys, marker=marker, s=13, facecolor=color, edgecolor="white", lw=0.4, label=f"vs {LABEL[b]}", zorder=3)
        quad.append(pd.DataFrame({"run": RUN_ORDER, "baseline": LABEL[b], "internal_diff": xs, "target_diff": ys}))
    lo = min(-0.01, min(q.internal_diff.min() for q in quad) - 0.005)
    hi = max(q[["internal_diff", "target_diff"]].max().max() for q in quad) + 0.005
    ax.axhspan(0, hi, xmin=0, xmax=1, color=S.TEAL_PALE, zorder=0)
    ax.axhline(0, color=S.INK2, lw=0.6)
    ax.axvline(0, color=S.INK2, lw=0.6)
    ax.set_xlim(lo, hi)
    ax.set_ylim(min(-0.01, min(q.target_diff.min() for q in quad) - 0.005), hi + 0.016)
    q = quad[0]
    n_both = int(((q.internal_diff >= 0) & (q.target_diff >= 0)).sum())
    ax.text(0.05, 0.97, f"{n_both}/16 runs ahead of\nNMIFS on both", transform=ax.transAxes, ha="left", va="top", fontsize=5.4, color=S.TEAL_DARK)
    ax.set_xlabel("internal difference")
    ax.set_ylabel("target difference")
    ax.legend(loc="lower right", fontsize=5.4, handletextpad=0.1, borderpad=0.1)
    S.title(ax, "Internal vs. targets")
    S.panel_label(ax, "c", dx=-0.32)
    src("e1c_quadrant", pd.concat(quad))

    # (d) rank frequency stacked bars --------------------------------------------------------------
    ax = fig.add_subplot(gs[3])
    methods = ["srs_ap_t", "srs_ap"] + list(S.BASELINES)
    piv = r[r.registry.isin(methods)].pivot_table(index="name", columns="registry", values="mean_target_auprc")
    ranks = piv.rank(axis=1, ascending=False, method="min")
    show = ["srs_ap_t", "srs_ap", "nmifs", "mrmr", "l1_dropout", "mrmr_filter_known", "dpp_ig", "cmim"]
    bins = [(1, 1, "1st"), (2, 2, "2nd"), (3, 3, "3rd"), (4, 5, "4-5th"), (6, 99, "6th+")]
    colors = [S.TEAL_DARK, S.TEAL, "#5DBAA6", S.TEAL_LIGHT, S.GREY_PALE]
    left = np.zeros(len(show))
    table = []
    for (a0, a1, text), color in zip(bins, colors):
        vals = np.array([((ranks[m] >= a0) & (ranks[m] <= a1)).sum() for m in show])
        ax.barh(range(len(show)), vals, left=left, color=color, height=0.68, edgecolor="white", lw=0.4, label=text)
        for k, v in enumerate(vals):
            if v >= 3:
                ax.text(left[k] + v / 2, k, str(v), ha="center", va="center", fontsize=5.0, color="white" if a0 <= 2 else S.INK)
        left += vals
        table.append(pd.Series(vals, index=[LABEL[m] for m in show], name=text))
    ax.set_yticks(range(len(show)), [LABEL[m] for m in show], fontsize=5.8)
    ax.invert_yaxis()
    ax.set_xlim(0, 16)
    ax.set_xticks([0, 4, 8, 12, 16])
    ax.legend(ncol=5, loc="upper center", bbox_to_anchor=(0.4, -0.2), fontsize=5.0, handlelength=0.7, columnspacing=0.5, handletextpad=0.2)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel("runs (of 16)", labelpad=1)
    S.title(ax, "Rank among 14 selectors")
    S.panel_label(ax, "d", dx=-0.42)
    src("e1d_rank_frequency", pd.concat(table, axis=1).reset_index().rename(columns={"index": "method"}))
    S.save(fig, "fig_e1_effectiveness")


from figs_b import fig_e2, fig_e3, fig_e4, fig_e5  # noqa: E402,F401
from figs_c import fig_e6, fig_e7, fig_e8, fig_e9  # noqa: E402,F401

if __name__ == "__main__":
    todo = sys.argv[1:] or [f"e{i}" for i in range(1, 10)]
    for name in todo:
        globals()[f"fig_{name}"]()
        print("built", name)
