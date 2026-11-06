"""Experiment figures X3-X7 of the PVLDB v3 paper (second design, continued from figs_v3.py): one multi-panel figure per
experiment, every panel a different chart type and a different claim than the tables.

Workflow of the cs-experiment-figure-studio skill: a figure contract in each docstring, source data of every panel in
paper/v3_assets/source_data/, PDF/SVG/PNG exports, Nature teal-green palette (style.py).

  PYTHONDONTWRITEBYTECODE=1 /d/BRCoo/miniconda/python scripts/v2/paper_v3/figs_v3b.py [x3 ... x7]
"""
from __future__ import annotations

import glob
import json
import sys

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
from figs_v3 import ART, LABEL, all_paired, rows_all, verdict

RNG = np.random.default_rng(5)
HYB = "#7E8FBF"            # NMIFS+AP (hybrid), a lighter blue next to NMIFS
ZERO = "#F1C9BE"           # zeroed coordinate (degradation, light warm)
CTRL = ("drop_labs", "drop_ventilation", "drop_bp", "drop_urine", "vitals_only")


def style_of(m: str) -> dict:
    if m == "nmifs_ap_t":
        return dict(color=HYB, ls=(0, (3, 1.2)), marker="s", lw=1.1, mfc="white")
    return method_style(m)


def with_run_keys(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["root"] = df.run.str.split("/").str[0]
    df["label"] = df.run.str.split("/").str[1]
    df["k"] = df.root.map(data.K_OF)
    df["large"] = df.root.str.startswith("union_large")
    return df


def sig(v: int) -> str:
    return "*" if v == 1 else ("°" if v == -1 else "")


# ================================================================================================= X3
TGT_ROWS = [("hospitals", "eICU hospitals"), ("eicu", "eICU"), ("hirid", "HiRID"), ("drop_labs", "no labs"),
            ("drop_ventilation", "no vent."), ("drop_bp", "no BP"), ("drop_urine", "no urine"), ("vitals_only", "vitals only")]


def per_target_verdicts() -> pd.DataFrame:
    """SRS+T against NMIFS and mRMR per target and run: the seven targets (extra_metrics) and the affected eICU hospitals."""
    pt = pd.concat([pd.read_csv(f) for f in glob.glob(str(ART / "extra_metrics/*/*/paired_targets.csv"))], ignore_index=True)
    _, hp, _ = data.hospital_all()
    h = hp[(hp.a == "srs_ap_t") & (hp.subset == "affected") & (hp["mode"] == "surrogate")].copy()
    h["run"] = h.root + "/" + h.label
    h["target"] = "hospitals"
    cols = ["run", "target", "a", "b", "point_diff", "low", "high"]
    out = pd.concat([pt[cols], h[cols]], ignore_index=True)
    out["verdict"] = verdict(out)
    return out


def fig_x3() -> None:
    """Contract. Claim (RQ3): under the schema shifts that occur in practice, SRS+T keeps its lead: it is ahead of mRMR
    on almost every target and of NMIFS on most, falls behind NMIFS where only vital signs remain, keeps the ranking under
    every deployment rule, and wins most draws of random feature or group loss. Evidence: (a) UpSet of the features absent at
    individual eICU hospitals, (b) diverging bars of per-run verdicts per target, (c) parallel coordinates over six
    deployment modes, (d) heatmap of strict win rates under random loss. Data: hospital_schemas_ap, extra_metrics
    paired_targets, deployment_modes_ap rows, stress_ap. Higher = better."""
    hr, hp, hm = data.hospital_all()
    fig = plt.figure(figsize=(S.FULL_W, 2.45))
    outer = GridSpec(1, 4, width_ratios=[0.92, 1.3, 1.0, 1.5], wspace=0.47, left=0.07, right=0.99, top=0.87, bottom=0.23)

    # (a) UpSet of the absence patterns (vasopressor 1 h cohort; the other cohorts in the source data)
    inner = GridSpecFromSubplotSpec(2, 1, subplot_spec=outer[0], height_ratios=[0.85, 1.0], hspace=0.08)
    summ = hm[("union", "f_mimic_iv_gap1h")]["summary"]
    items = sorted([(k, v) for k, v in summ["patterns"].items() if k != "-"], key=lambda kv: -kv[1])[:8]
    feats = ["Lactate", "pH", "sedation_on", "Bilirubin_Total", "urine_output"]
    fname = {"Lactate": "lactate", "pH": "pH", "sedation_on": "sedation", "Bilirubin_Total": "bilirubin", "urine_output": "urine out."}
    axb = fig.add_subplot(inner[0])
    axm = fig.add_subplot(inner[1], sharex=axb)
    xs = np.arange(len(items))
    axb.bar(xs, [v for _, v in items], color=S.TEAL, width=0.62)
    for x, (_, v) in zip(xs, items):
        axb.text(x, v + 18, str(v), ha="center", va="bottom", fontsize=4.6, color=S.INK2)
    axb.set_ylabel("episodes", fontsize=5.6)
    axb.spines["bottom"].set_visible(False)
    axb.tick_params(axis="x", length=0, labelbottom=False)
    axb.set_ylim(0, max(v for _, v in items) * 1.3)
    axb.text(0.66, 0.62, f"{summ['n_affected']:,} of {summ['n_episodes']:,}\nepisodes ({100 * summ['n_affected'] / summ['n_episodes']:.0f}%)\n"
             f"at {summ['n_hospitals']} hospitals", transform=axb.transAxes, ha="center", va="center", fontsize=4.8, color=S.INK2)
    for j, f in enumerate(feats):
        axm.axhspan(j - 0.5, j + 0.5, color="#F5F5F5" if j % 2 else "white", zorder=0, lw=0)
        for x, (k, _) in zip(xs, items):
            axm.scatter(x, j, s=9, color=S.TEAL_DARK if f in k.split(",") else S.GREY_PALE, zorder=2, lw=0)
    for x, (k, _) in zip(xs, items):
        on = [j for j, f in enumerate(feats) if f in k.split(",")]
        if len(on) > 1:
            axm.plot([x, x], [min(on), max(on)], color=S.TEAL_DARK, lw=0.9, zorder=1)
    axm.set_yticks(range(len(feats)), [fname[f] for f in feats], fontsize=5.2)
    axm.set_ylim(len(feats) - 0.5, -0.5)
    axm.set_xticks([])
    for spine in ("bottom", "left"):
        axm.spines[spine].set_visible(False)
    axm.tick_params(axis="y", length=0)
    S.title(axb, "Hospital schemas")
    S.panel_label(axb, "a", dx=-0.36)
    rows = []
    for (root, label), m in hm.items():
        for k, v in m["summary"]["patterns"].items():
            rows.append({"run": f"{root}/{label}", "pattern": k, "episodes": v, "n_affected": m["summary"]["n_affected"],
                         "n_episodes": m["summary"]["n_episodes"], "n_hospitals": m["summary"]["n_hospitals"]})
    src("x3a_hospital_patterns", pd.DataFrame(rows))

    # (b) diverging bars of verdicts per target
    V = per_target_verdicts()
    inner = GridSpecFromSubplotSpec(1, 2, subplot_spec=outer[1], wspace=0.12)
    out = []
    for ci, (b, head) in enumerate((("nmifs", "vs NMIFS"), ("mrmr", "vs mRMR"))):
        ax = fig.add_subplot(inner[ci])
        for i, (t, text) in enumerate(TGT_ROWS):
            s = V[(V.a == "srs_ap_t") & (V.b == b) & (V.target == t)]
            assert len(s) == 16, (b, t, len(s))
            w, n, bt = int((s.verdict == -1).sum()), int((s.verdict == 0).sum()), int((s.verdict == 1).sum())
            for left, width, color, ink in ((-(w + n / 2), w, S.WARM, "white"), (-n / 2, n, S.GREY_PALE, S.INK2), (n / 2, bt, S.TEAL, "white")):
                if width:
                    ax.barh(i, width, left=left, height=0.7, color=color, edgecolor="white", lw=0.5)
                    if width >= (3 if color == S.GREY_PALE else 2):
                        ax.text(left + width / 2, i, str(width), ha="center", va="center", fontsize=4.7, color=ink)
            out.append({"comparator": LABEL[b], "target": text, "better": bt, "ns": n, "worse": w})
        ax.axvline(0, color=S.INK2, lw=0.5, zorder=0)
        ax.axhline(2.5, color=S.INK2, lw=0.4, ls=(0, (1, 1.5)))
        ax.set_xlim(-14, 17)
        ax.set_xticks([-10, 0, 10], ["10", "0", "10"], fontsize=5.2)
        ax.set_ylim(len(TGT_ROWS) - 0.5, -0.5)
        ax.set_xlabel(f"{head} (runs)", fontsize=5.6)
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="y", length=0)
        if ci == 0:
            ax.set_yticks(range(len(TGT_ROWS)), [t for _, t in TGT_ROWS], fontsize=5.4)
            S.title(ax, "Verdicts per target (16 runs)")
            S.panel_label(ax, "b", dx=-0.62)
            ax.legend(handles=[Patch(color=S.TEAL, label="SRS+T better"), Patch(color=S.GREY_PALE, label="n.s."),
                               Patch(color=S.WARM, label="worse")], loc="upper center", bbox_to_anchor=(1.06, -0.2), ncol=3,
                      fontsize=4.9, handlelength=0.9, handletextpad=0.3, columnspacing=0.7)
        else:
            ax.set_yticks([])
    src("x3b_target_verdicts", pd.DataFrame(out))

    # (c) parallel coordinates over deployment modes
    dr, _ = data.deploy_all()
    modes = [("zero", "zeroed"), ("reduced", "re-fit"), ("robust", "dropout-trained"), ("surrogate", "surrogates (shipped)"),
             ("reduced_surrogate", "re-fit + surr."), ("soft", "soft surrogates")]
    tg = dr[~dr.target.isin(["internal_test"]) & dr["mode"].isin([m for m, _ in modes])]
    mt = tg.groupby(["registry", "mode", "name"]).auprc.mean().groupby(["registry", "mode"]).mean().unstack()
    ax = fig.add_subplot(outer[2])
    for m in ("srs_ap_t", "srs_ap", "nmifs", "mrmr", "l1_dropout"):
        st = style_of(m)
        ax.plot(range(len(modes)), [mt.loc[m, md] for md, _ in modes], color=st["color"], ls=st["ls"], marker=st["marker"],
                mfc=st["mfc"], mec=st["color"], lw=st["lw"], ms=3.0, label=LABEL[m])
    for x in range(len(modes)):
        ax.axvline(x, color="#EEEEEE", lw=0.6, zorder=0)
    ax.set_xticks(range(len(modes)), [t for _, t in modes], fontsize=4.9, rotation=32, ha="right", rotation_mode="anchor")
    ax.set_ylabel("mean target AUPRC (16 runs)")
    ax.legend(ncol=2, fontsize=4.8, loc="upper left", handlelength=1.6, columnspacing=0.6, borderpad=0.1, labelspacing=0.25)
    ax.set_ylim(0.664, 0.721)
    S.title(ax, "Six ways to deploy")
    S.panel_label(ax, "c", dx=-0.3)
    src("x3c_deployment_modes", mt.reset_index())

    # (d) strict win rates under random loss
    ax = fig.add_subplot(outer[3])
    st_ = data.stress_all()
    st_ = st_[st_.variant == "composite"]
    piv = st_.pivot_table(index=["family", "level", "draw", "name"], columns="registry", values="auprc")
    opps = ["nmifs", "mrmr", "l1_dropout", "mrmr_filter_known", "cmim", "fs_cv"]
    levels = sorted(st_.level.unique())
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
            ax.text(j, i, f"{100 * win[i, j]:.0f}", ha="center", va="center", fontsize=4.4, color="white" if win[i, j] > 0.7 else S.INK)
    ax.axvline(len(levels) - 0.5, color="white", lw=2.0)
    ax.set_xticks(range(2 * len(levels)), [f"{lv:.1f}".lstrip("0") for lv in levels] * 2, fontsize=4.5)
    ax.set_yticks(range(len(opps)), [LABEL[o] for o in opps], fontsize=5.4)
    ax.text(len(levels) / 2 - 0.5, len(opps) + 0.12, "features dropped, $l$", ha="center", va="top", fontsize=5.5)
    ax.text(1.5 * len(levels) - 0.5, len(opps) + 0.12, "groups dropped, $l$", ha="center", va="top", fontsize=5.5)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    S.title(ax, "Draws won strictly by SRS+T (%)")
    S.panel_label(ax, "d", dx=-0.25)
    cols = [f"{fam}_{lv}" for fam in ("feature", "group") for lv in levels]
    src("x3d_strict_wins", pd.concat([pd.DataFrame(win, index=[LABEL[o] for o in opps], columns=cols).assign(kind="strict win"),
                                      pd.DataFrame(tie, index=[LABEL[o] for o in opps], columns=cols).assign(kind="tie")]).reset_index())
    S.save(fig, "fig_x3_schema_shift")


# ================================================================================================= X4
def fig_x4() -> None:
    """Contract. Claim (RQ4): the lead grows with the budget and holds on 4.5x larger pools, and it comes from the
    heterogeneity of the union rather than from its size. Evidence: (a) mean target AUPRC vs K on the standard pools,
    (b) paired differences with CIs on the large pools, (c) dumbbell of union vs single-generator pools of 1,000,
    (d) heatmap of paired differences per setting and K against mRMR, NMIFS and NMIFS+AP. Data: extra_metrics rows_all,
    union_ap paired, nmifs_ap paired, restricted_pools rows."""
    ra = with_run_keys(rows_all())
    P = all_paired()
    fig = plt.figure(figsize=(S.FULL_W, 2.35))
    gs = GridSpec(1, 4, width_ratios=[0.95, 1.15, 1.1, 0.95], wspace=0.62, left=0.065, right=0.965, top=0.86, bottom=0.25)

    # (a) budget
    ax = fig.add_subplot(gs[0])
    std = ra[~ra.large]
    rows = []
    for m in ("srs_ap_t", "srs_ap", "nmifs_ap_t", "nmifs", "mrmr", "l1_dropout"):
        g = std[std.registry == m].groupby("k").mean_target_auprc.mean()
        sty = style_of(m)
        ax.plot(g.index, g.values, color=sty["color"], ls=sty["ls"], marker=sty["marker"], ms=2.8, mfc=sty["mfc"], mec=sty["color"],
                lw=sty["lw"], label=LABEL[m])
        rows.append(pd.DataFrame({"method": LABEL[m], "k": g.index, "mean_target": g.values}))
    ax.set_xscale("log", base=2)
    ax.set_xticks([5, 10, 20], ["5", "10", "20"])
    ax.minorticks_off()
    ax.set_xlabel("budget $K$")
    ax.set_ylabel("mean target AUPRC (4 settings)")
    ax.legend(fontsize=4.7, loc="upper center", bbox_to_anchor=(0.45, -0.24), ncol=3, handlelength=1.6, columnspacing=0.5, handletextpad=0.3)
    ax.grid(axis="y")
    S.title(ax, "Budget")
    S.panel_label(ax, "a", dx=-0.32)
    src("x4a_budget", pd.concat(rows))

    # (b) large pools: paired differences with CIs
    ax = fig.add_subplot(gs[1])
    cases = ["Vaso 1h L K10", "Vaso 1h L K20", "AKI L K10", "AKI L K20"]
    comps = [("nmifs_ap_t", HYB), ("nmifs", S.BLUE), ("mrmr", S.SKY), ("l1_dropout", S.GREY)]
    w = 0.19
    out = []
    for ci, case in enumerate(cases):
        for bi, (b, color) in enumerate(comps):
            s = P[(P.name == case) & (P.b == b) & (P.metric == "mean_target")].iloc[0]
            x = ci + (bi - 1.5) * w
            ax.bar(x, s.point_diff, width=w * 0.9, color=color, edgecolor="white", lw=0.3, label=LABEL[b] if ci == 0 else None)
            ax.vlines(x, s.low, s.high, color=S.INK, lw=0.6)
            if s.verdict != 0:
                ax.text(x, (s.high + 0.002) if s.point_diff > 0 else (s.low - 0.002), "*" if s.verdict == 1 else "°", ha="center",
                        va="bottom" if s.point_diff > 0 else "top", fontsize=5.6, color=S.INK)
            out.append({"case": case, "comparator": LABEL[b], "diff": s.point_diff, "low": s.low, "high": s.high, "verdict": s.verdict})
    ax.axhline(0, color=S.INK2, lw=0.6)
    ax.set_xticks(range(len(cases)), ["V1h\nK10", "V1h\nK20", "AKI\nK10", "AKI\nK20"], fontsize=5.4)
    ax.set_ylabel("SRS+T $-$ comparator (targets)")
    ax.legend(ncol=4, fontsize=4.7, loc="upper center", bbox_to_anchor=(0.45, -0.24), handlelength=0.8, columnspacing=0.6, handletextpad=0.3)
    ax.grid(axis="y")
    S.title(ax, "Large pools")
    S.panel_label(ax, "b", dx=-0.3)
    src("x4b_large_pools", pd.DataFrame(out))

    # (c) heterogeneity, not size
    ax = fig.add_subplot(gs[2])
    rp = data.restricted()
    r0 = data.rows_all()
    union = r0[(r0.root == "union") & (r0.registry == "srs_ap_t")].set_index("label")
    rows = []
    ylabels = []
    y = 0
    for label, task in (("f_mimic_iv_gap1h", "Vaso 1h"), ("f_mimic_iv_aki_gap6h", "AKI")):
        for metric, mtext in (("internal_auprc", "internal"), ("mean_target_auprc", "targets")):
            vals = {"union (~510)": float(union.loc[label, metric])}
            for gen, gtext in (("generator_erminer", "ERMiner (1,000)"), ("generator_prefixspan", "PrefixSpan (1,000)")):
                s = rp[(rp.restrict_dir == gen) & (rp.registry == "srs_ap_t") & (rp.run.str.contains(label))]
                vals[gtext] = float(s[metric].iloc[0])
            xs_ = list(vals.values())
            ax.plot([min(xs_), max(xs_)], [y, y], color=S.GREY_PALE, lw=2.6, zorder=1, solid_capstyle="round")
            for (name, v), (color, marker) in zip(vals.items(), ((S.TEAL, "o"), (S.TEAL_DARK, "s"), (S.SAND, "^"))):
                ax.scatter(v, y, s=15, color=color, marker=marker, zorder=3, label=name if y == 0 else None, lw=0)
            rows.append({"task": task, "metric": mtext, **vals})
            ylabels.append(f"{task}, {mtext}")
            y += 1
    ax.set_yticks(range(len(ylabels)), ylabels, fontsize=5.4)
    ax.set_ylim(len(ylabels) - 0.4, -0.6)
    ax.set_xlabel("AUPRC of SRS+T ($K = 10$)")
    ax.legend(fontsize=4.7, loc="upper center", bbox_to_anchor=(0.3, -0.24), ncol=2, handletextpad=0.1, columnspacing=0.6)
    ax.grid(axis="x")
    S.title(ax, "Heterogeneity, not size")
    S.panel_label(ax, "c", dx=-0.5)
    src("x4c_single_generator", pd.DataFrame(rows))

    # (d) heatmap per setting and K
    ax = fig.add_subplot(gs[3])
    settings = [("Vaso 1h", "V1h"), ("Vaso 2h", "V2h"), ("Vaso 4h", "V4h"), ("AKI", "AKI")]
    comps = [("mrmr", "mRMR"), ("nmifs", "NMIFS"), ("nmifs_ap_t", "NMIFS+AP")]
    mat = np.zeros((9, 4))
    ver = np.zeros_like(mat)
    for bi, (b, _) in enumerate(comps):
        for ki, k in enumerate((5, 10, 20)):
            for si, (task, _) in enumerate(settings):
                s = P[(P.name == f"{task} K{k}") & (P.b == b) & (P.metric == "mean_target")].iloc[0]
                mat[bi * 3 + ki, si], ver[bi * 3 + ki, si] = s.point_diff, s.verdict
    lim = 0.05
    ax.imshow(np.clip(mat, -lim, lim), cmap=S.DIV, vmin=-lim, vmax=lim, aspect="auto")
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax.text(j, i, f"{100 * mat[i, j]:+.1f}{sig(ver[i, j])}", ha="center", va="center", fontsize=4.6,
                    color="white" if abs(mat[i, j]) > 0.03 else S.INK)
    for yy in (2.5, 5.5):
        ax.axhline(yy, color="white", lw=2.0)
    ax.set_xticks(range(4), [t for _, t in settings], fontsize=5.3)
    ax.set_yticks(range(9), ["K5", "K10", "K20"] * 3, fontsize=5.0)
    for bi, (_, text) in enumerate(comps):
        ax.text(3.62, bi * 3 + 1, f"vs {text}", rotation=-90, ha="left", va="center", fontsize=5.2, color=S.INK2)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xlabel("target diff. ×100 (* better, ° worse)", fontsize=5.2)
    S.title(ax, "Edge per setting and $K$")
    S.panel_label(ax, "d", dx=-0.32)
    src("x4d_edge_heatmap", pd.DataFrame(mat, index=[f"{t}_K{k}" for _, t in comps for k in (5, 10, 20)],
                                         columns=[t for _, t in settings]).reset_index())
    S.save(fig, "fig_x4_budget_pools")


# ================================================================================================= X5
def fig_x5() -> None:
    """Contract. Claim (RQ5): selection is linear in |C| (10^5 candidates in about two hours on one machine), the
    surrogate chains are quadratic but computed once, the extra accuracy costs seconds to minutes of offline selection
    where filters take milliseconds, and the shipped registries run as SQL in seconds, also when the database exceeds
    memory. Evidence: (a) log-log selection time vs |C|, (b) 100% stacked phase breakdown, (c) cost-quality scatter with
    the Pareto frontier, (d) dumbbell of SQL time in-memory vs disk-backed. Data: selection_scalability, extra_metrics
    rows_all, sql_execution/disk_scale_*. Time: lower = better; AUPRC: higher = better."""
    sc = pd.read_csv(ART / "selection_scalability/selection_scalability.csv")
    sz = sc[(sc.axis == "size") & (sc.status == "ok")]
    fig = plt.figure(figsize=(S.FULL_W, 2.35))
    gs = GridSpec(1, 4, width_ratios=[1.2, 0.9, 1.1, 1.05], wspace=0.55, left=0.06, right=0.975, top=0.84, bottom=0.25)

    # (a) runtime vs |C|
    ax = fig.add_subplot(gs[0])
    lines = [("srs_ap_v", "SRS+T", S.TEAL, "-", "o", 1.5), ("srs_c_v", "SRS+T, lik. decision", S.TEAL_DARK, (0, (2, 1.2)), "o", 1.0),
             ("fast_q0", "fast screening, $q = 0$", "#7CC7B6", "-", "o", 1.0), ("srs_c_exact", "exact greedy, $q = 0$", "#7CC7B6", (0, (3, 1.5)), "s", 1.0),
             ("fs_cv", "Wrapper, lik.", S.GREY_LIGHT, "-", "D", 1.0),
             ("dpp", "DPP", S.SAND, "-", "x", 1.0), ("mmrfs", "MMRFS", "#C9B48F", (0, (1, 1)), "+", 1.0),
             ("l1_logistic", "L1", S.GREY, "-", ">", 1.0), ("cmim", "CMIM / JMI", HYB, "-", "v", 1.0),
             ("nmifs", "NMIFS", S.BLUE, "-", "s", 1.0), ("mrmr", "mRMR", S.SKY, "-", "^", 1.0)]
    rows = []
    for m, text, color, ls, marker, lw in lines:
        if m == "fast_q0":  # chains + the q = 0 selection of the same timed run, the work exact greedy does
            s = sz[sz.method == "srs_c_v"].sort_values("n_candidates").copy()
            s = s[s.n_candidates <= 5000]
            s["seconds"] = s.phase_chains + s.phase_select_q0
            s["method"] = "fast_q0"
        else:
            s = sz[sz.method == m].sort_values("n_candidates")
        ax.plot(s.n_candidates, s.seconds, color=color, ls=ls, marker=marker, ms=2.2, lw=lw, label=text, mfc=color)
        rows.append(s[["method", "n_candidates", "seconds", "peak_mb"]])
    xs = np.array([500, 1e5])
    ax.plot(xs, 30 * xs / 500 * 2.2, color=S.INK2, lw=0.5, ls=(0, (2, 2)))
    ax.text(2.4e4, 30 * 2.4e4 / 500 * 2.2 * 1.5, r"$\propto |C|$", fontsize=5.2, color=S.INK2, rotation=24, ha="center")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("candidates $|C|$")
    ax.set_ylabel("selection time (s)")
    ax.legend(fontsize=4.3, loc="upper center", bbox_to_anchor=(0.62, -0.22), handlelength=1.6, ncol=3, columnspacing=0.5,
              handletextpad=0.3, labelspacing=0.2)
    ax.grid(which="major")
    S.title(ax, "Selection time")
    S.panel_label(ax, "a", dx=-0.22)
    src("x5a_runtime", pd.concat(rows))

    # (b) phase breakdown
    ax = fig.add_subplot(gs[1])
    v = sz.set_index(["method", "n_candidates"])
    bars = [("srs_ap_v", 500, "500"), ("srs_ap_v", 2274, "2,274"), ("srs_ap_v", 10000, "$10^4$"), ("srs_ap_v", 20000, "$2{\\cdot}10^4$"),
            ("srs_c_v", 100000, "$10^5\\dagger$")]
    phases = [("phase_chains", "surrogate chains", S.SAND), ("phase_select_q0", "selection, $q=0$", S.TEAL_LIGHT),
              ("phase_select_q25", "selection, $q=0.25$", S.TEAL)]
    bottom = np.zeros(len(bars))
    out = []
    for col, text, color in phases:
        share = np.array([v.loc[(m, n), col] / v.loc[(m, n), "seconds"] for m, n, _ in bars])
        ax.bar(range(len(bars)), share, bottom=bottom, color=color, width=0.64, edgecolor="white", lw=0.4, label=text)
        bottom += share
    for k, (m, n, _) in enumerate(bars):
        t = v.loc[(m, n), "seconds"]
        ax.text(k, 1.02, f"{t:.0f} s" if t < 1000 else (f"{t / 60:.0f} m" if t < 3600 else f"{t / 3600:.1f} h"), ha="center", va="bottom",
                fontsize=4.8, color=S.INK2)
        out.append({"method": m, "n_candidates": n, **{c: v.loc[(m, n), c] for c, _, _ in phases}, "seconds": t})
    ax.set_xticks(range(len(bars)), [b[2] for b in bars], fontsize=5.0)
    ax.set_ylim(0, 1.13)
    ax.set_yticks([0, 0.5, 1.0], ["0", "50%", "100%"])
    ax.set_xlabel("candidates $|C|$")
    ax.legend(fontsize=4.6, loc="upper center", bbox_to_anchor=(0.45, -0.22), ncol=2, handlelength=0.8, columnspacing=0.5)
    S.title(ax, "Where the time goes")
    S.panel_label(ax, "b", dx=-0.32)
    src("x5b_phases", pd.DataFrame(out))

    # (c) cost-quality frontier at |C| = 500 (the size of the union pools)
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
    best = -1.0
    front = []
    for _, row in frame.iterrows():
        if row.auprc > best:
            front.append(row)
            best = row.auprc
    fx = [r.seconds for r in front]
    fy = [r.auprc for r in front]
    ax.step(fx + [200], fy + [fy[-1]], where="post", color=S.TEAL_LIGHT, lw=1.0, zorder=1)
    for _, row in frame.iterrows():
        ax.scatter(row.seconds, row.auprc, s=16, color=row.color, zorder=3, edgecolor="white", lw=0.4)
    offsets = {"SRS+T": (-3, 1, "right", "bottom"), "SRS+T, lik.": (-3, 3, "right", "bottom"), "Wrapper, lik.": (0, -4, "center", "top"),
               "DPP": (3, 0, "left", "center"), "MMRFS": (0, 3, "center", "bottom"), "L1": (4, 0, "left", "center"),
               "CMIM": (3, 0, "left", "center"), "JMI": (3, -1, "left", "center"), "NMIFS": (3, 1, "left", "bottom"),
               "mRMR": (3, -1, "left", "top")}
    for _, row in frame.iterrows():
        dx, dy, ha, va = offsets[row.method]
        ax.annotate(row.method, (row.seconds, row.auprc), xytext=(dx, dy), textcoords="offset points", ha=ha, va=va, fontsize=4.8,
                    color=S.INK2)
    ax.set_xscale("log")
    ax.set_xlim(3e-3, 150)
    ax.set_ylim(0.62, 0.714)
    ax.set_xlabel("selection time at $|C| = 500$ (s)")
    ax.set_ylabel("mean target AUPRC (16 runs)")
    ax.grid(which="major")
    S.title(ax, "Cost and quality")
    S.panel_label(ax, "c", dx=-0.3)
    src("x5c_cost_quality", frame.drop(columns="color"))

    # (d) SQL: in-memory vs disk-backed DuckDB (AKI, eICU episodes replicated)
    ax = fig.add_subplot(gs[3])
    mem = pd.DataFrame(json.loads((ART / "sql_execution/disk_scale_f_mimic_iv_aki_gap6h_memory.json").read_text(encoding="utf-8")))
    dsk = pd.DataFrame(json.loads((ART / "sql_execution/disk_scale_f_mimic_iv_aki_gap6h_disk.json").read_text(encoding="utf-8")))
    yl = []
    y = 0
    out = []
    for r_ in (1, 10, 30):
        a, b = mem[mem.replicate == r_].iloc[0], dsk[dsk.replicate == r_].iloc[0]
        for col, text, color in (("seconds_tokens", "event tokens", S.TEAL), ("seconds_registry", "registry", S.BLUE)):
            ax.plot([a[col], b[col]], [y, y], color=S.GREY_PALE, lw=2.4, zorder=1, solid_capstyle="round")
            ax.scatter(a[col], y, s=15, color=color, zorder=3, lw=0)
            ax.scatter(b[col], y, s=15, facecolor="white", edgecolor=color, lw=0.9, zorder=3)
            yl.append(f"×{r_} {text}")
            out.append({"replicate": r_, "n_episodes": a.n_episodes, "phase": text, "in_memory_s": a[col], "disk_s": b[col],
                        "peak_gb_memory": a.peak_rss_gb, "peak_gb_disk": b.peak_rss_gb})
            y += 1
        ax.text(260, y - 1.5, f"{a.peak_rss_gb:.1f} → {b.peak_rss_gb:.1f} GB", fontsize=4.7, color=S.INK2, va="center", ha="left")
        y += 0.45
    ax.set_yticks([0, 1, 2.45, 3.45, 4.9, 5.9], yl, fontsize=4.9)
    ax.set_ylim(y - 0.2, -0.7)
    ax.set_xscale("log")
    ax.set_xlim(0.1, 1500)
    ax.set_xticks([0.1, 1, 10, 100])
    ax.set_xlabel("seconds (AKI, eICU ×1 to ×30)")
    ax.legend(handles=[Line2D([], [], marker="o", ls="", color=S.INK2, label="in-memory"),
                       Line2D([], [], marker="o", ls="", mfc="white", mec=S.INK2, label="disk-backed, 8 GB limit")],
              fontsize=4.7, loc="upper center", bbox_to_anchor=(0.3, -0.22), ncol=2, handletextpad=0.1, columnspacing=0.6)
    ax.grid(axis="x", which="major")
    ax.text(0.98, 0.02, "0 mismatches in 16,932\nprogram × database checks", transform=ax.transAxes, ha="right", va="bottom",
            fontsize=4.6, color=S.TEAL_DARK)
    S.title(ax, "Registries as SQL")
    S.panel_label(ax, "d", dx=-0.42)
    src("x5d_sql", pd.DataFrame(out))
    S.save(fig, "fig_x5_efficiency")


# ================================================================================================= X6
def fig_x6() -> None:
    """Contract. Claim (RQ6): the conclusions survive new patient splits, a tree decoder, demographic subgroups and changes
    of every hyper-parameter, and greedy reaches 95-100% of the exact optimum of its objective; the exception is NMIFS on
    re-split vasopressor 1 h. Evidence: (a) strip plot over five re-splits, (b) forest of decoder-swapped differences,
    (c) categorical heatmap of subgroup verdicts, (d) tornado of hyper-parameter changes, (e) ECDF of greedy/optimum.
    Data: union_ap/union_resplit, union_robustness_ap, srs_sensitivity_ap, exact_small_pools."""
    rng = np.random.default_rng(2)
    fig = plt.figure(figsize=(S.FULL_W, 2.4))
    gs = GridSpec(1, 5, width_ratios=[1.05, 0.95, 1.05, 1.05, 0.85], wspace=0.62, left=0.055, right=0.985, top=0.87, bottom=0.24)

    # (a) re-splits
    ax = fig.add_subplot(gs[0])
    rr, _ = data.resplits()
    meths = ["srs_ap_t", "nmifs", "mrmr", "l1_dropout", "mrmr_filter_known"]
    rows = []
    for ti, task in enumerate(("Vaso 1h", "AKI")):
        for mi, m in enumerate(meths):
            v = rr[(rr.task == task) & (rr.registry == m)].mean_target_auprc.values
            x = ti * (len(meths) + 1) + mi
            ax.scatter(np.full(len(v), x) + rng.uniform(-0.18, 0.18, len(v)), v, s=7, color=S.METHOD_COLOR[m], lw=0, alpha=0.9)
            ax.hlines(v.mean(), x - 0.32, x + 0.32, color=S.INK, lw=0.9)
            rows.append({"task": task, "method": LABEL[m], **{f"split{i + 1}": vv for i, vv in enumerate(v)}})
    ax.set_xticks([2, 8], ["Vaso 1h", "AKI"])
    ax.set_ylabel("mean target AUPRC")
    ax.legend(handles=[Line2D([], [], marker="o", ls="", color=S.METHOD_COLOR[m], label=LABEL[m]) for m in meths], fontsize=4.6,
              loc="upper right", ncol=1, handletextpad=0.05, borderpad=0.1, labelspacing=0.15)
    ax.grid(axis="y")
    S.title(ax, "Five new patient splits")
    S.panel_label(ax, "a", dx=-0.3)
    src("x6a_resplits", pd.DataFrame(rows))

    # (b) decoder swap: gradient-boosted trees instead of the logistic probe
    ax = fig.add_subplot(gs[1])
    rb = data.robustness()
    y = 0
    ticks = []
    rows = []
    for label, task in zip(data.STANDARD, ("V1h", "V2h", "V4h", "AKI")):
        dp = rb[label]["decoder_paired"]
        for b, color, marker in (("mrmr", S.SKY, "^"), ("nmifs", S.BLUE, "s")):
            s_ = dp[(dp.a == "srs_ap_t") & (dp.b == b) & (dp.decoder == "gbdt") & (dp.target == "mean_over_targets")]
            ax.hlines(y, s_.low, s_.high, color=color, lw=0.9)
            ax.scatter(s_.point_diff, y, color=color, marker=marker, s=10, zorder=3)
            rows.append({"setting": task, "b": b, "diff": float(s_.point_diff.iloc[0]), "low": float(s_.low.iloc[0]), "high": float(s_.high.iloc[0])})
            y += 1
        ticks.append((y - 1.5, task))
        y += 0.6
    ax.set_yticks([t[0] for t in ticks], [t[1] for t in ticks], fontsize=5.5)
    ax.set_ylim(y - 0.4, -0.9)
    ax.axvline(0, color=S.INK2, lw=0.6, ls=(0, (2, 2)))
    ax.set_xlabel("SRS+T $-$ comparator (trees)")
    ax.legend(handles=[Line2D([], [], marker="^", ls="", color=S.SKY, label="vs mRMR"), Line2D([], [], marker="s", ls="", color=S.BLUE, label="vs NMIFS")],
              fontsize=4.8, loc="upper center", bbox_to_anchor=(0.45, -0.22), ncol=2, handletextpad=0.05, columnspacing=0.6)
    S.title(ax, "Tree decoder")
    S.panel_label(ax, "b", dx=-0.3)
    src("x6b_decoder", pd.DataFrame(rows))

    # (c) subgroup verdicts
    ax = fig.add_subplot(gs[2])
    first = rb[data.STANDARD[0]]["subgroup_paired"]
    sub_keys = [tuple(x) for x in first[(first.a == "srs_ap_t") & (first.b == "mrmr")][["cohort", "subgroup"]].drop_duplicates().values]
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
    ax.set_yticks(range(len(sub_keys)), [f"{cohort_name.get(c, c)} {g}" for c, g in sub_keys], fontsize=4.5)
    ax.set_xticks(range(8), ["1", "2", "4", "A"] * 2, fontsize=5.0)
    nb = (mat[:, :4] == 1).sum(), (mat[:, :4] == -1).sum(), (mat[:, 4:] == 1).sum(), (mat[:, 4:] == -1).sum()
    tr = blended_transform_factory(ax.transData, ax.transAxes)
    ax.text(1.5, -0.075, f"vs mRMR\n{nb[0]} better\n{nb[1]} worse", transform=tr, ha="center", va="top", fontsize=4.8, color=S.INK2)
    ax.text(5.5, -0.075, f"vs NMIFS\n{nb[2]} better\n{nb[3]} worse", transform=tr, ha="center", va="top", fontsize=4.8, color=S.INK2)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    S.title(ax, "Subgroups (sex, age)")
    S.panel_label(ax, "c", dx=-0.45)
    src("x6c_subgroups", pd.DataFrame(mat, index=[f"{c} {g}" for c, g in sub_keys],
                                      columns=[f"{b}_{l}" for b in ("mrmr", "nmifs") for l in data.STANDARD]).reset_index())

    # (d) tornado of hyper-parameter changes
    ax = fig.add_subplot(gs[3])
    sens = data.sensitivity()
    default = sens[sens.parameter == "default"].set_index("label").mean_target_auprc
    names = {"q": "q", "max_surrogates": "M", "n_folds": "V", "C": "1/η", "n_dropout": "D", "seed": "seed"}
    rows = []
    keep = sens[(sens.parameter != "default") & ~((sens.parameter == "q") & (sens.value.astype(str) == "0.25"))]
    for (param, value), g in keep.groupby(["parameter", "value"], sort=False):
        d = (g.set_index("label").mean_target_auprc - default).values
        rows.append({"variant": f"{names.get(param, param)} = {float(value):g}", "min": d.min(), "max": d.max(), "mean": d.mean()})
    tor = pd.DataFrame(rows).sort_values("min").reset_index(drop=True)
    for k, row in tor.iterrows():
        ax.hlines(k, row["min"], row["max"], color=S.TEAL_LIGHT, lw=3.2)
        ax.scatter(row["mean"], k, s=7, color=S.TEAL_DARK, zorder=3)
    ax.axvline(0, color=S.INK2, lw=0.6)
    ax.set_yticks(range(len(tor)), tor.variant, fontsize=4.6)
    ax.set_ylim(len(tor) - 0.4, -0.6)
    ax.set_xlabel("Δ target AUPRC vs default")
    S.title(ax, "Hyper-parameters")
    S.panel_label(ax, "d", dx=-0.35)
    src("x6d_sensitivity", tor)

    # (e) ECDF of greedy / optimum
    ax = fig.add_subplot(gs[4])
    ex = pd.concat([pd.read_csv(ART / f"exact_small_pools/rows_{t}.csv") for t in ("gap1h", "aki_gap6h")])
    rows = []
    for obj, color, ls, text in (("deployed_q0", S.TEAL, "-", "$q = 0$"), ("deployed_q25", S.BLUE, (0, (3, 1.5)), "$q = 0.25$"),
                                 ("analyzable", S.SAND, (0, (1, 1)), "analyzable")):
        v = np.sort(ex[ex.objective == obj].ratio.values)
        ax.step(v, np.arange(1, len(v) + 1) / len(v), where="post", color=color, ls=ls, lw=1.2, label=text)
        rows.append(pd.DataFrame({"objective": obj, "ratio": v}))
    ax.set_xlim(0.93, 1.004)
    ax.set_xlabel("greedy / optimum")
    ax.set_ylabel("share of 80 sub-pools")
    ax.legend(fontsize=4.7, loc="upper left", handlelength=1.6, borderpad=0.1)
    ax.grid()
    S.title(ax, "Near-optimal")
    S.panel_label(ax, "e", dx=-0.35)
    src("x6e_exact_ratio", pd.concat(rows))
    S.save(fig, "fig_x6_robustness")


# ================================================================================================= X7
def fig_x7() -> None:
    """Contract. Claim (RQ7): in a second, pre-specified domain (drive failure across manufacturers) SRS+T is ahead of the
    classic selectors at K = 10 and 20 but not at K = 5, where NMIFS and NMIFS+AP win; L1 with schema dropout ties it.
    Mechanism: SRS+T keeps more programs executable than the filters and fills most of the rest with surrogates.
    Evidence: (a) heatmap of paired differences per budget, (b) stacked bars of how the K = 10 programs run at the three
    targets, (c) target AUPRC vs K for variants and the strongest comparators, (d) forest of the pre-registered tests,
    pre-specified (likelihood decision) and post hoc (AUPRC decision). Data: union_ap/drive_k*, nmifs_ap/drive_k*,
    paper_v3_cache/drive_executability.csv, drive/union_k10 (pre-specified)."""
    rows, paired, abl = data.drives()
    hyb_rows = pd.concat([pd.read_csv(f).assign(k=int(f.split("drive_k")[1].split("\\")[0].split("/")[0]))
                          for f in glob.glob(str(ART / "nmifs_ap/drive_k*/*/rows.csv"))], ignore_index=True)
    hyb_pair = pd.concat([pd.read_csv(f).assign(k=int(f.split("drive_k")[1].split("\\")[0].split("/")[0]))
                          for f in glob.glob(str(ART / "nmifs_ap/drive_k*/*/paired.csv"))], ignore_index=True)
    rows = pd.concat([rows, hyb_rows[hyb_rows.registry == "nmifs_ap_t"]], ignore_index=True)
    hp = hyb_pair[(hyb_pair.a == "srs_ap_t") & (hyb_pair.b == "nmifs_ap_t")].copy()
    hp["verdict"] = verdict(hp)
    paired = pd.concat([paired, hp], ignore_index=True)
    fig = plt.figure(figsize=(S.FULL_W, 2.35))
    gs = GridSpec(1, 4, width_ratios=[1.25, 1.05, 0.95, 0.9], wspace=0.7, left=0.075, right=0.985, top=0.87, bottom=0.25)

    # (a) heatmap of paired differences
    ax = fig.add_subplot(gs[0])
    base = ["nmifs_ap_t", "nmifs", "mrmr", "mmrfs", "cmim", "jmi", "l1_logistic", "l1_dropout", "mrmr_filter_known"]
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
            ax.text(j, i, f"{100 * mat[i, j]:+.1f}{sig(ver[i, j])}", ha="center", va="center", fontsize=4.6,
                    color="white" if abs(mat[i, j]) > 0.07 else S.INK)
    for x in (1.5, 3.5):
        ax.axvline(x, color="white", lw=2.0)
    ax.set_xticks(range(6), ["int.", "tgt."] * 3, fontsize=5.1)
    for ki, k in enumerate((5, 10, 20)):
        ax.text(ki * 2 + 0.5, len(base) + 0.25, f"K = {k}" + ("\n(pre-spec.)" if k == 10 else ""), ha="center", va="top", fontsize=5.1,
                color=S.INK2)
    ax.set_yticks(range(len(base)), [LABEL[b] for b in base], fontsize=5.4)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    S.title(ax, "SRS+T $-$ comparator (×100)")
    S.panel_label(ax, "a", dx=-0.36)
    src("x7a_drive_heatmap", pd.DataFrame(mat, index=[LABEL[b] for b in base],
                                          columns=[f"K{k}_{m}" for k in (5, 10, 20) for m in ("internal", "target")]).reset_index())

    # (b) how the K = 10 programs run at the three manufacturers
    ax = fig.add_subplot(gs[1])
    ex = pd.read_csv(ART / "paper_v3_cache/drive_executability.csv")
    ex = ex[ex.k == 10]
    meths = ["srs_ap_t", "srs_ap", "nmifs_ap_t", "nmifs", "mrmr", "l1_dropout", "fs_cv"]
    tot = ex.groupby("registry")[["primary", "surrogate", "zeroed"]].sum().loc[meths]
    left = np.zeros(len(meths))
    for col, text, color in (("primary", "as selected", S.TEAL), ("surrogate", "via surrogate", S.TEAL_LIGHT), ("zeroed", "zeroed", ZERO)):
        ax.barh(range(len(meths)), tot[col].values, left=left, color=color, height=0.66, edgecolor="white", lw=0.5, label=text)
        for k_, v_ in enumerate(tot[col].values):
            if v_ >= 2:
                ax.text(left[k_] + v_ / 2, k_, str(int(v_)), ha="center", va="center", fontsize=4.8, color="white" if col == "primary" else S.INK)
        left += tot[col].values
    ax.set_yticks(range(len(meths)), [LABEL[m] for m in meths], fontsize=5.4)
    ax.set_ylim(len(meths) - 0.5, -0.6)
    ax.set_xlim(0, 30)
    ax.set_xticks([0, 10, 20, 30])
    ax.set_xlabel("programs at HGST + Toshiba + WDC ($K = 10$)")
    ax.legend(ncol=3, fontsize=4.7, loc="upper center", bbox_to_anchor=(0.4, -0.22), handlelength=0.8, columnspacing=0.6, handletextpad=0.3)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    S.title(ax, "How the programs run")
    S.panel_label(ax, "b", dx=-0.42)
    src("x7b_executability", ex)

    # (c) target AUPRC vs K
    ax = fig.add_subplot(gs[2])
    ref = rows[rows.registry == "srs_ap_t"].set_index("k").mean_target_auprc.loc[[5, 10, 20]]
    out = []
    for m in ("srs_ap_t", "srs_ap", "nmifs_ap_t", "nmifs", "l1_dropout"):
        vals = rows[rows.registry == m].set_index("k").mean_target_auprc.loc[[5, 10, 20]].values
        sty = style_of(m)
        ax.plot([5, 10, 20], vals, color=sty["color"], ls=sty["ls"], marker=sty["marker"], ms=2.8, mfc=sty["mfc"], mec=sty["color"],
                lw=sty["lw"], label=LABEL[m])
        out.append({"variant": m, "k5": vals[0], "k10": vals[1], "k20": vals[2]})
    s = abl[(abl.a == "srs_ap_t") & (abl.b == "srs_ap_t_nosur") & (abl.target == "mean_over_targets")].set_index("k").point_diff.loc[[5, 10, 20]]
    vals = ref.values - s.values
    ax.plot([5, 10, 20], vals, color=S.WARM, ls=(0, (3, 1.5)), marker="o", ms=2.8, mfc="white", mec=S.WARM, lw=1.0, label="no surr. in obj.")
    out.append({"variant": "srs_ap_t_nosur", "k5": vals[0], "k10": vals[1], "k20": vals[2]})
    ax.set_xscale("log", base=2)
    ax.set_xticks([5, 10, 20], ["5", "10", "20"])
    ax.minorticks_off()
    ax.set_xlabel("budget $K$")
    ax.set_ylabel("mean target AUPRC")
    ax.legend(fontsize=4.6, loc="upper center", bbox_to_anchor=(0.42, -0.22), ncol=2, handlelength=1.6, columnspacing=0.5, labelspacing=0.15)
    ax.grid(axis="y")
    S.title(ax, "Budget, variants")
    S.panel_label(ax, "c", dx=-0.42)
    src("x7c_variants", pd.DataFrame(out))

    # (d) pre-registered tests: pre-specified (likelihood) vs post hoc (AUPRC)
    ax = fig.add_subplot(gs[3])
    pre_dir = ART / "drive/union_k10/f_backblaze_seagate_gap2d"
    pi = pd.read_csv(pre_dir / "paired_internal.csv")
    pt = pd.read_csv(pre_dir / "paired_targets.csv")
    for df in (pi, pt):
        df["a"] = df["a"].str.replace(":composite", "")
        df["b"] = df["b"].str.replace(":composite", "")
    e = pd.read_csv(ART / "drive/union_extra_k10/f_backblaze_seagate_gap2d/paired_targets.csv")
    e["a"] = e["a"].str.replace(":composite", "")
    e["b"] = e["b"].str.replace(":composite", "")

    def ci(s):
        s = s.iloc[0]
        return float(s.point_diff), float(s.low), float(s.high)

    tests = [("H1 mRMR int.", ci(pi[(pi.a == "srs_c_val") & (pi.b == "mrmr")]),
              ci(paired[(paired.a == "srs_ap_t") & (paired.b == "mrmr") & (paired.k == 10) & (paired.metric == "internal")])),
             ("H1 mRMR tgt.", ci(pt[(pt.a == "srs_c_val") & (pt.b == "mrmr") & (pt.target == "mean_over_targets")]),
              ci(paired[(paired.a == "srs_ap_t") & (paired.b == "mrmr") & (paired.k == 10) & (paired.metric == "mean_target")])),
             ("H2 MMRFS int.", ci(pi[(pi.a == "srs_c_val") & (pi.b == "mmrfs")]),
              ci(paired[(paired.a == "srs_ap_t") & (paired.b == "mmrfs") & (paired.k == 10) & (paired.metric == "internal")])),
             ("H2 MMRFS tgt.", ci(pt[(pt.a == "srs_c_val") & (pt.b == "mmrfs") & (pt.target == "mean_over_targets")]),
              ci(paired[(paired.a == "srs_ap_t") & (paired.b == "mmrfs") & (paired.k == 10) & (paired.metric == "mean_target")])),
             ("H2 L1 int.", ci(pi[(pi.a == "srs_c_val") & (pi.b == "l1_logistic")]),
              ci(paired[(paired.a == "srs_ap_t") & (paired.b == "l1_logistic") & (paired.k == 10) & (paired.metric == "internal")])),
             ("H2 L1 tgt.", ci(pt[(pt.a == "srs_c_val") & (pt.b == "l1_logistic") & (pt.target == "mean_over_targets")]),
              ci(paired[(paired.a == "srs_ap_t") & (paired.b == "l1_logistic") & (paired.k == 10) & (paired.metric == "mean_target")])),
             ("H3 SRS tgt.", ci(e[(e.a == "srs_c_unknown_val") & (e.b == "mrmr") & (e.target == "mean_over_targets")]),
              ci(paired[(paired.a == "srs_ap") & (paired.b == "mrmr") & (paired.k == 10) & (paired.metric == "mean_target")]))]
    out = []
    for i, (name, a_pre, a_post) in enumerate(tests):
        for off, (vals, filled) in zip((-0.16, 0.16), ((a_pre, False), (a_post, True))):
            ax.hlines(i + off, vals[1], vals[2], color=S.TEAL if filled else S.GREY, lw=0.9)
            ax.scatter(vals[0], i + off, s=13, marker="o", facecolor=S.TEAL if filled else "white", edgecolor=S.TEAL if filled else S.GREY,
                       lw=0.9, zorder=3)
        out.append({"test": name, "pre_diff": a_pre[0], "pre_low": a_pre[1], "pre_high": a_pre[2], "post_diff": a_post[0],
                    "post_low": a_post[1], "post_high": a_post[2]})
    ax.axvline(0, color=S.INK2, lw=0.6, ls=(0, (2, 2)))
    ax.set_yticks(range(len(tests)), [t[0] for t in tests], fontsize=5.0)
    ax.set_ylim(len(tests) - 0.4, -0.6)
    ax.set_xlabel("AUPRC difference, K = 10")
    ax.legend(handles=[Line2D([], [], marker="o", ls="-", lw=0.9, mfc="white", mec=S.GREY, color=S.GREY, label="pre-specified (lik.)"),
                       Line2D([], [], marker="o", ls="-", lw=0.9, color=S.TEAL, label="post hoc (AUPRC)")],
              fontsize=4.7, loc="upper center", bbox_to_anchor=(0.2, -0.22), ncol=2, handletextpad=0.3, columnspacing=0.6, handlelength=1.4)
    ax.grid(axis="x")
    S.title(ax, "Pre-registered tests")
    S.panel_label(ax, "d", dx=-0.62)
    src("x7d_hypotheses", pd.DataFrame(out))
    S.save(fig, "fig_x7_drives")


if __name__ == "__main__":
    for name in sys.argv[1:] or ["x3", "x4", "x5", "x6", "x7"]:
        globals()[f"fig_{name}"]()
        print("built", name)
