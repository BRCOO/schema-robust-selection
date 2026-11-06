"""Experiment figures E6-E9 (imported by figures.py)."""
from __future__ import annotations

import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D

import data
import style as S
from figs_b import method_style, src
from style import LABEL

RNG = np.random.default_rng(2)


# ================================================================================================= E6
def fig_e6() -> None:
    """Contract. Claim: the edge over mRMR and NMIFS grows with the budget K and holds on large pools; it comes from the
    heterogeneity of the union, not from its size: a single generator's pool of twice the size does worse. Evidence:
    (a) mean target AUPRC vs K, (b) grouped bars on the large pools, (c) dot plot single-generator pools vs union,
    (d) heatmap of the edge per setting and K. Data: union_ap rows/paired, restricted_pools rows."""
    r = data.rows_all()
    p = data.paired_all()
    fig = plt.figure(figsize=(S.FULL_W, 2.25))
    gs = GridSpec(1, 4, width_ratios=[0.95, 1.1, 1.1, 1.15], wspace=0.6, left=0.06, right=0.965, top=0.84, bottom=0.22)

    # (a) target AUPRC vs K (standard pools)
    ax = fig.add_subplot(gs[0])
    std = r[~r.large]
    rows = []
    for m in ("srs_ap_t", "srs_ap", "nmifs", "mrmr", "l1_dropout", "mrmr_filter_known"):
        g = std[std.registry == m].groupby("k").mean_target_auprc.mean()
        sty = method_style(m)
        ax.plot(g.index, g.values, color=sty["color"], ls=sty["ls"], marker=sty["marker"], ms=3, mfc=sty["mfc"], mec=sty["color"],
                lw=sty["lw"], label=LABEL[m])
        rows.append(pd.DataFrame({"method": LABEL[m], "k": g.index, "mean_target": g.values}))
    ax.set_xticks([5, 10, 20])
    ax.set_xscale("log", base=2)
    ax.set_xticks([5, 10, 20], ["5", "10", "20"])
    ax.minorticks_off()
    ax.set_xlabel("budget $K$")
    ax.set_ylabel("mean target AUPRC (4 settings)")
    ax.legend(fontsize=4.7, loc="upper center", bbox_to_anchor=(0.45, -0.23), ncol=3, handlelength=1.5, columnspacing=0.5,
              handletextpad=0.3)
    ax.grid(axis="y")
    S.title(ax, "Budget")
    S.panel_label(ax, "a", dx=-0.3)
    src("e6a_budget", pd.concat(rows))

    # (b) grouped bars on the large pools
    ax = fig.add_subplot(gs[1])
    large = r[r.large]
    groups = [("union_large_k10", "f_mimic_iv_gap1h", "V1h K10"), ("union_large_k20", "f_mimic_iv_gap1h", "V1h K20"),
              ("union_large_k10", "f_mimic_iv_aki_gap6h", "AKI K10"), ("union_large_k20", "f_mimic_iv_aki_gap6h", "AKI K20")]
    meths = ["srs_ap_t", "nmifs", "mrmr", "l1_dropout"]
    w = 0.2
    out = []
    for gi, (root, label, text) in enumerate(groups):
        for mi, m in enumerate(meths):
            v = float(large[(large.root == root) & (large.label == label) & (large.registry == m)].mean_target_auprc.iloc[0])
            ax.bar(gi + (mi - 1.5) * w, v - 0.6, bottom=0.6, width=w * 0.92, color=S.METHOD_COLOR[m], edgecolor="white", lw=0.3,
                   label=LABEL[m] if gi == 0 else None)
            out.append({"group": text, "method": LABEL[m], "mean_target": v})
    ax.set_xticks(range(len(groups)), [g[2].replace(" ", "\n") for g in groups], fontsize=5.6)
    ax.set_ylim(0.6, 0.86)
    ax.set_ylabel("mean target AUPRC")
    ax.legend(ncol=2, fontsize=5.0, loc="upper left", handlelength=0.9, columnspacing=0.6, borderpad=0.1)
    ax.grid(axis="y")
    S.title(ax, "Large pools")
    S.panel_label(ax, "b", dx=-0.28)
    src("e6b_large_pools", pd.DataFrame(out))

    # (c) single-generator pools vs union (dot plot)
    ax = fig.add_subplot(gs[2])
    rp = data.restricted()
    union = r[(r.root == "union") & (r.registry == "srs_ap_t")].set_index("label")
    rows = []
    y = 0
    ylabels = []
    for label, task in (("f_mimic_iv_gap1h", "Vaso 1h"), ("f_mimic_iv_aki_gap6h", "AKI")):
        for metric, mtext in (("internal_auprc", "internal"), ("mean_target_auprc", "targets")):
            vals = {"union (~510)": float(union.loc[label, metric])}
            for gen, gtext in (("generator_erminer", "ERMiner (1,000)"), ("generator_prefixspan", "PrefixSpan (1,000)")):
                s = rp[(rp.restrict_dir == gen) & (rp.registry == "srs_ap_t") & (rp.run.str.contains(label))]
                vals[gtext] = float(s[metric].iloc[0])
            xs = list(vals.values())
            ax.plot([min(xs), max(xs)], [y, y], color=S.GREY_PALE, lw=2.4, zorder=1, solid_capstyle="round")
            for (name, v), (color, marker) in zip(vals.items(), ((S.TEAL, "o"), (S.BLUE, "s"), (S.SKY, "^"))):
                ax.scatter(v, y, s=16, color=color, marker=marker, zorder=3, label=name if y == 0 else None)
            rows.append({"task": task, "metric": mtext, **vals})
            ylabels.append(f"{task}, {mtext}")
            y += 1
    ax.set_yticks(range(len(ylabels)), ylabels, fontsize=5.5)
    ax.set_ylim(len(ylabels) - 0.4, -0.6)
    ax.set_xlabel("AUPRC of SRS+T ($K = 10$)")
    ax.legend(fontsize=4.9, loc="upper center", bbox_to_anchor=(0.4, -0.22), ncol=2, handletextpad=0.1, columnspacing=0.6)
    ax.grid(axis="x")
    S.title(ax, "Heterogeneity, not size")
    S.panel_label(ax, "c", dx=-0.42)
    src("e6c_single_generator", pd.DataFrame(rows))

    # (d) edge heatmap per setting x K
    ax = fig.add_subplot(gs[3])
    settings = [("f_mimic_iv_gap1h", False, "V1h"), ("f_mimic_iv_gap2h", False, "V2h"), ("f_mimic_iv_gap4h", False, "V4h"),
                ("f_mimic_iv_aki_gap6h", False, "AKI"), ("f_mimic_iv_gap1h", True, "V1h-L"), ("f_mimic_iv_aki_gap6h", True, "AKI-L")]
    ks = [5, 10, 20]
    mat = np.full((len(settings), 6), np.nan)
    ver = np.zeros_like(mat)
    for i, (label, is_large, _) in enumerate(settings):
        for bi, b in enumerate(("mrmr", "nmifs")):
            for ki, k in enumerate(ks):
                root = {5: "union_k5", 10: "union", 20: "union_k20"}[k] if not is_large else {10: "union_large_k10", 20: "union_large_k20"}.get(k)
                if root is None:
                    continue
                s = p[(p.root == root) & (p.label == label) & (p.a == "srs_ap_t") & (p.b == b) & (p.metric == "mean_target")]
                mat[i, bi * 3 + ki], ver[i, bi * 3 + ki] = s.point_diff.iloc[0], s.verdict.iloc[0]
    lim = 0.06
    ax.imshow(np.clip(np.nan_to_num(mat, nan=0), -lim, lim), cmap=S.DIV, vmin=-lim, vmax=lim, aspect="auto")
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            if np.isnan(mat[i, j]):
                ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, color=S.GREY_PALE, lw=0))
                continue
            star = "*" if ver[i, j] == 1 else ("°" if ver[i, j] == -1 else "")
            ax.text(j, i, f"{100 * mat[i, j]:+.1f}{star}", ha="center", va="center", fontsize=4.9,
                    color="white" if abs(mat[i, j]) > 0.035 else S.INK)
    ax.axvline(2.5, color="white", lw=2.0)
    ax.set_xticks(range(6), ["K5", "K10", "K20"] * 2, fontsize=5.3)
    ax.set_yticks(range(len(settings)), [s[2] for s in settings], fontsize=5.5)
    ax.text(1, -0.85, "vs mRMR", ha="center", fontsize=5.4, color=S.INK2)
    ax.text(4, -0.85, "vs NMIFS", ha="center", fontsize=5.4, color=S.INK2)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xlabel("target diff. ×100 (* significant)")
    ax.set_title("Edge by setting and budget", loc="left", fontsize=6.8, fontweight="bold", y=1.1)
    S.panel_label(ax, "d", dx=-0.25, dy=1.1)
    src("e6d_edge_heatmap", pd.DataFrame(mat, index=[s[2] for s in settings],
                                         columns=[f"{b}_K{k}" for b in ("mrmr", "nmifs") for k in ks]).reset_index())
    S.save(fig, "fig_e6_budget")


# ================================================================================================= E7
def fig_e7() -> None:
    """Contract. Claim: selection time is linear in the number of candidates (10^5 candidates in about two hours), the
    surrogate chains are quadratic but computed once, the AUPRC decision adds a constant 5-13 s, and registries execute as
    SQL in seconds, linear until the in-memory database runs out of memory. Evidence: (a) log-log runtime vs |C|,
    (b) 100% stacked phase breakdown, (c) lollipop of the AUPRC-decision overhead, (d) bubble chart of SQL time vs episodes.
    Data: selection_scalability/selection_scalability.csv; sql_execution/union/*/scale.csv. Lower = better."""
    sc = pd.read_csv(S.ART / "selection_scalability/selection_scalability.csv")
    sz = sc[(sc.axis == "size") & (sc.status == "ok")]
    fig = plt.figure(figsize=(S.FULL_W, 2.25))
    gs = GridSpec(1, 4, width_ratios=[1.25, 0.95, 0.9, 1.1], wspace=0.55, left=0.06, right=0.97, top=0.84, bottom=0.22)

    # (a) runtime vs |C|
    ax = fig.add_subplot(gs[0])
    lines = [("srs_c_v", "SRS+T, fast (both $q$)", S.TEAL, "-", "o"), ("srs_ap_v", "SRS+T, AUPRC decision", S.TEAL_DARK, (0, (1, 1)), "o"),
             ("srs_c_exact", "SRS+T, exact greedy", "#7CC7B6", (0, (3, 1.5)), "s"), ("fs_cv", "schema-unaware, fast", S.TEAL_LIGHT, "-", "D"),
             ("mrmr", "mRMR", S.SKY, "-", "^"), ("cmim", "CMIM", S.BLUE, "-", "v"), ("jmi", "JMI", "#7E8FBF", "-", "<"),
             ("l1_logistic", "L1", S.GREY, "-", ">"), ("mmrfs", "MMRFS", S.SAND, "-", "x")]
    rows = []
    for m, text, color, ls, marker in lines:
        s = sz[sz.method == m].sort_values("n_candidates")
        if s.empty:
            continue
        ax.plot(s.n_candidates, s.seconds, color=color, ls=ls, marker=marker, ms=2.4, lw=1.3 if m.startswith("srs_c_v") else 1.0,
                label=text, mfc=color)
        rows.append(s[["method", "n_candidates", "seconds", "peak_mb"]])
    xs = np.array([500, 1e5])
    ax.plot(xs, 4.0 * xs / 500 * 6, color=S.INK2, lw=0.5, ls=(0, (2, 2)))
    ax.text(3.0e4, 4.0 * 3e4 / 500 * 6 * 0.35, r"$\propto |C|$", fontsize=5.4, color=S.INK2, rotation=24)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("candidates |C|")
    ax.set_ylabel("selection time (s)")
    ax.legend(fontsize=4.5, loc="upper center", bbox_to_anchor=(0.45, -0.22), handlelength=1.6, ncol=3, columnspacing=0.5,
              handletextpad=0.3, labelspacing=0.2)
    ax.grid(which="major")
    S.title(ax, "Selection time")
    S.panel_label(ax, "a", dx=-0.22)
    src("e7a_runtime", pd.concat(rows))

    # (b) phase breakdown
    ax = fig.add_subplot(gs[1])
    v = sz[sz.method == "srs_c_v"].set_index("n_candidates")
    sizes = [500, 2274, 10000, 100000]
    phases = [("phase_chains", "surrogate chains", S.SAND), ("phase_select_q0", "selection, $q=0$", S.TEAL_LIGHT),
              ("phase_select_q25", "selection, $q=0.25$", S.TEAL)]
    bottom = np.zeros(len(sizes))
    for col, text, color in phases:
        share = np.array([v.loc[n, col] / v.loc[n, "seconds"] for n in sizes])
        ax.bar(range(len(sizes)), share, bottom=bottom, color=color, width=0.62, edgecolor="white", lw=0.4, label=text)
        bottom += share
    for k, n in enumerate(sizes):
        t = v.loc[n, "seconds"]
        ax.text(k, 1.02, f"{t:.0f} s" if t < 600 else f"{t / 3600:.2f} h", ha="center", va="bottom", fontsize=5.0, color=S.INK2)
    ax.set_xticks(range(len(sizes)), ["500", "2,274", "$10^4$", "$10^5$"], fontsize=5.6)
    ax.set_ylim(0, 1.12)
    ax.set_yticks([0, 0.5, 1.0], ["0", "50%", "100%"])
    ax.set_xlabel("candidates |C|")
    ax.legend(fontsize=4.8, loc="upper center", bbox_to_anchor=(0.45, -0.24), ncol=2, handlelength=0.8, columnspacing=0.5)
    S.title(ax, "Where the time goes")
    S.panel_label(ax, "b", dx=-0.3)
    src("e7b_phases", v.loc[sizes, [c for c, _, _ in phases] + ["seconds"]].reset_index())

    # (c) AUPRC-decision overhead
    ax = fig.add_subplot(gs[2])
    ap = sz[sz.method == "srs_ap_v"].set_index("n_candidates").seconds
    lk = sz[sz.method == "srs_c_v"].set_index("n_candidates").seconds
    ns = sorted(ap.index)
    rel = [100 * (ap[n] - lk[n]) / lk[n] for n in ns]
    absd = [ap[n] - lk[n] for n in ns]
    ax.vlines(ns, 0, rel, color=S.TEAL_LIGHT, lw=1.6)
    ax.scatter(ns, rel, s=14, color=S.TEAL, zorder=3)
    for n, rv, a in zip(ns, rel, absd):
        ax.text(n, rv + 1.8, f"+{a:.0f} s", ha="center", fontsize=4.8, color=S.INK2)
    ax.set_xscale("log")
    ax.set_xlim(330, 3.2e4)
    ax.set_ylim(0, 48)
    ax.set_xlabel("candidates |C|")
    ax.set_ylabel("extra time of AUPRC decision (%)")
    S.title(ax, "AUPRC decision cost")
    S.panel_label(ax, "c", dx=-0.35)
    src("e7c_ap_overhead", pd.DataFrame({"n_candidates": ns, "relative_pct": rel, "seconds": absd}))

    # (d) SQL bubbles
    ax = fig.add_subplot(gs[3])
    rows = []
    for label, task, marker in (("f_mimic_iv_gap1h", "Vaso 1h", "o"), ("f_mimic_iv_aki_gap6h", "AKI", "s")):
        s = pd.read_csv(S.ART / "sql_execution/union" / label / "scale.csv")
        for col, color, text in (("seconds_tokens", S.TEAL, "event tokens"), ("seconds_registry", S.BLUE, "registry")):
            ax.scatter(s.n_episodes, s[col], s=8 + s.n_obs_rows / 1.2e6, marker=marker, facecolor=color, alpha=0.75, edgecolor="white", lw=0.4)
            ax.plot(s.n_episodes, s[col], color=color, lw=0.6, alpha=0.6)
        s["task"] = task
        rows.append(s)
    big = rows[1].iloc[-1]
    ax.annotate("in-memory DB\nfills 21 GB", (big.n_episodes, big.seconds_tokens), xytext=(1.6e4, 60), fontsize=4.8, color=S.WARM,
                arrowprops=dict(arrowstyle="-", color=S.WARM, lw=0.5))
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("eICU episodes (replicated)")
    ax.set_xlim(4e3, 1.3e6)
    ax.set_ylim(0.07, 900)
    ax.set_ylabel("seconds")
    ax.legend(handles=[Line2D([], [], marker="o", ls="", color=S.TEAL, label="event tokens"),
                       Line2D([], [], marker="o", ls="", color=S.BLUE, label="registry"),
                       Line2D([], [], marker="o", ls="", mfc="white", mec=S.INK2, label="Vaso 1h"),
                       Line2D([], [], marker="s", ls="", mfc="white", mec=S.INK2, label="AKI")],
              fontsize=4.8, loc="lower right", ncol=2, handletextpad=0.1, columnspacing=0.5, borderpad=0.1)
    ax.grid(which="major")
    S.title(ax, "Registries as SQL (DuckDB)")
    S.panel_label(ax, "d", dx=-0.25)
    src("e7d_sql_scale", pd.concat(rows))
    S.save(fig, "fig_e7_efficiency")


# ================================================================================================= E8
def fig_e8() -> None:
    """Contract. Claim: the conclusions survive new patient splits, a tree decoder, demographic subgroups and changes of
    every hyper-parameter, and greedy is close to the exact optimum; the exception is NMIFS on re-split vasopressor 1 h.
    Evidence: (a) strip plot over five re-splits, (b) dot-range of decoder-swapped differences, (c) categorical heatmap of
    subgroup verdicts, (d) tornado of hyper-parameter changes, (e) ECDF of greedy/optimum ratios.
    Data: union_ap/union_resplit, union_robustness_ap, srs_sensitivity_ap, exact_small_pools."""
    fig = plt.figure(figsize=(S.FULL_W, 2.3))
    gs = GridSpec(1, 5, width_ratios=[1.05, 0.95, 1.05, 1.05, 0.85], wspace=0.62, left=0.055, right=0.985, top=0.84, bottom=0.22)

    # (a) re-splits strip plot
    ax = fig.add_subplot(gs[0])
    rr, _ = data.resplits()
    meths = ["srs_ap_t", "nmifs", "mrmr", "l1_dropout", "mrmr_filter_known"]
    rows = []
    for ti, task in enumerate(("Vaso 1h", "AKI")):
        for mi, m in enumerate(meths):
            v = rr[(rr.task == task) & (rr.registry == m)].mean_target_auprc.values
            x = ti * (len(meths) + 1) + mi
            ax.scatter(np.full(len(v), x) + RNG.uniform(-0.18, 0.18, len(v)), v, s=7, color=S.METHOD_COLOR[m], lw=0, alpha=0.9)
            ax.hlines(v.mean(), x - 0.32, x + 0.32, color=S.INK, lw=0.9)
            rows.append({"task": task, "method": LABEL[m], **{f"split{i + 1}": vv for i, vv in enumerate(v)}})
    ax.set_xticks([2, 8], ["Vaso 1h", "AKI"])
    ax.set_ylabel("mean target AUPRC")
    ax.legend(handles=[Line2D([], [], marker="o", ls="", color=S.METHOD_COLOR[m], label=LABEL[m]) for m in meths], fontsize=4.6,
              loc="upper right", ncol=1, handletextpad=0.05, borderpad=0.1, labelspacing=0.15)
    ax.grid(axis="y")
    S.title(ax, "Five new patient splits")
    S.panel_label(ax, "a", dx=-0.3)
    src("e8a_resplits", pd.DataFrame(rows))

    # (b) decoder swap: GBDT differences
    ax = fig.add_subplot(gs[1])
    rb = data.robustness()
    y = 0
    labels = []
    rows = []
    for label, task in zip(data.STANDARD, ("V1h", "V2h", "V4h", "AKI")):
        dp = rb[label]["decoder_paired"]
        for b, color, marker in (("mrmr", S.SKY, "^"), ("nmifs", S.BLUE, "s")):
            s = dp[(dp.a == "srs_ap_t") & (dp.b == b) & (dp.decoder == "gbdt") & (dp.target == "mean_over_targets")]
            if s.empty:
                continue
            ax.hlines(y, s.low, s.high, color=color, lw=0.9)
            ax.scatter(s.point_diff, y, color=color, marker=marker, s=10, zorder=3)
            rows.append({"setting": task, "b": b, "diff": float(s.point_diff.iloc[0]), "low": float(s.low.iloc[0]), "high": float(s.high.iloc[0])})
            y += 1
        labels.append((y - 1.5, task))
        y += 0.6
    ax.set_yticks([l[0] for l in labels], [l[1] for l in labels], fontsize=5.5)
    ax.set_ylim(y - 0.4, -0.8)
    ax.axvline(0, color=S.INK2, lw=0.6, ls=(0, (2, 2)))
    ax.set_xlabel("SRS+T $-$ baseline, trees")
    ax.legend(handles=[Line2D([], [], marker="^", ls="", color=S.SKY, label="vs mRMR"), Line2D([], [], marker="s", ls="", color=S.BLUE, label="vs NMIFS")],
              fontsize=4.8, loc="upper center", bbox_to_anchor=(0.45, -0.22), ncol=2, handletextpad=0.05, columnspacing=0.6)
    tau = [rb[l]["manifest"]["rank_agreement"]["mean_target_auprc"]["kendall_tau"] for l in data.STANDARD]
    ax.text(0.02, 1.0, f"Kendall τ {min(tau):.2f}–{max(tau):.2f}", transform=ax.transAxes, fontsize=4.9, va="top", color=S.INK2)
    S.title(ax, "Tree decoder")
    S.panel_label(ax, "b", dx=-0.3)
    src("e8b_decoder", pd.DataFrame(rows))

    # (c) subgroup verdict heatmap
    ax = fig.add_subplot(gs[2])
    cols, mats = [], []
    first = rb[data.STANDARD[0]]["subgroup_paired"]
    sub_keys = first[(first.a == "srs_ap_t") & (first.b == "mrmr")][["cohort", "subgroup"]].drop_duplicates()
    sub_keys = [tuple(x) for x in sub_keys.values]
    mat = np.zeros((len(sub_keys), 8))
    for bi, b in enumerate(("mrmr", "nmifs")):
        for li, label in enumerate(data.STANDARD):
            sp = rb[label]["subgroup_paired"]
            for si, (cohort, sg) in enumerate(sub_keys):
                s = sp[(sp.a == "srs_ap_t") & (sp.b == b) & (sp.cohort == cohort) & (sp.subgroup == sg)]
                mat[si, bi * 4 + li] = 1 if float(s.low.iloc[0]) > 0 else (-1 if float(s.high.iloc[0]) < 0 else 0)
    from matplotlib.colors import ListedColormap
    ax.imshow(mat, cmap=ListedColormap([S.WARM, "#EFEFEF", S.TEAL]), vmin=-1, vmax=1, aspect="auto")
    ax.axvline(3.5, color="white", lw=2.0)
    for x in range(8):
        ax.axvline(x + 0.5, color="white", lw=0.4)
    for yy in range(len(sub_keys)):
        ax.axhline(yy + 0.5, color="white", lw=0.4)
    cohort_name = {"mimic_iv_test": "MIMIC", "eicu": "eICU", "hirid": "HiRID"}
    ax.set_yticks(range(len(sub_keys)), [f"{cohort_name.get(c, c)} {g}" for c, g in sub_keys], fontsize=4.5)
    ax.set_xticks(range(8), ["1", "2", "4", "A"] * 2, fontsize=5.2)
    ax.text(1.5, -1.0, "vs mRMR", ha="center", fontsize=5.2, color=S.INK2)
    ax.text(5.5, -1.0, "vs NMIFS", ha="center", fontsize=5.2, color=S.INK2)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    nb = (mat[:, :4] == 1).sum(), (mat[:, :4] == -1).sum(), (mat[:, 4:] == 1).sum(), (mat[:, 4:] == -1).sum()
    ax.set_xlabel(f"better {nb[0]} / worse {nb[1]}   |   better {nb[2]} / worse {nb[3]}", fontsize=5.0)
    ax.set_title("Subgroups (sex, age)", loc="left", fontsize=6.8, fontweight="bold", y=1.1)
    S.panel_label(ax, "c", dx=-0.45, dy=1.1)
    src("e8c_subgroups", pd.DataFrame(mat, index=[f"{c} {g}" for c, g in sub_keys],
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
    src("e8d_sensitivity", tor)

    # (e) ECDF of greedy / optimum
    ax = fig.add_subplot(gs[4])
    ex = pd.concat([pd.read_csv(S.ART / f"exact_small_pools/rows_{t}.csv") for t in ("gap1h", "aki_gap6h")])
    rows = []
    for obj, color, ls, text in (("deployed_q0", S.TEAL, "-", "deployed, $q=0$"), ("deployed_q25", S.BLUE, (0, (3, 1.5)), "deployed, $q=0.25$"),
                                 ("analyzable", S.SAND, (0, (1, 1)), "analyzable")):
        v = np.sort(ex[ex.objective == obj].ratio.values)
        ax.step(v, np.arange(1, len(v) + 1) / len(v), where="post", color=color, ls=ls, lw=1.2, label=text)
        rows.append(pd.DataFrame({"objective": obj, "ratio": v}))
    ax.set_xlim(0.93, 1.002)
    ax.set_xlabel("greedy / optimum")
    ax.set_ylabel("share of 80 sub-pools")
    ax.legend(fontsize=4.6, loc="upper left", handlelength=1.6, borderpad=0.1)
    ax.grid()
    S.title(ax, "Near-optimal")
    S.panel_label(ax, "e", dx=-0.35)
    src("e8e_exact_ratio", pd.concat(rows))
    S.save(fig, "fig_e8_robustness")


# ================================================================================================= E9
def fig_e9() -> None:
    """Contract. Claim: in a second, pre-specified domain (drive failure across manufacturers) the pre-registered test with
    the likelihood decision replicated partially; with the AUPRC decision SRS+T is significantly better on the targets
    than every selector but L1 with schema dropout, and surrogates in the objective are essential. Evidence: (a) heatmap of
    paired differences per budget, (b) per-manufacturer grouped bars, (c) target AUPRC vs K for variants, (d) tile chart
    of the pre-registered hypotheses. Data: union_ap/drive_k*/ and drive/union_k10 (pre-specified, likelihood decision)."""
    rows, paired, abl = data.drives()
    fig = plt.figure(figsize=(S.FULL_W, 2.3))
    gs = GridSpec(1, 4, width_ratios=[1.3, 1.0, 0.9, 0.8], wspace=0.72, left=0.075, right=0.985, top=0.84, bottom=0.21)

    # (a) heatmap of paired differences
    ax = fig.add_subplot(gs[0])
    base = ["nmifs", "mrmr", "mmrfs", "cmim", "jmi", "l1_logistic", "l1_dropout", "mrmr_filter_known"]
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
            star = "*" if ver[i, j] == 1 else ("°" if ver[i, j] == -1 else "")
            ax.text(j, i, f"{100 * mat[i, j]:+.1f}{star}", ha="center", va="center", fontsize=4.8, color="white" if abs(mat[i, j]) > 0.07 else S.INK)
    for x in (1.5, 3.5):
        ax.axvline(x, color="white", lw=2.0)
    ax.set_xticks(range(6), ["int.", "tgt."] * 3, fontsize=5.2)
    for ki, k in enumerate((5, 10, 20)):
        ax.text(ki * 2 + 0.5, -0.9, f"$K={k}$" + (" (pre-spec.)" if k == 10 else ""), ha="center", fontsize=5.0, color=S.INK2)
    ax.set_yticks(range(len(base)), [LABEL[b] for b in base], fontsize=5.5)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xlabel("SRS+T $-$ baseline ×100 (* significant, ° worse)", fontsize=5.4)
    ax.set_title("Paired differences (drives)", loc="left", fontsize=6.8, fontweight="bold", y=1.1)
    S.panel_label(ax, "a", dx=-0.3, dy=1.1)
    src("e9a_drive_heatmap", pd.DataFrame(mat, index=[LABEL[b] for b in base],
                                          columns=[f"K{k}_{m}" for k in (5, 10, 20) for m in ("internal", "target")]).reset_index())

    # (b) per-manufacturer bars at K = 10
    ax = fig.add_subplot(gs[1])
    meths = ["srs_ap_t", "srs_ap", "nmifs", "mrmr", "l1_dropout"]
    mans = [("hgst_auprc", "HGST"), ("toshiba_auprc", "Toshiba"), ("wdc_auprc", "WDC")]
    w = 0.16
    out = []
    for mi, m in enumerate(meths):
        s = rows[(rows.registry == m) & (rows.k == 10)].iloc[0]
        vals = [s[c] for c, _ in mans]
        ax.bar(np.arange(len(mans)) + (mi - 2) * w, np.array(vals) - 0.4, bottom=0.4, width=w * 0.92, color=S.METHOD_COLOR[m],
               edgecolor="white", lw=0.3, label=LABEL[m])
        out.append({"method": LABEL[m], **{t: v for (_, t), v in zip(mans, vals)}})
    ax.set_xticks(range(len(mans)), [t for _, t in mans])
    ax.set_ylim(0.4, 0.9)
    ax.set_ylabel("target AUPRC ($K = 10$)")
    ax.legend(ncol=3, fontsize=4.8, loc="upper center", bbox_to_anchor=(0.5, -0.12), handlelength=0.8, columnspacing=0.5)
    ax.grid(axis="y")
    S.title(ax, "Per manufacturer")
    S.panel_label(ax, "b", dx=-0.3)
    src("e9b_manufacturers", pd.DataFrame(out))

    # (c) target AUPRC vs K for variants
    ax = fig.add_subplot(gs[2])
    ref = rows[rows.registry == "srs_ap_t"].set_index("k").mean_target_auprc
    series = [("srs_ap_t", ref.values, method_style("srs_ap_t"))]
    series.append(("srs_ap", rows[rows.registry == "srs_ap"].set_index("k").mean_target_auprc.loc[[5, 10, 20]].values, method_style("srs_ap")))
    for v, text, color, ls in (("srs_ap_t_nosur", "no surrogates in obj.", S.WARM, (0, (3, 1.5))), ("srs_ap_unaware", "no scenarios", S.SAND, (0, (1, 1)))):
        s = abl[(abl.a == "srs_ap_t") & (abl.b == v) & (abl.target == "mean_over_targets")].set_index("k").point_diff
        series.append((v, ref.loc[[5, 10, 20]].values - s.loc[[5, 10, 20]].values, dict(color=color, ls=ls, marker="o", lw=1.0, mfc="white")))
        LABEL.setdefault(v, text)
    series.append(("l1_dropout", rows[rows.registry == "l1_dropout"].set_index("k").mean_target_auprc.loc[[5, 10, 20]].values, method_style("l1_dropout")))
    out = []
    for m, vals, sty in series:
        ax.plot([5, 10, 20], vals, color=sty["color"], ls=sty["ls"], marker=sty["marker"], ms=3, mfc=sty.get("mfc", sty["color"]),
                mec=sty["color"], lw=sty["lw"], label={"srs_ap_t_nosur": "no surr. in obj.", "srs_ap_unaware": "no scenarios"}.get(m, LABEL[m]))
        out.append({"variant": m, "k5": vals[0], "k10": vals[1], "k20": vals[2]})
    ax.set_xscale("log", base=2)
    ax.set_xticks([5, 10, 20], ["5", "10", "20"])
    ax.minorticks_off()
    ax.set_xlabel("budget $K$")
    ax.set_ylabel("mean target AUPRC")
    ax.legend(fontsize=4.6, loc="upper center", bbox_to_anchor=(0.45, -0.22), ncol=2, handlelength=1.6, columnspacing=0.5,
              labelspacing=0.15)
    ax.grid(axis="y")
    S.title(ax, "What matters here")
    S.panel_label(ax, "c", dx=-0.38)
    src("e9c_variants", pd.DataFrame(out))

    # (d) pre-registered hypotheses
    ax = fig.add_subplot(gs[3])
    pre_dir = S.ART / "drive/union_k10/f_backblaze_seagate_gap2d"
    pi = pd.read_csv(pre_dir / "paired_internal.csv")
    pt = pd.read_csv(pre_dir / "paired_targets.csv")
    pt["a"] = pt["a"].str.replace(":composite", "")
    pt["b"] = pt["b"].str.replace(":composite", "")

    def pre(b, metric):
        if metric == "internal":
            s = pi[(pi.a == "srs_c_val") & (pi.b == b)]
        else:
            s = pt[(pt.a == "srs_c_val") & (pt.b == b) & (pt.target == "mean_over_targets")]
        return float(s.point_diff.iloc[0]), (1 if float(s.low.iloc[0]) > 0 else (-1 if float(s.high.iloc[0]) < 0 else 0))

    extra = S.ART / "drive/union_extra_k10/f_backblaze_seagate_gap2d"
    h3_pre = None
    if (extra / "paired_targets.csv").exists():
        e = pd.read_csv(extra / "paired_targets.csv")
        e["a"] = e["a"].str.replace(":composite", "")
        e["b"] = e["b"].str.replace(":composite", "")
        s = e[(e.a == "srs_c_unknown_val") & (e.b == "mrmr") & (e.target == "mean_over_targets")]
        if len(s):
            h3_pre = (float(s.point_diff.iloc[0]), 1 if float(s.low.iloc[0]) > 0 else (-1 if float(s.high.iloc[0]) < 0 else 0))

    def post(a, b, metric):
        s = paired[(paired.a == a) & (paired.b == b) & (paired.k == 10) & (paired.metric == metric)]
        return float(s.point_diff.iloc[0]), int(s.verdict.iloc[0])

    tests = [("H1 mRMR int.", ("mrmr", "internal")), ("H1 mRMR tgt.", ("mrmr", "mean_target")),
             ("H2 MMRFS int.", ("mmrfs", "internal")), ("H2 MMRFS tgt.", ("mmrfs", "mean_target")),
             ("H2 L1 int.", ("l1_logistic", "internal")), ("H2 L1 tgt.", ("l1_logistic", "mean_target")),
             ("H3 SRS tgt.", None)]
    grid = []
    for name, spec in tests:
        if spec is None:
            a_pre = h3_pre
            a_post = post("srs_ap", "mrmr", "mean_target")
        else:
            a_pre = pre(*spec)
            a_post = post("srs_ap_t", spec[0], spec[1])
        grid.append((name, a_pre, a_post))
    from matplotlib.colors import ListedColormap
    vmat = np.array([[g[1][1] if g[1] else 0, g[2][1]] for g in grid])
    ax.imshow(vmat, cmap=ListedColormap([S.WARM, "#EFEFEF", S.TEAL_LIGHT]), vmin=-1, vmax=1, aspect="auto")
    for i, (name, a_pre, a_post) in enumerate(grid):
        for j, val in enumerate((a_pre, a_post)):
            if val is None:
                continue
            ax.text(j, i, f"{100 * val[0]:+.1f}" + ("*" if val[1] == 1 else ""), ha="center", va="center", fontsize=5.0, color=S.INK)
    for x in (0.5,):
        ax.axvline(x, color="white", lw=2.0)
    for yy in range(len(grid)):
        ax.axhline(yy + 0.5, color="white", lw=0.6)
    ax.set_xticks([0, 1], ["pre-spec.\n(lik.)", "post hoc\n(AUPRC)"], fontsize=5.2)
    ax.set_yticks(range(len(grid)), [g[0] for g in grid], fontsize=5.0)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xlabel("$K = 10$, ×100, * significant", fontsize=5.2)
    S.title(ax, "Pre-registered tests")
    S.panel_label(ax, "d", dx=-0.62)
    src("e9d_hypotheses", pd.DataFrame([{"test": g[0], "pre_diff": g[1][0] if g[1] else None, "pre_verdict": g[1][1] if g[1] else None,
                                          "post_diff": g[2][0], "post_verdict": g[2][1]} for g in grid]))
    S.save(fig, "fig_e9_drives")
