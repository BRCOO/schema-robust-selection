"""Experiment figures E2-E5 (imported by figures.py)."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D

import data
import style as S
from style import LABEL

RNG = np.random.default_rng(1)
GEN_COLOR = {"ERMiner": S.TEAL_DARK, "PrefixSpan": "#3FAE9A", "Motif": S.BLUE, "Shapelet": S.SKY, "FMG": S.SAND}
GEN_ORDER = ["ERMiner", "PrefixSpan", "Motif", "Shapelet", "FMG"]
PROXY = {"objective": ("SRS estimate", S.TEAL), "nmifs": ("NMIFS score", S.BLUE), "mrmr": ("mRMR score", S.SKY)}
TASKS = [("f_mimic_iv_gap1h", "Vaso 1h"), ("f_mimic_iv_gap2h", "Vaso 2h"), ("f_mimic_iv_gap4h", "Vaso 4h"), ("f_mimic_iv_aki_gap6h", "AKI")]


def src(name: str, frame: pd.DataFrame) -> None:
    frame.to_csv(S.OUT_SRC / f"{name}.csv", index=False)


def method_style(m: str) -> dict:
    return {"srs_ap_t": dict(color=S.TEAL, ls="-", marker="o", lw=1.6, mfc=S.TEAL),
            "srs_ap": dict(color=S.TEAL_MID, ls=(0, (3, 1.5)), marker="o", lw=1.2, mfc="white"),
            "nmifs": dict(color=S.BLUE, ls="-", marker="s", lw=1.1, mfc=S.BLUE),
            "mrmr": dict(color=S.SKY, ls="-", marker="^", lw=1.1, mfc=S.SKY),
            "l1_dropout": dict(color=S.GREY, ls=(0, (1, 1)), marker="D", lw=1.0, mfc="white"),
            "mrmr_filter_known": dict(color=S.SAND, ls="-", marker="v", lw=1.0, mfc=S.SAND),
            "fs_cv": dict(color=S.GREY_LIGHT, ls="-", marker="x", lw=1.0, mfc=S.GREY_LIGHT),
            "srs_c_val": dict(color="#7CC7B6", ls=(0, (4, 1.5)), marker="o", lw=1.0, mfc="white")}[m]


# ================================================================================================= E2
def fig_e2() -> None:
    """Contract. Claim: on heterogeneous union pools, relevance/redundancy proxies rank candidates far worse than SRS's
    cross-validated estimate of deployed value, because generators fire at very different rates; selected registries mix
    generators. Evidence: (a) dumbbell within-generator vs union rank correlation per proxy and task, (b) violin of TRAIN
    firing rates by generator, (c) correlation by selection step, (d) generator composition of registries.
    Data: proxy_correlation/heldout_*.csv (steps 1-3, truth = VAL AUPRC gain); source_data/firing_rates.csv;
    registries.json of the K = 10 / 20 runs. Higher correlation = better proxy."""
    rows = []
    for label, task in TASKS:
        d = pd.read_csv(S.ART / "proxy_correlation" / f"heldout_{label}.csv")
        d = d[(d.step > 0) & (d.truth == "val_auprc")].dropna(subset=["spearman"])
        d["task"] = task
        rows.append(d)
    pc = pd.concat(rows, ignore_index=True)
    pc["where"] = np.where(pc.scope == "union", "union", "within")
    agg = pc.groupby(["proxy", "task", "where"]).spearman.mean().unstack().reset_index()
    src("e2a_proxy_dumbbell", agg)

    fig = plt.figure(figsize=(S.FULL_W, 2.3))
    gs = GridSpec(1, 4, width_ratios=[1.15, 1.2, 0.95, 1.15], wspace=0.5, left=0.075, right=0.985, top=0.83, bottom=0.2)

    # (a) dumbbell
    ax = fig.add_subplot(gs[0])
    ylab, y = [], 0
    for proxy in ("objective", "nmifs", "mrmr"):
        name, color = PROXY[proxy]
        for _, task in TASKS:
            row = agg[(agg.proxy == proxy) & (agg.task == task)].iloc[0]
            yy = y + 1.3 * (y // 4 + 1)
            ax.plot([row["within"], row["union"]], [yy, yy], color=color, lw=1.0, alpha=0.6, zorder=1)
            ax.scatter(row["within"], yy, s=13, facecolor="white", edgecolor=color, lw=0.9, zorder=2)
            ax.scatter(row["union"], yy, s=15, color=color, zorder=3)
            ylab.append(task)
            y += 1
    pos = [i + 1.3 * (i // 4 + 1) for i in range(12)]
    ax.set_yticks(pos, ylab, fontsize=5.5)
    ax.set_ylim(pos[-1] + 0.6, 0.0)
    for k, proxy in enumerate(("objective", "nmifs", "mrmr")):
        ax.text(0.305, pos[4 * k] - 0.72, PROXY[proxy][0], fontsize=5.6, color=PROXY[proxy][1], fontweight="bold", va="bottom")
    ax.set_xlim(0.3, 0.82)
    ax.set_xlabel("Spearman ρ with held-out gain")
    ax.legend(handles=[Line2D([], [], marker="o", ls="", mfc="white", mec=S.INK2, label="within a generator"),
                       Line2D([], [], marker="o", ls="", color=S.INK2, label="union pool")],
              loc="lower right", fontsize=5.3, handletextpad=0.1, borderpad=0.1)
    ax.grid(axis="x")
    S.title(ax, "Score fidelity")
    S.panel_label(ax, "a", dx=-0.3)

    # (b) violin of firing rates
    ax = fig.add_subplot(gs[1])
    fr = pd.read_csv(S.OUT_SRC / "firing_rates.csv")
    fr = fr[fr.task == "Vaso 1h"]
    vals = [fr[fr.generator == g].train_rate.values for g in GEN_ORDER]
    parts = ax.violinplot(vals, positions=range(len(GEN_ORDER)), vert=False, widths=0.85, showextrema=False, bw_method=0.25)
    for body, g in zip(parts["bodies"], GEN_ORDER):
        body.set_facecolor(GEN_COLOR[g])
        body.set_edgecolor("none")
        body.set_alpha(0.55)
    for k, (g, v) in enumerate(zip(GEN_ORDER, vals)):
        ax.scatter(v, np.full(len(v), k) + RNG.uniform(-0.12, 0.12, len(v)), s=1.5, color=GEN_COLOR[g], lw=0, alpha=0.8)
        ax.text(1.02, k, f"n={len(v)}", fontsize=5.0, va="center", ha="left", color=S.INK2, transform=ax.get_yaxis_transform())
    ax.set_yticks(range(len(GEN_ORDER)), GEN_ORDER, fontsize=5.8)
    ax.set_ylim(len(GEN_ORDER) - 0.4, -0.6)
    ax.set_xlim(-0.02, 1.0)
    ax.set_xlabel("TRAIN firing rate (vasopressor 1 h pool)")
    S.title(ax, "Heterogeneous candidates")
    S.panel_label(ax, "b", dx=-0.25)
    src("e2b_firing_rates", fr)

    # (c) correlation by step
    ax = fig.add_subplot(gs[2])
    u = pc[pc.scope == "union"].groupby(["proxy", "step", "task"]).spearman.mean().reset_index()
    rows = []
    for proxy in ("objective", "nmifs", "mrmr"):
        name, color = PROXY[proxy]
        g = u[u.proxy == proxy].groupby("step").spearman
        mean, lo, hi = g.mean(), g.min(), g.max()
        ax.fill_between(mean.index, lo, hi, color=color, alpha=0.15, lw=0)
        ax.plot(mean.index, mean.values, color=color, marker="o", ms=3, lw=1.3)
        ax.text(3.08, mean.values[-1] + {"objective": 0.0, "nmifs": 0.018, "mrmr": -0.018}[proxy], name.split()[0], fontsize=5.3,
                color=color, va="center")
        rows.append(pd.DataFrame({"proxy": proxy, "step": mean.index, "mean": mean.values, "min": lo.values, "max": hi.values}))
    ax.set_xticks([1, 2, 3])
    ax.set_xlim(0.85, 3.6)
    ax.set_xlabel("forward-selection step")
    ax.set_ylabel("ρ on the union pool")
    ax.grid(axis="y")
    S.title(ax, "Across steps")
    S.panel_label(ax, "c", dx=-0.3)
    src("e2c_proxy_steps", pd.concat(rows))

    # (d) generator composition
    ax = fig.add_subplot(gs[3])
    comp = []
    for k in (10, 20):
        root = "union" if k == 10 else "union_k20"
        for m in ("srs_ap_t", "nmifs", "mrmr", "l1_dropout"):
            counts = {g: 0 for g in GEN_ORDER}
            for label in data.STANDARD:
                for pid in data.registries(root, label)[m]:
                    counts[data.generator_of(pid)] = counts.get(data.generator_of(pid), 0) + 1
            total = sum(counts.values())
            comp.append({"k": k, "method": m, **{g: counts[g] / total for g in GEN_ORDER},
                         "n_generators": np.mean([len({data.generator_of(p) for p in data.registries(root, l)[m]}) for l in data.STANDARD])})
    comp = pd.DataFrame(comp)
    ylabels = []
    for i, row in comp.iterrows():
        left = 0.0
        for g in GEN_ORDER:
            ax.barh(i + (0.5 if row.k == 20 else 0), row[g], left=left, color=GEN_COLOR[g], height=0.72, edgecolor="white", lw=0.4)
            left += row[g]
        ax.text(1.02, i + (0.5 if row.k == 20 else 0), f"{row.n_generators:.1f}", fontsize=5.0, va="center", color=S.INK2)
        ylabels.append(f"{LABEL[row.method]}")
    ypos = [i + (0.5 if k == 20 else 0) for i, k in enumerate(comp.k)]
    ax.set_yticks(ypos, ylabels, fontsize=5.6)
    ax.set_ylim(max(ypos) + 0.6, -0.6)
    ax.text(1.02, -0.85, "gens", fontsize=5.0, color=S.INK2, fontweight="bold")
    ax.axhline(3.75, color=S.INK2, lw=0.4, ls=(0, (2, 2)))
    ax.text(0.5, 3.75, "K=10 above, K=20 below", fontsize=4.8, ha="center", va="center", color=S.INK2,
            bbox=dict(facecolor="white", edgecolor="none", pad=0.6))
    ax.set_xlim(0, 1)
    ax.set_xlabel("share of selected programs")
    ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=GEN_COLOR[g]) for g in GEN_ORDER], labels=GEN_ORDER, ncol=3,
              loc="upper center", bbox_to_anchor=(0.45, -0.22), fontsize=5.0, handlelength=0.8, columnspacing=0.6, handletextpad=0.3)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    S.title(ax, "Generators in a registry")
    S.panel_label(ax, "d", dx=-0.3)
    src("e2d_generator_composition", comp)
    S.save(fig, "fig_e2_mechanism")


# ================================================================================================= E3
def fig_e3() -> None:
    """Contract. Claim: the deployed-value objective gives most of the gain over mRMR; deciding by AUPRC and the schema
    scenarios add smaller, target-specific gains (scenarios help where programs lose executability). Evidence: (a) waterfall
    of mean target AUPRC along the component path, (b) per-target effect heatmap of each component, (c) slope chart of the
    decision per run, (d) bubble scatter of the scenario gain against executability. Data: union_ap rows.csv,
    ablation_paired.csv, ablation_manifest.json. Higher = better."""
    r = data.rows_all()
    a = data.ablation_all()
    mean = lambda reg: r[r.registry == reg].mean_target_auprc.mean()  # noqa: E731
    ref = r[r.registry == "srs_ap_t"].set_index("name")
    s_un = a[(a.a == "srs_ap_t") & (a.b == "srs_ap_unaware") & (a.target == "mean_over_targets")].set_index("name")
    v_unaware = float((ref.loc[s_un.index, "mean_target_auprc"] - s_un.point_diff).mean())
    steps = [("mRMR", mean("mrmr")), ("wrapper\n(lik.)", mean("fs_cv")), ("+AUPRC\ndecision", v_unaware),
             ("+schema\nscenarios", mean("srs_ap_t"))]
    src("e3a_waterfall", pd.DataFrame(steps, columns=["step", "mean_target_auprc"]))

    fig = plt.figure(figsize=(S.FULL_W, 2.3))
    gs = GridSpec(1, 4, width_ratios=[0.9, 1.45, 0.72, 1.05], wspace=0.78, left=0.065, right=0.985, top=0.83, bottom=0.25)

    # (a) waterfall
    ax = fig.add_subplot(gs[0])
    prev = None
    for i, (name, v) in enumerate(steps):
        if prev is None:
            ax.bar(i, v - 0.66, bottom=0.66, color=S.SKY, width=0.62)
        else:
            d = v - prev
            ax.bar(i, d, bottom=prev, color=S.TEAL if d >= 0 else S.WARM, width=0.62)
            ax.plot([i - 1 + 0.31, i - 0.31], [prev, prev], color=S.INK2, lw=0.5, ls=(0, (1.5, 1.5)))
            ax.text(i, max(v, prev) + 0.0012, f"{d:+.3f}", ha="center", va="bottom", fontsize=5.4, color=S.TEAL_DARK)
        prev = v
    ax.text(0, steps[0][1] + 0.0012, f"{steps[0][1]:.3f}", ha="center", va="bottom", fontsize=5.4, color=S.INK2)
    ax.text(3, steps[-1][1] + 0.0065, f"= {steps[-1][1]:.3f}", ha="center", va="bottom", fontsize=5.4, color=S.TEAL_DARK, fontweight="bold")
    ax.set_xticks(range(len(steps)), [s[0].replace("\n", " ") for s in steps], fontsize=5.0, rotation=32, ha="right",
                  rotation_mode="anchor")
    ax.set_ylim(0.66, 0.722)
    ax.set_ylabel("mean target AUPRC (16 runs)")
    S.title(ax, "Gain path")
    S.panel_label(ax, "a", dx=-0.25)

    # (b) per-target effect heatmap
    ax = fig.add_subplot(gs[1])
    cols = ["internal", "eicu", "hirid", "drop_labs", "drop_ventilation", "drop_bp", "drop_urine", "vitals_only"]
    comps = [("AUPRC decision", None), ("schema scenarios", "srs_ap_unaware"), ("surrogates in obj.", "srs_ap_t_nosur")]
    mat = np.zeros((len(comps), len(cols)))
    sig = np.zeros_like(mat)
    for i, (name, abl) in enumerate(comps):
        for j, c in enumerate(cols):
            if abl is None:
                key = "internal_auprc" if c == "internal" else f"{c}_auprc"
                x = r[r.registry == "srs_ap_t"].set_index("name")[key] - r[r.registry == "srs_c_val"].set_index("name")[key]
                mat[i, j] = x.mean()
                sig[i, j] = np.nan
            else:
                sub = a[(a.a == "srs_ap_t") & (a.b == abl) & (a.target == c)]
                mat[i, j] = sub.point_diff.mean()
                sig[i, j] = (sub.verdict == 1).sum() - (sub.verdict == -1).sum()
    lim = 0.02
    ax.imshow(np.clip(mat, -lim, lim), cmap=S.DIV, vmin=-lim, vmax=lim, aspect="auto")
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax.text(j, i, f"{100 * mat[i, j]:+.1f}", ha="center", va="center", fontsize=5.4,
                    color="white" if abs(mat[i, j]) > 0.012 else S.INK)
    ax.set_xticks(range(len(cols)), ["internal"] + [data.TARGET_LABEL[c] for c in cols[1:]], rotation=35, ha="right", fontsize=5.4)
    ax.set_yticks(range(len(comps)), [c[0] for c in comps], fontsize=5.6)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.axvline(0.5, color="white", lw=1.6)
    ax.axvline(2.5, color="white", lw=1.6)
    S.title(ax, "Component effect (×100)")
    S.panel_label(ax, "b", dx=-0.32)
    src("e3b_component_by_target", pd.DataFrame(mat, index=[c[0] for c in comps], columns=cols).reset_index())

    # (c) slope chart likelihood -> AUPRC decision
    ax = fig.add_subplot(gs[2])
    lik = r[r.registry == "srs_c_val"].set_index("name").mean_target_auprc
    ap = r[r.registry == "srs_ap_t"].set_index("name").mean_target_auprc
    for name in lik.index:
        up = ap[name] >= lik[name]
        ax.plot([0, 1], [lik[name], ap[name]], color=S.TEAL if up else S.WARM, lw=0.8, alpha=0.85, marker="o", ms=2.2)
    ax.set_xticks([0, 1], ["likelihood", "AUPRC"], fontsize=5.8)
    ax.set_xlim(-0.25, 1.25)
    ax.set_ylabel("mean target AUPRC")
    n_up = int((ap >= lik).sum())
    ax.set_xlabel(f"{n_up}/16 runs improve", fontsize=5.6, color=S.TEAL_DARK)
    S.title(ax, "Decision rule")
    S.panel_label(ax, "c", dx=-0.42)
    src("e3c_decision_slope", pd.DataFrame({"run": lik.index, "likelihood": lik.values, "auprc": ap.loc[lik.index].values}))

    # (d) bubble: scenario gain vs executability of the schema-unaware registry
    ax = fig.add_subplot(gs[3])
    sub = a[(a.a == "srs_ap_t") & (a.b == "srs_ap_unaware") & (a.target.isin(data.TARGETS))].copy()
    colors = {"eicu": S.TEAL_DARK, "hirid": "#3FAE9A", "drop_labs": S.BLUE, "drop_ventilation": S.SKY, "drop_bp": "#7E8FBF",
              "drop_urine": S.SAND, "vitals_only": S.GREY}
    for t in data.TARGETS:
        s = sub[sub.target == t]
        ax.scatter(s.exec_share_unaware + RNG.uniform(-0.012, 0.012, len(s)), s.point_diff, s=6 + 220 * s.point_diff.abs(),
                   color=colors[t], alpha=0.7, edgecolor="white", lw=0.3, label=data.TARGET_LABEL[t])
    ax.axhline(0, color=S.INK2, lw=0.6)
    ax.set_xlabel("executable share of the schema-unaware registry")
    ax.set_ylabel("gain from scenarios")
    ax.legend(ncol=4, fontsize=4.9, loc="upper center", bbox_to_anchor=(0.45, -0.28), handletextpad=0.1, columnspacing=0.5, markerscale=0.8)
    S.title(ax, "Scenarios help where programs break")
    S.panel_label(ax, "d", dx=-0.3)
    src("e3d_scenario_gain", sub[["name", "target", "exec_share_unaware", "point_diff", "low", "high"]])
    S.save(fig, "fig_e3_ablation")


# ================================================================================================= E4
def fig_e4() -> None:
    """Contract. Claim: under the real schemas of individual eICU hospitals, which nobody gave to the selectors, SRS+T stays
    ahead; deploying the same registries in six different ways changes AUPRC but not the ranking, so the gain comes from
    selection. Evidence: (a) UpSet of missing-feature patterns at eICU hospitals, (b) per-run differences on the affected
    hospitals with CIs, (c) parallel coordinates over deployment modes, (d) box plot of the surrogate gain over zeroing.
    Data: hospital_schemas_ap (manifest patterns, paired.csv subset=affected), deployment_modes_ap rows.csv."""
    hr, hp, hm = data.hospital_all()
    fig = plt.figure(figsize=(S.FULL_W, 2.4))
    gs = GridSpec(2, 4, width_ratios=[1.15, 1.2, 1.25, 0.95], height_ratios=[0.8, 1.0], wspace=0.55, hspace=0.06,
                  left=0.075, right=0.985, top=0.85, bottom=0.2)

    # (a) UpSet of patterns (vasopressor 1 h cohort)
    pat = hm[("union", "f_mimic_iv_gap1h")]["summary"]["patterns"]
    items = sorted([(k, v) for k, v in pat.items() if k != "-"], key=lambda kv: -kv[1])[:8]
    feats = ["Lactate", "pH", "sedation_on", "Bilirubin_Total", "urine_output"]
    fname = {"Lactate": "lactate", "pH": "pH", "sedation_on": "sedation", "Bilirubin_Total": "bilirubin", "urine_output": "urine out.",
             "Creatinine": "creatinine"}
    axb = fig.add_subplot(gs[0, 0])
    axm = fig.add_subplot(gs[1, 0], sharex=axb)
    xs = np.arange(len(items))
    axb.bar(xs, [v for _, v in items], color=S.TEAL, width=0.6)
    for x, (_, v) in zip(xs, items):
        axb.text(x, v + 15, str(v), ha="center", va="bottom", fontsize=4.8, color=S.INK2)
    axb.set_ylabel("episodes", fontsize=5.6)
    axb.spines["bottom"].set_visible(False)
    axb.tick_params(axis="x", length=0, labelbottom=False)
    axb.set_ylim(0, max(v for _, v in items) * 1.25)
    for j, f in enumerate(feats):
        for x, (k, _) in zip(xs, items):
            on = f in k.split(",")
            axm.scatter(x, j, s=10, color=S.TEAL_DARK if on else S.GREY_PALE, zorder=2)
        axm.axhspan(j - 0.5, j + 0.5, color="#F7F7F7" if j % 2 else "white", zorder=0)
    for x, (k, _) in zip(xs, items):
        on = [j for j, f in enumerate(feats) if f in k.split(",")]
        if len(on) > 1:
            axm.plot([x, x], [min(on), max(on)], color=S.TEAL_DARK, lw=0.9, zorder=1)
    axm.set_yticks(range(len(feats)), [fname[f] for f in feats], fontsize=5.3)
    axm.set_ylim(len(feats) - 0.5, -0.5)
    axm.set_xticks([])
    for spine in ("bottom", "left"):
        axm.spines[spine].set_visible(False)
    axm.tick_params(axis="y", length=0)
    n_aff = hm[("union", "f_mimic_iv_gap1h")]["summary"]["n_affected"]
    n_ep = hm[("union", "f_mimic_iv_gap1h")]["summary"]["n_episodes"]
    S.title(axb, "Absent at eICU hospitals")
    S.panel_label(axb, "a", dx=-0.3)
    src("e4a_hospital_patterns", pd.DataFrame(items, columns=["pattern", "episodes"]))

    # (b) per-run differences on affected hospitals
    ax = fig.add_subplot(gs[:, 1])
    sub = hp[(hp.a == "srs_ap_t") & (hp.subset == "affected") & (hp["mode"] == "surrogate")]
    order = [data.run_name(r_, l_) for r_, l_ in data.ICU_RUNS]
    rows = []
    for off, (b, color, marker) in enumerate((("mrmr", S.SKY, "^"), ("nmifs", S.BLUE, "s"))):
        s = sub[sub.b == b].set_index("name").loc[order]
        y = np.arange(len(order)) + (off - 0.5) * 0.32
        ax.hlines(y, s.low, s.high, color=color, lw=0.8, alpha=0.9)
        ax.scatter(s.point_diff, y, marker=marker, s=9, color=color, zorder=3, label=f"vs {LABEL[b]}")
        rows.append(s.reset_index()[["name", "b", "point_diff", "low", "high", "verdict"]])
    ax.axvline(0, color=S.INK2, lw=0.6, ls=(0, (2, 2)))
    ax.set_yticks(range(len(order)), order, fontsize=5.0)
    ax.set_ylim(len(order) - 0.4, -0.6)
    ax.set_xlabel("SRS+T $-$ baseline (affected hospitals)", labelpad=1)
    ax.legend(loc="upper center", bbox_to_anchor=(0.45, -0.14), ncol=2, fontsize=5.3, handletextpad=0.1, columnspacing=0.8)
    ax.grid(axis="x")
    S.title(ax, "Real per-hospital schemas")
    S.panel_label(ax, "b", dx=-0.38)
    src("e4b_affected_differences", pd.concat(rows))

    # (c) parallel coordinates over deployment modes
    dr, dp = data.deploy_all()
    modes = [("zero", "zeroed"), ("reduced", "re-fit"), ("robust", "dropout-trained"), ("surrogate", "surrogates (shipped)"),
             ("reduced_surrogate", "re-fit + surr."), ("soft", "soft surrogates")]
    tg = dr[~dr.target.isin(["internal_test"]) & dr["mode"].isin([m for m, _ in modes])]
    mt = tg.groupby(["registry", "mode", "name"]).auprc.mean().groupby(["registry", "mode"]).mean().unstack()
    ax = fig.add_subplot(gs[:, 2])
    for m in ("srs_ap_t", "srs_ap", "nmifs", "mrmr", "l1_dropout"):
        st = method_style(m)
        vals = [mt.loc[m, md] for md, _ in modes]
        ax.plot(range(len(modes)), vals, color=st["color"], ls=st["ls"], marker=st["marker"], mfc=st["mfc"], mec=st["color"],
                lw=st["lw"], ms=3.2, label=LABEL[m])
    for x in range(len(modes)):
        ax.axvline(x, color="#EEEEEE", lw=0.6, zorder=0)
    ax.set_xticks(range(len(modes)), [t for _, t in modes], fontsize=5.0, rotation=30, ha="right", rotation_mode="anchor")
    ax.set_ylabel("mean target AUPRC (16 runs)")
    ax.legend(ncol=2, fontsize=5.0, loc="upper left", handlelength=1.6, columnspacing=0.6, borderpad=0.1)
    S.title(ax, "Six ways to deploy")
    S.panel_label(ax, "c", dx=-0.22)
    src("e4c_deployment_modes", mt.reset_index())

    # (d) surrogate gain over zeroing (box)
    ax = fig.add_subplot(gs[:, 3])
    per = tg.groupby(["registry", "mode", "name"]).auprc.mean().unstack("mode")
    meths = ["srs_ap_t", "nmifs", "mrmr", "l1_dropout", "fs_cv"]
    gains = [(per.loc[m]["surrogate"] - per.loc[m]["zero"]).values for m in meths]
    bp = ax.boxplot(gains, vert=False, widths=0.55, patch_artist=True, showfliers=False, medianprops=dict(color="white", lw=1.0),
                    whiskerprops=dict(lw=0.6), capprops=dict(lw=0.6))
    for patch, m in zip(bp["boxes"], meths):
        patch.set_facecolor(S.METHOD_COLOR.get(m, S.GREY_LIGHT) if m != "fs_cv" else S.GREY_LIGHT)
        patch.set_edgecolor("none")
    for k, g in enumerate(gains):
        ax.scatter(g, np.full(len(g), k + 1) + RNG.uniform(-0.15, 0.15, len(g)), s=2.5, color=S.INK2, alpha=0.6, lw=0, zorder=3)
    ax.set_yticks(range(1, len(meths) + 1), [LABEL[m] if m != "fs_cv" else "wrapper (lik.)" for m in meths], fontsize=5.5)
    ax.set_ylim(len(meths) + 0.6, 0.4)
    ax.axvline(0, color=S.INK2, lw=0.6, ls=(0, (2, 2)))
    ax.set_xlabel("surrogate gain (AUPRC)")
    S.title(ax, "Surrogate gain")
    S.panel_label(ax, "d", dx=-0.42)
    src("e4d_surrogate_gain", pd.DataFrame({LABEL.get(m, m): pd.Series(g) for m, g in zip(meths, gains)}))
    S.save(fig, "fig_e4_real_shift")


# ================================================================================================= E5
def fig_e5() -> None:
    """Contract. Claim: under random loss of features or feature groups, SRS+T keeps the highest mean AUPRC at every loss
    level and wins most paired draws against every baseline, but it degrades at the same rate. Evidence: (a) mean AUPRC
    vs loss with run-to-run band, (b) win-rate heatmap per opponent and level, (c) violin of paired differences vs NMIFS,
    (d) area under the stress curve per method. Data: stress_ap/<run>/stress.csv (16 runs x 9 levels x 30 draws)."""
    st = data.stress_all()
    st = st[st.variant == "composite"]
    meths = ["srs_ap_t", "srs_ap", "nmifs", "mrmr", "l1_dropout", "mrmr_filter_known"]
    fig = plt.figure(figsize=(S.FULL_W, 2.25))
    gs = GridSpec(1, 4, width_ratios=[1.1, 1.45, 1.05, 0.95], wspace=0.5, left=0.06, right=0.955, top=0.84, bottom=0.21)

    # (a) curves (feature loss)
    ax = fig.add_subplot(gs[0])
    f = st[st.family == "feature"]
    per_run = f.groupby(["registry", "level", "name"]).auprc.mean()
    rows = []
    for m in meths:
        pr = per_run.loc[m].unstack("name")
        mean = pr.mean(axis=1)
        sty = method_style(m)
        if m == "srs_ap_t":
            ax.fill_between(mean.index, pr.quantile(0.25, axis=1), pr.quantile(0.75, axis=1), color=S.TEAL, alpha=0.13, lw=0)
        ax.plot(mean.index, mean.values, color=sty["color"], ls=sty["ls"], lw=sty["lw"], marker=sty["marker"], ms=2.4,
                mfc=sty["mfc"], mec=sty["color"], label=LABEL[m])
        rows.append(pd.DataFrame({"method": LABEL[m], "level": mean.index, "mean_auprc": mean.values}))
    ax.set_xlabel("share of features dropped, $l$")
    ax.set_ylabel("AUPRC (mean of 16 runs)")
    ax.legend(fontsize=4.9, loc="lower left", handlelength=1.6, borderpad=0.1, labelspacing=0.25)
    ax.grid(axis="y")
    S.title(ax, "Random feature loss")
    S.panel_label(ax, "a", dx=-0.25)
    src("e5a_feature_curves", pd.concat(rows))

    # (b) win-rate heatmap
    ax = fig.add_subplot(gs[1])
    opps = ["nmifs", "mrmr", "l1_dropout", "mrmr_filter_known", "cmim", "srs_c_val"]
    levels = sorted(st.level.unique())
    piv = st.pivot_table(index=["family", "level", "draw", "name"], columns="registry", values="auprc")
    mat = np.zeros((len(opps), 2 * len(levels)))
    for i, o in enumerate(opps):
        for fi, fam in enumerate(("feature", "group")):
            for li, lv in enumerate(levels):
                sub = piv.loc[(fam, lv)]
                d = sub["srs_ap_t"] - sub[o]
                mat[i, fi * len(levels) + li] = ((d > 1e-12).sum() + 0.5 * (d.abs() <= 1e-12).sum()) / len(d)
    im = ax.imshow(mat, cmap=S.SEQ, vmin=0.3, vmax=1.0, aspect="auto")
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax.text(j, i, f"{100 * mat[i, j]:.0f}", ha="center", va="center", fontsize=4.6, color="white" if mat[i, j] > 0.75 else S.INK)
    ax.axvline(len(levels) - 0.5, color="white", lw=2.0)
    ax.set_xticks(range(2 * len(levels)), [f"{lv:.1f}".lstrip("0") for lv in levels] * 2, fontsize=4.8)
    ax.set_yticks(range(len(opps)), [LABEL[o] if o != "srs_c_val" else "lik. decision" for o in opps], fontsize=5.5)
    ax.text(len(levels) / 2 - 0.5, -0.85, "feature loss", ha="center", fontsize=5.4, color=S.INK2)
    ax.text(1.5 * len(levels) - 0.5, -0.85, "group loss", ha="center", fontsize=5.4, color=S.INK2)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xlabel("loss level $l$")
    ax.set_title("Draws won by SRS+T (%)", loc="left", fontsize=6.8, fontweight="bold", y=1.1)
    S.panel_label(ax, "b", dx=-0.27, dy=1.1)
    src("e5b_win_rates", pd.DataFrame(mat, index=[LABEL[o] for o in opps],
                                      columns=[f"{fam}_{lv}" for fam in ("feature", "group") for lv in levels]).reset_index())

    # (c) violin of paired differences vs NMIFS (group loss)
    ax = fig.add_subplot(gs[2])
    g = piv.loc["group"]
    sel = [0.1, 0.3, 0.5, 0.7, 0.9]
    diffs = [(g.loc[lv]["srs_ap_t"] - g.loc[lv]["nmifs"]).values for lv in sel]
    parts = ax.violinplot(diffs, positions=range(len(sel)), widths=0.8, showextrema=False, bw_method=0.3)
    for body in parts["bodies"]:
        body.set_facecolor(S.TEAL)
        body.set_alpha(0.35)
        body.set_edgecolor("none")
    for k, d in enumerate(diffs):
        q1, med, q3 = np.percentile(d, [25, 50, 75])
        ax.vlines(k, q1, q3, color=S.TEAL_DARK, lw=2.2)
        ax.scatter(k, med, s=8, color="white", zorder=3, edgecolor=S.TEAL_DARK, lw=0.5)
    ax.axhline(0, color=S.INK2, lw=0.6, ls=(0, (2, 2)))
    ax.set_xticks(range(len(sel)), [f"{lv:.1f}" for lv in sel])
    ax.set_ylim(-0.08, 0.1)
    ax.set_xlabel("group loss $l$")
    ax.set_ylabel("SRS+T $-$ NMIFS per draw")
    S.title(ax, "Paired draws vs. NMIFS")
    S.panel_label(ax, "c", dx=-0.32)
    src("e5c_paired_vs_nmifs", pd.DataFrame({f"group_{lv}": pd.Series(d) for lv, d in zip(sel, diffs)}))

    # (d) area under the stress curve
    ax = fig.add_subplot(gs[3])
    auc = st.groupby(["registry", "name"]).auprc.mean().unstack("registry")
    show = ["srs_ap_t", "srs_ap", "nmifs", "l1_dropout", "mrmr", "mrmr_filter_known", "cmim"]
    means = auc[show].mean()
    sds = auc[show].std()
    order = means.sort_values().index
    ax.barh(range(len(order)), means[order] - 0.55, left=0.55, xerr=sds[order] / 4, color=[S.METHOD_COLOR.get(m, S.GREY_LIGHT) for m in order],
            height=0.62, error_kw=dict(lw=0.6, capsize=1.2, ecolor=S.INK2))
    for k, m in enumerate(order):
        ax.text(means[m] + sds[m] / 4 + 0.003, k, f"{means[m]:.3f}", va="center", fontsize=5.0, color=S.INK2)
    ax.set_yticks(range(len(order)), [LABEL[m] for m in order], fontsize=5.5)
    ax.set_xlim(0.55, 0.695)
    ax.set_xticks([0.55, 0.60, 0.65])
    ax.set_xlabel("mean AUPRC, all levels")
    S.title(ax, "Area under the curve")
    S.panel_label(ax, "d", dx=-0.42)
    src("e5d_stress_area", pd.DataFrame({"method": [LABEL[m] for m in show], "mean": means[show].values, "sd": sds[show].values}))
    S.save(fig, "fig_e5_stress")
