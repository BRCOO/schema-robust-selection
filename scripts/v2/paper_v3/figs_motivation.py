"""Challenge figures of the PVLDB v3 paper (Section 1: one figure per challenge). Single-column figures, every text element
at 7 pt or more, Nature teal-green family, source data in paper/v3_assets/source_data/.

  PYTHONDONTWRITEBYTECODE=1 /d/BRCoo/miniconda/python scripts/v2/paper_v3/figs_motivation.py
"""
from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import data
import style as S
from figs_b import GEN_COLOR, GEN_ORDER, src

S.setup()
FS = 7.0
mpl.rcParams.update({"font.size": FS, "axes.labelsize": FS, "xtick.labelsize": FS, "ytick.labelsize": FS, "legend.fontsize": FS})
TGT = ["hirid", "eicu", "drop_urine", "drop_ventilation", "drop_bp", "drop_labs", "vitals_only"]


def title(ax, text: str) -> None:
    ax.set_title(text, loc="left", fontsize=7.5, fontweight="bold", y=1.03, pad=0)


def letter(ax, text: str, dx: float) -> None:
    ax.text(dx, 1.03, text, transform=ax.transAxes, fontsize=9, fontweight="bold", va="bottom", ha="right", color=S.INK)


def fig_c1() -> None:
    """Contract. Claim: a registry selected without regard to the target loses programs at targets with fewer features,
    and its frozen probe silently loses accuracy; surrogates recover only part of it. Evidence: (a) share of an mRMR
    registry's programs that cannot execute, per target (bars); (b) AUPRC at the source vs at the controlled targets (same
    TEST patients, features removed) with zeroing and with surrogates (dumbbell). Data: deployment_modes_ap rows (16 runs)."""
    dr, _ = data.deploy_all()
    m = dr[dr.registry == "mrmr"]
    share = (1 - m[m["mode"] == "zero"].groupby("target").apply(lambda s: (s.n_executable / s.n_programs).mean())).reindex(TGT)
    internal = m[m["mode"] == "internal_frozen"].auprc.mean()
    zero = m[m["mode"] == "zero"].groupby("target").auprc.mean().reindex(TGT)
    sur = m[m["mode"] == "surrogate"].groupby("target").auprc.mean().reindex(TGT)
    fig = plt.figure(figsize=(S.COL_W, 2.15))
    bottom, h = 0.3, 0.58
    ax = fig.add_axes([0.235, bottom, 0.28, h])
    y = np.arange(len(TGT))
    ax.barh(y, share.values, color=S.WARM, height=0.64)
    for k, v in enumerate(share.values):
        ax.text(v + 0.03, k, f"{100 * v:.0f}%", va="center", fontsize=FS, color=S.INK2)
    ax.set_yticks(y, [data.TARGET_LABEL[t] for t in TGT])
    ax.set_ylim(len(TGT) - 0.5, -0.6)
    ax.set_xlim(0, 1.25)
    ax.set_xticks([0, 0.5, 1], ["0", "50", "100"])
    ax.set_xlabel("programs that cannot run (%)")
    ax.tick_params(axis="y", length=0)
    title(ax, "Broken programs")
    letter(ax, "a", dx=-0.62)
    ax = fig.add_axes([0.57, bottom, 0.42, h])
    CTRL = [t for t in TGT if t.startswith("drop") or t == "vitals_only"]
    ax.axvline(internal, color=S.TEAL_DARK, lw=0.8, ls=(0, (2, 2)))
    ax.text(internal - 0.005, 0.2, f"source\n{internal:.3f}", fontsize=FS, color=S.TEAL_DARK, ha="right", va="center")
    for k, t in enumerate(TGT):
        if t not in CTRL:
            ax.text(0.505, k, "other patients", fontsize=FS, color=S.GREY, style="italic", va="center", ha="left")
            continue
        ax.plot([zero[t], internal], [k, k], color=S.GREY_PALE, lw=2.2, solid_capstyle="round", zorder=1)
        ax.scatter(zero[t], k - 0.16, s=15, color=S.WARM, zorder=3, label="zeroed" if t == CTRL[0] else None)
        ax.scatter(sur[t], k + 0.16, s=15, marker="D", facecolor="white", edgecolor=S.TEAL, lw=0.8, zorder=4, label="surrogates" if t == CTRL[0] else None)
    ax.set_yticks(y, [])
    ax.tick_params(axis="y", length=0)
    ax.set_ylim(len(TGT) - 0.5, -0.6)
    ax.set_xlim(0.5, 0.79)
    ax.set_xticks([0.5, 0.6, 0.7])
    ax.set_xlabel("AUPRC, same TEST patients")
    ax.legend(loc="upper center", bbox_to_anchor=(0.45, -0.28), ncol=2, handletextpad=0.1, borderpad=0.2, columnspacing=0.8)
    ax.grid(axis="x")
    title(ax, "Silent loss")
    letter(ax, "b", dx=-0.04)
    src("c1_broken_programs", pd.DataFrame({"target": TGT, "non_executable_share": share.values, "auprc_zeroed": zero.values,
                                            "auprc_surrogates": sur.values, "auprc_source": internal}))
    S.save_exact(fig, "fig_c1_broken")


def fig_c2() -> None:
    """Contract. Claim: on a union pool, mRMR's relevance-redundancy score ranks candidates poorly by their held-out value,
    across generators, whereas a cross-validated estimate of the deployed value ranks them well. Evidence: per-candidate
    scatter at the second pick (vasopressor 1 h), coloured by generator, for both scores. Data:
    source_data/proxy_candidates_gap1h.csv (dump_proxy_candidates.py; Spearman as in proxy_correlation)."""
    d = pd.read_csv(S.OUT_SRC / "proxy_candidates_gap1h.csv")
    d = d[d.step == 1]
    fig = plt.figure(figsize=(S.COL_W, 2.15))
    bottom, h = 0.38, 0.51
    marker = {"ERMiner": "o", "PrefixSpan": "s", "Motif": "^", "Shapelet": "D", "FMG": "v"}
    for k, (col, name) in enumerate((("mrmr", "mRMR score"), ("srs_estimate", "estimated value (bits)"))):
        ax = fig.add_axes([0.16 + k * 0.44, bottom, 0.37, h])
        for g in GEN_ORDER:
            s = d[d.generator == g]
            ax.scatter(s[col], 100 * s.val_auprc_gain, s=6, marker=marker[g], color=GEN_COLOR[g], alpha=0.75, lw=0, label=g)
        top = d.nlargest(10, col)
        ax.scatter(top[col], 100 * top.val_auprc_gain, s=26, facecolor="none", edgecolor=S.INK, lw=0.6, zorder=4)
        rho = spearmanr(d[col], d.val_auprc_gain).correlation
        ax.text(0.04, 0.96, f"$\\rho$ = {rho:.2f}", transform=ax.transAxes, fontsize=FS, va="top", fontweight="bold",
                color=S.TEAL_DARK if col == "srs_estimate" else S.WARM)
        ax.set_xlabel(name)
        if k == 1:
            ax.set_xticks([0, 0.02, 0.04])
        if k == 0:
            ax.set_ylabel("VAL gain (AUPRC $\\times$100)")
        ax.axhline(0, color=S.GREY_LIGHT, lw=0.5)
        title(ax, "Proxy score" if k == 0 else "Estimated value")
        letter(ax, "ab"[k], dx=-0.3 if k == 0 else -0.1)
    fig.legend(handles=[plt.Line2D([], [], marker=marker[g], ls="", color=GEN_COLOR[g], ms=3.5, label=g) for g in GEN_ORDER] +
               [plt.Line2D([], [], marker="o", ls="", mfc="none", mec=S.INK, ms=5, label="top 10 by score")],
               loc="lower center", ncol=3, handletextpad=0.05, columnspacing=0.8, bbox_to_anchor=(0.55, 0.0), labelspacing=0.2)
    src("c2_proxy_scatter", d)
    S.save_exact(fig, "fig_c2_proxies")


def fig_c3() -> None:
    """Contract. Claim: measuring a candidate's deployed value exactly needs a model refit per candidate, fold and step, so
    exact greedy selection takes minutes on our pools and grows with the pool; screening does the same work several times
    faster. Evidence: measured exact-greedy and screening times against the pool size, median of the repeats with their
    range (lines and bands, no extrapolation); speed-up per size (bars). Data: selection_scalability (+ _repeats):
    srs_c_exact, srs_c_v phases (chains + selection at omega = 0)."""
    from figs_v5 import scalability_repeats

    reps = scalability_repeats()
    ex = reps[reps.method == "srs_c_exact"].groupby("n_candidates").seconds.agg(["median", "min", "max", "size"]).reset_index()
    fq = reps[reps.method == "srs_c_v"].copy()
    fq["seconds"] = fq.phase_chains + fq.phase_select_q0
    fq = fq.groupby("n_candidates").seconds.agg(["median", "min", "max", "size"]).reset_index()
    fig = plt.figure(figsize=(S.COL_W, 2.0))
    bottom, h = 0.3, 0.58
    ax = fig.add_axes([0.16, bottom, 0.4, h])
    for d, color, marker, text in ((ex, S.WARM, "s", "exact"), (fq, S.TEAL, "o", "screening")):
        ax.fill_between(d.n_candidates, d["min"], d["max"], color=color, alpha=0.18, lw=0)
        ax.plot(d.n_candidates, d["median"], color=color, marker=marker, ms=3, lw=1.3, label=text)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xticks([1e3, 1e4, 1e5], ["1k", "10k", "100k"])
    ax.set_yticks([10, 100, 1000], ["10", "100", "1,000"])
    ax.minorticks_off()
    ax.set_xlabel("candidates $|\\mathcal{C}|$")
    ax.set_ylabel("one selection (s)")
    ax.legend(loc="upper left", handlelength=1.4, borderpad=0.1, labelspacing=0.2)
    ax.grid(which="major")
    title(ax, "Time")
    letter(ax, "a", dx=-0.3)
    ax = fig.add_axes([0.72, bottom, 0.27, h])
    e_i = ex.set_index("n_candidates")["median"]
    f_i = fq.set_index("n_candidates")["median"]
    sizes = [n for n in e_i.index if n in f_i.index]
    speed = [e_i[n] / f_i[n] for n in sizes]
    ax.bar(range(len(sizes)), speed, width=0.62, color=S.TEAL)
    for k, v in enumerate(speed):
        ax.text(k, v + 0.2, f"{v:.1f}", ha="center", va="bottom", fontsize=FS, color=S.INK2)
    labels = {500: "500", 1000: "1k", 2274: "2.3k", 5000: "5k", 10000: "10k", 20000: "20k"}
    ax.set_xticks(range(len(sizes)), [labels.get(n, f"{n:,}") for n in sizes], rotation=90)
    ax.set_xlabel("candidates $|\\mathcal{C}|$")
    ax.set_ylabel("exact / screening")
    ax.set_ylim(0, max(speed) * 1.3)
    title(ax, "Speed-up")
    letter(ax, "b", dx=-0.42)
    src("c3_cost", pd.concat([ex.assign(method="srs_c_exact"), fq.assign(method="screening_q0_with_chains")]))
    S.save_exact(fig, "fig_c3_cost")


if __name__ == "__main__":
    fig_c1()
    fig_c2()
    fig_c3()
    print("built c1-c3")
