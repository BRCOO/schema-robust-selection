"""Round-2 re-analysis of the v3 experiments (review of 2026-11-05): real-target endpoint, pooled effects over the six
independent pools, random-loss win/tie/loss with mean differences, lift retention, Brier, extended proxy fidelity with
shortlist recall, re-splits with NMIFS+AP and the ablations, compilable-only registries, wrapper decomposition and seed-aware
summaries. Reads artifacts/v2 only; writes paper/v3_assets/source_data/q2_*.csv and prints the numbers the text quotes.

  PYTHONDONTWRITEBYTECODE=1 /d/BRCoo/miniconda/python scripts/v2/paper_v3/round2.py [part ...]
"""
from __future__ import annotations

import glob
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
import data  # noqa: E402
from style import ART  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "paper/v3_assets/source_data"
REAL = ("eicu", "hirid")
CONTROLLED = ("drop_labs", "drop_ventilation", "drop_bp", "drop_urine", "vitals_only")
NAMES = {"srs_ap_t": "SRS+T", "srs_ap": "SRS", "nmifs_ap_t": "NMIFS+AP", "srs_ap_unaware": "Wrapper, AUPRC", "fs_cv": "Wrapper, lik.",
         "nmifs": "NMIFS", "mrmr": "mRMR", "mrmr_stratified": "mRMR strat.", "jmi": "JMI", "cmim": "CMIM", "mmrfs": "MMRFS",
         "mmr_ig": "MMR", "mms_xin_ig": "MMS", "dpp_ig": "DPP", "mmr": "MMR", "mms_xin": "MMS", "dpp": "DPP", "l1_logistic": "L1",
         "l1_dropout": "L1+dropout", "mrmr_filter_known": "Filter+mRMR", "filter_nmifs": "Filter+NMIFS", "srs_c_val": "SRS+T, lik.",
         "srs_ap_t_nosur": "no surr. in objective", "srs_ap_t_nosurr": "no surr. in objective"}
#: SRS+T knows these target schemas (no ventilation shares eICU's schema); every other target is unseen by it
KNOWN_T = ("eicu", "hirid", "drop_ventilation")


def pool_of(run: str) -> str:
    root, label = run.split("/")
    return ("L " if root.startswith("union_large") else "") + data.TASK[label]


def verdict(low, high):
    return np.where(np.asarray(low) > 0, 1, np.where(np.asarray(high) < 0, -1, 0))


def tally(v) -> str:
    v = np.asarray(v)
    return f"{int((v == 1).sum())}/{int((v == 0).sum())}/{int((v == -1).sum())}"


def hksj(effects: np.ndarray, ses: np.ndarray) -> tuple[float, float, float, float]:
    """Random-effects mean (DerSimonian-Laird tau^2) with the Hartung-Knapp-Sidik-Jonkman interval; returns mu, low, high, tau."""
    effects, ses = np.asarray(effects, float), np.maximum(np.asarray(ses, float), 1e-9)
    k = len(effects)
    w = 1 / ses ** 2
    mu_fe = np.sum(w * effects) / np.sum(w)
    q = np.sum(w * (effects - mu_fe) ** 2)
    tau2 = max(0.0, (q - (k - 1)) / (np.sum(w) - np.sum(w ** 2) / np.sum(w))) if k > 1 else 0.0
    wr = 1 / (ses ** 2 + tau2)
    mu = np.sum(wr * effects) / np.sum(wr)
    if k < 2:
        return mu, mu - 1.96 * ses[0], mu + 1.96 * ses[0], math.sqrt(tau2)
    qs = max(np.sum(wr * (effects - mu) ** 2) / (k - 1), 1.0)  # truncated HKSJ: never narrower than the DL interval
    se = math.sqrt(qs / np.sum(wr))
    t = stats.t.ppf(0.975, k - 1)
    return float(mu), float(mu - t * se), float(mu + t * se), math.sqrt(tau2)


def pooled(df: pd.DataFrame) -> dict:
    """df: one row per run with point_diff, low, high; pools the runs of a pool (mean effect, mean SE), then HKSJ over pools."""
    d = df.copy()
    d["se"] = (d["high"] - d["low"]) / (2 * 1.96)
    d["pool"] = d["run"].map(pool_of)
    per = d.groupby("pool").agg(effect=("point_diff", "mean"), se=("se", "mean"), n=("run", "size")).reset_index()
    mu, lo, hi, tau = hksj(per["effect"].to_numpy(), per["se"].to_numpy())
    return {"pooled": mu, "low": lo, "high": hi, "tau": tau, "n_pools": len(per), "n_runs": len(d), "run_mean": d["point_diff"].mean(),
            "tally": tally(verdict(d["low"], d["high"]))}


# ---------------------------------------------------------------- real targets: AUPRC, lift and tallies
def real_targets() -> None:
    lifts, prev = [], {}
    for root, label in data.ICU_RUNS:
        base = ART / "real_targets" / root / label
        df = pd.read_csv(base / "lift.csv")
        p = json.loads((base / "prevalence.json").read_text(encoding="utf-8"))
        for t in ("internal",) + REAL + CONTROLLED:
            key = "internal_test" if t == "internal" else t
            df[f"{t}_auprc"] = df[f"{t}_lift"] + p[key]
        lifts.append(df)
    L = pd.concat(lifts, ignore_index=True)
    L["real_auprc"] = L[[f"{t}_auprc" for t in REAL]].mean(axis=1)
    L["controlled_auprc"] = L[[f"{t}_auprc" for t in CONTROLLED]].mean(axis=1)
    L["targets_auprc"] = L[[f"{t}_auprc" for t in REAL + CONTROLLED]].mean(axis=1)
    L["real_lift"] = L[[f"{t}_lift" for t in REAL]].mean(axis=1)
    L["targets_lift"] = L[[f"{t}_lift" for t in REAL + CONTROLLED]].mean(axis=1)
    g = L.groupby("registry")
    summ = g[["internal_auprc", "eicu_auprc", "hirid_auprc", "real_auprc", "controlled_auprc", "targets_auprc", "internal_lift", "real_lift",
              "targets_lift", "eicu_lift", "hirid_lift"]].mean()
    summ["retention_auprc_targets"] = summ["targets_auprc"] / summ["internal_auprc"]
    summ["retention_lift_targets"] = summ["targets_lift"] / summ["internal_lift"]
    summ["retention_lift_real"] = summ["real_lift"] / summ["internal_lift"]
    summ["retention_lift_eicu"] = summ["eicu_lift"] / summ["internal_lift"]
    summ["retention_lift_hirid"] = summ["hirid_lift"] / summ["internal_lift"]
    summ = summ.sort_values("real_auprc", ascending=False)
    summ.insert(0, "name", [NAMES.get(n, n) for n in summ.index])
    summ.to_csv(SRC / "q2_real_targets.csv")
    print("\n== real targets: mean over 16 runs (AUPRC; lift retention = mean target lift / mean internal lift)")
    print(summ[["name", "internal_auprc", "eicu_auprc", "hirid_auprc", "real_auprc", "controlled_auprc", "targets_auprc", "retention_auprc_targets",
                "retention_lift_targets", "retention_lift_real"]].round(3).to_string())
    # rank of each registry at each real target (mean over runs)
    for t in REAL + ("real",):
        col = f"{t}_auprc"
        order = summ[col].sort_values(ascending=False)
        print(f"  best at {t}: " + ", ".join(f"{NAMES.get(n, n)} {v:.3f}" for n, v in order.head(5).items()))
    P = pd.concat([pd.read_csv(ART / "real_targets" / r / l / "paired.csv") for r, l in data.ICU_RUNS], ignore_index=True)
    P["v"] = verdict(P["low"], P["high"])
    P["target"] = P["target"].replace({"mean_over_targets": "real_mean"})
    rows = []
    for (a, b, t), grp in P.groupby(["a", "b", "target"]):
        rows.append({"a": a, "b": b, "target": t, "tally": tally(grp["v"]), "mean_diff": grp["point_diff"].mean()})
    T = pd.DataFrame(rows)
    T.to_csv(SRC / "q2_real_tallies.csv", index=False)
    piv = T.pivot_table(index=["a", "b"], columns="target", values="tally", aggfunc="first")
    piv["mean_diff_real"] = T[T.target == "real_mean"].set_index(["a", "b"])["mean_diff"]
    print("\n== real-target tallies (b/n.s./w over 16 runs) and mean difference over the two real databases")
    print(piv.to_string())
    # pooled effect on the real mean
    out = []
    for (a, b), grp in P[P.target == "real_mean"].groupby(["a", "b"]):
        out.append({"a": a, "b": b, "endpoint": "real", **pooled(grp)})
    pd.DataFrame(out).to_csv(SRC / "q2_pooled_real.csv", index=False)


# ---------------------------------------------------------------- pooled effects (internal, all targets)
def pair_frame() -> pd.DataFrame:
    """SRS+T and SRS against every comparator: internal and mean over all targets, one row per run and pair (first source wins)."""
    frames = []
    for root, label in data.ICU_RUNS:
        run = f"{root}/{label}"
        for path in (ART / "union_ap" / run / "paired.csv", ART / "union_ap" / run / "ablation_paired.csv", ART / "nmifs_ap" / run / "paired.csv",
                     ART / "references" / run / "paired.csv", ART / "paired_extra" / run / "paired.csv", ART / "decomposition" / run / "paired.csv"):
            if not path.exists():
                continue
            d = pd.read_csv(path)
            if "scope" in d.columns:
                d = d[d["scope"] != "real_targets"]
            d["run"] = run
            d["endpoint"] = np.where(d["target"].fillna("internal").isin(["internal"]) | (d.get("metric", "") == "internal"), "internal",
                                     np.where(d["target"] == "mean_over_targets", "targets", "per_target"))
            d["source"] = path.parts[-4] if path.parts[-4] != "v2" else path.parts[-3]
            frames.append(d[["run", "a", "b", "endpoint", "target", "point_diff", "low", "high", "source"]])
    F = pd.concat(frames, ignore_index=True)
    F = F[F["endpoint"] != "per_target"]
    return F.drop_duplicates(["run", "a", "b", "endpoint"], keep="first")


def pooled_effects() -> None:
    F = pair_frame()
    rows = []
    for (a, b, e), grp in F.groupby(["a", "b", "endpoint"]):
        if grp["run"].nunique() < 16:
            continue
        rows.append({"a": a, "b": b, "endpoint": e, **pooled(grp)})
    R = pd.DataFrame(rows)
    R.to_csv(SRC / "q2_pooled.csv", index=False)
    show = R[R.a.isin(["srs_ap_t", "srs_ap"])].copy()
    show["b"] = show["b"].map(lambda n: NAMES.get(n, n))
    print("\n== pooled effects over the six pools (HKSJ random effects; within a pool, runs averaged)")
    print(show[["a", "b", "endpoint", "pooled", "low", "high", "tau", "tally"]].round(4).to_string(index=False))


# ---------------------------------------------------------------- random loss: W/T/L and mean differences
def stress() -> None:
    S = data.stress_all()
    S = S[S["variant"] == "composite"] if "variant" in S.columns else S
    wide = S.pivot_table(index=["run", "family", "level", "draw"], columns="registry", values="auprc").reset_index()
    rows = []
    for b in [c for c in wide.columns if c not in ("run", "family", "level", "draw", "srs_ap_t")]:
        d = wide["srs_ap_t"] - wide[b]
        for (fam, lev), idx in wide.groupby(["family", "level"]).groups.items():
            x = d.loc[idx].dropna()
            per_run = (wide.loc[idx].assign(diff=x).groupby("run")["diff"].mean())
            rng = np.random.default_rng(0)
            boots = [per_run.sample(len(per_run), replace=True, random_state=int(rng.integers(1 << 31))).mean() for _ in range(1000)]
            rows.append({"b": b, "family": fam, "level": lev, "n": len(x), "win": float((x > 1e-12).mean()), "tie": float((x.abs() <= 1e-12).mean()),
                         "loss": float((x < -1e-12).mean()), "mean_diff": float(x.mean()), "low": float(np.percentile(boots, 2.5)),
                         "high": float(np.percentile(boots, 97.5)), "runs_positive": int((per_run > 0).sum()), "n_runs": len(per_run)})
    R = pd.DataFrame(rows)
    R.to_csv(SRC / "q2_stress_wtl.csv", index=False)
    print("\n== random loss: SRS+T minus comparator, W/T/L shares of draws and mean difference (run-bootstrap CI)")
    for b in ("nmifs", "mrmr", "srs_ap", "l1_dropout", "mrmr_filter_known"):
        sub = R[R.b == b]
        if sub.empty:
            continue
        print(f"  vs {NAMES.get(b, b)}")
        print(sub[["family", "level", "win", "tie", "loss", "mean_diff", "low", "high", "runs_positive"]].round(3).to_string(index=False))


# ---------------------------------------------------------------- Brier
def brier() -> None:
    frames = [pd.read_csv(p) for p in glob.glob(str(ART / "extra_metrics/*/*/rows.csv"))]
    B = pd.concat(frames, ignore_index=True)
    cols = [c for c in B.columns if c.endswith("_brier")]
    g = B.groupby("registry")[cols].mean()
    g["n_runs"] = B.groupby("registry").size()
    g.to_csv(SRC / "q2_brier.csv")
    print("\n== Brier (mean over runs with extra metrics)")
    print(g[["internal_brier", "eicu_brier", "hirid_brier", "mean_target_brier", "vitals_only_brier", "n_runs"]].round(3).to_string())
    prev = {}
    for root, label in data.ICU_RUNS:
        p = json.loads((ART / "real_targets" / root / label / "prevalence.json").read_text(encoding="utf-8"))
        prev[f"{root}/{label}"] = p
    const = {t: np.mean([p[t] * (1 - p[t]) for p in prev.values()]) for t in ("internal_test", "eicu", "hirid", "vitals_only")}
    print("  constant forecast (prevalence) Brier: " + ", ".join(f"{k} {v:.3f}" for k, v in const.items()))


# ---------------------------------------------------------------- proxy fidelity, picks 1-20, per generator, shortlist recall
def fidelity() -> None:
    C = pd.concat([pd.read_csv(p) for p in glob.glob(str(ART / "proxy_correlation/ext_corr_*.csv"))], ignore_index=True)
    Rc = pd.concat([pd.read_csv(p) for p in glob.glob(str(ART / "proxy_correlation/ext_recall_*.csv"))], ignore_index=True)
    bins = [(1, 1, "1"), (2, 4, "2-4"), (5, 7, "5-7"), (8, 11, "8-11"), (12, 20, "12-20")]

    def binned(df, value):
        out = []
        for lo, hi, name in bins:
            sub = df[(df.pick >= lo) & (df.pick <= hi)]
            out.append(sub.groupby([c for c in df.columns if c in ("tag", "scope", "proxy")])[value].mean().rename(name))
        return pd.concat(out, axis=1)

    U = C[C.scope == "union"]
    tab = binned(U, "spearman").groupby("proxy").agg(["min", "max", "mean"])
    print("\n== fidelity on the union pools: Spearman by pick range (min/max/mean over the four tasks)")
    print(tab.round(2).to_string())
    G = C[C.scope != "union"]
    gen = binned(G, "spearman").reset_index()
    print("\n== fidelity within generators (mean over tasks), picks 2-4 and 8-11")
    print(gen.groupby(["scope", "proxy"])[["2-4", "8-11"]].mean().round(2).unstack("proxy").to_string())
    # width of the Fisher intervals at picks 2-4 on the union
    w = U[(U.pick >= 2) & (U.pick <= 4)]
    print(f"  Fisher 95% interval half-width on the union, picks 2-4: median {np.median((w.high - w.low) / 2):.2f}")
    binned(C, "spearman").reset_index().to_csv(SRC / "q2_fidelity.csv", index=False)
    Rc["regret"] = Rc["best_gain"] - Rc["chosen_gain"]
    rec = []
    for lo, hi, name in bins:
        sub = Rc[(Rc.pick >= lo) & (Rc.pick <= hi)]
        g = sub.groupby("proxy").agg(recall=("best_in_top10", "mean"), regret=("regret", "mean"), top10_gain=("top10_mean_gain", "mean"),
                                     best_gain=("best_gain", "mean"))
        g["picks"] = name
        rec.append(g.reset_index())
    R = pd.concat(rec, ignore_index=True)
    R.to_csv(SRC / "q2_recall.csv", index=False)
    print("\n== shortlist recall (VAL-best candidate in the top 10 of each score) and regret of each score's top choice")
    print(R.pivot_table(index="proxy", columns="picks", values=["recall", "regret"]).round(4).to_string())


# ---------------------------------------------------------------- re-splits with NMIFS+AP and the ablations
def resplits() -> None:
    rows = []
    for task in ("gap1h", "aki_gap6h"):
        for i in range(1, 6):
            run = f"union_resplit/r{i}_mimic_iv_{task}"
            for path in (ART / "union_ap" / run / "paired.csv", ART / "nmifs_ap" / run / "paired.csv", ART / "union_ap" / run / "ablation_paired.csv"):
                if not path.exists():
                    continue
                d = pd.read_csv(path)
                d["task"], d["split"] = ("Vaso 1h" if task == "gap1h" else "AKI"), i
                d["endpoint"] = np.where(d["target"].fillna("internal") == "internal", "internal",
                                         np.where(d["target"] == "mean_over_targets", "targets", "per_target"))
                rows.append(d)
    D = pd.concat(rows, ignore_index=True)
    D = D[D.endpoint != "per_target"].drop_duplicates(["task", "split", "a", "b", "endpoint"])
    D["v"] = verdict(D["low"], D["high"])
    T = D.groupby(["a", "b", "task", "endpoint"]).agg(tally=("v", tally), mean_diff=("point_diff", "mean")).reset_index()
    T.to_csv(SRC / "q2_resplits.csv", index=False)
    print("\n== re-splits (5 per task): tallies and mean differences")
    show = T[T.b.isin(["nmifs", "mrmr", "nmifs_ap_t", "srs_ap_unaware", "srs_ap_t_nosur", "srs_ap"])]
    print(show.to_string(index=False))


# ---------------------------------------------------------------- compilable-only registries at K = 20
def compilable() -> None:
    frames = []
    for label in data.STANDARD:
        d = pd.read_csv(ART / "compilable_only/union_k20" / label / "paired.csv")
        d["label"] = label
        frames.append(d)
    D = pd.concat(frames, ignore_index=True)
    D["v"] = verdict(D["low"], D["high"])
    D.to_csv(SRC / "q2_compilable.csv", index=False)
    print("\n== SRS+T (shipped) minus SRS+T on compilable candidates only, K = 20")
    print(D.groupby("target").agg(tally=("v", tally), mean=("point_diff", "mean"), lo=("point_diff", "min"), hi=("point_diff", "max")).round(4).to_string())
    R = pd.concat([pd.read_csv(ART / "compilable_only/union_k20" / l / "rows.csv").assign(label=l) for l in data.STANDARD], ignore_index=True)
    print(R.groupby("registry")[[c for c in R.columns if c in ("internal_auprc", "mean_target_auprc")]].mean().round(4).to_string())


# ---------------------------------------------------------------- wrapper decomposition
def decomposition() -> None:
    files = glob.glob(str(ART / "decomposition/*/*/paired.csv"))
    if not files:
        print("\n== decomposition: not computed yet")
        return
    D = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    D["endpoint"] = np.where(D["scope"] == "internal", "internal", np.where(D["scope"] == "real_targets", "real",
                                                                          np.where(D["target"] == "mean_over_targets", "targets", "per_target")))
    D = D[D.endpoint != "per_target"]
    rows = []
    for (a, b, e), grp in D.groupby(["a", "b", "endpoint"]):
        rows.append({"a": a, "b": b, "endpoint": e, **pooled(grp)})
    R = pd.DataFrame(rows)
    R.to_csv(SRC / "q2_decomposition.csv", index=False)
    print(f"\n== decomposition over {D.run.nunique()} runs (pooled over pools; tally over runs)")
    print(R.round(4).to_string(index=False))


# ---------------------------------------------------------------- seeds
def seeds() -> None:
    frames = []
    for f in glob.glob(str(ART / "seed_replicates/**/rows*.csv"), recursive=True):
        d = pd.read_csv(f)
        frames.append(d)
    if not frames:
        print("\n== seeds: none")
        return
    S = pd.concat(frames, ignore_index=True).drop_duplicates(["run", "registry", "seed"])
    base = data.rows_all()
    base = base[base.registry.isin(["srs_ap_t", "srs_ap", "srs_ap_unaware"])][["run", "registry", "internal_auprc", "mean_target_auprc"]].assign(seed=0)
    hyb = pd.concat([pd.read_csv(ART / "nmifs_ap" / r / l / "rows.csv").assign(run=f"{r}/{l}") for r, l in data.ICU_RUNS], ignore_index=True)
    hyb = hyb[hyb.registry == "nmifs_ap_t"][["run", "registry", "internal_auprc", "mean_target_auprc"]].assign(seed=0)
    allseeds = pd.concat([base, hyb, S[["run", "registry", "internal_auprc", "mean_target_auprc", "seed"]]], ignore_index=True)
    cnt = allseeds.groupby(["run", "registry"]).seed.nunique().unstack()
    print("\n== seeds available per run and registry")
    print(cnt.to_string())
    allseeds.to_csv(SRC / "q2_seeds_available.csv", index=False)  # q2_seeds_long.csv is seeds_pooled's


def seed_table() -> pd.DataFrame:
    """Long table of every registry's AUPRC per run and CV seed (seed 0: the shipped registries, from real_targets lift.csv;
    seeds 1-4: seed_replicates); endpoints internal, real (mean of eICU and HiRID) and targets (mean of all seven)."""
    rows = []
    for root, label in data.ICU_RUNS:
        base = ART / "real_targets" / root / label
        df = pd.read_csv(base / "lift.csv")
        prev = json.loads((base / "prevalence.json").read_text(encoding="utf-8"))
        for _, r in df.iterrows():
            rec = {"run": f"{root}/{label}", "registry": r.registry, "seed": 0, "internal": r.internal_lift + prev["internal_test"]}
            tg = {t: r[f"{t}_lift"] + prev[t] for t in REAL + CONTROLLED}
            rec["real"] = float(np.mean([tg[t] for t in REAL]))
            rec["targets"] = float(np.mean(list(tg.values())))
            rec["eicu"], rec["hirid"] = tg["eicu"], tg["hirid"]
            rec["controlled"] = float(np.mean([tg[t] for t in CONTROLLED]))
            rec.update({t: tg[t] for t in CONTROLLED})
            rows.append(rec)
    S0 = pd.DataFrame(rows)
    # seed-0 rows of the registries the real-target analysis did not cover (the likelihood versions), from the E3 rows
    ra = data.rows_all()
    extra = []
    for _, r in ra[ra.registry.isin(["srs_c_val", "srs_c_unknown_val"])].iterrows():
        run = r.run
        if ((S0.run == run) & (S0.registry == r.registry)).any():
            continue
        tg = {t: float(r[f"{t}_auprc"]) for t in REAL + CONTROLLED}
        extra.append({"run": run, "registry": r.registry, "seed": 0, "internal": float(r.internal_auprc),
                      "real": float(np.mean([tg[t] for t in REAL])), "targets": float(np.mean(list(tg.values()))), "eicu": tg["eicu"],
                      "hirid": tg["hirid"], "controlled": float(np.mean([tg[t] for t in CONTROLLED])), **{t: tg[t] for t in CONTROLLED}})
    S0 = pd.concat([S0, pd.DataFrame(extra)], ignore_index=True)
    frames = [pd.read_csv(f) for f in glob.glob(str(ART / "seed_replicates/**/rows*.csv"), recursive=True)]
    R = pd.concat(frames, ignore_index=True).drop_duplicates(["run", "registry", "seed"])
    R["real"] = R[[f"{t}_auprc" for t in REAL]].mean(axis=1)
    R["targets"] = R[[f"{t}_auprc" for t in REAL + CONTROLLED]].mean(axis=1)
    R["controlled"] = R[[f"{t}_auprc" for t in CONTROLLED]].mean(axis=1)
    R = R.rename(columns={"internal_auprc": "internal", "eicu_auprc": "eicu", "hirid_auprc": "hirid", **{f"{t}_auprc": t for t in CONTROLLED}})
    R = R[["run", "registry", "seed", "internal", "real", "targets", "eicu", "hirid", "controlled", *CONTROLLED]]
    return pd.concat([S0, R], ignore_index=True)


def seeds_pooled() -> None:
    """Seed-aware comparison of SRS+T with every comparator: per run, the difference of seed means, with variance = squared
    patient-bootstrap SE of the shipped pair (seed 0) + between-seed variance of each mean; pooled over the six pools (HKSJ).
    A run's verdict is significant when its normal interval excludes zero."""
    T = seed_table()
    T.to_csv(SRC / "q2_seeds_long.csv", index=False)
    g = T.groupby(["run", "registry"])
    stat = g[["internal", "real", "targets"]].agg(["mean", "var", "count"])
    # bootstrap SEs of the shipped pairs: internal and all targets from the first paired source, real from real_targets
    F = pair_frame()
    F = F[F.a == "srs_ap_t"].copy()
    F["se"] = (F["high"] - F["low"]) / (2 * 1.96)
    se = {("internal", r.run, r.b): r.se for r in F[F.endpoint == "internal"].itertuples()}
    se.update({("targets", r.run, r.b): r.se for r in F[F.endpoint == "targets"].itertuples()})
    P = pd.concat([pd.read_csv(ART / "real_targets" / r / l / "paired.csv") for r, l in data.ICU_RUNS], ignore_index=True)
    P = P[(P.a == "srs_ap_t") & (P.target == "mean_over_targets")]
    se.update({("real", r.run, r.b): (r.high - r.low) / (2 * 1.96) for r in P.itertuples()})
    comparators = sorted(set(T.registry) - {"srs_ap_t"})
    out, per_run = [], []
    for b in comparators:
        for e in ("internal", "real", "targets"):
            recs = []
            for root, label in data.ICU_RUNS:
                run = f"{root}/{label}"
                if (run, b) not in stat.index or (e, run, b) not in se:
                    continue
                a_m, a_v, a_n = stat.loc[(run, "srs_ap_t"), (e, "mean")], stat.loc[(run, "srs_ap_t"), (e, "var")], stat.loc[(run, "srs_ap_t"), (e, "count")]
                b_m, b_v, b_n = stat.loc[(run, b), (e, "mean")], stat.loc[(run, b), (e, "var")], stat.loc[(run, b), (e, "count")]
                var = se[(e, run, b)] ** 2 + (0 if a_n < 2 else a_v / a_n) + (0 if b_n < 2 else b_v / b_n)
                d = a_m - b_m
                rec = {"run": run, "b": b, "endpoint": e, "point_diff": d, "low": d - 1.96 * math.sqrt(var), "high": d + 1.96 * math.sqrt(var),
                       "n_seeds_a": int(a_n), "n_seeds_b": int(b_n)}
                recs.append(rec)
                per_run.append(rec)
            if len(recs) < 16:
                continue
            res = pooled(pd.DataFrame(recs))
            out.append({"a": "srs_ap_t", "b": b, "endpoint": e, **res, "min_seeds_a": min(r["n_seeds_a"] for r in recs),
                        "min_seeds_b": min(r["n_seeds_b"] for r in recs)})
    R = pd.DataFrame(out)
    R.to_csv(SRC / "q2_pooled_seeds.csv", index=False)
    pd.DataFrame(per_run).to_csv(SRC / "q2_seeds_per_run.csv", index=False)
    show = R.copy()
    show["b"] = show["b"].map(lambda n: NAMES.get(n, n))
    print("\n== seed-aware pooled effects of SRS+T (difference of seed means; patient + seed variance; HKSJ over pools)")
    print(show[["b", "endpoint", "pooled", "low", "high", "tau", "tally", "min_seeds_a", "min_seeds_b"]].round(4).to_string(index=False))
    M = T.groupby(["registry", "seed"])[["internal", "real", "targets"]].mean()
    for reg in ("srs_ap_t", "srs_ap", "nmifs_ap_t", "srs_ap_unaware"):
        if reg in M.index.get_level_values(0):
            m = M.loc[reg]
            print(f"  {NAMES[reg]}: seed means over the runs, internal {m.internal.mean():.4f} (sd {m.internal.std():.4f}), real {m.real.mean():.4f} "
                  f"(sd {m.real.std():.4f}), targets {m.targets.mean():.4f} (sd {m.targets.std():.4f}), seeds {len(m)}")


PARTS = {"seeds_pooled": seeds_pooled, "real": real_targets, "pooled": pooled_effects, "stress": stress, "brier": brier, "fidelity": fidelity, "resplits": resplits,
         "compilable": compilable, "decomposition": decomposition, "seeds": seeds}

if __name__ == "__main__":
    pd.set_option("display.width", 250)
    for part in (sys.argv[1:] or PARTS):
        PARTS[part]()
