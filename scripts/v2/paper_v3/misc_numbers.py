"""Text-level numbers for the v3 experiments that are not panels of a figure (review of 2026-11-05): q per run and the
runs where SRS equals Wrapper, AUPRC; ablation tallies per target; the portability residual (target difference regressed
on internal difference); writes paper/v3_assets/source_data/misc_*.csv.

  PYTHONDONTWRITEBYTECODE=1 python scripts/v2/paper_v3/misc_numbers.py
"""
import glob
import json

import numpy as np
import pandas as pd

import data
from figs_v3 import all_paired
from style import ART, OUT_SRC

# ---- q per run, SRS == Wrapper, AUPRC
rows = []
for root, label in data.ICU_RUNS:
    sel = json.loads((ART / "union_ap" / root / label / "select.json").read_text(encoding="utf-8"))
    regs = json.loads((ART / "union_ap" / root / label / "registries.json").read_text(encoding="utf-8"))
    abl = json.loads((ART / "union_ap" / root / label / "ablations.json").read_text(encoding="utf-8"))
    same = regs["srs_ap"]["primaries"] == abl["srs_ap_unaware"]["primaries"]
    rows.append({"run": data.run_name(root, label), "q_srs_t": sel["srs_ap_t"]["q"], "q_srs": sel["srs_ap"]["q"], "srs_equals_wrapper_auprc": same,
                 "n_programs_srs_t": sel["srs_ap_t"]["n_programs"], "n_programs_srs": sel["srs_ap"]["n_programs"]})
for k in (5, 10, 20):
    base = ART / f"union_ap/drive_k{k}/f_backblaze_seagate_gap2d"
    sel = json.loads((base / "select.json").read_text(encoding="utf-8"))
    regs = json.loads((base / "registries.json").read_text(encoding="utf-8"))
    abl = json.loads((base / "ablations.json").read_text(encoding="utf-8"))
    rows.append({"run": f"drive K{k}", "q_srs_t": sel["srs_ap_t"]["q"], "q_srs": sel["srs_ap"]["q"],
                 "srs_equals_wrapper_auprc": regs["srs_ap"]["primaries"] == abl["srs_ap_unaware"]["primaries"],
                 "n_programs_srs_t": sel["srs_ap_t"]["n_programs"], "n_programs_srs": sel["srs_ap"]["n_programs"]})
q = pd.DataFrame(rows)
q.to_csv(OUT_SRC / "misc_q_per_run.csv", index=False)
print(q.to_string())

# ---- ablation tallies (SRS+T against each variant), mean target and per target
a = data.ablation_all()
a = a[a.a == "srs_ap_t"]
tal = a.groupby(["b", "target"]).verdict.agg(better=lambda v: int((v == 1).sum()), ns=lambda v: int((v == 0).sum()),
                                              worse=lambda v: int((v == -1).sum())).reset_index()
p = data.paired_all()
lik = p[(p.a == "srs_ap_t") & (p.b == "srs_c_val")].groupby("metric").verdict.agg(
    better=lambda v: int((v == 1).sum()), ns=lambda v: int((v == 0).sum()), worse=lambda v: int((v == -1).sum())).reset_index()
lik["b"] = "srs_c_val"
lik = lik.rename(columns={"metric": "target"})
tal = pd.concat([tal, lik], ignore_index=True)
tal.to_csv(OUT_SRC / "misc_ablation_tallies.csv", index=False)
print(tal[tal.target.isin(["internal", "mean_over_targets", "mean_target", "vitals_only", "drop_urine", "drop_bp", "drop_ventilation", "eicu"])].to_string())

# ---- portability residual: per run, target difference ~ a + b * internal difference (OLS, 95% CI by run bootstrap)
P = all_paired()
out = []
rng = np.random.default_rng(0)
for b in ("nmifs", "nmifs_ap_t", "mrmr", "srs_ap_unaware", "l1_dropout"):
    i = P[(P.b == b) & (P.metric == "internal")].set_index("name").point_diff
    t = P[(P.b == b) & (P.metric == "mean_target")].set_index("name").point_diff.loc[i.index]
    x, y = i.values, t.values

    def fit(xx, yy):
        A = np.c_[np.ones_like(xx), xx]
        return np.linalg.lstsq(A, yy, rcond=None)[0]

    coef = fit(x, y)
    boots = np.array([fit(x[s], y[s]) for s in (rng.integers(0, len(x), len(x)) for _ in range(2000))])
    lo, hi = np.percentile(boots[:, 0], [2.5, 97.5])
    out.append({"comparator": b, "intercept": coef[0], "intercept_low": lo, "intercept_high": hi, "slope": coef[1],
                "mean_internal_diff": x.mean(), "mean_target_diff": y.mean()})
res = pd.DataFrame(out)
res.to_csv(OUT_SRC / "misc_portability_residual.csv", index=False)
print(res.round(4).to_string())
