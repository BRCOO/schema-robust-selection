"""How much do floating-point tie breaks move the reported AUPRC? (review of 2026-11-09; appendix, Statistics)

Average precision as evaluated (sklearn on the probe's probabilities) splits ties between episodes with identical coordinate
rows by last-bit BLAS noise, which depends on the BLAS configuration. Every probability is here rounded to 12 decimals, so that
mathematically tied scores tie exactly (the paper's definition: tied scores as one threshold), and both values are compared:

* ``icu``: the joint-bootstrap dumps (CV seed 0, 16 runs x 20 registries x 8 datasets), artifacts/v2/joint_bootstrap/*/*.npz;
* ``confirm``: the dumps of both confirmatory studies, artifacts/v2/eicu_source{,2}/dump/*/*.npz (every contrast of SRS+T);
* ``drives``: every drive registry redeployed as in run_union_ap.py; run with four BLAS threads to reproduce the evaluation.

  PYTHONDONTWRITEBYTECODE=1 /d/BRCoo/miniconda/python scripts/v2/paper_v3/tie_check.py icu|confirm|drives

Writes paper/v3_assets/source_data/q7_tie_check{,_confirm,_drives}.csv.
"""
import glob
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "paper/v3_assets/source_data"


def ap_both(y, p) -> tuple[float, float]:
    return average_precision_score(y, p), average_precision_score(y, np.round(p, 12))


def icu() -> None:
    rows = []
    for f in sorted(glob.glob(str(ROOT / "artifacts/v2/joint_bootstrap/*/*.npz"))):
        z = np.load(f, allow_pickle=True)
        run = Path(f).parent.name + "/" + Path(f).stem
        for ds in [str(d) for d in z["datasets"]]:
            y, p = z[f"y_{ds}"], z[f"p_{ds}"]
            for j, m in enumerate(str(x) for x in z["methods"]):
                raw, tied = ap_both(y, p[:, j])
                rows.append({"run": run, "registry": m, "dataset": ds, "ap_raw": raw, "ap_tied": tied})
    d = pd.DataFrame(rows)
    d.to_csv(OUT / "q7_tie_check.csv", index=False)
    d["endpoint"] = np.where(d.dataset.isin(["eicu", "hirid"]), "real", np.where(d.dataset == "internal", "internal", "controlled"))
    m = d.groupby(["endpoint", "run", "registry"])[["ap_raw", "ap_tied"]].mean().groupby(["endpoint", "registry"]).mean()
    print("16-run means, largest |tied - raw| x100 per endpoint:",
          (100 * (m.ap_tied - m.ap_raw)).abs().groupby("endpoint").max().round(4).to_dict())


def confirm() -> None:
    rows = []
    for study in ("eicu_source", "eicu_source2"):
        for f in sorted(glob.glob(str(ROOT / f"artifacts/v2/{study}/dump/*/*.npz"))):
            z = np.load(f, allow_pickle=True)
            for ds in ["sites"] + [str(x) for x in z["datasets"]]:
                if f"p_{ds}" not in z:
                    continue
                y, p = z[f"y_{ds}"], z[f"p_{ds}"]
                for j, m in enumerate(str(x) for x in z["methods"]):
                    raw, tied = ap_both(y, p[:, j])
                    rows.append({"study": study, "run": f, "dataset": ds, "registry": m.split("@")[0], "raw": raw, "tied": tied})
    d = pd.DataFrame(rows)
    d.to_csv(OUT / "q7_tie_check_confirm.csv", index=False)
    d["endpoint"] = np.where(d.dataset.isin(["mimic_iv", "hirid"]), "databases", d.dataset)
    for study, s in d.groupby("study"):
        g = s.groupby(["endpoint", "run", "registry"])[["raw", "tied"]].mean().groupby(["endpoint", "registry"]).mean()
        worst = 0.0
        for ep in g.index.get_level_values(0).unique():
            a = g.loc[(ep, "srs_ap_t")]
            for b, row in g.loc[ep].iterrows():
                if b != "srs_ap_t":
                    worst = max(worst, abs((a.tied - row.tied) - (a.raw - row.raw)))
        print(f"{study}: largest change of an SRS+T contrast (x100): {100 * worst:.4f}")


def drives() -> None:
    sys.path.insert(0, str(ROOT))
    spec = importlib.util.spec_from_file_location("run_union_ap", ROOT / "scripts/v2/run_union_ap.py")
    ap = importlib.util.module_from_spec(spec)
    sys.modules["run_union_ap"] = ap
    spec.loader.exec_module(ap)
    ctx = ap.ctx
    rows = []
    for k in (5, 10, 20):
        label = f"drive_k{k}/f_backblaze_seagate_gap2d"
        run = ctx.load_run(*ctx.drive_dirs(k))
        regs = dict(run.registries)
        hyb = ROOT / "artifacts/v2/nmifs_ap" / label / "registries.json"
        if hyb.exists():
            for name, reg in json.loads(hyb.read_text(encoding="utf-8")).items():
                regs.setdefault(name, ctx._composite(reg, run.by_id))
        if "nmifs" not in regs:  # as run_union_ap.py's evaluate phase
            regs["nmifs"] = run.composite(ap.nmifs(run.inp, run.k))
        y_test = np.asarray(run.labels["test"])
        for name, comp in sorted(regs.items()):
            p_test, per = ap.deploy(run, comp)
            sets = [("internal", y_test, p_test)] + [(t.name, np.array([e.label for e in t.episodes], dtype=int), per[t.name]) for t in run.targets]
            for ds, y, p in sets:
                raw, tied = ap_both(y, p)
                rows.append({"k": k, "registry": name, "dataset": ds, "raw": raw, "tied": tied})
        print(f"K={k}: {len(regs)} registries", flush=True)
    d = pd.DataFrame(rows)
    d.to_csv(OUT / "q7_tie_check_drives.csv", index=False)
    t = d[d.dataset != "internal"].groupby(["k", "registry"])[["raw", "tied"]].mean()
    t["diff_x100"] = 100 * (t.tied - t.raw)
    print(t[t.diff_x100.abs() > 0.01].round(4).to_string())


if __name__ == "__main__":
    {"icu": icu, "confirm": confirm, "drives": drives}[sys.argv[1]]()
