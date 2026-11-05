"""Loaders and aggregations over artifacts/v2 for the PVLDB v3 figures and tables (SRS-AP results, exact-tie fix)."""
from __future__ import annotations

import json
from functools import lru_cache

import numpy as np
import pandas as pd

from style import ART

STANDARD = ("f_mimic_iv_gap1h", "f_mimic_iv_gap2h", "f_mimic_iv_gap4h", "f_mimic_iv_aki_gap6h")
LARGE = ("f_mimic_iv_gap1h", "f_mimic_iv_aki_gap6h")
ICU_RUNS = tuple((r, l) for r in ("union_k5", "union", "union_k20") for l in STANDARD) + \
    tuple((r, l) for r in ("union_large_k10", "union_large_k20") for l in LARGE)
K_OF = {"union_k5": 5, "union": 10, "union_k20": 20, "union_large_k10": 10, "union_large_k20": 20}
TASK = {"f_mimic_iv_gap1h": "Vaso 1h", "f_mimic_iv_gap2h": "Vaso 2h", "f_mimic_iv_gap4h": "Vaso 4h", "f_mimic_iv_aki_gap6h": "AKI"}
TARGETS = ("eicu", "hirid", "drop_labs", "drop_ventilation", "drop_bp", "drop_urine", "vitals_only")
TARGET_LABEL = {"eicu": "eICU", "hirid": "HiRID", "drop_labs": "no labs", "drop_ventilation": "no vent.", "drop_bp": "no BP",
                "drop_urine": "no urine", "vitals_only": "vitals only"}


def run_name(root: str, label: str) -> str:
    large = "L" if root.startswith("union_large") else ""
    return f"{TASK[label]}{' ' + large if large else ''} K{K_OF[root]}"


def generator_of(pid: str) -> str:
    for prefix, name in (("erminer_", "ERMiner"), ("prefixspan_", "PrefixSpan"), ("shapelet:", "Shapelet"),
                         ("matrix_profile:", "Motif"), ("program_", "FMG")):
        if pid.startswith(prefix):
            return name
    return "other"


@lru_cache(maxsize=None)
def rows_all() -> pd.DataFrame:
    """internal / per-target / mean-target AUPRC of every registry in the 16 ICU runs (union_ap/rows.csv)."""
    frames = []
    for root, label in ICU_RUNS:
        df = pd.read_csv(ART / "union_ap" / root / label / "rows.csv")
        df["root"], df["label"], df["k"], df["name"] = root, label, K_OF[root], run_name(root, label)
        df["large"] = root.startswith("union_large")
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


@lru_cache(maxsize=None)
def paired_all() -> pd.DataFrame:
    frames = []
    for root, label in ICU_RUNS:
        df = pd.read_csv(ART / "union_ap" / root / label / "paired.csv")
        df["root"], df["label"], df["k"], df["name"] = root, label, K_OF[root], run_name(root, label)
        frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    out["verdict"] = np.where(out["low"] > 0, 1, np.where(out["high"] < 0, -1, 0))
    return out


@lru_cache(maxsize=None)
def ablation_all() -> pd.DataFrame:
    frames = []
    for root, label in ICU_RUNS:
        df = pd.read_csv(ART / "union_ap" / root / label / "ablation_paired.csv")
        df["root"], df["label"], df["k"], df["name"] = root, label, K_OF[root], run_name(root, label)
        man = json.loads((ART / "union_ap" / root / label / "ablation_manifest.json").read_text(encoding="utf-8"))
        df["exec_share_unaware"] = df["target"].map(man["executable_share_unaware"])
        frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    out["verdict"] = np.where(out["low"] > 0, 1, np.where(out["high"] < 0, -1, 0))
    return out


@lru_cache(maxsize=None)
def stress_all() -> pd.DataFrame:
    frames = []
    for root, label in ICU_RUNS:
        df = pd.read_csv(ART / "stress_ap" / root / label / "stress.csv")
        df["root"], df["label"], df["name"] = root, label, run_name(root, label)
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


@lru_cache(maxsize=None)
def deploy_all() -> tuple[pd.DataFrame, pd.DataFrame]:
    rows, paired = [], []
    for root, label in ICU_RUNS:
        base = ART / "deployment_modes_ap" / root / label
        r, p = pd.read_csv(base / "rows.csv"), pd.read_csv(base / "paired.csv")
        for df in (r, p):
            df["root"], df["label"], df["name"] = root, label, run_name(root, label)
        rows.append(r)
        paired.append(p)
    p = pd.concat(paired, ignore_index=True)
    p["verdict"] = np.where(p["low"] > 0, 1, np.where(p["high"] < 0, -1, 0))
    return pd.concat(rows, ignore_index=True), p


@lru_cache(maxsize=None)
def hospital_all() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    rows, paired, manifests = [], [], {}
    for root, label in ICU_RUNS:
        base = ART / "hospital_schemas_ap/cov0.05_min1" / root / label
        r, p = pd.read_csv(base / "rows.csv"), pd.read_csv(base / "paired.csv")
        for df in (r, p):
            df["root"], df["label"], df["name"] = root, label, run_name(root, label)
        rows.append(r)
        paired.append(p)
        manifests[(root, label)] = json.loads((base / "manifest.json").read_text(encoding="utf-8"))
    p = pd.concat(paired, ignore_index=True)
    p["verdict"] = np.where(p["low"] > 0, 1, np.where(p["high"] < 0, -1, 0))
    return pd.concat(rows, ignore_index=True), p, manifests


def registries(root: str, label: str) -> dict[str, list[str]]:
    """primaries of every registry of a run: E3, fairness re-runs and SRS-AP."""
    out = {}
    for base in (ART / root / label, ART / "union_fairness" / root / label, ART / "union_extra" / root / label, ART / "union_ap" / root / label):
        path = base / "registries.json"
        if path.exists():
            for name, reg in json.loads(path.read_text(encoding="utf-8")).items():
                out.setdefault(name, reg["primaries"])
    for name, reg in json.loads((ART / "union_ap" / root / label / "registries.json").read_text(encoding="utf-8")).items():
        out[name] = reg["primaries"]
    return out


def drives() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows, paired, abl = [], [], []
    for k in (5, 10, 20):
        base = ART / f"union_ap/drive_k{k}/f_backblaze_seagate_gap2d"
        for target, name in ((rows, "rows.csv"), (paired, "paired.csv"), (abl, "ablation_paired.csv")):
            df = pd.read_csv(base / name)
            df["k"] = k
            target.append(df)
    p = pd.concat(paired, ignore_index=True)
    p["verdict"] = np.where(p["low"] > 0, 1, np.where(p["high"] < 0, -1, 0))
    a = pd.concat(abl, ignore_index=True)
    a["verdict"] = np.where(a["low"] > 0, 1, np.where(a["high"] < 0, -1, 0))
    return pd.concat(rows, ignore_index=True), p, a


def resplits() -> tuple[pd.DataFrame, pd.DataFrame]:
    rows, paired = [], []
    for task in ("gap1h", "aki_gap6h"):
        for i in range(1, 6):
            base = ART / f"union_ap/union_resplit/r{i}_mimic_iv_{task}"
            r, p = pd.read_csv(base / "rows.csv"), pd.read_csv(base / "paired.csv")
            for df in (r, p):
                df["task"], df["split"] = ("Vaso 1h" if task == "gap1h" else "AKI"), i
            rows.append(r)
            paired.append(p)
    p = pd.concat(paired, ignore_index=True)
    p["verdict"] = np.where(p["low"] > 0, 1, np.where(p["high"] < 0, -1, 0))
    return pd.concat(rows, ignore_index=True), p


def robustness() -> dict:
    out = {}
    for label in STANDARD:
        base = ART / "union_robustness_ap" / label
        out[label] = {name: pd.read_csv(base / f"{name}.csv") for name in ("decoder", "decoder_paired", "subgroups", "subgroup_paired")}
        out[label]["manifest"] = json.loads((base / "manifest.json").read_text(encoding="utf-8"))
    return out


def sensitivity() -> pd.DataFrame:
    frames = []
    for label in STANDARD:
        df = pd.read_csv(ART / "srs_sensitivity_ap" / label / "sensitivity.csv")
        df["label"] = label
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def restricted() -> pd.DataFrame:
    frames = []
    for path in sorted((ART / "restricted_pools").rglob("rows.csv")):
        df = pd.read_csv(path)
        df["restrict_dir"] = path.parts[-4]
        frames.append(df)
    return pd.concat(frames, ignore_index=True)
