#!/usr/bin/env python3
"""End-to-end v2 track for one prediction gap (resumable; each step skips if its output exists).

Steps for gap g (full cohorts from scripts/v2/extract_v2_cohort.py, dirs <src>_gap<g>h_full;
with --task aki_onset the AKI cohorts <src>_aki_gap<g>h_full and output tags aki_gap<g>h):
  split      internal patient-level split + external episode lists (+ split copies for gating)
  gates      leakage gate on all three sources
  fmg        internal FMG per-episode mining -> support pre-filter -> train-only v1 selection,
             then shrink the internal cache to the candidates' rule ids and delete the big files
  fmg_ext    external FMG mining keeping only the candidates' rule ids
  compare    discovery comparison (5 methods x native/v2, real + controlled targets), pools saved
  supervised supervised references (internal test + frozen external evaluation if supported)

  PYTHONDONTWRITEBYTECODE=1 /d/BRCoo/miniconda/python scripts/v2/run_full_track.py --gap 1
  PYTHONDONTWRITEBYTECODE=1 /d/BRCoo/miniconda/python scripts/v2/run_full_track.py --gap 6 --task aki_onset
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PY = sys.executable
SOURCES = ("mimic_iv", "eicu", "hirid")
EXTERNALS = ("eicu", "hirid")
LABS = ["Lactate", "pH", "Creatinine", "Glucose", "Bilirubin_Total", "WBC", "Platelets"]
ENV = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", PYTHONIOENCODING="utf-8", CUDA_VISIBLE_DEVICES="")


def sh(args: list[str], log: Path) -> None:
    started = time.perf_counter()
    with open(log, "a", encoding="utf-8", errors="replace") as handle:
        handle.write(f"\n$ {' '.join(args)}\n")
        handle.flush()
        code = subprocess.run(args, cwd=str(ROOT), env=ENV, stdout=handle, stderr=subprocess.STDOUT).returncode
        handle.write(f"[exit {code} after {time.perf_counter() - started:.0f}s]\n")
    if code != 0:
        raise SystemExit(f"step failed ({code}): {' '.join(args)}; see {log}")


TASK_NOTES = {
    "vasopressor_initiation": "landmark never-vasopressor controls",
    "aki_onset": "AKI onset (KDIGO creatinine), landmark at-risk never-AKI controls",
}


def paths(gap: int, task: str = "vasopressor_initiation") -> dict[str, Path]:
    tag = f"gap{gap}h" if task == "vasopressor_initiation" else f"aki_gap{gap}h"
    x = ROOT / "data/processed/v2/extract"
    return {
        "extract": {s: x / f"{s}_{tag}_full" for s in SOURCES},
        "task": {s: ROOT / f"artifacts/v2/task/f_{s}_{tag}" for s in SOURCES},
        "gate_split": {s: ROOT / f"artifacts/v2/task/f_{s}_{tag}_gatesplit" for s in EXTERNALS},
        "mining": {s: ROOT / f"artifacts/v2/fmg_mining/f_{s}_{tag}" for s in SOURCES},
        "fmg_sel": ROOT / f"artifacts/v2/fmg_train_only/f_mimic_iv_{tag}",
        "comparison": ROOT / f"artifacts/v2/comparison/f_mimic_iv_{tag}",
        "config": ROOT / f"artifacts/v2/comparison/configs/f_mimic_iv_{tag}.json",
        "supervised": ROOT / f"artifacts/v2/supervised/f_mimic_iv_{tag}",
        "gates": ROOT / "artifacts/v2/leakage_audit",
        "tag": tag,
        "task_name": task,
    }


def step_split(p: dict, log: Path) -> None:
    if not (p["task"]["mimic_iv"] / "episodes_split.json").exists():
        sh([PY, "scripts/v2/split_extracted_cohort.py", "--extract-dir", str(p["extract"]["mimic_iv"]), "--out-dir", str(p["task"]["mimic_iv"]), "--seed", "20261021"], log)
    for s in EXTERNALS:
        if not (p["task"][s] / "episodes_all.json").exists():
            sh([PY, "scripts/v2/split_extracted_cohort.py", "--extract-dir", str(p["extract"][s]), "--out-dir", str(p["task"][s]), "--external"], log)
        if not (p["gate_split"][s] / "episodes_split.json").exists():
            sh([PY, "scripts/v2/split_extracted_cohort.py", "--extract-dir", str(p["extract"][s]), "--out-dir", str(p["gate_split"][s]), "--seed", "1"], log)


def step_gates(p: dict, log: Path) -> None:
    for s in SOURCES:
        out = p["gates"] / f"f_{s}_{p['tag']}_audit.json"
        if out.exists():
            continue
        split = p["task"][s] / "episodes_split.json" if s == "mimic_iv" else p["gate_split"][s] / "episodes_split.json"
        args = [PY, "scripts/v2/run_cohort_leakage_audit.py", "--episodes", str(split), "--episode-dir", str(p["extract"][s]), "--out-json", str(out)]
        with open(log, "a", encoding="utf-8") as handle:
            handle.write(f"\n$ {' '.join(args)}\n")
            handle.flush()
            code = subprocess.run(args, cwd=str(ROOT), env=ENV, stdout=handle, stderr=subprocess.STDOUT).returncode
        if code not in (0, 2):
            raise SystemExit(f"gate failed for {s}")
        if code == 2:
            raise SystemExit(f"LEAKAGE GATE FLAGGED {s} ({out}); stopping before any model is run")


def step_fmg(p: dict, log: Path) -> None:
    m = p["mining"]["mimic_iv"]
    candidates = m / "contrastive_inputs_candidates.json"
    keep = m / "fmg_candidate_rule_ids.json"
    if candidates.exists() and (p["fmg_sel"] / "fmg_programs_selected.json").exists():
        return
    full = m / "contrastive_inputs.json"
    prefiltered = m / "contrastive_inputs_train_prefiltered.json"
    split = p["task"]["mimic_iv"] / "episodes_split.json"
    if not full.exists():
        sh([PY, "scripts/v2/run_fmg_mining_sharded.py", "--input", str(p["extract"]["mimic_iv"]) + "_fmg", "--out-dir", str(m), "--shards", "32", "--workers", "6", "--delete-shard-caches"], log)
    if not (p["fmg_sel"] / "fmg_programs_selected.json").exists():
        if not prefiltered.exists():
            sh([PY, "scripts/v2/prefilter_fmg_inputs.py", "--contrastive-inputs", str(full), "--split-manifest", str(split), "--out", str(prefiltered)], log)
        sh([PY, "scripts/v2/run_fmg_train_only.py", "--contrastive-inputs", str(prefiltered), "--split-manifest", str(split), "--output-dir", str(p["fmg_sel"]), "--compact-memory"], log)
    ranked = json.loads((p["fmg_sel"] / "fmg_programs_ranked.json").read_text(encoding="utf-8"))
    ids = set()
    for program in ranked:
        ids.update(str(r) for r in program.get("member_rule_ids", []) or [])
        if program.get("representative_rule_id"):
            ids.add(str(program["representative_rule_id"]))
    keep.write_text(json.dumps(sorted(ids)), encoding="utf-8")
    sys.path.insert(0, str(ROOT))
    import importlib.util

    spec = importlib.util.spec_from_file_location("sharded", ROOT / "scripts/v2/run_fmg_mining_sharded.py")
    sharded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sharded)
    sharded.merge_shard_caches([full], candidates, frozenset(ids))
    full.unlink()
    prefiltered.unlink(missing_ok=True)


def step_fmg_ext(p: dict, log: Path) -> None:
    keep = p["mining"]["mimic_iv"] / "fmg_candidate_rule_ids.json"
    for s in EXTERNALS:
        if (p["mining"][s] / "contrastive_inputs.json").exists():
            continue
        sh([PY, "scripts/v2/run_fmg_mining_sharded.py", "--input", str(p["extract"][s]) + "_fmg", "--out-dir", str(p["mining"][s]), "--shards", "24", "--workers", "6", "--delete-shard-caches", "--keep-rule-ids", str(keep)], log)


def write_config(p: dict) -> None:
    avail = {s: json.loads((p["extract"][s] / "extraction_manifest.json").read_text(encoding="utf-8"))["available_features"] for s in SOURCES}
    cfg = {
        "_note": f"Final-track run, {p['tag']}: full eligible cohorts, fixed mappings, {TASK_NOTES[p['task_name']]}.",
        "cohort": str(p["task"]["mimic_iv"] / "episodes_split.json"),
        "episode_dir": str(p["extract"]["mimic_iv"]),
        "out_dir": str(p["comparison"]),
        "k": 10, "n_boot": 2000, "seed": 0,
        "representation": {"n_bins": 4, "observation_tokens": True, "trend_tokens": False},
        "internal_features": avail["mimic_iv"],
        "methods": {
            "fmg_bridge": {"fmg_dir": str(p["fmg_sel"]), "contrastive_inputs": str(p["mining"]["mimic_iv"] / "contrastive_inputs_candidates.json")},
            "prefixspan": {"minsup": 0.2, "max_pattern_length": 3, "pool_cap": 200},
            "erminer": {"minsup": 0.2, "minconf": 0.6, "max_antecedent": 2, "max_consequent": 1, "pool_cap": 200},
            "matrix_profile": {"lengths": [3, 4, 5], "top_k_per_feature": 3},
            "shapelet": {"lengths": [3, 4, 5], "top_k_per_feature": 3},
        },
        "targets": {
            s: {
                "episodes": str(p["task"][s] / "episodes_all.json"),
                "episode_dir": str(p["extract"][s]),
                "features": avail[s],
                "fmg_contrastive_inputs": str(p["mining"][s] / "contrastive_inputs.json"),
            }
            for s in EXTERNALS
        },
        "schema_shift_targets": {
            "drop_labs": {"drop": LABS},
            "drop_ventilation": {"drop": ["ventilation_on"]},
            "drop_bp": {"drop": ["SBP", "DBP", "MAP"]},
            "drop_urine": {"drop": ["urine_output"]},
            # AKI task: vasopressor_on is an input, so "vitals only" drops it as well
            "vitals_only": {"drop": LABS + ["ventilation_on", "sedation_on", "urine_output"] + (["vasopressor_on"] if p["task_name"] == "aki_onset" else [])},
        },
        "selector_targets": list(EXTERNALS),
        # v2 main selector: the v1 "no care-process program in the top 5" hard rule is off (see run_selector_analysis.BASE_OVERRIDES)
        "selector": {"care_process_cap": None},
        "reference": ["fmg_bridge", "native"],
    }
    p["config"].parent.mkdir(parents=True, exist_ok=True)
    p["config"].write_text(json.dumps(cfg, indent=2), encoding="utf-8")


def step_compare(p: dict, log: Path) -> None:
    if (p["comparison"] / "comparison_rows.csv").exists():
        return
    write_config(p)
    sh([PY, "scripts/v2/run_discovery_comparison.py", "--config", str(p["config"]), "--save-pools", str(p["comparison"] / "pools.pkl")], log)


def step_supervised(p: dict, log: Path) -> None:
    if (p["supervised"] / "supervised_v2_summary.json").exists():
        return
    args = [
        PY, "scripts/v2/run_supervised_v2.py", "--episodes", str(p["task"]["mimic_iv"] / "episodes_split.json"),
        "--data-dir", str(p["extract"]["mimic_iv"]), "--window", "stored", "--exclude-features", "",
        "--out-dir", str(p["supervised"]), "--num-threads", "6",
    ]
    help_text = subprocess.run([PY, "scripts/v2/run_supervised_v2.py", "--help"], cwd=str(ROOT), env=ENV, capture_output=True, text=True).stdout
    if "--external" in help_text:
        for s in EXTERNALS:
            args += ["--external", f"{s}={p['task'][s] / 'episodes_all.json'}::{p['extract'][s]}"]
    sh(args, log)


STEPS = {"split": step_split, "gates": step_gates, "fmg": step_fmg, "fmg_ext": step_fmg_ext, "compare": step_compare, "supervised": step_supervised}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--gap", type=int, required=True)
    parser.add_argument("--steps", default=",".join(STEPS))
    parser.add_argument("--task", choices=sorted(TASK_NOTES), default="vasopressor_initiation")
    args = parser.parse_args(argv)
    p = paths(args.gap, args.task)
    log = ROOT / f"artifacts/v2/logs/full_track_{p['tag']}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    for name in [s for s in args.steps.split(",") if s]:
        started = time.perf_counter()
        STEPS[name](p, log)
        free = shutil.disk_usage(str(ROOT)).free / 1e9
        print(f"[full-track {p['tag']}] {name} done in {time.perf_counter() - started:.0f}s; disk free {free:.0f} GB", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
