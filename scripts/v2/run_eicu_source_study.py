#!/usr/bin/env python3
"""Confirmatory study with a second source and real site schemas (protocol, committed before the first run:
docs/reviews/v2_confirmatory_eicu_protocol_2026-11-07.md).

eICU hospitals whose schema holds every harmonized feature are the source; hospitals whose schema lacks features are the
real site targets, grouped by schema and each deployed under its own schema; MIMIC-IV and HiRID are further targets; four
controlled targets are built from source TEST. Method and comparators are the paper's, frozen at the protocol's commit.

* ``--phase build``: cohorts, site groups and comparison configs (label-free design; split seed 20261107).
* ``--phase run``: the paper's drivers -- pools of the four non-FMG generators, union selection for K = 5, 10, 20, add-on
  comparators, NMIFS / stratified mRMR / IG-relevance selectors, SRS with the AUPRC decision. Resumable.
* ``--phase dump``: every registry deployed with surrogate chains (and by zeroing at the site targets); probabilities.
* ``--phase boot``: joint patient bootstrap (eICU, MIMIC-IV and HiRID patients drawn once per replicate for every run).
* ``--phase summarize``: differences of the 6-run means with 95% percentile intervals, per-run verdicts, H1-H5.

``--variant stat`` is the second study (protocol docs/reviews/v2_confirmatory_eicu_stat_protocol_2026-11-08.md): the same
design and cohorts, the pools extended by the window-statistic predicates (``add_statistic_predicates.py``), CV seeds 0-4
for the CV-based selectors (SRS+T, SRS, SRS+T with the likelihood decision; also SRS+T on the first study's pools), and the
statistic models of ``run_stat_baselines.py`` trained on the eICU source (``--phase stats``). Outputs under
artifacts/v2/eicu_source2 and the roots union_esrc2_k*; summarize gives H1-H7 with intervals that add the CV-seed variance.

  CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1 python scripts/v2/run_eicu_source_study.py --phase build
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
ART = ROOT / "artifacts/v2"
OUT = ART / "eicu_source"
#: --variant: "five" = the first study (protocol of 2026-11-07, four non-FMG generators); "stat" = the second study (protocol of
#: 2026-11-08, the pools extended by the window-statistic predicates); main() points OUT at artifacts/v2/eicu_source2 for it
VARIANT = "five"
SEEDED = ("srs_ap_t", "srs_ap", "srs_c_val")  # second study: CV-based selectors, averaged over CV seeds 0-4
CV_SEEDS = (1, 2, 3, 4)
STAT_MODELS = ("lr_stats", "gbdt_stats", "kstats")  # kstats = the K statistics of the L1 path at the run's budget K
ZERO = ("srs_ap_t", "nmifs", "mrmr")  # second study: also deployed by zeroing (H4)
NI_MARGIN = 0.02  # second study, H7: non-inferiority margin in AUPRC
SRC_DATA = ROOT / "paper/v3_assets/source_data"
TAGS = ("gap1h", "aki_gap6h")
BUDGETS = (5, 10, 20)
SEED = 20261107
MIN_EPISODES = 20
COVERAGE = 0.05
STATUS = ("sedation_on", "ventilation_on", "vasopressor_on")
SHORT = {"Lactate": "lactate", "pH": "ph", "sedation_on": "sedation", "Bilirubin_Total": "bilirubin", "urine_output": "urine",
         "vasopressor_on": "vasopressor", "Creatinine": "creatinine", "Glucose": "glucose", "WBC": "wbc", "Platelets": "platelets",
         "HR": "hr", "SBP": "sbp", "DBP": "dbp", "MAP": "map", "RR": "rr", "SpO2": "spo2", "Temp": "temp"}
CONTROLLED = ("drop_labs", "drop_bp", "drop_urine", "vitals_only")
METHODS = ("srs_ap_t", "srs_ap", "srs_c_val", "nmifs", "mrmr", "mrmr_stratified", "jmi", "cmim", "mmrfs", "mmr_ig", "mms_xin_ig",
           "dpp_ig", "l1_logistic", "l1_dropout", "mrmr_filter_known")
ESTABLISHED = ("nmifs", "mrmr", "mrmr_stratified", "jmi", "cmim", "mmrfs", "mmr_ig", "mms_xin_ig", "dpp_ig", "l1_logistic", "l1_dropout",
               "mrmr_filter_known")
PY = sys.executable
ENV = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", PYTHONIOENCODING="utf-8", CUDA_VISIBLE_DEVICES="")


def label_of(tag: str) -> str:
    return f"e_eicu_{tag}"


def root_of(k: int) -> str:
    return f"union_esrc_k{k}" if VARIANT == "five" else f"union_esrc2_k{k}"


def runs() -> list[tuple[str, str, str]]:
    return [(root_of(k), label_of(tag), tag) for tag in TAGS for k in BUDGETS]


def _module(name: str, path: Path):
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


# ----------------------------------------------------------------------------------------------------------------- build
def hospital_schemas(cov: dict, pooled: frozenset) -> dict[str, frozenset]:
    """As run_hospital_schemas.py: a measured feature when observed in >= 5% of the hospital's windows, a status feature
    when it is "on" in at least one window."""
    out = {}
    for h, n in cov["count"].items():
        keep = set()
        for f in pooled:
            if f in STATUS:
                if cov["on"].get(h, {}).get(f, 0) > 0:
                    keep.add(f)
            elif cov["observed"].get(h, {}).get(f, 0) / n >= COVERAGE:
                keep.add(f)
        out[h] = frozenset(keep)
    return out


def phase_build(args) -> None:
    from src.icu.v2.extract.common import episode_records
    from src.icu.v2.splits import patient_level_split, split_summary

    OUT.mkdir(parents=True, exist_ok=True)
    for tag in TAGS:
        label = label_of(tag)
        task_dir = ART / "task" / label
        cfg_path = ART / "comparison/configs" / f"{label}.json"
        if cfg_path.exists() and (task_dir / "episodes_split.json").exists():
            print(f"[build] {label}: done, skipped", flush=True)
            continue
        cfg0 = json.loads((ART / f"comparison/f_mimic_iv_{tag}/config.json").read_text(encoding="utf-8"))
        eicu = cfg0["targets"]["eicu"]
        pooled = frozenset(eicu["features"])
        cov = json.loads((ART / f"hospital_schemas_ap/coverage/{tag}.json").read_text(encoding="utf-8"))
        schema = hospital_schemas(cov, pooled)
        count = cov["count"]
        source_h = {h for h, n in count.items() if n >= MIN_EPISODES and schema[h] == pooled}
        site_h = {h for h, n in count.items() if n >= MIN_EPISODES and schema[h] != pooled}
        hosp = cov["episode_hospital"]
        episodes = episode_records(Path(eicu["episode_dir"]) / "shock_episode_index.json")
        assert {e.episode_id for e in episodes} == set(hosp), "the coverage cache and the extraction disagree"
        source = [e for e in episodes if hosp[e.episode_id] in source_h]
        split, split_manifest = patient_level_split(source, fractions=(0.7, 0.1, 0.2), seed=SEED)
        task_dir.mkdir(parents=True, exist_ok=True)
        (task_dir / "episodes_split.json").write_text(json.dumps([asdict(e) for e in split], ensure_ascii=False), encoding="utf-8")
        (task_dir / "split_manifest.json").write_text(json.dumps(split_manifest, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
        subjects = {e.subject_id for e in source}
        groups: dict[tuple, list[str]] = {}
        for h in sorted(site_h):
            groups.setdefault(tuple(sorted(pooled - schema[h])), []).append(h)
        targets, design_groups = {}, []
        for missing, hs in sorted(groups.items(), key=lambda kv: (len(kv[0]), kv[0])):
            name = "site_" + "_".join(SHORT[f] for f in missing)
            members = [e for e in episodes if hosp[e.episode_id] in set(hs)]
            rows = [asdict(replace(e, split="external")) for e in members if e.subject_id not in subjects]
            path = task_dir / f"target_{name}.json"
            path.write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
            targets[name] = {"episodes": str(path), "episode_dir": eicu["episode_dir"], "features": sorted(pooled - set(missing))}
            design_groups.append({"target": name, "missing": list(missing), "hospitals": hs, "episodes": len(rows),
                                  "removed_shared_patients": len(members) - len(rows), "positives": int(sum(r["label"] for r in rows))})
        mimic = [dict(r, split="external") for r in json.loads((ART / f"task/f_mimic_iv_{tag}/episodes_split.json").read_text(encoding="utf-8"))]
        path = task_dir / "target_mimic_iv.json"
        path.write_text(json.dumps(mimic, ensure_ascii=False), encoding="utf-8")
        targets["mimic_iv"] = {"episodes": str(path), "episode_dir": cfg0["episode_dir"], "features": list(cfg0["internal_features"])}
        targets["hirid"] = {k: v for k, v in cfg0["targets"]["hirid"].items() if k != "fmg_contrastive_inputs"}
        shift = {name: {"drop": [f for f in cfg0["schema_shift_targets"][name]["drop"] if f in pooled]} for name in CONTROLLED}
        known = [g["target"] for g in design_groups] + ["mimic_iv", "hirid"]
        config = dict(cfg0)
        config.update({
            "_note": f"Confirmatory study (docs/reviews/v2_confirmatory_eicu_protocol_2026-11-07.md), {tag}: eICU hospitals with the full "
                     "harmonized schema are the source, hospitals lacking features are site targets; four non-FMG generators.",
            "cohort": str(task_dir / "episodes_split.json"), "episode_dir": eicu["episode_dir"], "out_dir": str(ART / "comparison" / label),
            "internal_features": list(eicu["features"]), "methods": {m: v for m, v in cfg0["methods"].items() if m != "fmg_bridge"},
            "targets": targets, "schema_shift_targets": shift, "selector_targets": known, "reference": None, "n_boot": 200})
        cfg_path.parent.mkdir(parents=True, exist_ok=True)
        cfg_path.write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")
        design = {"tag": tag, "protocol": "docs/reviews/v2_confirmatory_eicu_protocol_2026-11-07.md", "coverage": COVERAGE,
                  "min_episodes": MIN_EPISODES, "seed": SEED, "source_hospitals": sorted(source_h), "source_episodes": len(source),
                  "split": split_summary(split), "site_groups": design_groups,
                  "excluded_hospitals": {h: n for h, n in count.items() if n < MIN_EPISODES}, "known_targets": known,
                  "controlled": shift}
        (OUT / f"design_{tag}.json").write_text(json.dumps(design, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
        print(f"[build] {label}: source {len(source_h)} hospitals / {len(source)} episodes; sites "
              + ", ".join(f"{g['target']} {g['episodes']}" for g in design_groups), flush=True)


# ------------------------------------------------------------------------------------------------------------------- run
def sh(cmd: list[str], log: Path) -> None:
    started = time.perf_counter()
    with open(log, "a", encoding="utf-8", errors="replace") as handle:
        handle.write(f"\n$ {' '.join(cmd)}\n")
        handle.flush()
        code = subprocess.run(cmd, cwd=str(ROOT), env=ENV, stdout=handle, stderr=subprocess.STDOUT).returncode
        handle.write(f"[exit {code} after {time.perf_counter() - started:.0f}s]\n")
    print(f"[run] {' '.join(cmd[1:3])} ... exit {code} after {time.perf_counter() - started:.0f}s", flush=True)
    if code != 0:
        raise SystemExit(f"step failed ({code}): {' '.join(cmd)}; see {log}")


def phase_run(args) -> None:
    (ART / "logs").mkdir(parents=True, exist_ok=True)
    budgets = [k for k in BUDGETS if not args.budgets or str(k) in args.budgets.split(",")]
    for tag in [t for t in TAGS if not args.tags or t in args.tags.split(",")]:
        label = label_of(tag)
        cfg_path = ART / "comparison/configs" / f"{label}.json"
        known = json.loads(cfg_path.read_text(encoding="utf-8"))["selector_targets"]
        log = ART / f"logs/eicu_source{'' if VARIANT == 'five' else '2'}_{tag}.log"
        comparison = ART / "comparison" / label
        if not (comparison / "pools.pkl").exists():
            sh([PY, "scripts/v2/run_discovery_comparison.py", "--config", str(cfg_path), "--save-pools", str(comparison / "pools.pkl")], log)
        pools = comparison / "pools.pkl"
        if VARIANT == "stat":  # the first study's pools plus the window-statistic predicates, mined on the same source TRAIN
            pools = comparison / "pools_stat.pkl"
            if not pools.exists():
                sh([PY, "scripts/v2/add_statistic_predicates.py", "--comparison", str(comparison)], log)
        for k in budgets:
            root = root_of(k)
            e3, extra = ART / root / label, ART / "union_extra" / root / label
            if not (e3 / "manifest.json").exists():
                sh([PY, "scripts/v2/run_union_selection.py", "--pools", str(pools), "--e1-dir", "", "--k", str(k),
                    "--out-dir", str(e3), "--known-targets", ",".join(known)], log)
            if not (extra / "manifest.json").exists():
                sh([PY, "scripts/v2/run_union_extra.py", "--e3-dir", str(e3), "--out-dir", str(extra), "--stress-levels", "0.5",
                    "--stress-draws", "1"], log)
            if not (ART / "union_fairness" / root / label / "registries.json").exists():
                sh([PY, "scripts/v2/run_union_fairness.py", "--runs", f"{root}:{label}"], log)
            if not (ART / "union_ap" / root / label / "select.json").exists():
                sh([PY, "scripts/v2/run_union_ap.py", "--phase", "select", "--runs", f"{root}:{label}", "--drive-k", ""], log)
            if VARIANT == "stat":  # CV seeds 1-4 of the CV-based selectors, and of SRS+T on the first study's pools (H6)
                seeds, five = OUT / "seeds", f"union_esrc_k{k}"
                if not (seeds / root / label / "rows_esrc.csv").exists():
                    sh([PY, "scripts/v2/run_reference_registries.py", "--phase", "seeds", "--root", root, "--runs", f"{root}:{label}",
                        "--configs", ",".join(SEEDED), "--tag", "esrc", "--seeds-out", str(seeds)], log)
                if not (seeds / five / label / "rows_esrc_five.csv").exists():
                    sh([PY, "scripts/v2/run_reference_registries.py", "--phase", "seeds", "--root", five, "--runs", f"{five}:{label}",
                        "--configs", "srs_ap_t", "--tag", "esrc_five", "--seeds-out", str(seeds)], log)


# ----------------------------------------------------------------------------------------------------------------- stats
def phase_stats(args) -> None:
    """Second study: the statistic models of run_stat_baselines.py (every window statistic with L2 logistic regression or
    gradient-boosted trees; the K statistics first chosen on an L1 path, refitted, K = 5, 10, 20) trained on the eICU source
    TRAIN with every choice on VAL, scored on each dataset of the design: source TEST; the site targets, each episode with its
    hospital group's absent features missing in every bin; MIMIC-IV and HiRID with the features outside their schemas missing
    in every bin; the controlled targets (source TEST with the dropped features missing). Missing statistics take the TRAIN
    medians, as in run_stat_baselines.py. Writes OUT/stats_<tag>.npz: episode ids and probabilities per model and dataset."""
    sb = _module("run_stat_baselines", ROOT / "scripts/v2/run_stat_baselines.py")
    from src.icu.v2.supervised import FileArrayLoader, build_supervised_inputs, load_episode_list, read_feature_names

    OUT.mkdir(parents=True, exist_ok=True)
    for tag in [t for t in TAGS if not args.tags or t in args.tags.split(",")]:
        out = OUT / f"stats_{tag}.npz"
        if out.exists():
            print(f"[stats] {tag}: done, skipped", flush=True)
            continue
        t0 = time.perf_counter()
        cfg = json.loads((ART / "comparison/configs" / f"{label_of(tag)}.json").read_text(encoding="utf-8"))
        schema = set(cfg["internal_features"])
        src_dir = Path(cfg["episode_dir"])
        episodes = load_episode_list(Path(cfg["cohort"]))
        names = read_feature_names(src_dir, episodes[0].episode_id)
        loader = FileArrayLoader(src_dir)
        inputs = build_supervised_inputs(episodes, names, loader, excluded_features=tuple(f for f in names if f not in schema))
        tr, va, te = (inputs.splits[s] for s in ("train", "val", "test"))
        order = {e: i for i, e in enumerate(te.episode_ids)}
        test_eps = sorted((e for e in episodes if e.split == "test"), key=lambda e: order[e.episode_id])
        data = {"internal": (list(te.episode_ids), te.tabular)}
        site_ids, site_rows = [], []
        for name, spec in cfg["targets"].items():
            eps = load_episode_list(Path(spec["episodes"]))
            t_dir = Path(spec["episode_dir"])
            t_names = read_feature_names(t_dir, eps[0].episode_id)
            X = sb.masked_test(inputs, eps, FileArrayLoader(t_dir), t_names, schema - set(spec["features"]))
            if name.startswith("site_"):
                site_ids += [e.episode_id for e in eps]
                site_rows.append(X)
            else:
                data[name] = ([e.episode_id for e in eps], X)
        data["sites"] = (site_ids, np.vstack(site_rows))
        for name, spec in cfg["schema_shift_targets"].items():
            data[name] = ([e.episode_id for e in test_eps], sb.masked_test(inputs, test_eps, loader, names, set(spec["drop"])))
        print(f"[stats] {tag}: inputs {tr.tabular.shape}, datasets " + ", ".join(f"{d} {len(i)}" for d, (i, _) in data.items())
              + f" ({time.perf_counter() - t0:.0f}s)", flush=True)
        models, details = {}, {}
        model, details["lr_stats"] = sb.fit_lr(tr.tabular, tr.y, va.tabular, va.y)
        models["lr_stats"] = (lambda m: (lambda X: m.predict_proba(X)[:, 1]))(model)
        models["gbdt_stats"], details["gbdt_stats"] = sb.fit_gbdt(tr.tabular, tr.y, va.tabular, va.y)
        for k in BUDGETS:
            cols = sb.l1_columns(tr.tabular, tr.y, k)
            model_k, info = sb.fit_lr(tr.tabular[:, cols], tr.y, va.tabular[:, cols], va.y)
            info["columns"] = [inputs.tabular_feature_names[c] for c in cols]
            models[f"kstats_k{k}"], details[f"kstats_k{k}"] = (lambda m, c: (lambda X: m.predict_proba(X[:, c])[:, 1]))(model_k, cols), info
        arrays = {"models": np.array(list(models)), "datasets": np.array(list(data))}
        for d, (ids, X) in data.items():
            arrays[f"ids_{d}"] = np.array(ids)
            for m, predict in models.items():
                arrays[f"p_{m}_{d}"] = np.asarray(predict(X), dtype=np.float64)
        np.savez_compressed(out, **arrays)
        (OUT / f"stats_{tag}.json").write_text(json.dumps(details, indent=1, default=str), encoding="utf-8")
        print(f"[stats] {tag}: done ({time.perf_counter() - t0:.0f}s); internal AUPRC "
              + ", ".join(f"{m} {sb.auprc(te.y, arrays[f'p_{m}_internal']):.3f}" for m in models), flush=True)


# ------------------------------------------------------------------------------------------------------------------ dump
def phase_dump(args) -> None:
    ctx = _module("e3_context", ROOT / "scripts/v2/e3_context.py")
    from src.icu.v2.bridge import refire_external
    from src.icu.v2.surrogates import refire_composite

    for root, label, tag in runs():
        out = OUT / "dump" / root / f"{label}.npz"
        if out.exists():
            print(f"[dump] {root}/{label}: done, skipped", flush=True)
            continue
        t0 = time.perf_counter()
        run = ctx.load_run(*ctx.run_dirs(root, label))
        missing = [m for m in METHODS if m not in run.registries]
        if missing:
            raise SystemExit(f"{root}/{label}: registries missing: {missing}")
        # column -> (the run whose pool the registry indexes, registry)
        columns = {m: (run, run.registries[m]) for m in METHODS}
        zero_methods, stats, k = list(METHODS), None, int(root.rsplit("_k", 1)[1])
        if VARIANT == "stat":
            seeds = json.loads((OUT / "seeds" / root / label / "registries_esrc.json").read_text(encoding="utf-8"))
            for m in SEEDED:
                for s in CV_SEEDS:
                    columns[f"{m}@s{s}"] = (run, ctx._composite(seeds[f"{m}_seed{s}"], run.by_id))
            five_root = f"union_esrc_k{k}"
            run5 = ctx.load_run(*ctx.run_dirs(five_root, label))
            episodes_of = lambda r: (r.ids, [(t.name, [e.episode_id for e in t.episodes]) for t in r.targets])  # noqa: E731
            if episodes_of(run5) != episodes_of(run):
                raise SystemExit(f"{root}/{label}: the first study's run has other episodes")
            five = json.loads((OUT / "seeds" / five_root / label / "registries_esrc_five.json").read_text(encoding="utf-8"))
            columns["srs_ap_t_five"] = (run5, run5.registries["srs_ap_t"])
            for s in CV_SEEDS:
                columns[f"srs_ap_t_five@s{s}"] = (run5, ctx._composite(five[f"srs_ap_t_seed{s}"], run5.by_id))
            zero_methods = [m for m in columns if m.split("@")[0] in ZERO]
            stats = np.load(OUT / f"stats_{tag}.npz", allow_pickle=False)
        sites = [t for t in run.targets if t.name.startswith("site_")]
        others = [t for t in run.targets if not t.name.startswith("site_")]
        datasets = {"internal": run.grouped["test"], **{t.name: t.episodes for t in others},
                    "sites": [e for t in sites for e in t.episodes]}
        probs = {d: [] for d in datasets}
        zero = []
        for m, (r, comp) in columns.items():
            bridge = r.fit(list(comp.primaries))
            bundle = bridge["model_bundle"]
            probs["internal"].append(np.array([bridge["test_probabilities"][e] for e in r.ids["test"]]))
            for t in others:
                eids = [e.episode_id for e in t.episodes]
                p = refire_composite(bundle, comp, t.features, r.external[t.name], eids)["probabilities"]
                probs[t.name].append(np.array([p[e] for e in eids]))
            ps, pz = [], []
            for t in sites:
                eids = [e.episode_id for e in t.episodes]
                p = refire_composite(bundle, comp, t.features, r.external[t.name], eids)["probabilities"]
                ps.append(np.array([p[e] for e in eids]))
                if m in zero_methods:
                    q = refire_external(bundle, list(comp.primaries), t.features, r.external[t.name], eids)["probabilities"]
                    pz.append(np.array([q[e] for e in eids]))
            probs["sites"].append(np.concatenate(ps))
            if m in zero_methods:
                zero.append(np.concatenate(pz))
        names = list(columns)
        if stats is not None:  # the statistic models, scored per episode by phase_stats
            for m in STAT_MODELS:
                model = f"kstats_k{k}" if m == "kstats" else m
                for d, eps in datasets.items():
                    lookup = dict(zip((str(x) for x in stats[f"ids_{d}"]), stats[f"p_{model}_{d}"]))
                    probs[d].append(np.array([lookup[e.episode_id] for e in eps]))
            names += list(STAT_MODELS)
        arrays = {"methods": np.array(names), "zero_methods": np.array(zero_methods), "datasets": np.array(list(datasets)),
                  "site_targets": np.array([t.name for t in sites]), "site_sizes": np.array([len(t.episodes) for t in sites])}
        for d, eps in datasets.items():
            arrays[f"y_{d}"] = np.array([e.label for e in eps], dtype=np.int8)
            arrays[f"g_{d}"] = np.array([str(e.subject_id) for e in eps])
            arrays[f"p_{d}"] = np.column_stack(probs[d]).astype(np.float64)
        arrays["p_sites_zero"] = np.column_stack(zero).astype(np.float64)
        out.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out, **arrays)
        print(f"[dump] {root}/{label} ({time.perf_counter() - t0:.0f}s): " + ", ".join(f"{d} {len(e)}" for d, e in datasets.items()), flush=True)


# ------------------------------------------------------------------------------------------------------------------ boot
def database_of(dataset: str) -> str:
    return {"mimic_iv": "mimic", "hirid": "hirid"}.get(dataset, "eicu")


def phase_boot(args) -> None:
    jb = _module("run_joint_bootstrap", ROOT / "scripts/v2/run_joint_bootstrap.py")
    loaded = [(f"{root}/{label}", np.load(OUT / "dump" / root / f"{label}.npz", allow_pickle=False)) for root, label, _ in runs()]
    methods = [str(m) for m in loaded[0][1]["methods"]]
    zero_methods = [str(m) for m in loaded[0][1]["zero_methods"]] if "zero_methods" in loaded[0][1].files else methods
    datasets = [str(d) for d in loaded[0][1]["datasets"]]
    for _, z in loaded:
        assert [str(d) for d in z["datasets"]] == datasets and [str(m) for m in z["methods"]] == methods
    index: dict[str, dict[str, int]] = {"eicu": {}, "mimic": {}, "hirid": {}}
    for _, z in loaded:
        for d in datasets:
            ix = index[database_of(d)]
            for g in z[f"g_{d}"]:
                ix.setdefault(str(g), len(ix))
    sizes = {db: len(ix) for db, ix in index.items()}
    print(f"[boot] patients: {sizes}", flush=True)
    # columns: the methods with surrogates, then (sites only) the methods deployed by zeroing
    cols = methods + [f"{m}:zero" for m in zero_methods]
    prep = []
    for run, z in loaded:
        for d in datasets:
            y = z[f"y_{d}"].astype(int)
            g = np.array([index[database_of(d)][str(x)] for x in z[f"g_{d}"]])
            P = z[f"p_{d}"]
            if d == "sites":
                P = np.column_stack([P, z["p_sites_zero"]])
            prep.append((run, d, g, [jb.ap_prepare(y, P[:, j]) for j in range(P.shape[1])]))
    R, M, D = len(loaded), len(cols), len(datasets)
    point = np.full((R, M, D), np.nan)
    for i, (run, d, g, per_m) in enumerate(prep):
        r, k = divmod(i, D)
        for j, (order, ends, ys) in enumerate(per_m):
            point[r, j, k] = jb.weighted_ap(np.ones((1, len(g))), order, ends, ys)[0]
    rng = np.random.default_rng(args.seed)
    B, chunk = args.n_boot, args.chunk
    boot = np.full((B, R, M, D), np.nan)
    t0 = time.perf_counter()
    for start in range(0, B, chunk):
        nb = min(chunk, B - start)
        counts = {db: np.stack([np.bincount(rng.integers(0, n, n), minlength=n) for _ in range(nb)]).astype(np.float64) for db, n in sizes.items()}
        for i, (run, d, g, per_m) in enumerate(prep):
            r, k = divmod(i, D)
            W = counts[database_of(d)][:, g]
            for j, (order, ends, ys) in enumerate(per_m):
                boot[start:start + nb, r, j, k] = jb.weighted_ap(W, order, ends, ys)
        print(f"[boot] {start + nb}/{B} ({time.perf_counter() - t0:.0f}s)", flush=True)
    np.savez_compressed(OUT / "boot.npz", boot=boot, point=point, runs=np.array([r for r, _ in loaded]), cols=np.array(cols),
                        datasets=np.array(datasets), seed=args.seed)
    print(f"[boot] wrote {OUT / 'boot.npz'}", flush=True)


# ------------------------------------------------------------------------------------------------------------- summarize
def endpoint(arr: np.ndarray, datasets: list[str], name: str) -> np.ndarray:
    idx = {d: i for i, d in enumerate(datasets)}
    groups = {"sites": ("sites",), "databases": ("mimic_iv", "hirid"), "controlled": CONTROLLED, "internal": ("internal",)}[name]
    return np.mean(np.stack([arr[..., idx[d]] for d in groups], axis=-1), axis=-1)


HYPOTHESES = (("H1", "srs_ap_t", "nmifs", "sites"), ("H2", "srs_ap_t", "mrmr", "sites"), ("H3", "srs_ap_t", "nmifs", "databases"),
              ("H4a", "srs_ap_t", "srs_ap_t:zero", "sites"), ("H4b", "nmifs", "nmifs:zero", "sites"), ("H4c", "mrmr", "mrmr:zero", "sites"),
              ("H5", "srs_ap", "nmifs", "sites"))


def phase_summarize(args) -> None:
    import pandas as pd

    if VARIANT == "stat":
        summarize_stat()
        return
    z = np.load(OUT / "boot.npz", allow_pickle=False)
    boot, point = z["boot"], z["point"]
    run_names, cols, datasets = [str(x) for x in z["runs"]], [str(x) for x in z["cols"]], [str(x) for x in z["datasets"]]
    ci = {c: i for i, c in enumerate(cols)}
    pairs = [(h, a, b, e) for h, a, b, e in HYPOTHESES]
    pairs += [("desc", "srs_ap_t", b, e) for b in ESTABLISHED + ("srs_ap", "srs_c_val") for e in ("sites", "databases", "controlled", "internal")
              if not any(p[1] == "srs_ap_t" and p[2] == b and p[3] == e for p in pairs)]
    pairs += [("desc", "srs_c_val", "nmifs", e) for e in ("sites", "databases", "controlled", "internal")]
    pairs += [("desc", "srs_ap", "nmifs", e) for e in ("databases", "controlled", "internal")]
    rows, per_run = [], []
    for h, a, b, e in pairs:
        va, vb = endpoint(point[:, ci[a]], datasets, e), endpoint(point[:, ci[b]], datasets, e)
        ba, bb = endpoint(boot[:, :, ci[a]], datasets, e), endpoint(boot[:, :, ci[b]], datasets, e)
        d_runs, d_boot = va - vb, ba - bb
        lo, hi = np.percentile(d_boot.mean(axis=1), [2.5, 97.5])
        lo_r, hi_r = np.percentile(d_boot, [2.5, 97.5], axis=0)
        verdict = np.where(lo_r > 0, 1, np.where(hi_r < 0, -1, 0))
        rows.append({"hypothesis": h, "a": a, "b": b, "endpoint": e, "diff": float(d_runs.mean()), "low": float(lo), "high": float(hi),
                     "holds": bool(lo > 0) if h != "desc" else None, "better": int((verdict == 1).sum()), "ns": int((verdict == 0).sum()),
                     "worse": int((verdict == -1).sum()), "mean_a": float(va.mean()), "mean_b": float(vb.mean())})
        for r, run in enumerate(run_names):
            per_run.append({"hypothesis": h, "a": a, "b": b, "endpoint": e, "run": run, "diff": float(d_runs[r]), "low": float(lo_r[r]),
                            "high": float(hi_r[r]), "a_value": float(va[r]), "b_value": float(vb[r])})
    S = pd.DataFrame(rows)
    S.to_csv(OUT / "summary.csv", index=False)
    pd.DataFrame(per_run).to_csv(OUT / "per_run.csv", index=False)
    # means per method and endpoint (6-run means) for the paper's table
    means = []
    for c in cols:
        for e in ("sites", "databases", "controlled", "internal"):
            v = endpoint(point[:, ci[c]], datasets, e)
            if np.isfinite(v).all():
                means.append({"registry": c, "endpoint": e, "mean": float(v.mean()), **{run: float(x) for run, x in zip(run_names, v)}})
    pd.DataFrame(means).to_csv(OUT / "means.csv", index=False)
    SRC_DATA.mkdir(parents=True, exist_ok=True)
    S.to_csv(SRC_DATA / "q4_confirmatory_summary.csv", index=False)
    pd.DataFrame(means).to_csv(SRC_DATA / "q4_confirmatory_means.csv", index=False)
    show = S[S.hypothesis != "desc"].copy()
    for c in ("diff", "low", "high"):
        show[c] = (100 * show[c]).round(2)
    pd.set_option("display.width", 220)
    print(show[["hypothesis", "a", "b", "endpoint", "diff", "low", "high", "holds", "better", "ns", "worse"]].to_string(index=False))


#: second study: (hypothesis, a, b, endpoint, margin); a hypothesis holds when the interval's lower bound exceeds the margin
HYPOTHESES2 = (("H1", "srs_ap_t", "nmifs", "sites", 0.0), ("H2", "srs_ap_t", "mrmr", "sites", 0.0),
               ("H3", "srs_ap_t", "nmifs", "databases", 0.0), ("H4a", "srs_ap_t", "srs_ap_t:zero", "sites", 0.0),
               ("H4b", "nmifs", "nmifs:zero", "sites", 0.0), ("H4c", "mrmr", "mrmr:zero", "sites", 0.0),
               ("H5", "srs_ap", "nmifs", "sites", 0.0), ("H6", "srs_ap_t", "srs_ap_t_five", "sites", 0.0),
               ("H7", "srs_ap_t", "kstats", "sites", -NI_MARGIN))


def members(name: str) -> list[str]:
    """The columns averaged for a method: CV seeds 0-4 for the CV-based selectors (and for SRS+T's zeroing), else itself."""
    base, _, suffix = name.partition(":")
    if base in SEEDED + ("srs_ap_t_five",):
        return [name] + [f"{base}@s{s}" + (f":{suffix}" if suffix else "") for s in CV_SEEDS]
    return [name]


def summarize_stat() -> None:
    """Second study: per comparison, the difference of the 6-run means (seed means for the CV-based selectors) with the 95%
    interval point +- 1.96 sqrt(v_boot + v_seed): v_boot the joint patient-bootstrap variance of the difference, v_seed the
    between-seed variance of each seed mean (sum over runs / 36). Per run: the same with the run's own variances."""
    import pandas as pd

    z = np.load(OUT / "boot.npz", allow_pickle=False)
    boot, point = z["boot"], z["point"]
    run_names, cols, datasets = [str(x) for x in z["runs"]], [str(x) for x in z["cols"]], [str(x) for x in z["datasets"]]
    ci = {c: i for i, c in enumerate(cols)}
    R = len(run_names)

    def values(name: str, e: str):
        J = [ci[c] for c in members(name)]
        pe = np.stack([endpoint(point[:, j], datasets, e) for j in J], axis=-1)  # runs x seeds
        be = np.stack([endpoint(boot[:, :, j], datasets, e) for j in J], axis=-1)  # replicates x runs x seeds
        sv = pe.var(axis=1, ddof=1) / len(J) if len(J) > 1 else np.zeros(R)
        return pe.mean(axis=1), be.mean(axis=2), sv

    pairs = [(h, a, b, e, m) for h, a, b, e, m in HYPOTHESES2]
    comparators = ESTABLISHED + ("srs_ap", "srs_c_val", "srs_ap_t_five") + STAT_MODELS
    pairs += [("desc", "srs_ap_t", b, e, 0.0) for b in comparators for e in ("sites", "databases", "controlled", "internal")
              if not any(p[1] == "srs_ap_t" and p[2] == b and p[3] == e for p in pairs)]
    pairs += [("desc", a, b, e, 0.0) for a, b in (("srs_ap", "nmifs"), ("nmifs", "kstats"), ("srs_ap_t_five", "nmifs"), ("lr_stats", "kstats"))
              for e in ("sites", "databases", "controlled", "internal") if not (a == "srs_ap" and e == "sites")]
    rows, per_run = [], []
    for h, a, b, e, margin in pairs:
        va, ba, sva = values(a, e)
        vb, bb, svb = values(b, e)
        d_runs, d_boot = va - vb, ba - bb
        diff = float(d_runs.mean())
        se = float(np.sqrt(d_boot.mean(axis=1).var(ddof=1) + (sva + svb).sum() / R ** 2))
        lo, hi = diff - 1.96 * se, diff + 1.96 * se
        p_lo, p_hi = np.percentile(d_boot.mean(axis=1), [2.5, 97.5])
        se_r = np.sqrt(d_boot.var(axis=0, ddof=1) + sva + svb)
        lo_r, hi_r = d_runs - 1.96 * se_r, d_runs + 1.96 * se_r
        verdict = np.where(lo_r > 0, 1, np.where(hi_r < 0, -1, 0))
        rows.append({"hypothesis": h, "a": a, "b": b, "endpoint": e, "margin": margin, "diff": diff, "low": lo, "high": hi,
                     "holds": bool(lo > margin) if h != "desc" else None, "better": int((verdict == 1).sum()), "ns": int((verdict == 0).sum()),
                     "worse": int((verdict == -1).sum()), "mean_a": float(va.mean()), "mean_b": float(vb.mean()), "se": se,
                     "low_patients_only": float(p_lo), "high_patients_only": float(p_hi), "n_seeds_a": len(members(a)), "n_seeds_b": len(members(b))})
        for r, run in enumerate(run_names):
            per_run.append({"hypothesis": h, "a": a, "b": b, "endpoint": e, "run": run, "diff": float(d_runs[r]), "low": float(lo_r[r]),
                            "high": float(hi_r[r]), "a_value": float(va[r]), "b_value": float(vb[r])})
    S = pd.DataFrame(rows)
    S.to_csv(OUT / "summary.csv", index=False)
    pd.DataFrame(per_run).to_csv(OUT / "per_run.csv", index=False)
    means = []
    shown = [c for c in cols if "@" not in c]
    for c in shown:
        for e in ("sites", "databases", "controlled", "internal"):
            v = values(c, e)[0]
            if np.isfinite(v).all():
                means.append({"registry": c, "endpoint": e, "n_seeds": len(members(c)), "mean": float(v.mean()),
                              **{run: float(x) for run, x in zip(run_names, v)}})
    pd.DataFrame(means).to_csv(OUT / "means.csv", index=False)
    SRC_DATA.mkdir(parents=True, exist_ok=True)
    S.to_csv(SRC_DATA / "q5_confirmatory2_summary.csv", index=False)
    pd.DataFrame(means).to_csv(SRC_DATA / "q5_confirmatory2_means.csv", index=False)
    show = S[S.hypothesis != "desc"].copy()
    for c in ("diff", "low", "high"):
        show[c] = (100 * show[c]).round(2)
    pd.set_option("display.width", 220)
    print(show[["hypothesis", "a", "b", "endpoint", "diff", "low", "high", "holds", "better", "ns", "worse"]].to_string(index=False))


def main(argv=None) -> int:
    global OUT, VARIANT
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--phase", choices=("build", "run", "stats", "dump", "boot", "summarize"), required=True)
    parser.add_argument("--variant", choices=("five", "stat"), default="five", help="five = the first study, stat = the second")
    parser.add_argument("--tags", default="", help="run / stats phase: restrict to these task tags (comma-separated)")
    parser.add_argument("--budgets", default="", help="run phase: restrict to these budgets (comma-separated)")
    parser.add_argument("--n-boot", type=int, default=1000)
    parser.add_argument("--chunk", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    VARIANT = args.variant
    if VARIANT == "stat":
        OUT = ART / "eicu_source2"
    {"build": phase_build, "run": phase_run, "stats": phase_stats, "dump": phase_dump, "boot": phase_boot,
     "summarize": phase_summarize}[args.phase](args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
