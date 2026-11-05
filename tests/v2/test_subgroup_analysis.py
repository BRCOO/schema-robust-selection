from __future__ import annotations

import csv
import importlib.util
import json
import math
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import average_precision_score, roc_auc_score

from src.icu.v2.pipeline import TargetCohort, compare_methods, evaluate_registry, native_registry
from src.icu.v2.selector import SelectorConfig
from src.icu.v2.splits import patient_level_split
from src.icu.v2.types import EpisodeRecord

from tests.v2.test_pipeline import FEATURES, _episodes, _pool

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "v2" / "run_subgroup_analysis.py"


@pytest.fixture(scope="module")
def mod():
    spec = importlib.util.spec_from_file_location("run_subgroup_analysis", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # dataclasses with string annotations look their module up here
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------------- derivations


def test_sex_normalization(mod):
    assert [mod.normalize_sex(v) for v in ("F", "f", "Female", " female ", "M", "Male", "MALE")] == ["F"] * 4 + ["M"] * 3
    assert [mod.normalize_sex(v) for v in ("Unknown", "Other", "", None, float("nan"))] == [None] * 5


def test_eicu_age_text(mod):
    assert mod.eicu_age("> 89") == 90.0
    assert mod.eicu_age(" 64 ") == 64.0 and mod.eicu_age("65") == 65.0
    assert all(math.isnan(mod.eicu_age(v)) for v in ("", None, float("nan"), "unknown"))
    assert mod.age_group(mod.eicu_age("> 89")) == ">=65"


def test_age_groups_and_hirid_bins(mod):
    assert [mod.age_group(v) for v in (18, 64.9, 65, 65.0, 91)] == ["<65", "<65", ">=65", ">=65", ">=65"]
    assert mod.age_group(float("nan")) is None and mod.age_group(None) is None
    # HiRID bin labels are lower bounds: 60 = 60-64, 65 = 65-69, 90 = 90+
    assert [mod.age_group(mod.hirid_age(v)) for v in (20, 60, 65, 90)] == ["<65", "<65", ">=65", ">=65"]


def test_mimic_age_formula(mod):
    ages = mod.mimic_ages(
        [60, 64, 91, 50],
        [2150, 2150, 2180, 2120],
        pd.to_datetime(["2155-03-01 00:00", "2150-12-31 23:00", "2181-01-01 00:00", None]),
    )
    assert ages[:3].tolist() == [65.0, 64.0, 92.0] and math.isnan(ages[3])


def test_demographic_frames(mod):
    stays = pd.DataFrame(
        {
            "stay_id": [1, 2, 3],
            "subject_id": [10, 10, 20],
            "intime": pd.to_datetime(["2150-01-01", "2160-06-01", "2130-01-01"]),
            "anchor_age": [60, 60, 40],
            "anchor_year": [2150, 2150, 2130],
        }
    )
    mimic = mod.mimic_demographics(stays, pd.DataFrame({"subject_id": [10, 20], "gender": ["F", "M"]}))
    assert mimic["age"].tolist() == [60.0, 70.0, 40.0] and mimic["sex"].tolist() == ["F", "F", "M"]

    patient = pd.DataFrame({"patientunitstayid": [5, 6, 7, 8], "gender": ["Female", "Male", "Unknown", None], "age": ["> 89", "45", "", "70"]})
    eicu = mod.eicu_demographics(patient)
    assert eicu["sex"].tolist() == ["F", "M", None, None]
    assert eicu["age"].iloc[[0, 1, 3]].tolist() == [90.0, 45.0, 70.0] and math.isnan(eicu["age"].iloc[2])

    hirid = mod.hirid_demographics(pd.DataFrame({"patientid": [1, 2], "sex": ["M", "F"], "age": [60, 65]}))
    assert hirid.to_dict("list") == {"stay_id": [1, 2], "sex": ["M", "F"], "age": [60.0, 65.0]}


def test_raw_loaders_on_tiny_files(mod, tmp_path):
    from src.icu.v2.extract.eicu import RAW_FILES

    eicu_dir = tmp_path / "eicu"
    eicu_dir.mkdir()
    pd.DataFrame(
        {"patientunitstayid": [5, 6], "uniquepid": ["002-1", "002-2"], "gender": ["Female", "Male"], "age": ["> 89", "30"], "unitdischargeoffset": [100, 200]}
    ).to_csv(eicu_dir / RAW_FILES["patient"], index=False, compression="gzip")
    assert mod.load_eicu_demographics({"raw_dir": eicu_dir}).to_dict("list") == {"stay_id": [5, 6], "sex": ["F", "M"], "age": [90.0, 30.0]}

    hirid_dir = tmp_path / "hirid"
    hirid_dir.mkdir()
    pd.DataFrame(
        {"patientid": [1, 2], "admissiontime": ["2100-01-01", "2100-01-02"], "sex": ["F", "M"], "age": [85, 20], "discharge_status": ["alive", "dead"]}
    ).to_csv(hirid_dir / "general_table.csv", index=False)
    assert mod.load_hirid_demographics({"raw_dir": hirid_dir}).to_dict("list") == {"stay_id": [1, 2], "sex": ["F", "M"], "age": [85.0, 20.0]}


def test_attach_demographics_counts_unknowns_and_checks_subjects(mod):
    episodes = [EpisodeRecord(f"e{i}", subject_id=100 + i // 2, stay_id=i // 2, label=i % 2) for i in range(8)]
    demo = pd.DataFrame({"stay_id": [0, 1, 2], "subject_id": [100, 101, 102], "sex": ["F", "M", None], "age": [70.0, 40.0, float("nan")]})
    cohort = mod.attach_demographics("c", episodes, demo)
    assert cohort.audit["stays_without_demographics"] == 1  # stay 3
    assert cohort.audit["unknown_sex_episodes"] == 4 and cohort.audit["unknown_age_episodes"] == 4
    assert cohort.masks[("sex", "F")].tolist() == [True, True] + [False] * 6
    assert cohort.masks[("age", ">=65")].sum() == 2 and cohort.masks[("age", "<65")].sum() == 2
    assert cohort.masks[("all", "all")].all()
    with pytest.raises(ValueError, match="another subject_id"):
        mod.attach_demographics("c", episodes, demo.assign(subject_id=[100, 999, 102]))


# ---------------------------------------------------------------------------------- metrics


def test_score_subgroup_matches_sklearn_and_flags_single_class(mod):
    rng = np.random.default_rng(3)
    y = np.array([1, 0] * 40)
    p = 0.3 * y + 0.7 * rng.random(80)
    groups = np.repeat(np.arange(40), 2)
    out = mod.score_subgroup(y, p, groups, n_boot=200, seed=0)
    assert (out["n_episodes"], out["n_patients"], out["n_positive"], out["prevalence"]) == (80, 40, 40, 0.5)
    assert out["auprc"] == pytest.approx(average_precision_score(y, p), abs=1e-12)
    assert out["auroc"] == pytest.approx(roc_auc_score(y, p), abs=1e-12)
    assert out["auprc_low"] <= out["auprc_high"] and out["auroc_low"] <= out["auroc_high"] and out["n_boot_used"] == 200
    single = mod.score_subgroup(np.ones(5, dtype=int), np.linspace(0, 1, 5), np.arange(5), n_boot=10, seed=0)
    assert single["error"] == "single-class subgroup" and "auprc" not in single
    assert mod.score_subgroup([], [], [], n_boot=10, seed=0)["error"] == "empty subgroup"


def test_subgroup_rows_score_each_subgroup_on_its_own_episodes(mod):
    episodes = [EpisodeRecord(f"e{i}", subject_id=i // 2, stay_id=i // 2, label=i % 2) for i in range(60)]
    demo = pd.DataFrame({"stay_id": range(30), "sex": ["F" if s % 3 else "M" for s in range(30)], "age": [50.0 + s for s in range(30)]})
    cohort = mod.attach_demographics("c", episodes, demo)
    p = np.random.default_rng(0).random(60)
    rows = {(r["subgroup_var"], r["subgroup"]): r for r in mod.subgroup_rows(cohort, p, n_boot=50, seed=0)}
    assert list(rows) == list(mod.SUBGROUPS)
    for key, row in rows.items():
        mask = cohort.masks[key]
        assert row["n_episodes"] == mask.sum() and row["n_patients"] == len({e.subject_id for e, m in zip(episodes, mask) if m})
        assert row["auprc"] == pytest.approx(average_precision_score(cohort.y[mask], p[mask]), abs=1e-12)
        assert row["auroc"] == pytest.approx(roc_auc_score(cohort.y[mask], p[mask]), abs=1e-12)
    assert rows[("sex", "F")]["n_episodes"] == 40 and rows[("sex", "M")]["n_episodes"] == 20
    assert rows[("age", "<65")]["n_episodes"] == 30 and rows[("age", ">=65")]["n_episodes"] == 30


def test_firing_stats(mod):
    active = np.array([[1, 0, 0], [0, 0, 0], [0, 1, 0], [1, 1, 0]], dtype=bool)
    assert mod.firing_stats(active) == {"n_programs": 3, "firing_coverage": 0.75, "n_programs_firing": 2, "firing_survival": 2 / 3}
    empty = mod.firing_stats(np.zeros((0, 3), dtype=bool))
    assert math.isnan(empty["firing_coverage"]) and empty["n_programs_firing"] == 0


# ---------------------------------------------------------------------------------- bridge and end to end


def _state():
    episodes, _ = patient_level_split(_episodes(), fractions=(0.6, 0.2, 0.2), seed=1)
    eicu = TargetCohort(name="eicu", features=frozenset(f for f in FEATURES if f != "Lactate"), episodes=tuple(_episodes(25, prefix="x", seed=7)))
    hirid = TargetCohort(name="hirid", features=frozenset(FEATURES), episodes=tuple(_episodes(20, prefix="h", seed=8)))
    pools = [
        _pool("strong", episodes, signal=0.45, seed=1, external=[eicu, hirid]),
        _pool("noise", episodes, signal=0.0, seed=2, external=[eicu, hirid]),
    ]
    return {"pools": pools, "episodes": episodes, "targets": [eicu, hirid], "timings": {}, "n_candidates": {}}


def test_bridge_scores_reproduce_evaluate_registry(mod):
    state = _state()
    pool = state["pools"][0]
    registry = native_registry(pool, 10)
    reference = evaluate_registry(registry, pool.firings, state["episodes"], targets=state["targets"], external_firings=pool.external_firings, n_boot=20)
    scores, info = mod.bridge_scores(registry, pool, state["episodes"], state["targets"])
    internal = scores[mod.INTERNAL_COHORT]
    assert dict(zip(internal["episode_ids"], internal["probabilities"].tolist())) == reference["test_probabilities"]
    assert info["selected_c"] == reference["bridge"]["selected_c"]
    for target in state["targets"]:
        block, ref = scores[target.name], reference["external"][target.name]
        y = [episode.label for episode in target.episodes]
        assert average_precision_score(y, block["probabilities"]) == ref["auprc"]
        assert roc_auc_score(y, block["probabilities"]) == ref["auroc"]
        assert block["schema_survival"] == ref["schema_survival"]
        assert mod.firing_stats(block["active"])["firing_survival"] == ref["firing_survival"]
    # Lactate programs do not survive the eicu schema: their coordinates stay off although firings were reported
    lactate = [j for j, program in enumerate(registry) if "Lactate" in program.features]
    assert lactate and scores["eicu"]["suppressed_firings"] and not scores["eicu"]["active"][:, lactate].any()


def _write_inputs(tmp_path: Path, state: dict) -> dict:
    folder = tmp_path / "comparison"
    folder.mkdir()
    pools = folder / "pools.pkl"
    with open(pools, "wb") as handle:
        pickle.dump(state, handle)
    (folder / "config.json").write_text(json.dumps({"k": 10, "selector": {"care_process_cap": None}, "selector_targets": ["eicu", "hirid"]}), encoding="utf-8")
    result = compare_methods(
        state["pools"], state["episodes"], targets=state["targets"], selector_config=SelectorConfig(k=10, care_process_cap=None),
        k=10, n_boot=20, selector_target_names=["eicu", "hirid"],
    )
    columns = sorted({key for row in result["rows"] for key in row})
    with open(folder / "comparison_rows.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(result["rows"])

    test = [e for e in state["episodes"] if e.split == "test"]
    cohorts = {"mimic_iv": test, "eicu": list(state["targets"][0].episodes), "hirid": list(state["targets"][1].episodes)}
    demo = [
        {"source": source, "stay_id": stay, "sex": ["F", "M", ""][stay % 3], "age": 40 + (7 * stay) % 50}
        for source, episodes in cohorts.items()
        for stay in sorted({e.stay_id for e in episodes})
    ]
    pd.DataFrame(demo).to_csv(tmp_path / "demographics.csv", index=False)

    rng = np.random.default_rng(5)
    sup = tmp_path / "supervised"
    sup.mkdir()

    def predictions(episodes, cohort=None):
        rows = []
        for model, strength in (("good", 0.6), ("weak", 0.1)):
            for seed in (0, 1):
                for e in episodes:
                    row = {"model": model, "seed": seed, "episode_id": e.episode_id, "subject_id": e.subject_id, "label": e.label, "prob": strength * e.label + rng.random()}
                    rows.append(row if cohort is None else {"cohort": cohort, **row})
        return pd.DataFrame(rows)

    internal = predictions(test)
    external = pd.concat([predictions(cohorts["eicu"], "eicu"), predictions(cohorts["hirid"], "hirid")], ignore_index=True)
    internal.to_csv(sup / "supervised_v2_test_predictions.csv", index=False)
    external.to_csv(sup / "supervised_v2_external_predictions.csv", index=False)
    good0 = internal[(internal["model"] == "good") & (internal["seed"] == 0)]
    eicu0 = external[(external["model"] == "good") & (external["seed"] == 0) & (external["cohort"] == "eicu")]
    summary = {
        "models": {
            "good": {
                "over_seeds": {"val_auprc_mean": 0.9},
                "seeds": [
                    {
                        "seed": 0,
                        "test_metrics": {"auprc": average_precision_score(good0["label"], good0["prob"]), "auroc": roc_auc_score(good0["label"], good0["prob"])},
                        "external_metrics": {"eicu": {"auprc": average_precision_score(eicu0["label"], eicu0["prob"])}},
                    }
                ],
            },
            "weak": {"over_seeds": {"val_auprc_mean": 0.6}, "seeds": []},
        }
    }
    (sup / "supervised_v2_summary.json").write_text(json.dumps(summary), encoding="utf-8")
    return {"pools": pools, "supervised": sup, "demographics": tmp_path / "demographics.csv", "comparison_rows": result["rows"], "folder": folder}


def _run(mod, paths: dict, out: Path) -> int:
    return mod.main(
        [
            "--pools", str(paths["pools"]), "--supervised-dir", str(paths["supervised"]), "--out-dir", str(out),
            "--demographics", str(paths["demographics"]), "--n-boot", "30", "--headline", "strong:native,noise:v2_selector",
        ]
    )


def _csv(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def test_main_end_to_end_on_synthetic_inputs(mod, tmp_path):
    paths = _write_inputs(tmp_path, _state())
    out = tmp_path / "out"
    assert _run(mod, paths, out) == 0
    rows = _csv(out / "subgroups.csv")
    discovery = [r for r in rows if r["family"] == "discovery"]
    supervised = [r for r in rows if r["family"] == "supervised"]
    assert len(discovery) == 2 * 2 * 3 * 5 and len(supervised) == 2 * 3 * 5
    assert not any(r["error"] for r in rows)

    reference = {(r["method"], r["selection"]): r for r in paths["comparison_rows"]}
    for r in discovery:
        if r["subgroup_var"] == "all":
            prefix = "" if r["cohort"] == "mimic_iv_test" else r["cohort"] + "_"
            assert abs(float(r["auprc"]) - reference[(r["method"], r["selection"])][prefix + "auprc"]) <= 1e-12
        assert 0.0 <= float(r["firing_coverage"]) <= 1.0
    for r in supervised:
        assert r["n_seeds"] == "2" and r["ci_seed"] == "0" and r["auprc_seed_sd"] != ""

    demo = _csv(out / "subgroup_demographics.csv")

    def count(cohort, variable, level, label="all"):
        return int(next(r for r in demo if (r["cohort"], r["subgroup_var"], r["subgroup"], r["label"]) == (cohort, variable, level, label))["n_episodes"])

    for cohort in ("mimic_iv_test", "eicu", "hirid"):
        total = count(cohort, "all", "all")
        assert count(cohort, "sex", "F") + count(cohort, "sex", "M") + count(cohort, "sex", "unknown") == total
        assert count(cohort, "age", "<65") + count(cohort, "age", ">=65") + count(cohort, "age", "unknown") == total
        assert count(cohort, "sex", "unknown") > 0 and count(cohort, "all", "all", "1") + count(cohort, "all", "all", "0") == total
        sex_rows = [r for r in discovery if r["cohort"] == cohort and r["subgroup"] in ("F", "M") and r["method"] == "strong" and r["selection"] == "native"]
        assert sum(int(r["n_episodes"]) for r in sex_rows) == total - count(cohort, "sex", "unknown")

    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["sanity_check"]["passed"] and manifest["best_supervised"]["model"] == "good"
    assert manifest["sanity_check"]["by_reference"]["supervised_v2_summary.json"]["n_checks"] == 3
    summary = (out / "summary.md").read_text(encoding="utf-8")
    assert "## Main observations" in summary and "good (supervised)" in summary and "strong (native)" in summary


def test_main_writes_nothing_when_the_sanity_check_fails(mod, tmp_path):
    paths = _write_inputs(tmp_path, _state())
    rows = _csv(paths["folder"] / "comparison_rows.csv")
    rows[0]["auprc"] = str(float(rows[0]["auprc"]) + 1e-6)
    with open(paths["folder"] / "comparison_rows.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    out = tmp_path / "out"
    assert _run(mod, paths, out) == 3
    assert not (out / "subgroups.csv").exists() and not (out / "summary.md").exists()
