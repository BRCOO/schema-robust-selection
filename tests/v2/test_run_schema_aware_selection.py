from __future__ import annotations

import csv
import importlib.util
import pickle
from pathlib import Path

from src.icu.v2.pipeline import TargetCohort
from src.icu.v2.splits import patient_level_split

from tests.v2.test_pipeline import FEATURES, _episodes, _pool

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "v2" / "run_schema_aware_selection.py"


def _module():
    spec = importlib.util.spec_from_file_location("run_schema_aware_selection", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _rows(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def test_schema_aware_selection_end_to_end(tmp_path):
    episodes, _ = patient_level_split(_episodes(80), fractions=(0.6, 0.2, 0.2), seed=3)
    ext = TargetCohort(name="ext", features=frozenset(f for f in FEATURES if f != "Lactate"), episodes=tuple(_episodes(25, prefix="x", seed=9)))
    narrow = TargetCohort(name="narrow", features=frozenset({"HR", "MAP", "RR", "SpO2"}), episodes=tuple(_episodes(25, prefix="y", seed=11)))
    pools = [_pool("strong", episodes, signal=0.45, seed=1, external=[ext, narrow]), _pool("weak", episodes, signal=0.1, seed=2, external=[ext, narrow])]
    with open(tmp_path / "pools.pkl", "wb") as handle:
        pickle.dump({"pools": pools, "episodes": episodes, "targets": [ext, narrow], "timings": {}, "n_candidates": {}}, handle)
    sa = tmp_path / "sa"
    sa.mkdir()
    with open(sa / "tuned.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["method", "selection", "w_redundancy", "w_family", "w_anchor", "w_schema"])
        writer.writeheader()
        for method in ("strong", "weak"):
            writer.writerow({"method": method, "selection": "v2_selector", "w_redundancy": 0.0, "w_family": 0.8, "w_anchor": 0.3, "w_schema": 0.0})
    out = tmp_path / "out"
    assert _module().main(["--pools", str(tmp_path / "pools.pkl"), "--selector-analysis-dir", str(sa), "--out-dir", str(out), "--agnostic-targets", "ext", "--n-boot", "30"]) == 0

    single = _rows(out / "single_target.csv")
    names = {"native", "v2_agnostic", "fs_cv", "filter_topk", "v2_aware", "v2_filter", "mmr_filter", "srs_aware", "mrmr_filter", "mmrfs_filter", "l1_logistic_filter"}
    for method in ("strong", "weak"):
        for target in ("ext", "narrow"):
            got = {r["registry"] for r in single if r["method"] == method and r["target"] == target}
            assert got == names
    for r in single:
        if r["registry"] in ("filter_topk", "v2_filter", "mmr_filter", "mrmr_filter", "mmrfs_filter", "l1_logistic_filter") and not r.get("error"):
            assert float(r["target_schema_survival"]) == 1.0  # hard filters only keep executable programs
    multi = _rows(out / "multi_target.csv")
    assert {r["registry"] for r in multi if r["method"] == "strong"} == {"native", "v2_agnostic", "fs_cv", "filter_topk_intersection", "v2_aware_all", "v2_filter_intersection", "srs_all"}
    for r in multi:
        if not r.get("error"):
            assert float(r["min_target_auprc"]) <= float(r["mean_target_auprc"])
    paired = _rows(out / "single_target_paired.csv")
    allowed = {f"{a} - {b}" for a, b in (("v2_aware", "filter_topk"), ("v2_aware", "native"), ("v2_aware", "v2_agnostic"), ("srs_aware", "filter_topk"), ("srs_aware", "mmr_filter"), ("srs_aware", "v2_aware"), ("srs_aware", "native"), ("srs_aware", "fs_cv"),
                                         ("srs_aware", "mrmr_filter"), ("srs_aware", "mmrfs_filter"), ("srs_aware", "l1_logistic_filter"))}
    assert {p["comparison"] for p in paired} <= allowed and any(p["comparison"].startswith("srs_aware") for p in paired)


def test_unaware_registries_are_paired_with_srs(tmp_path):
    import json

    episodes, _ = patient_level_split(_episodes(80), fractions=(0.6, 0.2, 0.2), seed=3)
    ext = TargetCohort(name="ext", features=frozenset(f for f in FEATURES if f != "Lactate"), episodes=tuple(_episodes(25, prefix="x", seed=9)))
    pools = [_pool("strong", episodes, signal=0.45, seed=1, external=[ext])]
    with open(tmp_path / "pools.pkl", "wb") as handle:
        pickle.dump({"pools": pools, "episodes": episodes, "targets": [ext], "timings": {}, "n_candidates": {}}, handle)
    sa = tmp_path / "sa"
    sa.mkdir()
    with open(sa / "tuned.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["method", "selection", "w_redundancy", "w_family", "w_anchor", "w_schema"])
        writer.writeheader()
        writer.writerow({"method": "strong", "selection": "v2_selector", "w_redundancy": 0.0, "w_family": 0.8, "w_anchor": 0.3, "w_schema": 0.0})
    e1 = tmp_path / "e1_registries.json"
    e1.write_text(json.dumps({"strong": {"mrmr": [c.program_id for c in pools[0].candidates[:4]]}}), encoding="utf-8")
    out = tmp_path / "out"
    args = ["--pools", str(tmp_path / "pools.pkl"), "--selector-analysis-dir", str(sa), "--out-dir", str(out), "--agnostic-targets", "ext",
            "--n-boot", "30", "--unaware-registries", str(e1), "--unaware-selectors", "mrmr"]
    assert _module().main(args) == 0
    assert "mrmr" in {r["registry"] for r in _rows(out / "single_target.csv")}
    assert "srs_aware - mrmr" in {p["comparison"] for p in _rows(out / "single_target_paired.csv")}
    assert "srs_all - mrmr" in {p["comparison"] for p in _rows(out / "multi_target_paired.csv")}
    registries = json.loads((out / "registries.json").read_text(encoding="utf-8"))["strong"]
    assert registries["mrmr"] == [c.program_id for c in pools[0].candidates[:4]]
    assert "srs_aware@ext" in registries and "srs_all@all" in registries
