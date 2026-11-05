from __future__ import annotations

import math

import numpy as np
import pytest

from src.icu.v2.bridge import (
    DEFAULT_C_GRID,
    compile_bridge,
    coordinate_manifest,
    fit_frozen_bridge_v2,
    refire_external,
    select_probe_on_validation,
)
from src.icu.v2.types import make_symbolic_program


def _rule(pid: str, antecedent: str, consequent: str, score: float = 1.0):
    return make_symbolic_program(
        pid, "fmg", "ordered_rule", (frozenset({antecedent}),), (frozenset({consequent}),), score=score
    )


LAC_MAP = _rule("lac_map", "Lactate:q3", "MAP:q0", score=2.0)
HR_RR = _rule("hr_rr", "HR:q1", "RR:level_0", score=1.5)
CR_TEMP = _rule("cr_temp", "Creatinine:sparse_measurement", "Temp:q0", score=0.5)  # never fires on train
REGISTRY = [LAC_MAP, HR_RR, CR_TEMP]

TRAIN = ["t1", "t2", "t3", "t4", "t5", "t6", "t7", "t8"]
VAL = ["v1", "v2", "v3", "v4"]
TEST = ["s1", "s2", "s3", "s4", "s5"]
LABELS = {"train": [1, 1, 1, 1, 0, 0, 0, 0], "val": [1, 1, 0, 0], "test": [1, 1, 0, 1, 0]}
FIRINGS = {
    "train": {"lac_map": frozenset({"t1", "t2", "t3"}), "hr_rr": frozenset({"t4", "t5"})},
    "val": {"lac_map": frozenset({"v1"}), "hr_rr": frozenset({"v2", "v3"}), "cr_temp": frozenset({"v4"})},
    # s4 fires only the zero-train program, s5 fires nothing.
    "test": {"lac_map": frozenset({"s1"}), "hr_rr": frozenset({"s2", "s3"}), "cr_temp": frozenset({"s4"})},
}
EPISODES = {"train": TRAIN, "val": VAL, "test": TEST}


def _fit(**kwargs):
    return fit_frozen_bridge_v2(REGISTRY, FIRINGS, EPISODES, LABELS, **kwargs)


def test_compile_bridge_has_one_column_per_registry_program_in_registry_order():
    registry = [CR_TEMP, LAC_MAP, HR_RR]
    firings = {**FIRINGS["train"], "not_in_registry": frozenset({"t1"}), "lac_map": frozenset({"t1", "zz"})}
    matrix, names = compile_bridge(registry, firings, TRAIN)
    assert names == ["cr_temp", "lac_map", "hr_rr"]
    assert matrix.shape == (len(TRAIN), 3)
    assert not matrix[:, 0].any()  # zero-firing program still has its column
    assert matrix[:, 1].tolist() == [1.0, 0, 0, 0, 0, 0, 0, 0]  # unknown episode "zz" ignored
    assert matrix[:, 2].tolist() == [0, 0, 0, 1.0, 1.0, 0, 0, 0]


def test_compile_bridge_score_encoding_and_edge_cases():
    matrix, _ = compile_bridge(REGISTRY, FIRINGS["test"], TEST, encoding="score")
    assert matrix[0, 0] == 2.0 and matrix[3, 2] == 0.5
    empty_rows, names = compile_bridge(REGISTRY, {}, [])
    assert empty_rows.shape == (0, 3) and names == ["lac_map", "hr_rr", "cr_temp"]
    no_columns, no_names = compile_bridge([], FIRINGS["train"], TRAIN)
    assert no_columns.shape == (len(TRAIN), 0) and no_names == []
    with pytest.raises(ValueError, match="encoding"):
        compile_bridge(REGISTRY, {}, TRAIN, encoding="counts")
    with pytest.raises(ValueError, match="duplicate program_id"):
        compile_bridge([LAC_MAP, LAC_MAP], {}, TRAIN)
    with pytest.raises(ValueError, match="duplicate episode"):
        compile_bridge(REGISTRY, {}, ["t1", "t1"])


def test_coordinate_manifest_declares_order_and_zero_train_columns():
    # Firings outside the training ids must not count as training support.
    train_firings = {**FIRINGS["train"], "cr_temp": frozenset({"s4"})}
    manifest = coordinate_manifest(REGISTRY, train_firings, TRAIN)
    assert manifest["names"] == ["lac_map", "hr_rr", "cr_temp"]
    assert manifest["order"] == {"lac_map": 0, "hr_rr": 1, "cr_temp": 2}
    assert manifest["n_coordinates"] == 3
    assert manifest["zero_train_columns"] == ["cr_temp"]
    assert manifest["train_firing_counts"] == {"lac_map": 3, "hr_rr": 2, "cr_temp": 0}
    assert manifest["coordinates"][0]["features"] == ["Lactate", "MAP"]
    assert any("cr_temp" in note for note in manifest["notes"])
    assert coordinate_manifest([], {}, TRAIN)["zero_train_columns"] == []


def test_zero_firing_program_keeps_its_coordinate_and_cannot_move_predictions():
    result = _fit()
    assert result["n_coordinates"] == 3
    assert result["coordinate_manifest"]["zero_train_columns"] == ["cr_temp"]
    bundle = result["model_bundle"]
    assert bundle["coordinate_names"] == ["lac_map", "hr_rr", "cr_temp"]
    assert len(bundle["coef"]) == 3
    assert abs(bundle["coef"][2]) < 1e-12
    probabilities = result["test_probabilities"]
    assert list(probabilities) == TEST
    assert probabilities["s4"] == pytest.approx(probabilities["s5"], abs=1e-12)
    assert result["selected_c"] in DEFAULT_C_GRID
    assert set(result["test_metrics"]) >= {"auprc", "auroc"}
    assert 0.0 <= result["test_metrics"]["auprc"] <= 1.0
    assert result["n_episodes"] == {"train": 8, "val": 4, "test": 5}


def test_select_probe_raises_on_empty_validation_instead_of_using_training():
    x_train, _ = compile_bridge(REGISTRY, FIRINGS["train"], TRAIN)
    with pytest.raises(ValueError, match="validation split is empty"):
        select_probe_on_validation(x_train, LABELS["train"], np.zeros((0, 3)), [])
    with pytest.raises(ValueError, match="validation split is empty"):
        fit_frozen_bridge_v2(REGISTRY, FIRINGS, {**EPISODES, "val": []}, {**LABELS, "val": []})


def test_select_probe_raises_on_single_class_validation():
    x_train, _ = compile_bridge(REGISTRY, FIRINGS["train"], TRAIN)
    x_val, _ = compile_bridge(REGISTRY, FIRINGS["val"], VAL)
    with pytest.raises(ValueError, match="single class"):
        select_probe_on_validation(x_train, LABELS["train"], x_val, [1, 1, 1, 1])


def test_select_probe_tie_break_takes_smallest_c_regardless_of_grid_order():
    # One perfectly separating coordinate: every C ranks validation identically (AUPRC 1.0).
    x_train = np.array([[1.0], [1.0], [0.0], [0.0]])
    x_val = np.array([[1.0], [0.0], [1.0], [0.0]])
    for grid in ((0.01, 0.1, 1.0, 10.0), (10.0, 1.0, 0.1, 0.01)):
        result = select_probe_on_validation(x_train, [1, 1, 0, 0], x_val, [1, 0, 1, 0], c_grid=grid)
        assert result["selected_c"] == 0.01
        assert result["tied_c"] == [0.01, 0.1, 1.0, 10.0]
        assert result["validation_score"] == pytest.approx(1.0)
        assert result["model"].C == 0.01
    auroc = select_probe_on_validation(x_train, [1, 1, 0, 0], x_val, [1, 0, 1, 0], metric="auroc")
    assert auroc["metric"] == "auroc" and auroc["validation_score"] == pytest.approx(1.0)


def test_select_probe_rejects_bad_inputs():
    x = np.array([[1.0], [0.0]])
    with pytest.raises(ValueError, match="metric"):
        select_probe_on_validation(x, [1, 0], x, [1, 0], metric="accuracy")
    with pytest.raises(ValueError, match="empty"):
        select_probe_on_validation(x, [1, 0], x, [1, 0], c_grid=())
    with pytest.raises(ValueError, match="> 0"):
        select_probe_on_validation(x, [1, 0], x, [1, 0], c_grid=(0.0, 1.0))
    with pytest.raises(ValueError, match="both classes"):
        select_probe_on_validation(x, [1, 1], x, [1, 0])
    with pytest.raises(ValueError, match="no bridge coordinates"):
        select_probe_on_validation(np.zeros((2, 0)), [1, 0], np.zeros((2, 0)), [1, 0])
    with pytest.raises(ValueError, match="0/1"):
        select_probe_on_validation(x, [1, 2], x, [1, 0])
    with pytest.raises(ValueError, match="coordinates"):
        select_probe_on_validation(x, [1, 0], np.zeros((2, 2)), [1, 0])


def test_fit_frozen_bridge_v2_rejects_split_errors():
    with pytest.raises(ValueError, match="missing splits"):
        fit_frozen_bridge_v2(REGISTRY, {"train": {}, "val": {}}, EPISODES, LABELS)
    with pytest.raises(ValueError, match="appears in both"):
        fit_frozen_bridge_v2(REGISTRY, FIRINGS, {**EPISODES, "test": TEST + ["t1"]}, {**LABELS, "test": LABELS["test"] + [1]})
    with pytest.raises(ValueError, match="labels"):
        fit_frozen_bridge_v2(REGISTRY, FIRINGS, EPISODES, {**LABELS, "test": [1, 0]})
    with pytest.raises(ValueError, match="no bridge coordinates"):
        fit_frozen_bridge_v2([], FIRINGS, EPISODES, LABELS)


def test_refire_external_zeroes_nonsurvivable_coordinates_without_refit():
    result = _fit()
    target = {"HR", "RR", "Creatinine", "Temp", "Lactate"}  # MAP missing -> lac_map is not survivable
    external_ids = ["x1", "x2", "x3"]
    # A firing reported for lac_map (e.g. produced by a target-specific substitute) must be ignored.
    external_firings = {"lac_map": frozenset({"x1"}), "hr_rr": frozenset({"x2"})}
    out = refire_external(result["model_bundle"], REGISTRY, target, external_firings, external_ids, labels=[1, 1, 0])
    assert out["survival_mask"] == [False, True, True]
    assert out["nonsurvivable_ids"] == ["lac_map"]
    assert out["schema_survival"] == pytest.approx(2 / 3)
    assert out["suppressed_firings"] == {"lac_map": 1}
    assert out["firing_ids"] == ["hr_rr"] and out["firing_survival"] == pytest.approx(1 / 3)
    assert out["probabilities"]["x1"] == pytest.approx(out["probabilities"]["x3"], abs=1e-12)
    assert out["probabilities"]["x2"] != pytest.approx(out["probabilities"]["x3"])

    model = result["model_bundle"]["model"]
    zeroed = model.predict_proba(np.array([[0.0, 0.0, 0.0]]))[0, 1]
    assert out["probabilities"]["x1"] == pytest.approx(zeroed)
    assert not math.isnan(out["metrics"]["auroc"])

    full_target = refire_external(result, REGISTRY, {"HR", "RR", "Creatinine", "Temp", "Lactate", "MAP"},
                                  external_firings, external_ids)
    assert full_target["survival_mask"] == [True, True, True] and full_target["suppressed_firings"] == {}
    assert full_target["probabilities"]["x1"] != pytest.approx(full_target["probabilities"]["x3"])


def test_refire_external_rejects_a_changed_registry_and_handles_no_episodes():
    result = _fit()
    with pytest.raises(ValueError, match="does not match"):
        refire_external(result["model_bundle"], [HR_RR, LAC_MAP, CR_TEMP], {"HR"}, {}, ["x1"])
    with pytest.raises(ValueError, match="does not match"):
        refire_external(result["model_bundle"], REGISTRY[:2], {"HR"}, {}, ["x1"])
    empty = refire_external(result["model_bundle"], REGISTRY, set(), {}, [])
    assert empty["probabilities"] == {} and empty["n_episodes"] == 0
    assert empty["schema_survival"] == 0.0
