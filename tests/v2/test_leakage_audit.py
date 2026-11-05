from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import average_precision_score, roc_auc_score

from src.icu.v2.leakage_audit import (
    CANDIDATE_SPECS,
    MULTIVARIATE_VIEWS,
    WINDOW_KEY,
    anchor_context_counts,
    audit_episode_ids,
    collect_summaries,
    episode_stem,
    find_probe,
    flag_leaks,
    format_probe_table,
    json_safe,
    load_episode,
    load_episodes,
    main,
    multivariate_probe,
    read_split_manifest,
    run_split_audit,
    single_feature_probes,
    summarize_episode,
)
from src.icu.v2.task import REASON_SHORT_WINDOW, TaskSpec

FEATURES = ["HR", "MAP", "Lactate", "vasopressor_on", "urine_output"]
VASO = FEATURES.index("vasopressor_on")
MAP = FEATURES.index("MAP")
ZERO_FILL = [FEATURES.index("vasopressor_on"), FEATURES.index("urine_output")]
H = pd.Timedelta(hours=1)


# ------------------------------------------------------------------------------ summaries


def test_summarize_episode() -> None:
    values = np.array([[1.0, 5.0, 0.0], [2.0, 5.0, 0.0], [3.0, 6.0, 0.0], [4.0, 6.0, 1.0]])
    mask = np.array([[0, 1, 0], [0, 0, 0], [0, 1, 0], [0, 1, 0]], dtype=bool)
    out = summarize_episode(values, mask, ["HR", "Lactate", "vasopressor_on"])
    assert out["HR"] == {"last": 4.0, "mean": 2.5, "max": 4.0, "observed_last": 1.0}
    assert out["Lactate"] == {"last": 5.0, "mean": 5.5, "max": 6.0, "observed_last": 0.0}
    assert out["vasopressor_on"]["last"] == 1.0 and out["vasopressor_on"]["mean"] == 0.25
    assert out[WINDOW_KEY] == {"n_rows": 4.0}

    never = summarize_episode(values[:, :1], np.ones((4, 1), dtype=bool), ["Lactate"])
    assert math.isnan(never["Lactate"]["last"]) and never["Lactate"]["observed_last"] == 0.0

    empty = summarize_episode(np.zeros((0, 2)), np.zeros((0, 2), dtype=bool), ["HR", "MAP"])
    assert empty[WINDOW_KEY] == {"n_rows": 0.0}
    assert all(math.isnan(value) for value in empty["HR"].values())

    with pytest.raises(ValueError):
        summarize_episode(values, mask, ["HR", "Lactate"])
    with pytest.raises(ValueError):
        summarize_episode(values[:, :1], mask[:, :1], [WINDOW_KEY])


# --------------------------------------------------------------------------------- probes


def _summaries(scores_by_feature: dict[str, np.ndarray]) -> dict[str, dict]:
    n = len(next(iter(scores_by_feature.values())))
    return {
        f"ep{i:03d}": {feature: {"x": float(scores[i])} for feature, scores in scores_by_feature.items()}
        for i in range(n)
    }


def test_single_feature_probes_match_sklearn_with_ties_and_nan() -> None:
    rng = np.random.default_rng(0)
    y = np.array([1, 0] * 30)
    tied = rng.integers(0, 4, size=y.size).astype(float) + y  # ties, higher in positives
    tied[[3, 10, 11]] = np.nan
    inverse = -tied.copy()
    inverse[[0, 5]] = np.nan
    summaries = _summaries({"A": tied, "B": inverse})
    labels = {f"ep{i:03d}": int(label) for i, label in enumerate(y)}
    rows = single_feature_probes(summaries, labels)
    assert [row["separability"] for row in rows] == sorted((row["separability"] for row in rows), reverse=True)
    by_feature = {row["feature"]: row for row in rows}
    for feature, raw in (("A", tied), ("B", inverse)):
        reference = np.where(np.isnan(raw), np.nanmin(raw) - 1.0, raw)  # NaN -> lowest
        row = by_feature[feature]
        assert row["auroc"] == pytest.approx(roc_auc_score(y, reference))
        assert row["auprc"] == pytest.approx(average_precision_score(y, reference))
        assert row["separability"] == pytest.approx(max(row["auroc"], 1 - row["auroc"]))
        assert row["n_nan"] == int(np.isnan(raw).sum())
        assert row["n_nan_pos"] == int(np.isnan(raw[y == 1]).sum())
        assert row["n_nan_neg"] == int(np.isnan(raw[y == 0]).sum())
        assert row["mean_pos"] == pytest.approx(np.nanmean(raw[y == 1]))
        assert row["n_pos"] == 30 and row["n_neg"] == 30 and row["prevalence"] == 0.5
    assert by_feature["A"]["direction"] == "higher_in_positive"
    assert by_feature["B"]["direction"] == "lower_in_positive"
    reference_b = np.where(np.isnan(inverse), np.nanmin(inverse) - 1.0, inverse)
    assert by_feature["B"]["auprc_oriented"] == pytest.approx(average_precision_score(y, -reference_b))


def test_single_feature_probes_perfect_constant_and_errors() -> None:
    y = np.array([1, 1, 0, 0, 0])
    summaries = _summaries({"perfect": y * 2.0, "constant": np.ones(5)})
    labels = {f"ep{i:03d}": int(label) for i, label in enumerate(y)}
    rows = single_feature_probes(summaries, labels)
    assert rows[0]["feature"] == "perfect" and rows[0]["separability"] == 1.0 and rows[0]["auprc"] == 1.0
    constant = rows[1]
    assert constant["auroc"] == 0.5 and constant["direction"] == "none"
    assert constant["auprc"] == pytest.approx(0.4)  # prevalence
    assert constant["mean_pos"] == 1.0 and constant["mean_neg"] == 1.0
    with pytest.raises(ValueError):
        single_feature_probes(summaries, {key: 1 for key in labels})
    with pytest.raises(ValueError):
        single_feature_probes(summaries, {key: value for key, value in list(labels.items())[:-1]})


def test_flag_leaks_threshold() -> None:
    rows = [
        {"feature": "a", "summary": "last", "separability": 0.95},
        {"feature": "b", "summary": "last", "separability": 0.80},
        {"feature": "c", "summary": "last", "separability": 0.7999},
        {"feature": "d", "summary": "last", "separability": float("nan")},
    ]
    flagged = flag_leaks(rows)
    assert [row["feature"] for row in flagged] == ["a", "b"]
    assert all(row["flag"] == "suspected_leak" for row in flagged)
    assert [row["feature"] for row in flag_leaks(rows, separability_threshold=0.9)] == ["a"]
    assert flag_leaks([]) == []


# ---------------------------------------------------------------------- multivariate probe


def _flat_summaries(columns: dict[str, np.ndarray], stat: str = "last") -> dict[str, dict]:
    n = len(next(iter(columns.values())))
    return {
        f"ep{i:04d}": {feature: {stat: float(values[i])} for feature, values in columns.items()}
        for i in range(n)
    }


def test_multivariate_probe_tree_detects_planted_interaction() -> None:
    rng = np.random.default_rng(11)
    n = 600
    a_bit = rng.random(n) < 0.7
    b_bit = rng.random(n) < 0.5
    y = (a_bit ^ b_bit).astype(int)  # label needs both features; A alone is uninformative
    summaries = _flat_summaries(
        {
            "A": a_bit + rng.normal(0, 0.1, n),
            "B": b_bit + rng.normal(0, 0.1, n),
            "C": rng.normal(size=n),
            "D": rng.normal(size=n),
        }
    )
    labels = {f"ep{i:04d}": int(label) for i, label in enumerate(y)}
    ids = sorted(summaries)
    assert flag_leaks(single_feature_probes(summaries, labels)) == []  # invisible to single features

    result = multivariate_probe(summaries, labels, ids[:400], ids[400:], n_bootstrap=200)
    tree, lr = result["decision_tree"], result["logistic_regression"]
    assert tree["test_auroc"] >= 0.95
    assert lr["test_auroc"] <= 0.85  # a linear model cannot express the interaction
    assert {"A.last", "B.last"} <= {split["feature"] for split in tree["splits"]}
    assert tree["test_auroc_ci95"][0] <= tree["test_auroc"] <= tree["test_auroc_ci95"][1]
    assert result["n_train"] == 400 and result["n_test"] == 200
    assert result["test_prevalence"] == pytest.approx(y[400:].mean())
    assert result["cv"]["splitter"] == "StratifiedKFold(5)"
    assert result["cv"]["chosen_c"] in result["cv"]["c_grid"]
    assert len(lr["top_coefficients"]) == 4 and lr["converged"]


def test_multivariate_probe_null_dataset_stays_near_chance() -> None:
    rng = np.random.default_rng(5)
    n = 500
    columns = {f"F{j}": rng.normal(size=n) for j in range(6)}
    columns["F0"][rng.random(n) < 0.2] = np.nan
    summaries = _flat_summaries(columns)
    labels = {f"ep{i:04d}": int(label) for i, label in enumerate(rng.random(n) < 0.4)}
    ids = sorted(summaries)
    result = multivariate_probe(summaries, labels, ids[:300], ids[300:], n_bootstrap=0)
    assert abs(result["logistic_regression"]["test_auroc"] - 0.5) < 0.12
    assert abs(result["decision_tree"]["test_auroc"] - 0.5) < 0.12
    assert "test_auroc_ci95" not in result["logistic_regression"]  # bootstrap disabled
    assert "F0.last.missing" in result["columns"]


def test_multivariate_probe_views_indicators_and_errors() -> None:
    rng = np.random.default_rng(3)
    n = 40
    y = np.array([1, 0] * (n // 2))
    lactate = rng.normal(size=n)
    lactate[[0, 3, 5]] = np.nan  # missing in train
    lactate[35] = np.nan  # missing in test only: imputed, no extra indicator needed
    summaries = {
        f"ep{i:03d}": {
            "Lactate": {"last": float(lactate[i]), "mean": 0.1 * i, "max": 0.2 * i, "observed_last": float(i % 3 == 0)},
            "HR": {"last": float(rng.normal()), "mean": float(rng.normal()), "max": 1.0, "observed_last": float(i % 2)},
            "vasopressor_on": {"last": float(y[i]), "mean": 0.0, "max": 1.0, "observed_last": 1.0},
            WINDOW_KEY: {"n_rows": 13.0 - (i % 2)},
        }
        for i in range(n)
    }
    labels = {f"ep{i:03d}": int(label) for i, label in enumerate(y)}
    ids = sorted(summaries)
    train, test = ids[:30], ids[30:]
    groups = {eid: i // 2 for i, eid in enumerate(ids)}  # pairs of episodes share a patient

    full = multivariate_probe(summaries, labels, train, test, groups=groups, n_bootstrap=50)
    columns = set(full["columns"])
    assert {"Lactate.last", "Lactate.last.missing", "HR.observed_last", f"{WINDOW_KEY}.n_rows"} <= columns
    assert "vasopressor_on.observed_last" not in columns and "HR.max" not in columns  # constant on train
    assert full["n_indicator_columns"] == 1 and full["n_columns_dropped"] >= 3
    assert full["cv"]["splitter"].startswith("StratifiedGroupKFold") and full["bootstrap"]["unit"] == "group"
    assert full["logistic_regression"]["test_auroc"] == 1.0  # vasopressor_on.last is a planted leak

    observation = multivariate_probe(summaries, labels, train, test, view="observation_only", n_bootstrap=0)
    assert set(observation["columns"]) == {"Lactate.observed_last", "HR.observed_last", "Lactate.last.missing"}
    values = multivariate_probe(summaries, labels, train, test, view="values_only", n_bootstrap=0)
    assert not any(name.endswith((".missing", ".observed_last")) for name in values["columns"])
    assert f"{WINDOW_KEY}.n_rows" in values["columns"]
    subset = multivariate_probe(summaries, labels, train, test, feature_subset={"HR"}, n_bootstrap=0)
    assert {name.split(".")[0] for name in subset["columns"]} <= {"HR", WINDOW_KEY}
    assert subset["feature_subset"] == ["HR"]

    with pytest.raises(ValueError):
        multivariate_probe(summaries, labels, train, test + train[:1])  # overlap
    with pytest.raises(ValueError):
        multivariate_probe(summaries, labels, [eid for eid in train if labels[eid] == 1], test)  # one class
    with pytest.raises(ValueError):
        multivariate_probe(summaries, labels, train, test, view="everything")
    with pytest.raises(ValueError):
        multivariate_probe(summaries, labels, train, test + ["missing_episode"])
    with pytest.raises(ValueError):
        multivariate_probe(summaries, labels, train, test, groups={eid: 0 for eid in train})
    with pytest.raises(ValueError):  # the window column is kept in "all", so use a view without it
        multivariate_probe(summaries, labels, train, test, feature_subset={"unknown"}, view="observation_only")


def test_collect_summaries_shares_eligibility_and_transform(tmp_path: Path) -> None:
    ids = _cohort(tmp_path, n_pairs=3)
    anchor = pd.Timestamp("2151-01-01T09:40:00")
    meta = _meta("positive", 300, anchor, anchor - 5 * H, anchor)  # short window
    ids.append(_write_episode(tmp_path, meta, np.zeros((6, len(FEATURES))), np.zeros((6, len(FEATURES)), dtype=bool)))
    episodes = load_episodes(tmp_path, ids)

    raw = collect_summaries(episodes)
    assert raw["eligibility"] is False and raw["transform"] is None and len(raw["summaries"]) == 7
    v2 = collect_summaries(episodes, TaskSpec())
    assert len(v2["summaries"]) == 6 and [row["reason"] for row in v2["excluded"]] == ["short_window"]
    assert "vasopressor_on" not in next(iter(v2["summaries"].values()))
    assert v2["labels"] == {eid: int(eid.endswith("positive")) for eid in v2["summaries"]}


# ---------------------------------------------------------------------- synthetic cohort


def _write_episode(directory: Path, meta: dict, values: np.ndarray, mask: np.ndarray) -> str:
    stem = episode_stem(meta["episode_id"])
    np.save(directory / f"{stem}.npy", values)
    np.save(directory / f"{stem}_missing_mask.npy", mask)
    meta = {**meta, "missing_mask_file": f"{stem}_missing_mask.npy"}
    (directory / f"{stem}.json").write_text(json.dumps(meta), encoding="utf-8")
    return meta["episode_id"]


def _meta(episode_type: str, stay: int, anchor: pd.Timestamp, window_start: pd.Timestamp, window_end: pd.Timestamp) -> dict:
    start, end = window_start.floor("60min"), window_end.floor("60min")
    return {
        "episode_id": f"stay_{stay}_{anchor:%Y%m%d%H%M%S}_{episode_type}",
        "episode_type": episode_type,
        "stay_id": stay,
        "anchor_kind": "initiation",
        "anchor_time": anchor.isoformat(),
        "window_start": window_start.isoformat(),
        "window_end": window_end.isoformat(),
        "start_time": start.isoformat(),
        "end_time": end.isoformat(),
        "shape": [int((end - start) / H) + 1, len(FEATURES)],
        "resample_minutes": 60,
        "feature_names": list(FEATURES),
    }


def _cohort(directory: Path, n_pairs: int = 12) -> list[str]:
    """Paired positives/controls with identical pre-cutoff rows and two planted leaks in the
    anchor bin of positives: vasopressor_on = 1 and an observed MAP spike."""
    rng = np.random.default_rng(7)
    ids: list[str] = []
    for i in range(n_pairs):
        base = rng.normal(size=(13, len(FEATURES)))
        base[:, ZERO_FILL] = rng.integers(0, 2, size=(13, len(ZERO_FILL)))
        base[11:, VASO] = 0.0
        mask = rng.random((13, len(FEATURES))) < 0.3
        mask[:, ZERO_FILL] = False
        mask[0, :] = False

        anchor = pd.Timestamp("2150-01-01T09:40:00") + i * 24 * H
        pos_values, pos_mask = base.copy(), mask.copy()
        pos_values[12, VASO] = 1.0
        pos_values[12, MAP], pos_mask[12, MAP] = 25.0, False
        ids.append(_write_episode(directory, _meta("positive", 100 + i, anchor, anchor - 12 * H, anchor), pos_values, pos_mask))

        control_end = anchor - 30 * H  # paired anchor 30 h after the control: eligible
        ids.append(
            _write_episode(
                directory, _meta("control", 200 + i, anchor, control_end - 12 * H, control_end), base.copy(), mask.copy()
            )
        )
    return ids


def test_audit_flags_planted_leak_but_not_after_transform(tmp_path: Path) -> None:
    ids = _cohort(tmp_path)
    decoy = _meta("positive", 999, pd.Timestamp("2150-06-01T09:40:00"), pd.Timestamp("2150-05-31T21:40:00"), pd.Timestamp("2150-06-01T09:40:00"))
    _write_episode(tmp_path, decoy, np.ones((13, len(FEATURES))), np.zeros((13, len(FEATURES)), dtype=bool))

    v1 = audit_episode_ids(tmp_path, ids, None)
    assert v1["n_loaded"] == len(ids) == 24  # only the requested ids, not the decoy
    assert v1["label_counts"] == {"positive": 12, "control": 12}
    flagged = {(row["feature"], row["summary"]) for row in v1["flagged"]}
    assert {("vasopressor_on", "last"), ("MAP", "last"), ("MAP", "max")} <= flagged
    vaso = find_probe(v1, "vasopressor_on", "last")
    assert vaso["separability"] == 1.0 and vaso["auprc"] == 1.0  # ties with the MAP probes at 1.0
    assert vaso["mean_pos"] == 1.0 and vaso["mean_neg"] == 0.0

    for spec in (TaskSpec(), TaskSpec(reimpute_after_cutoff=False)):
        v2 = audit_episode_ids(tmp_path, ids, spec)
        assert v2["flagged"] == []
        assert v2["label_counts"] == {"positive": 12, "control": 12}
        assert v2["exclusions"]["n_excluded"] == 0
        assert all(row["separability"] == pytest.approx(0.5) for row in v2["probes"])
        assert "vasopressor_on" not in {row["feature"] for row in v2["probes"]}
        assert v2["rows_by_label"] == {"positive": {"11": 12}, "control": {"11": 12}}
        assert v2["transform"]["dropped_features"] == ["vasopressor_on"]


def test_audit_excludes_short_positives_and_reports_window_length(tmp_path: Path) -> None:
    ids = _cohort(tmp_path, n_pairs=6)
    for i in range(3):  # positives admitted 5 h before the anchor: 6 rows, no controls
        anchor = pd.Timestamp("2151-01-01T09:40:00") + i * 24 * H
        meta = _meta("positive", 300 + i, anchor, anchor - 5 * H, anchor)
        values = np.zeros((6, len(FEATURES)))
        values[-1, VASO] = 1.0
        ids.append(_write_episode(tmp_path, meta, values, np.zeros_like(values, dtype=bool)))

    v1 = audit_episode_ids(tmp_path, ids, None)
    length = find_probe(v1, WINDOW_KEY, "n_rows")
    assert length["direction"] == "lower_in_positive" and length["separability"] > 0.5
    assert v1["rows_by_label"]["positive"] == {"6": 3, "13": 6}

    v2 = audit_episode_ids(tmp_path, ids, TaskSpec())
    assert v2["exclusions"]["primary"] == {REASON_SHORT_WINDOW: 3}
    assert v2["label_counts"] == {"positive": 6, "control": 6}
    assert {row["reason"] for row in v2["excluded"]} == {REASON_SHORT_WINDOW}
    assert v2["flagged"] == []

    no_eligibility = audit_episode_ids(tmp_path, ids, TaskSpec(), apply_eligibility=False)
    assert no_eligibility["exclusions"] is None and no_eligibility["n_audited"] == len(ids)
    assert no_eligibility["transform"]["audited_not_full_window"] == 3


def test_load_episode_by_constructed_path(tmp_path: Path) -> None:
    ids = _cohort(tmp_path, n_pairs=1)
    assert episode_stem(ids[0]) == f"shock_episode_{ids[0]}"
    assert episode_stem(episode_stem(ids[0])) == f"shock_episode_{ids[0]}"
    plain = load_episode(tmp_path, ids[0])
    prefixed = load_episode(tmp_path, episode_stem(ids[0]))
    assert plain.episode_id == prefixed.episode_id == ids[0]
    assert plain.values.shape == plain.mask.shape == (13, len(FEATURES)) and plain.mask.dtype == bool
    assert [episode.episode_id for episode in load_episodes(tmp_path, [ids[1], ids[0], ids[1]])] == [ids[1], ids[0]]
    with pytest.raises(FileNotFoundError):
        load_episode(tmp_path, "stay_1_21000101000000_positive")


def test_anchor_context_counts() -> None:
    anchor = pd.Timestamp("2136-01-15T09:40:00")
    positive = {"episode_type": "positive", "stay_id": 1, "anchor_time": anchor.isoformat()}
    recent = {"episode_type": "control", "stay_id": 1, "window_start": (anchor + 5 * H).isoformat()}
    old = {"episode_type": "control", "stay_id": 1, "window_start": (anchor + 40 * H).isoformat()}
    clean = {"episode_type": "control", "stay_id": 2, "window_start": anchor.isoformat()}
    index = {1: [anchor - 3 * H, anchor]}
    counts = anchor_context_counts([positive, recent, old, clean], index, TaskSpec())
    assert counts == {
        "positives": 1,
        "positives_with_earlier_anchor_in_window": 1,
        "controls": 3,
        "controls_after_recent_anchor": 1,
        "controls_after_any_anchor": 2,
    }
    assert anchor_context_counts([], {}, TaskSpec())["positives"] == 0


# ---------------------------------------------------------------------------- split runner


def test_run_split_audit_and_cli(tmp_path: Path) -> None:
    episode_dir = tmp_path / "episodes"
    episode_dir.mkdir()
    ids = _cohort(episode_dir, n_pairs=8)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"name": "toy", "split": {"seed": 3, "train": ids[:10], "test": ids[10:]}}), encoding="utf-8")
    assert read_split_manifest(manifest) == {"train": ids[:10], "test": ids[10:]}

    result = run_split_audit(manifest, episode_dir, multivariate_bootstrap=50)
    assert result["n_ids"] == 16 and result["split_counts"] == {"train": 10, "test": 6}
    assert set(result["audits"]) == {
        "v1_raw",
        "v2_proposed",
        "v2_transform_only",
        "v2_without_refit",
        "v2_with_control_paired_anchors",
        "v1_raw_train",
        "v1_raw_test",
    }
    headline = result["headline"]
    assert headline["v1_n_flagged"] > 0 and headline["v2_n_flagged"] == 0
    assert headline["v1_vasopressor_on_last"]["separability"] == 1.0
    assert headline["v1_test_vasopressor_on_last"]["n"] == 6
    assert result["anchor_context_v2_kept"]["positive_anchor_index"]["positives"] == 8
    assert set(result["candidate_specs"]) == set(CANDIDATE_SPECS)
    initiation = result["candidate_specs"]["initiation_only"]
    assert initiation["label_counts"] == {"positive": 8, "control": 8} and initiation["n_flagged"] == 0
    gap = result["candidate_specs"]["gap_2h"]
    assert gap["rows_by_label"] == {"positive": {"10": 8}, "control": {"10": 8}}

    escalation_only = run_split_audit(
        manifest,
        episode_dir,
        candidate_specs={"escalation_only": TaskSpec(anchor_kinds=("escalation",))},
        multivariate_bootstrap=0,
    )["candidate_specs"]["escalation_only"]
    assert escalation_only["label_counts"] == {"positive": 0, "control": 8}
    assert escalation_only["exclusions"]["primary"] == {"anchor_kind": 8}
    assert escalation_only["probe_error"] is not None and escalation_only["top_probes"] == []

    multivariate = result["multivariate"]
    assert multivariate["train_splits"] == ["train"] and multivariate["test_split"] == "test"
    assert set(multivariate["settings"]) == {"v1_raw", "v2_proposed", "v2_transform_only"}
    assert all(set(views) == set(MULTIVARIATE_VIEWS) for views in multivariate["settings"].values())
    # v1 still holds the planted anchor-bin leaks; v2 positives and controls are identical
    assert headline["multivariate"]["v1_raw/all"]["tree_test_auroc"] == 1.0
    assert headline["multivariate"]["v2_proposed/all"]["lr_test_auroc"] == 0.5
    assert headline["multivariate"]["v2_proposed/all"]["tree_test_auroc"] == 0.5
    assert headline["multivariate"]["v2_proposed/values_only"]["n_test"] == 6

    out_json = tmp_path / "out" / "audit.json"
    assert main(["--manifest", str(manifest), "--episode-dir", str(episode_dir), "--out-json", str(out_json)]) == 0

    def reject(constant: str) -> None:
        raise AssertionError(f"non-strict JSON constant {constant}")

    written = json.loads(out_json.read_text(encoding="utf-8"), parse_constant=reject)
    assert written["headline"]["v2_n_flagged"] == 0


def test_json_safe_and_probe_table() -> None:
    converted = json_safe({"a": np.float64("nan"), "b": (np.int64(3), np.bool_(True)), "c": pd.Timestamp("2136-01-01")})
    assert converted == {"a": None, "b": [3, True], "c": "2136-01-01T00:00:00"}
    rows = [
        {
            "feature": "vasopressor_on",
            "summary": "last",
            "separability": 0.9,
            "auroc": 0.9,
            "auprc_oriented": 0.8,
            "direction": "higher_in_positive",
            "mean_pos": 1.0,
            "mean_neg": float("nan"),
        }
    ]
    table = format_probe_table(rows, k=10).splitlines()
    assert len(table) == 3 and "vasopressor_on" in table[2] and "| nan |" in table[2]
