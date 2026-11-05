from __future__ import annotations

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

from src.icu.v2.robust_selection import SchemaScenario, _fit_logistic, grouped_folds, scenario_set, schema_robust_select
from src.icu.v2.selection_baselines import SelectionInput
from src.icu.v2.types import make_symbolic_program

ALL = frozenset({"HR", "MAP", "Lactate", "pH", "RR", "SpO2"})


def test_newton_logistic_matches_sklearn():
    rng = np.random.default_rng(0)
    X = rng.integers(0, 2, (500, 4)).astype(float)
    y = (rng.random(500) < 1 / (1 + np.exp(-(X @ np.array([1.5, -1.0, 0.5, 0.0]) - 0.3)))).astype(float)
    beta = _fit_logistic(X, y, C=1.0)
    ref = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000, tol=1e-10).fit(X, y)
    assert np.allclose(beta[1:], ref.coef_.ravel(), atol=1e-4)
    assert abs(beta[0] - ref.intercept_[0]) < 1e-4


def test_grouped_folds_keep_patients_together():
    ids = [f"e{i}" for i in range(30)]
    subject = {e: i // 3 for i, e in enumerate(ids)}  # 3 episodes per patient
    folds = grouped_folds(ids, subject, 3, seed=1)
    assert sorted(np.concatenate([test for _, test in folds]).tolist()) == list(range(30))
    for train, test in folds:
        assert not ({subject[ids[i]] for i in train} & {subject[ids[i]] for i in test})
    assert [t.tolist() for _, t in folds] == [t.tolist() for _, t in grouped_folds(ids, subject, 3, seed=1)]


def test_scenario_weights():
    scenarios = scenario_set(ALL, {"ext": ALL - {"Lactate"}}, dropout_q=0.5, n_dropout=10, seed=3)
    assert abs(sum(s.weight for s in scenarios) - 1.0) < 1e-12
    source = [s for s in scenarios if s.name == "source"][0]
    assert abs(source.weight - 1 / 3) < 1e-12
    assert all(s.features for s in scenarios)  # a dropout schema is never empty
    plain = scenario_set(ALL, {}, dropout_q=0.0)
    assert [s.name for s in plain] == ["source"] and plain[0].weight == 1.0


def _toy(n=600, seed=0):
    rng = np.random.default_rng(seed)
    pos = {f"p{i}" for i in range(n // 2)}
    ids = [f"p{i}" for i in range(n // 2)] + [f"n{i}" for i in range(n // 2)]
    subject = {e: e for e in ids}

    def fire(rate_pos, rate_neg, generator):
        return frozenset(e for e in ids if generator.random() < (rate_pos if e in pos else rate_neg))

    lab = make_symbolic_program("lab", "toy", "sequence", ({"Lactate:q3"},), score=3.0)  # strongest, needs labs
    lab_copy = make_symbolic_program("lab_copy", "toy", "sequence", ({"pH:q3"},), score=2.9)
    vital = make_symbolic_program("vital", "toy", "sequence", ({"MAP:q0"},), score=1.0)  # weaker, always executable
    noise = make_symbolic_program("noise", "toy", "sequence", ({"RR:q1"},), score=0.5)
    f_lab = fire(0.8, 0.1, rng)
    firings = {"lab": f_lab, "lab_copy": f_lab, "vital": fire(0.6, 0.25, rng), "noise": fire(0.3, 0.3, rng)}
    inp = SelectionInput(candidates=(lab, lab_copy, vital, noise), train_firings=firings, train_ids=frozenset(ids), train_positives=frozenset(pos))
    return inp, subject


def test_srs_prefers_executable_programs_when_targets_lack_labs():
    inp, subject = _toy()
    source_only = schema_robust_select(inp, 1, [SchemaScenario("source", ALL, 1.0)], subject)
    assert [p.program_id for p in source_only.registry] == ["lab"]
    no_labs = [SchemaScenario("source", ALL, 0.1), SchemaScenario("no_labs", ALL - {"Lactate", "pH"}, 0.9)]
    robust = schema_robust_select(inp, 1, no_labs, subject)
    assert [p.program_id for p in robust.registry] == ["vital"]


def test_srs_never_adds_an_exact_duplicate_and_stops_without_gain():
    inp, subject = _toy()
    result = schema_robust_select(inp, 4, [SchemaScenario("source", ALL, 1.0)], subject)
    ids = [p.program_id for p in result.registry]
    assert ids[0] == "lab" and not ({"lab", "lab_copy"} <= set(ids))
    assert all(b > a for a, b in zip(result.objective, result.objective[1:]))  # strictly increasing J
    assert len(ids) <= 4 and result.stop_reason


def test_projected_and_in_sample_modes():
    inp, subject = _toy()
    scenarios = [SchemaScenario("source", ALL, 0.5), SchemaScenario("no_labs", ALL - {"Lactate", "pH"}, 0.5)]
    for mode in ("frozen", "projected"):
        for cv in (True, False):
            result = schema_robust_select(inp, 3, scenarios, subject, mode=mode, cv=cv)
            assert result.registry and all(b > a for a, b in zip(result.objective, result.objective[1:]))
    # the projected objective refits per schema: a no-labs scenario only sees executable programs,
    # so the strong lab program still pays off on the source half and is picked first
    projected = schema_robust_select(inp, 1, scenarios, subject, mode="projected")
    assert [p.program_id for p in projected.registry] == ["lab"]
    with pytest.raises(ValueError):
        schema_robust_select(inp, 1, scenarios, subject, mode="other")
