import numpy as np
import pandas as pd
import pytest

from dsp import evaluate, models


class _Linear:
    """Score = weight * feature: increasing if weight > 0."""

    def __init__(self, feature, weight):
        self.feature, self.weight = feature, weight

    def decision_function(self, X):
        return self.weight * X[self.feature].to_numpy(dtype=float)


def _frame(n=50):
    rng = np.random.default_rng(0)
    return pd.DataFrame({
        "euribor3m": rng.uniform(0.6, 5.0, n),
        "previous": rng.integers(0, 3, n),
        "prev_success": rng.integers(0, 2, n),
        "contacted_before": rng.integers(0, 2, n),
        "days_since_contact": rng.uniform(0, 25, n),
    })


def test_directional_test_finds_and_clears_violations():
    X = _frame()
    ok = evaluate.directional_test(_Linear("euribor3m", 1.0), X, X, "euribor3m", "increasing")
    bad = evaluate.directional_test(_Linear("euribor3m", -1.0), X, X, "euribor3m", "increasing")
    assert ok["violation_share"] == 0.0
    assert bad["violation_share"] == 1.0


def test_directional_test_uses_realistic_subgroup():
    X = _frame()
    r = evaluate.directional_test(_Linear("prev_success", 1.0), X, X, "prev_success", "increasing")
    assert r["clients"] == int((X["previous"] > 0).sum())


def test_precision_at_top_ignores_row_order_for_ties():
    y = np.array([1, 0, 1, 0, 0, 0, 0, 0, 0, 0] * 10)
    score = np.zeros(100)  # all tied: expected precision is the base rate
    assert models.precision_at_top(y, score) == pytest.approx(y.mean())
    assert models.precision_at_top(y[::-1], score) == pytest.approx(y.mean())


def test_bootstrap_interval_contains_point_and_is_zero_for_identical_models():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 300)
    s = rng.normal(size=300) + y
    boot = evaluate.paired_bootstrap(y, {"a": s, "b": s.copy()}, [("a", "b")], n=200)
    assert (boot["difference"] == 0).all()
    assert (boot["ci_low"] == 0).all() and (boot["ci_high"] == 0).all()


def test_test_block_guard_refuses_without_committed_hypotheses(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # not a repository: hypotheses.md is not tracked
    with pytest.raises(RuntimeError):
        evaluate.assert_test_allowed()
