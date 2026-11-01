import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from dsp import data, reasoning, robust


def test_delong_difference_matches_sklearn():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 400)
    a, b = rng.normal(size=400) + y, rng.normal(size=400) + 0.5 * y
    d = robust.delong(y, a, b)
    assert d["difference"] == pytest.approx(roc_auc_score(y, a) - roc_auc_score(y, b), abs=1e-4)
    assert d["ci_low"] < d["difference"] < d["ci_high"]


def test_delong_identical_models_have_zero_difference():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 200)
    s = rng.normal(size=200) + y
    d = robust.delong(y, s, s.copy())
    assert d["difference"] == 0 and d["se"] == 0


@pytest.fixture(scope="module")
def raw():
    return data.rebuild_dates(data.load_raw())


def test_pdays_999_is_not_never_contacted(raw):
    # The audit's finding: rows with pdays = 999 but earlier contacts exist.
    assert ((raw["pdays"] == 999) & (raw["previous"] > 0)).sum() == 4110


def test_contact_flag_from_previous(raw):
    X, _, _ = data.prepare(raw, contact_flag="previous")
    assert (X["contacted_before"] == (raw["previous"] > 0).astype(int).to_numpy()).all()


def test_lagged_macro_uses_previous_month(raw):
    lagged = data.lag_macro(raw)
    months = sorted(raw["ym"].unique())
    second = raw["ym"] == months[1]
    first_mean = raw.loc[raw["ym"] == months[0], "euribor3m"].mean()
    assert np.allclose(lagged.loc[second, "euribor3m"], first_mean)


def test_restrict_drops_empty_factors(raw):
    X, _, _ = data.prepare(raw, drop_features=("contact",))
    rm = reasoning.load().restrict(X.columns)
    assert "reachability" not in rm.factors
    rm.validate(X.columns)
