import pandas as pd
import pytest

from dsp import data, reasoning, splits


@pytest.fixture(scope="module")
def prepared():
    df = data.rebuild_dates(data.load_raw())
    return df, *data.prepare(df)


def test_row_count_and_date_span(prepared):
    df, *_ = prepared
    assert len(df) == 41188
    assert df["ym"].iloc[0] == "2008-05"
    assert df["ym"].iloc[-1] == "2010-11"
    assert df["ym"].is_monotonic_increasing


def test_banned_fields_never_reach_a_model(prepared):
    _, X, _, _ = prepared
    assert not set(X.columns) & set(data.BANNED)


def test_banned_check_fails_when_a_field_leaks():
    with pytest.raises(ValueError):
        data.assert_no_banned(["age", "duration"])


def test_pdays_recode(prepared):
    df, X, _, _ = prepared
    never = df["pdays"] == 999
    assert (X.loc[never.to_numpy(), "contacted_before"] == 0).all()
    assert X.loc[never.to_numpy(), "days_since_contact"].isna().all()


def test_every_feature_in_exactly_one_factor(prepared):
    _, X, _, _ = prepared
    reasoning.load().validate(X.columns)


def test_reasoning_validation_catches_errors(prepared):
    _, X, _, _ = prepared
    rm = reasoning.load()
    rm.factors["capacity"]["features"].append("age")  # age is already in life_stage
    with pytest.raises(ValueError):
        rm.validate(X.columns)

    rm = reasoning.load()
    rm.factors["reachability"]["directions"] = {"contact": "increasing"}  # no source
    with pytest.raises(ValueError):
        rm.validate(X.columns)


def test_blocks_disjoint_ordered_and_sized(prepared):
    _, _, _, meta = prepared
    masks = splits.block_masks(meta, splits.load_blocks())
    total = sum(m.astype(int) for m in masks.values())
    assert total.max() == 1
    counts = {k: int(m.sum()) for k, m in masks.items()}
    assert counts == {"train": 36224, "calibration": 1663, "pilot": 1243, "test": 2058}


def test_overlapping_blocks_rejected():
    meta = pd.DataFrame({"ym": ["2008-05", "2008-06"]})
    with pytest.raises(ValueError):
        splits.block_masks(meta, {"a": ["2008-05", "2008-06"], "b": ["2008-06", "2008-07"]})
