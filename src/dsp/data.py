"""Load UCI Bank Marketing, rebuild call dates and apply the availability audit.

Scoring moment: before the first call of a campaign to the client. Fields not
known at that moment never reach a model.
"""
from pathlib import Path

import numpy as np
import pandas as pd

RAW_PATH = Path("data/raw/bank-additional/bank-additional-full.csv")

# Known only after the call, leaking through the stopping rule, or the date of
# the last call rather than a fact about the client.
BANNED = ("duration", "campaign", "month", "day_of_week")

MONTHS = ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]
FIRST_YEAR = 2008

CATEGORICAL = ["job", "marital", "education", "default", "housing", "loan", "contact"]


def load_raw(path: Path = RAW_PATH) -> pd.DataFrame:
    df = pd.read_csv(path, sep=";")
    df.insert(0, "row_id", np.arange(len(df)))
    return df


def rebuild_dates(df: pd.DataFrame) -> pd.DataFrame:
    """Add `ym` (YYYY-MM). Rows are in date order, so the year goes up whenever
    the month number goes down."""
    month = df["month"].map({m: i for i, m in enumerate(MONTHS, 1)})
    if month.isna().any():
        raise ValueError("unknown month value")
    year = FIRST_YEAR + (month.diff() < 0).cumsum()
    out = df.copy()
    out["ym"] = [f"{y}-{m:02d}" for y, m in zip(year, month)]
    return out


def assert_no_banned(columns) -> None:
    leaked = sorted(set(columns) & set(BANNED))
    if leaked:
        raise ValueError(f"banned fields reached a model: {leaked}")


MACRO = ["emp.var.rate", "cons.price.idx", "cons.conf.idx", "euribor3m", "nr.employed"]


def lag_macro(df: pd.DataFrame) -> pd.DataFrame:
    """Robustness check: replace each month's market indicators with the
    previous recorded month's average, so no value can come from after the
    first call. The first month keeps its own values."""
    monthly = df.groupby("ym")[MACRO].mean().sort_index()
    lagged = monthly.shift(1).fillna(monthly)
    out = df.copy()
    out[MACRO] = lagged.loc[out["ym"]].to_numpy()
    return out


def prepare(df: pd.DataFrame, contact_flag: str = "pdays", drop_duplicates: bool = False,
            drop_features: tuple = (), lagged_macro: bool = False):
    """Return (X, y, meta). X holds model features only; meta keeps row_id and
    ym for splitting and reporting.

    The defaults are the pre-registered design. The options exist only for the
    post-hoc robustness checks in scripts/run_robustness.py."""
    if "ym" not in df:
        df = rebuild_dates(df)
    if drop_duplicates:
        df = df.drop_duplicates(subset=[c for c in df.columns if c not in ("row_id",)])
    if lagged_macro:
        df = lag_macro(df)
    X = df.drop(columns=["row_id", "ym", "y", *BANNED]).copy()

    # pdays = 999 means "no day count recorded". The documentation says it
    # means never contacted, but 4,110 rows have 999 with earlier contacts and a
    # failed outcome. Pre-registered design (contact_flag="pdays"): the flag
    # is set when a day count exists, so `contacted_before` really means "has a
    # recorded contact date". Robustness check (contact_flag="previous"): the
    # flag is set when there were earlier contacts (previous > 0).
    has_days = X["pdays"] != 999
    if contact_flag == "pdays":
        X["contacted_before"] = has_days.astype(int)
    elif contact_flag == "previous":
        X["contacted_before"] = (X["previous"] > 0).astype(int)
    else:
        raise ValueError(contact_flag)
    X["days_since_contact"] = X["pdays"].where(has_days, np.nan)
    X = X.drop(columns=["pdays"])

    # Previous campaign outcome as a flag, so a direction can be enforced on it
    # (LightGBM cannot constrain a categorical feature). A failure is still
    # visible as previous > 0 with prev_success = 0.
    X["prev_success"] = (X["poutcome"] == "success").astype(int)
    X = X.drop(columns=["poutcome"])

    # "unknown" stays its own category. Categories are fixed from the full file
    # so every block encodes them the same way.
    for col in CATEGORICAL:
        if col in X:
            X[col] = pd.Categorical(X[col], categories=sorted(df[col].unique()))
    X = X.drop(columns=list(drop_features))

    assert_no_banned(X.columns)
    y = (df["y"] == "yes").astype(int).to_numpy()
    meta = df[["row_id", "ym"]].reset_index(drop=True)
    return X.reset_index(drop=True), y, meta
