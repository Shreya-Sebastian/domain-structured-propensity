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

CATEGORICAL = ["job", "marital", "education", "default", "housing", "loan", "contact", "poutcome"]


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


def prepare(df: pd.DataFrame):
    """Return (X, y, meta). X holds model features only; meta keeps row_id and
    ym for splitting and reporting."""
    if "ym" not in df:
        df = rebuild_dates(df)
    X = df.drop(columns=["row_id", "ym", "y", *BANNED]).copy()

    # pdays = 999 means never contacted before: split into a flag and a number.
    never = X["pdays"] == 999
    X["contacted_before"] = (~never).astype(int)
    X["days_since_contact"] = X["pdays"].where(~never, np.nan)
    X = X.drop(columns=["pdays"])

    # "unknown" stays its own category. Categories are fixed from the full file
    # so every block encodes them the same way.
    for col in CATEGORICAL:
        X[col] = pd.Categorical(X[col], categories=sorted(df[col].unique()))

    assert_no_banned(X.columns)
    y = (df["y"] == "yes").astype(int).to_numpy()
    meta = df[["row_id", "ym"]].reset_index(drop=True)
    return X.reset_index(drop=True), y, meta
