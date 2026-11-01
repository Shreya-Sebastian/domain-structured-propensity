"""Metrics, calibration, paired bootstrap and directional tests."""
import subprocess

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from dsp.models import precision_at_top

N_BOOT = 1000
BOOT_SEED = 20261101


def assert_test_allowed() -> None:
    """The test block may only be scored once hypotheses.md is committed and
    unchanged since that commit."""
    tracked = subprocess.run(["git", "ls-files", "--error-unmatch", "hypotheses.md"],
                             capture_output=True)
    dirty = subprocess.run(["git", "status", "--porcelain", "hypotheses.md"],
                           capture_output=True, text=True).stdout.strip()
    if tracked.returncode != 0 or dirty:
        raise RuntimeError("commit hypotheses.md before scoring the test block")


def calibrate(score_cal, y_cal) -> IsotonicRegression:
    return IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(score_cal, y_cal)


def ece(y, p, bins=10) -> float:
    """Expected calibration error with equal-count bins."""
    order = np.argsort(p)
    err = 0.0
    for chunk in np.array_split(order, bins):
        if len(chunk):
            err += len(chunk) * abs(p[chunk].mean() - y[chunk].mean())
    return float(err / len(y))


def metrics(y, score, prob=None) -> dict:
    out = {
        "auc": roc_auc_score(y, score),
        "average_precision": average_precision_score(y, score),
        "precision_top10": precision_at_top(y, score),
    }
    out["lift_top10"] = out["precision_top10"] / y.mean()
    if prob is not None:
        out["brier"] = brier_score_loss(y, prob)
        out["ece"] = ece(y, prob)
    return {k: round(float(v), 4) for k, v in out.items()}


def paired_bootstrap(y, scores: dict, pairs: list, n=N_BOOT, seed=BOOT_SEED) -> pd.DataFrame:
    """Difference (model minus reference) with a 95% percentile interval, for
    each (model, reference) pair, resampling the same clients for both."""
    rng = np.random.default_rng(seed)
    idx = [rng.integers(0, len(y), len(y)) for _ in range(n)]
    rows = []
    for model, ref in pairs:
        for name, fn in [("auc", roc_auc_score), ("average_precision", average_precision_score),
                         ("precision_top10", precision_at_top)]:
            point = fn(y, scores[model]) - fn(y, scores[ref])
            diffs = np.array([fn(y[i], scores[model][i]) - fn(y[i], scores[ref][i]) for i in idx])
            lo, hi = np.percentile(diffs, [2.5, 97.5])
            rows.append({"model": model, "reference": ref, "metric": name,
                         "difference": round(point, 4), "ci_low": round(lo, 4),
                         "ci_high": round(hi, 4)})
    return pd.DataFrame(rows)


def _grid(train_values: pd.Series) -> np.ndarray:
    values = np.sort(train_values.dropna().unique())
    if len(values) > 20:
        values = np.unique(np.quantile(train_values.dropna(), np.linspace(0.05, 0.95, 10)))
    return values


# Clients for whom changing the feature is realistic: success only among
# clients contacted in an earlier campaign; recency only for contacted clients.
SUBGROUP = {
    "prev_success": lambda X: X["previous"] > 0,
    "days_since_contact": lambda X: X["contacted_before"] == 1,
}


def directional_test(model, X, X_train, feature, direction, tol=1e-9) -> dict:
    """Move one feature over a grid for every client in its subgroup and count
    clients whose score moves against the declared direction."""
    keep = SUBGROUP.get(feature, lambda X: X[feature].notna())(X).to_numpy()
    Xs = X[keep].reset_index(drop=True)
    grid = _grid(X_train[feature])
    scores = []
    for v in grid:
        Xv = Xs.copy()
        Xv[feature] = v
        scores.append(model.decision_function(Xv))
    steps = np.diff(np.vstack(scores), axis=0)
    if direction == "decreasing":
        steps = -steps
    violated = (steps < -tol).any(axis=0)
    # Change in score from the lowest to the highest grid value, in the declared
    # direction: shows whether a violation is large or a near-zero wrong sign.
    total = steps.sum(axis=0)
    return {"feature": feature, "direction": direction, "clients": int(len(Xs)),
            "grid_points": int(len(grid)), "violation_share": round(float(violated.mean()), 4),
            "mean_change": round(float(total.mean()), 4),
            "min_change": round(float(total.min()), 4)}


def directional_test_realistic(model, X, X_train, tol=1e-9) -> dict:
    """Realistic variant for `prev_success`: in training, every previous success
    has a recorded contact date. So for clients contacted in an earlier
    campaign, compare 'contacted, no success' with 'contacted, success', both
    with a contact date (the client's own, or the training median for successes)."""
    keep = (X["previous"] > 0).to_numpy()
    Xs = X[keep].reset_index(drop=True)
    success_days = X_train.loc[X_train["prev_success"] == 1, "days_since_contact"].median()
    base = Xs.copy()
    base["contacted_before"] = 1
    base["days_since_contact"] = base["days_since_contact"].fillna(success_days)
    lo, hi = base.copy(), base.copy()
    lo["prev_success"], hi["prev_success"] = 0, 1
    change = model.decision_function(hi) - model.decision_function(lo)
    return {"feature": "prev_success (with contact date)", "direction": "increasing",
            "clients": int(len(Xs)), "grid_points": 2,
            "violation_share": round(float((change < -tol).mean()), 4),
            "mean_change": round(float(change.mean()), 4),
            "min_change": round(float(change.min()), 4)}
