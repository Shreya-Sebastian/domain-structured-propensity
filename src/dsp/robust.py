"""Tools for the post-hoc robustness checks (audit of 1 November 2026).

Nothing here changes the pre-registered analysis; scripts/run_robustness.py
reports these checks next to it.
"""
import numpy as np
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score


# ---------- DeLong test for two correlated AUCs (DeLong et al. 1988) ----------

def _midrank(x):
    order = np.argsort(x, kind="mergesort")
    xs = x[order]
    n = len(x)
    ranks = np.empty(n)
    i = 0
    while i < n:
        j = i
        while j < n and xs[j] == xs[i]:
            j += 1
        ranks[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    out = np.empty(n)
    out[order] = ranks
    return out


def delong(y, a, b):
    """AUC(a) - AUC(b) on the same clients: difference, standard error,
    two-sided p-value and 95% normal interval (Sun and Xu's fast algorithm)."""
    y = np.asarray(y)
    pos, neg = y == 1, y == 0
    m, n = pos.sum(), neg.sum()
    aucs, v01, v10 = [], [], []
    for s in (np.asarray(a, float), np.asarray(b, float)):
        sp, sn = s[pos], s[neg]
        tx, ty = _midrank(sp), _midrank(sn)
        tz = _midrank(np.concatenate([sp, sn]))
        aucs.append((tz[:m].sum() - m * (m + 1) / 2) / (m * n))
        v01.append((tz[:m] - tx) / n)
        v10.append(1 - (tz[m:] - ty) / m)
    s01, s10 = np.cov(np.vstack(v01)), np.cov(np.vstack(v10))
    cov = s01 / m + s10 / n
    diff = aucs[0] - aucs[1]
    se = float(np.sqrt(max(cov[0, 0] + cov[1, 1] - 2 * cov[0, 1], 0)))
    z = diff / se if se > 0 else 0.0
    return {"difference": round(diff, 4), "se": round(se, 4),
            "p_value": round(float(2 * stats.norm.sf(abs(z))), 4),
            "ci_low": round(diff - 1.96 * se, 4), "ci_high": round(diff + 1.96 * se, 4)}


# ---------- BCa bootstrap interval for a paired difference ----------

def bca_difference(y, a, b, metric=roc_auc_score, n_resamples=10_000, seed=20261101):
    """Paired BCa interval (scipy.stats.bootstrap) for metric(a) - metric(b)."""
    y, a, b = np.asarray(y), np.asarray(a), np.asarray(b)

    def stat(idx):
        idx = idx.astype(int)
        return metric(y[idx], a[idx]) - metric(y[idx], b[idx])

    res = stats.bootstrap((np.arange(len(y)),), stat, vectorized=False, n_resamples=n_resamples,
                          method="BCa", random_state=np.random.default_rng(seed))
    return {"difference": round(float(stat(np.arange(len(y)))), 4),
            "ci_low": round(float(res.confidence_interval.low), 4),
            "ci_high": round(float(res.confidence_interval.high), 4)}


# ---------- Calibration ----------

def platt(score_cal, y_cal):
    """Sigmoid (Platt) calibration fitted on the calibration block."""
    lr = LogisticRegression(C=1e6, max_iter=1000).fit(np.asarray(score_cal).reshape(-1, 1), y_cal)
    return lambda s: lr.predict_proba(np.asarray(s).reshape(-1, 1))[:, 1]


def calibration_in_the_large(y, p):
    """Observed minus mean predicted rate (0 = right on average)."""
    return round(float(np.mean(y) - np.mean(p)), 4)


def calibration_slope(y, p, eps=1e-6):
    """Slope of a logistic regression of the outcome on logit(p) (1 = ideal)."""
    p = np.clip(np.asarray(p), eps, 1 - eps)
    lp = np.log(p / (1 - p)).reshape(-1, 1)
    return round(float(LogisticRegression(C=1e6, max_iter=1000).fit(lp, y).coef_[0, 0]), 4)
