"""RQ3: are factor-level reasons more stable than feature-level reasons?

The model is refitted on bootstrap resamples of the training months. For each
evaluated client and each refit we record the main reason the brief would
show: the factor that raises the score most (factor level) and the single
feature that raises it most (feature level). Market context is left out of
both, because the brief shows it as context, never as a reason.

Agreement for a client = share of refits that give its most common answer.
"""
import numpy as np
import pandas as pd

from dsp import models, records

N_REFITS = 30
REFIT_SEED = 20261102


def _top(frame: pd.DataFrame) -> np.ndarray:
    return frame.to_numpy().argmax(axis=1)


def _agreement(tops: np.ndarray) -> np.ndarray:
    """tops: refits x clients of category codes -> per-client modal share."""
    out = np.empty(tops.shape[1])
    for j in range(tops.shape[1]):
        _, counts = np.unique(tops[:, j], return_counts=True)
        out[j] = counts.max() / tops.shape[0]
    return out


def reason_columns(rm, columns):
    """Factors and features that may be shown as reasons (not context)."""
    factors = [f for f, s in rm.factors.items() if not s.get("context_only")]
    features = [c for c in columns if rm.feature_to_factor[c] in factors]
    return factors, features


def random_partition(rm, columns, seed):
    """Control: the same reason features split into random groups with the
    same sizes as the real factors."""
    factors, features = reason_columns(rm, columns)
    sizes = [len(rm.factors[f]["features"]) for f in factors]
    perm = np.random.default_rng(seed).permutation(features)
    groups, start = {}, 0
    for i, size in enumerate(sizes):
        groups[f"random_{i}"] = list(perm[start:start + size])
        start += size
    return groups


def run(rm, Xtr, ytr, Xev, n_refits=N_REFITS, seed=REFIT_SEED, n_random=5):
    """Return per-client agreement for each model and level."""
    cols = list(Xtr.columns)
    factors, features = reason_columns(rm, cols)
    specs = {
        "lgbm_structured": dict(interaction_groups=rm.interaction_groups(cols),
                                monotone=rm.monotone_vector(cols)),
        "lgbm_unconstrained": dict(),
    }
    partitions = [random_partition(rm, cols, seed + k) for k in range(n_random)]
    rng = np.random.default_rng(seed)
    tops = {(m, lvl): [] for m in specs for lvl in ["factor", "feature"]}
    tops.update({("lgbm_unconstrained", f"random_{k}"): [] for k in range(n_random)})

    for _ in range(n_refits):
        idx = rng.integers(0, len(Xtr), len(Xtr))
        Xb, yb = Xtr.iloc[idx].reset_index(drop=True), ytr[idx]
        for name, kw in specs.items():
            model = models.fit(Xb, yb, **kw)
            fac, feat, _, _ = records.factor_contributions(model, Xev, rm)
            tops[(name, "factor")].append(_top(fac[factors]))
            tops[(name, "feature")].append(_top(feat[features]))
            if name == "lgbm_unconstrained":
                for k, groups in enumerate(partitions):
                    rand = pd.DataFrame({g: feat[fs].sum(axis=1) for g, fs in groups.items()})
                    tops[(name, f"random_{k}")].append(_top(rand))

    agreement = {key: _agreement(np.vstack(v)) for key, v in tops.items()}
    return agreement


def summarise(agreement, n_boot=1000, seed=20261101):
    """Mean agreement per (model, level), and the pre-registered paired
    difference factor minus feature for each model, with a bootstrap interval."""
    rng = np.random.default_rng(seed)
    n = len(next(iter(agreement.values())))
    idx = [rng.integers(0, n, n) for _ in range(n_boot)]
    rows = [{"model": m, "level": lvl, "mean_top1_agreement": round(float(a.mean()), 4)}
            for (m, lvl), a in agreement.items()]

    def diff(a, b):
        d = agreement[a] - agreement[b]
        boots = np.array([d[i].mean() for i in idx])
        lo, hi = np.percentile(boots, [2.5, 97.5])
        return {"comparison": f"{a[0]}:{a[1]} minus {b[0]}:{b[1]}", "difference": round(float(d.mean()), 4),
                "ci_low": round(float(lo), 4), "ci_high": round(float(hi), 4)}

    random_mean = np.mean([agreement[k] for k in agreement if k[1].startswith("random_")], axis=0)
    agreement[("lgbm_unconstrained", "random_mean")] = random_mean
    diffs = [
        diff(("lgbm_structured", "factor"), ("lgbm_structured", "feature")),        # H4 (primary for RQ3)
        diff(("lgbm_unconstrained", "factor"), ("lgbm_unconstrained", "feature")),
        diff(("lgbm_structured", "factor"), ("lgbm_unconstrained", "factor")),
        diff(("lgbm_unconstrained", "factor"), ("lgbm_unconstrained", "random_mean")),  # control
    ]
    rows.append({"model": "lgbm_unconstrained", "level": "random_mean",
                 "mean_top1_agreement": round(float(random_mean.mean()), 4)})
    return pd.DataFrame(rows), pd.DataFrame(diffs)
