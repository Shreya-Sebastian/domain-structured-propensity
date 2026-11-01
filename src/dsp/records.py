"""Factor contributions and decision records.

LightGBM's per-feature contributions (TreeSHAP) always add up to the raw
score, for any tree model, so summing them per factor always adds up too:
that is only a sanity check.

What factor grouping adds is separation. If every tree splits on features
from one factor only, the raw score is a sum of one function per factor,
score = f_1(x_1) + ... + f_5(x_5), and each factor's share is that factor's
own function (relative to the average training client). Changing one
factor's facts can then never move another factor's share. In an
unconstrained model, interactions spread across factors, so changing age can
move the relationship share. `separation_test` checks this.
"""
import math

import numpy as np
import pandas as pd

TOP_SHARE_PCT = 10


def factor_contributions(model, X, reasoning):
    """Return (factor table, feature table, base, raw score)."""
    contrib = model.predict(X, pred_contrib=True)
    features = pd.DataFrame(contrib[:, :-1], columns=X.columns)
    base = contrib[:, -1]
    raw = model.predict(X, raw_score=True)
    factors = pd.DataFrame(
        {fid: features[spec["features"]].sum(axis=1) for fid, spec in reasoning.factors.items()}
    )
    return factors, features, base, raw


def separation_test(model, X, reasoning, seed=0, tol=1e-9) -> dict:
    """For each factor, shuffle that factor's features across clients and
    count clients whose OTHER factors' shares move. A separable model moves
    none. Only clients whose own factor values actually changed are counted."""
    rng = np.random.default_rng(seed)
    before, *_ = factor_contributions(model, X, reasoning)
    out = {}
    for fid, spec in reasoning.factors.items():
        cols = spec["features"]
        Xp = X.copy()
        Xp[cols] = X[cols].iloc[rng.permutation(len(X))].reset_index(drop=True)
        same = (Xp[cols].astype(object) == X[cols].astype(object)) | (Xp[cols].isna() & X[cols].isna())
        changed = ~same.all(axis=1).to_numpy()
        after, *_ = factor_contributions(model, Xp, reasoning)
        others = [k for k in reasoning.factors if k != fid]
        move = np.abs(after[others].to_numpy() - before[others].to_numpy())[changed]
        own = np.abs(after[fid].to_numpy() - before[fid].to_numpy())[changed]
        out[fid] = {
            "clients_changed": int(changed.sum()),
            "share_with_other_factors_moved": round(float((move > tol).any(axis=1).mean()), 4),
            "max_other_factor_move": float(move.max()) if len(move) else 0.0,
            # If the factor's own share never moves, the shuffle tested nothing.
            "max_own_factor_move": float(own.max()) if len(own) else 0.0,
        }
    return out


def _plain(value):
    """JSON-safe value; missing becomes None."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return round(float(value), 3)
    return value


def build_records(X, meta, block, factors, features, base, raw, probability, reasoning, model_info):
    """One decision record per client, as plain dicts."""
    n = len(X)
    order = np.argsort(-raw, kind="stable")
    rank = np.empty(n, dtype=int)
    rank[order] = np.arange(1, n + 1)
    records = []
    for i in range(n):
        rank_pct = int(math.ceil(100 * rank[i] / n))
        factor_list = []
        for fid, spec in reasoning.factors.items():
            # Show the factor's two features that moved the score most for this client.
            shown = features.loc[i, spec["features"]].abs().sort_values(ascending=False).index[:2]
            factor_list.append({
                "id": fid,
                "contribution": round(float(factors.loc[i, fid]), 3),
                "features": {f: _plain(X.loc[i, f]) for f in shown},
                "talking_point": bool(spec.get("talking_point", False)),
                "context_only": bool(spec.get("context_only", False)),
            })
        factor_list.sort(key=lambda d: d["contribution"], reverse=True)
        total = float(base[i] + factors.loc[i].sum())
        records.append({
            "client_ref": f"{block}-{meta.loc[i, 'ym']}-{int(meta.loc[i, 'row_id']):05d}",
            "model": model_info,
            "scored_at_block": block,
            "score": {
                "probability_pct": int(round(100 * probability[i])),
                "interval": None,
                "rank_top_pct": rank_pct,
                "threshold_top_pct": TOP_SHARE_PCT,
                "in_top": bool(rank_pct <= TOP_SHARE_PCT),
            },
            "abstain": False,
            "base_logit": round(float(base[i]), 3),
            "factors": factor_list,
            "check": {
                "sum_of_factors_plus_base": round(total, 6),
                "model_logit": round(float(raw[i]), 6),
                "exact": bool(abs(total - raw[i]) < 1e-6),
            },
            "counterfactual": None,
            "evidence": None,
            "not_in_demo": ["interval", "abstain rule", "counterfactual", "evidence", "directions"],
        })
    return records
