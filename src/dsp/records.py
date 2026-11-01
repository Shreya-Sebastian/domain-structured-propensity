"""Exact factor contributions and decision records.

If every tree splits on features from one factor only, the model's raw score
is a sum of one function per factor. The per-feature contributions LightGBM
returns (TreeSHAP) then add up, within each factor, to that factor's exact
share of the score relative to the average training client.
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
