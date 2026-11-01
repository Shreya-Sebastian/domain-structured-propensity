"""Demo pipeline: factor-separable model plus advisor briefs.

Uses the train, calibration and pilot blocks only. The test block is never
scored here.
"""
import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import average_precision_score, roc_auc_score

from dsp import data, models, records, reasoning, splits


def run():
    rm = reasoning.load()
    df = data.rebuild_dates(data.load_raw())
    X, y, meta = data.prepare(df)
    rm.validate(X.columns)
    blocks = splits.load_blocks()
    masks = splits.block_masks(meta, blocks)

    def take(name):
        m = masks[name]
        return (X[m].reset_index(drop=True), y[m], meta[m].reset_index(drop=True))

    Xtr, ytr, _ = take("train")
    Xcal, ycal, _ = take("calibration")
    Xpi, ypi, mpi = take("pilot")

    fitted = {
        "unconstrained": models.fit(Xtr, ytr),
        "structured": models.fit(Xtr, ytr, rm.interaction_groups(X.columns),
                                 rm.monotone_vector(X.columns)),
    }

    cols = list(X.columns)
    pilot = {}
    for name, model in fitted.items():
        raw = model.predict(Xpi, raw_score=True)
        pilot[name] = {
            "auc": round(float(roc_auc_score(ypi, raw)), 4),
            "average_precision": round(float(average_precision_score(ypi, raw)), 4),
            "precision_top10": round(models.precision_at_top(ypi, raw), 4),
            "trees": len(models.tree_feature_sets(model)),
            "trees_mixing_factors": models.trees_mixing_factors(model, cols, rm.feature_to_factor),
        }

    model = fitted["structured"]
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1)
    iso.fit(model.predict(Xcal, raw_score=True), ycal)

    separation = {name: records.separation_test(m, Xpi, rm) for name, m in fitted.items()}
    separable = all(r["share_with_other_factors_moved"] == 0
                    for r in separation["structured"].values())

    factors, features, base, raw = records.factor_contributions(model, Xpi, rm)
    probability = iso.predict(raw)
    info = {
        "name": "lgbm_factor_grouped",
        "version": "0.1.0-demo",
        "trained_on": f"{blocks['train'][0]} to {blocks['train'][1]}",
        "calibrated_on": f"{blocks['calibration'][0]} to {blocks['calibration'][1]}",
        "separable_by_factor": separable,
    }
    recs = records.build_records(Xpi, mpi, "pilot", factors, features, base, raw,
                                 probability, rm, info)

    gap = np.abs(base + factors.sum(axis=1).to_numpy() - raw)
    checks = {
        "banned_fields_absent": not (set(X.columns) & set(data.BANNED)),
        "structured_trees_mixing_factors": pilot["structured"]["trees_mixing_factors"],
        "unconstrained_trees_mixing_factors": pilot["unconstrained"]["trees_mixing_factors"],
        # Separation: the property factor grouping adds.
        "separation_structured": separation["structured"],
        "separation_unconstrained": separation["unconstrained"],
        # Sanity only: TreeSHAP shares add up to the score for any tree model.
        "sanity_max_abs_gap_factor_sum_vs_score": float(gap.max()),
        "sanity_all_records_add_up": all(r["check"]["exact"] for r in recs),
    }
    counts = {name: int(m.sum()) for name, m in masks.items()}
    base_rates = {name: round(float(y[m].mean()), 4) for name, m in masks.items() if name != "test"}

    return {
        "reasoning": rm, "models": fitted, "pilot_metrics": pilot, "records": recs,
        "checks": checks, "block_counts": counts, "base_rates": base_rates,
        "X_columns": cols,
    }


def example_records(recs):
    """Three clients chosen by rule, not by hand: the top-ranked client, the one
    at the top-10% boundary and the median client."""
    by_rank = sorted(recs, key=lambda r: (r["check"]["model_logit"]), reverse=True)
    n = len(by_rank)
    return [by_rank[0], by_rank[int(0.10 * n) - 1], by_rank[n // 2]]
