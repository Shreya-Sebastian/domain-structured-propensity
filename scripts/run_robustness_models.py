"""Post-hoc robustness checks on modelling choices (audit of 1 November 2026).

Writes results/robustness/models_*.csv. Like run_robustness.py, everything
here was decided after the test block was scored.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

from dsp import data, evaluate, models, reasoning, robust, splits  # noqa: E402

OUT = Path("results/robustness")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rm = reasoning.load()
    df = data.rebuild_dates(data.load_raw())
    X, y, meta = data.prepare(df)
    masks = splits.block_masks(meta, splits.load_blocks())
    Xtr, ytr = X[masks["train"]].reset_index(drop=True), y[masks["train"]]
    Xte, yte = X[masks["test"]].reset_index(drop=True), y[masks["test"]]
    cols = list(X.columns)
    kws = {"lgbm_unconstrained": {}, "lgbm_directions": {"monotone": rm.monotone_vector(cols)},
           "lgbm_structured": {"interaction_groups": rm.interaction_groups(cols),
                               "monotone": rm.monotone_vector(cols)}}
    old = dict(models.PARAMS)

    def fit_with(params, method=None):
        models.PARAMS.clear()
        models.PARAMS.update(params)
        saved = models.MONOTONE_METHOD
        if method:
            models.MONOTONE_METHOD = method
        try:
            return {n: models.fit(Xtr, ytr, **kw).predict(Xte, raw_score=True) for n, kw in kws.items()}
        finally:
            models.MONOTONE_METHOD = saved
            models.PARAMS.clear()
            models.PARAMS.update(old)

    rows = []
    # M2: LightGBM's library defaults instead of the fixed settings.
    defaults = {k: v for k, v in old.items() if k not in ("n_estimators", "learning_rate", "num_leaves",
                                                          "min_child_samples")}
    variants = {"pre-registered settings": fit_with(old),
                "LightGBM library defaults": fit_with(defaults)}
    # M3: the three methods for enforcing directions.
    for method in ("basic", "intermediate"):
        variants[f"direction method: {method}"] = fit_with(old, method)
    for label, sc in variants.items():
        for a, b in [("lgbm_directions", "lgbm_unconstrained"), ("lgbm_structured", "lgbm_directions"),
                     ("lgbm_structured", "lgbm_unconstrained")]:
            d = robust.delong(yte, sc[a], sc[b])
            rows.append({"variant": label, "model": a, "reference": b,
                         "auc_model": round(roc_auc_score(yte, sc[a]), 4),
                         "auc_reference": round(roc_auc_score(yte, sc[b]), 4),
                         "auc_diff": d["difference"], "delong_ci": f"[{d['ci_low']}, {d['ci_high']}]"})
    pd.DataFrame(rows).to_csv(OUT / "models_settings_and_methods.csv", index=False)

    # M9: share of test clients whose market values lie outside the training range.
    rng_rows = []
    for c in data.MACRO:
        lo, hi = Xtr[c].min(), Xtr[c].max()
        out = (Xte[c] < lo) | (Xte[c] > hi)
        rng_rows.append({"indicator": c, "train_min": lo, "train_max": hi,
                         "test_min": Xte[c].min(), "test_max": Xte[c].max(),
                         "test_share_outside": round(float(out.mean()), 4)})
    pd.DataFrame(rng_rows).to_csv(OUT / "models_market_range.csv", index=False)

    # M5: pairs in the factor-grouped EBM that contain a feature with a direction
    # (monotonize does not change pair terms).
    directed = set(rm.directions)
    pairs = [p for p in rm.factor_pairs(cols) if directed & set(p)]
    # M6: correlation behind logistic regression's sign on prev_success.
    corr = float(np.corrcoef(Xtr["prev_success"], Xtr["contacted_before"])[0, 1])
    (OUT / "models_notes.json").write_text(json.dumps({
        "ebm_pairs_with_directed_feature": pairs,
        "corr_prev_success_contacted_before_train": round(corr, 4),
    }, indent=2))

    pd.set_option("display.width", 200)
    for f in ["models_settings_and_methods.csv", "models_market_range.csv"]:
        print(f"\n== {f}")
        print(pd.read_csv(OUT / f).to_string(index=False))
    print((OUT / "models_notes.json").read_text())


if __name__ == "__main__":
    np.seterr(all="ignore")
    main()
