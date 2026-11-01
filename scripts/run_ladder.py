"""Fit the model ladder on the training months, calibrate on the calibration
months and evaluate on one block. Writes results/ladder/<block>/.

Usage, from the repository root:
    .venv/Scripts/python scripts/run_ladder.py --block pilot
    .venv/Scripts/python scripts/run_ladder.py --block test   # refused until hypotheses.md is committed
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from dsp import data, evaluate, models, reasoning, records, splits, stability  # noqa: E402

# Comparisons fixed before any result was seen: (model, reference).
PAIRS = [
    ("lgbm_directions", "lgbm_unconstrained"),   # RQ1a: cost of directions
    ("lgbm_structured", "lgbm_directions"),      # RQ1b: cost of factor grouping on top
    ("lgbm_structured", "lgbm_unconstrained"),   # total cost of structure
    ("ebm_structured", "ebm_unconstrained"),     # same question for EBMs
    ("lgbm_structured", "logistic"),             # structured model against a simple reference
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--block", choices=["pilot", "test"], default="pilot")
    args = ap.parse_args()
    if args.block == "test":
        evaluate.assert_test_allowed()

    rm = reasoning.load()
    df = data.rebuild_dates(data.load_raw())
    X, y, meta = data.prepare(df)
    rm.validate(X.columns)
    masks = splits.block_masks(meta, splits.load_blocks())
    Xtr, ytr = X[masks["train"]].reset_index(drop=True), y[masks["train"]]
    Xcal, ycal = X[masks["calibration"]].reset_index(drop=True), y[masks["calibration"]]
    Xev, yev = X[masks[args.block]].reset_index(drop=True), y[masks[args.block]]

    cols = list(X.columns)
    fitted, scores, rows, timing = {}, {}, [], {}
    for name, model in models.ladder(rm, cols).items():
        t0 = time.time()
        model.fit(Xtr, ytr)
        timing[name] = round(time.time() - t0, 1)
        fitted[name] = model
        iso = evaluate.calibrate(model.decision_function(Xcal), ycal)
        s = model.decision_function(Xev)
        scores[name] = s
        rows.append({"model": name, **evaluate.metrics(yev, s, iso.predict(s))})
        print(f"{name}: fitted in {timing[name]}s")

    table = pd.DataFrame(rows)
    boot = evaluate.paired_bootstrap(yev, scores, PAIRS)

    directional = []
    for feature, direction in rm.directions.items():
        for name, model in fitted.items():
            if name == "rule_baseline":
                continue
            r = evaluate.directional_test(model, Xev, Xtr, feature, direction)
            directional.append({"model": name, **r})
    if "prev_success" in rm.directions:  # H2c: realistic variant, descriptive
        for name, model in fitted.items():
            if name != "rule_baseline":
                directional.append({"model": name,
                                    **evaluate.directional_test_realistic(model, Xev, Xtr)})
    directional = pd.DataFrame(directional)

    mixing = {name: models.trees_mixing_factors(m.model, cols, rm.feature_to_factor)
              for name, m in fitted.items() if name.startswith("lgbm")}

    # H3: separation (the benefit of factor grouping) for the LightGBM models,
    # plus the sum of factor shares as a sanity check.
    separation = []
    for name in ["lgbm_unconstrained", "lgbm_directions", "lgbm_structured"]:
        for factor, r in records.separation_test(fitted[name].model, Xev, rm).items():
            separation.append({"model": name, "shuffled_factor": factor, **r,
                               "informative": r["max_own_factor_move"] > 0})
    separation = pd.DataFrame(separation)
    factors, _, base, raw = records.factor_contributions(fitted["lgbm_structured"].model, Xev, rm)
    exact_gap = float(np.abs(base + factors.sum(axis=1).to_numpy() - raw).max())

    # Cost of dropping the life-stage factor (structured model without it).
    life = rm.factors["life_stage"]["features"]
    keep = [c for c in cols if c not in life]
    rm_wo = reasoning.ReasoningModel({k: v for k, v in rm.factors.items() if k != "life_stage"},
                                     rm.feature_labels, rm.sources, rm.flag_features)
    wo = models.LGBM(rm_wo.interaction_groups(keep), rm_wo.monotone_vector(keep)).fit(Xtr[keep], ytr)
    scores["lgbm_structured_no_life_stage"] = wo.decision_function(Xev[keep])
    iso = evaluate.calibrate(wo.decision_function(Xcal[keep]), ycal)
    table = pd.concat([table, pd.DataFrame([{
        "model": "lgbm_structured_no_life_stage",
        **evaluate.metrics(yev, scores["lgbm_structured_no_life_stage"],
                           iso.predict(scores["lgbm_structured_no_life_stage"]))}])])
    boot = pd.concat([boot, evaluate.paired_bootstrap(
        yev, scores, [("lgbm_structured_no_life_stage", "lgbm_structured")])])

    # Outcomes by group in the top 10%.
    groups = []
    age_band = pd.cut(Xev["age"], [0, 29, 59, 200], labels=["under 30", "30 to 59", "60 and over"])
    for name in ["lgbm_structured", "lgbm_unconstrained"]:
        s = scores[name]
        top = s >= np.sort(s)[::-1][max(1, round(0.1 * len(s))) - 1]
        for col, values in [("age_band", age_band), ("job", Xev["job"])]:
            for g in pd.unique(values.dropna()):
                m = (values == g).to_numpy()
                groups.append({"model": name, "group_by": col, "group": str(g), "clients": int(m.sum()),
                               "selection_rate_top10": round(float(top[m].mean()), 4),
                               "precision_in_top10": (round(float(yev[m & top].mean()), 4)
                                                      if (m & top).any() else None)})
    groups = pd.DataFrame(groups)

    # H4 (RQ3): stability of the main reason across bootstrap refits.
    t0 = time.time()
    agreement = stability.run(rm, Xtr, ytr, Xev)
    stab_means, stab_diffs = stability.summarise(agreement)
    timing["stability"] = round(time.time() - t0, 1)

    out = Path("results/ladder") / args.block
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / "metrics.csv", index=False)
    boot.to_csv(out / "bootstrap.csv", index=False)
    directional.to_csv(out / "directional.csv", index=False)
    groups.to_csv(out / "groups.csv", index=False)
    separation.to_csv(out / "separation.csv", index=False)
    stab_means.to_csv(out / "stability_agreement.csv", index=False)
    stab_diffs.to_csv(out / "stability_differences.csv", index=False)
    summary = {
        "block": args.block,
        "rows": int(len(yev)),
        "yes_rate": round(float(yev.mean()), 4),
        "directions": rm.directions,
        "trees_mixing_factors": mixing,
        "sanity_max_abs_gap_factor_sum_vs_score": exact_gap,
        "fit_seconds": timing,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))

    pd.set_option("display.width", 160)
    print(json.dumps(summary, indent=2))
    print(table.to_string(index=False))
    print(boot.to_string(index=False))
    if len(directional):
        print(directional.to_string(index=False))
    print(separation.to_string(index=False))
    print(stab_means.to_string(index=False))
    print(stab_diffs.to_string(index=False))


if __name__ == "__main__":
    np.seterr(all="ignore")
    main()
