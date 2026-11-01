"""Post-hoc robustness checks from the decision audit (1 November 2026).

The pre-registered results in results/ladder/test/ stand as they are. This
script reruns the key comparisons under the alternatives the audit found
better supported, and writes results/robustness/. Everything here was decided
after the test block had been scored, so it is reported as robustness, never
as a replacement.

Usage, from the repository root:
    .venv/Scripts/python scripts/run_robustness.py
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import brier_score_loss, roc_auc_score  # noqa: E402

from dsp import data, evaluate, models, reasoning, robust, splits, stability  # noqa: E402

OUT = Path("results/robustness")
LADDER = ["lgbm_unconstrained", "lgbm_directions", "lgbm_structured"]
PAIRS = [("lgbm_directions", "lgbm_unconstrained"), ("lgbm_structured", "lgbm_directions"),
         ("lgbm_structured", "lgbm_unconstrained")]


def blocks_for(X, y, meta):
    masks = splits.block_masks(meta, splits.load_blocks())
    take = lambda b: (X[masks[b]].reset_index(drop=True), y[masks[b]],  # noqa: E731
                      meta[masks[b]].reset_index(drop=True))
    return {b: take(b) for b in masks}


def fit_ladder(rm, B, params=None):
    """Fit the three LightGBM rungs and the rule baseline on train; return the
    fitted models and raw scores on the calibration, pilot and test blocks."""
    Xtr, ytr, _ = B["train"]
    cols = list(Xtr.columns)
    specs = {"lgbm_unconstrained": {}, "lgbm_directions": {"monotone": rm.monotone_vector(cols)},
             "lgbm_structured": {"interaction_groups": rm.interaction_groups(cols),
                                 "monotone": rm.monotone_vector(cols)}}
    fitted, scores = {}, {b: {} for b in ("calibration", "pilot", "test")}
    for name, kw in specs.items():
        p = (params or {}).get(name)
        old = dict(models.PARAMS)
        if p:
            models.PARAMS.update(p)
        try:
            fitted[name] = models.fit(Xtr, ytr, **kw)
        finally:
            models.PARAMS.clear()
            models.PARAMS.update(old)
        for b in scores:
            scores[b][name] = fitted[name].predict(B[b][0], raw_score=True)
    if {"prev_success", "contacted_before", "days_since_contact"} <= set(cols):
        rule = models.RuleBaseline()
        for b in scores:
            scores[b]["rule_baseline"] = rule.decision_function(B[b][0])
    return fitted, scores


def compare(y, scores, pairs=PAIRS, n_boot=2000):
    rows = []
    for a, b in pairs:
        d = robust.delong(y, scores[a], scores[b])
        boot = evaluate.paired_bootstrap(y, scores, [(a, b)], n=n_boot)
        bauc = boot[boot.metric == "auc"].iloc[0]
        rows.append({"model": a, "reference": b, "auc_diff": d["difference"],
                     "boot_ci_low": bauc.ci_low, "boot_ci_high": bauc.ci_high,
                     "delong_ci_low": d["ci_low"], "delong_ci_high": d["ci_high"],
                     "delong_p": d["p_value"]})
    return rows


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    np.seterr(all="ignore")
    t0 = time.time()
    rm_full = reasoning.load()
    df = data.rebuild_dates(data.load_raw())

    # ---- A. Data variants: rerun the ladder under each alternative ----
    variants = {
        "pre_registered": {},
        "drop_12_duplicates": {"drop_duplicates": True},
        "contact_flag_from_previous": {"contact_flag": "previous"},
        "without_contact_type": {"drop_features": ("contact",)},
        "market_lagged_one_month": {"lagged_macro": True},
        "without_market_context": {"drop_features": tuple(data.MACRO)},
        "all_data_fixes": {"drop_duplicates": True, "contact_flag": "previous",
                           "drop_features": ("contact",), "lagged_macro": True},
    }
    var_rows, var_metrics = [], []
    pre = None
    for vname, opts in variants.items():
        X, y, meta = data.prepare(df, **opts)
        rm = rm_full.restrict(X.columns)
        rm.validate(X.columns)
        B = blocks_for(X, y, meta)
        fitted, scores = fit_ladder(rm, B)
        yte = B["test"][1]
        if vname == "pre_registered":
            pre = (rm, B, fitted, scores)
        for r in compare(yte, scores["test"]):
            var_rows.append({"variant": vname, **r})
        for name, s in scores["test"].items():
            var_metrics.append({"variant": vname, "model": name,
                                "auc": round(roc_auc_score(yte, s), 4),
                                "precision_top10": round(models.precision_at_top(yte, s), 4)})
        print(f"variant {vname} done ({time.time() - t0:.0f}s)")

    # Leaky upper bound: add `campaign` back (known only after calling stops).
    X, y, meta = data.prepare(df)
    X = X.assign(campaign=df["campaign"].to_numpy())
    B = blocks_for(X, y, meta)
    leaky = models.fit(B["train"][0], B["train"][1])
    leaky_auc = round(roc_auc_score(B["test"][1], leaky.predict(B["test"][0], raw_score=True)), 4)
    var_metrics.append({"variant": "leaky_with_campaign", "model": "lgbm_unconstrained",
                        "auc": leaky_auc, "precision_top10": None})
    pd.DataFrame(var_rows).to_csv(OUT / "data_variants_differences.csv", index=False)
    pd.DataFrame(var_metrics).to_csv(OUT / "data_variants_metrics.csv", index=False)

    rm, B, fitted, scores = pre
    yte, ycal, ypi = B["test"][1], B["calibration"][1], B["pilot"][1]

    # ---- B. Statistics on the pre-registered models ----
    stat_rows = []
    for a, b in PAIRS:
        d = robust.delong(yte, scores["test"][a], scores["test"][b])
        bca = robust.bca_difference(yte, scores["test"][a], scores["test"][b])
        stat_rows.append({"model": a, "reference": b, "auc_diff": d["difference"],
                          "delong_se": d["se"], "delong_ci": f"[{d['ci_low']}, {d['ci_high']}]",
                          "delong_p": d["p_value"], "bca_10000_ci": f"[{bca['ci_low']}, {bca['ci_high']}]"})
    stats_df = pd.DataFrame(stat_rows)
    stats_df.to_csv(OUT / "statistics.csv", index=False)

    # Margin sensitivity for the primary comparison (H1b), bootstrap interval.
    h1b = pd.read_csv("results/ladder/test/bootstrap.csv")
    h1b = h1b[(h1b.model == "lgbm_structured") & (h1b.reference == "lgbm_directions") & (h1b.metric == "auc")].iloc[0]
    margins = [{"margin": m, "lower_bound": h1b.ci_low,
                "verdict": "non-inferior" if h1b.ci_low > -m else ("inferior" if h1b.ci_high < -m else "inconclusive")}
               for m in (0.005, 0.01, 0.02, 0.03)]
    pd.DataFrame(margins).to_csv(OUT / "margin_sensitivity.csv", index=False)

    # Determinism: the LightGBM settings use no bagging or feature sampling, so
    # the seed should not matter. Check it.
    Xtr, ytr = B["train"][0], B["train"][1]
    old = dict(models.PARAMS)
    models.PARAMS["random_state"] = 12345
    alt = models.fit(Xtr, ytr, rm.interaction_groups(list(Xtr.columns)), rm.monotone_vector(list(Xtr.columns)))
    models.PARAMS.clear()
    models.PARAMS.update(old)
    seed_gap = float(np.abs(alt.predict(B["test"][0], raw_score=True) - scores["test"]["lgbm_structured"]).max())

    # ---- C. Calibration ----
    cal_rows = []
    for name in ["lgbm_structured", "lgbm_unconstrained"]:
        s_cal, s_pi, s_te = scores["calibration"][name], scores["pilot"][name], scores["test"][name]
        maps = {
            "isotonic, calibration block (pre-registered)": evaluate.calibrate(s_cal, ycal).predict,
            "Platt, calibration block": robust.platt(s_cal, ycal),
            "Platt, refit on pilot block": robust.platt(s_pi, ypi),
        }
        for label, f in maps.items():
            p = f(s_te)
            rng = np.random.default_rng(1)
            boots = [evaluate.ece(yte[i], p[i]) for i in (rng.integers(0, len(yte), len(yte)) for _ in range(500))]
            cal_rows.append({"model": name, "calibration": label,
                             "ece": round(evaluate.ece(yte, p), 4),
                             "ece_ci": f"[{np.percentile(boots, 2.5):.4f}, {np.percentile(boots, 97.5):.4f}]",
                             "brier": round(brier_score_loss(yte, p), 4),
                             "calibration_in_the_large": robust.calibration_in_the_large(yte, p),
                             "calibration_slope": robust.calibration_slope(yte, p)})
    pd.DataFrame(cal_rows).to_csv(OUT / "calibration.csv", index=False)
    print(f"statistics and calibration done ({time.time() - t0:.0f}s)")

    # ---- D. Equal-budget tuning per rung, chosen on the pilot block ----
    rng = np.random.default_rng(7)
    space = lambda: {"num_leaves": int(rng.choice([7, 15, 31, 63])),  # noqa: E731
                     "learning_rate": float(np.exp(rng.uniform(np.log(0.02), np.log(0.2)))),
                     "n_estimators": int(rng.choice([100, 200, 300, 500, 800])),
                     "min_child_samples": int(rng.choice([10, 20, 50, 100, 200])),
                     "reg_lambda": float(rng.choice([0.0, 1.0, 5.0, 10.0]))}
    trials = [space() for _ in range(30)]  # the same 30 settings for every rung
    best, tune_rows = {}, []
    cols = list(Xtr.columns)
    kws = {"lgbm_unconstrained": {}, "lgbm_directions": {"monotone": rm.monotone_vector(cols)},
           "lgbm_structured": {"interaction_groups": rm.interaction_groups(cols), "monotone": rm.monotone_vector(cols)}}
    for name, kw in kws.items():
        scores_pi = []
        for t in trials:
            models.PARAMS.update(t)
            m = models.fit(Xtr, ytr, **kw)
            models.PARAMS.clear()
            models.PARAMS.update(old)
            scores_pi.append(roc_auc_score(ypi, m.predict(B["pilot"][0], raw_score=True)))
        k = int(np.argmax(scores_pi))
        best[name] = trials[k]
        tune_rows.append({"model": name, "best_pilot_auc": round(scores_pi[k], 4), **trials[k]})
    _, tuned = fit_ladder(rm, B, params=best)
    tuned_metrics = [{"model": n, "test_auc": round(roc_auc_score(yte, tuned["test"][n]), 4),
                      "test_precision_top10": round(models.precision_at_top(yte, tuned["test"][n]), 4)}
                     for n in LADDER]
    pd.DataFrame(tune_rows).to_csv(OUT / "tuning_choices.csv", index=False)
    pd.DataFrame(tuned_metrics).to_csv(OUT / "tuned_metrics.csv", index=False)
    pd.DataFrame(compare(yte, tuned["test"])).to_csv(OUT / "tuned_differences.csv", index=False)
    print(f"tuning done ({time.time() - t0:.0f}s)")

    # ---- E. Month by month (pilot and test months), pre-registered models ----
    month_rows = []
    for b in ("pilot", "test"):
        Xb, yb, mb = B[b]
        for ym in sorted(mb["ym"].unique()):
            m = (mb["ym"] == ym).to_numpy()
            if yb[m].min() == yb[m].max():
                continue
            row = {"block": b, "month": ym, "clients": int(m.sum()), "yes_rate": round(float(yb[m].mean()), 3)}
            for name in LADDER + ["rule_baseline"]:
                row[name] = round(roc_auc_score(yb[m], scores[b][name][m]), 4)
            month_rows.append(row)
    pd.DataFrame(month_rows).to_csv(OUT / "monthly_auc.csv", index=False)

    # ---- F. Stability, extended ----
    agreement, gap, corr = stability.run_extended(rm, Xtr, ytr, B["test"][0])
    rnd = np.mean([agreement[k] for k in agreement if k[1].startswith("random_")], axis=0)
    rnd_each = [float(agreement[k].mean()) for k in agreement if k[1].startswith("random_")]
    stab = [{"model": m, "level": l, "mean_agreement": round(float(a.mean()), 4)}
            for (m, l), a in agreement.items() if not l.startswith("random_")]
    stab.append({"model": "lgbm_unconstrained", "level": "random (mean of 50)", "mean_agreement": round(float(rnd.mean()), 4)})
    stab.append({"model": "lgbm_unconstrained", "level": "random (range of 50)",
                 "mean_agreement": f"{min(rnd_each):.4f} to {max(rnd_each):.4f}"})
    terc = np.quantile(gap, [1 / 3, 2 / 3])
    band = np.digitize(gap, terc)
    for k, lab in enumerate(["smallest gap", "middle", "largest gap"]):
        stab.append({"model": "lgbm_structured", "level": f"factor, {lab} between top two factors",
                     "mean_agreement": round(float(agreement[("lgbm_structured", "factor")][band == k].mean()), 4)})
    pd.DataFrame(stab).to_csv(OUT / "stability_extended.csv", index=False)
    (OUT / "correlation_groups.json").write_text(json.dumps(corr, indent=2))

    # ---- G. Groups: add recall in the top 10% and base rates ----
    grp = []
    age_band = pd.cut(B["test"][0]["age"], [0, 29, 59, 200], labels=["under 30", "30 to 59", "60 and over"])
    for name in ["lgbm_structured", "lgbm_unconstrained"]:
        s = scores["test"][name]
        top = s >= np.sort(s)[::-1][max(1, round(0.1 * len(s))) - 1]
        for g in age_band.cat.categories:
            m = (age_band == g).to_numpy()
            pos = m & (yte == 1)
            grp.append({"model": name, "age_band": g, "clients": int(m.sum()),
                        "base_rate": round(float(yte[m].mean()), 4),
                        "selection_rate_top10": round(float(top[m].mean()), 4),
                        "precision_in_top10": round(float(yte[m & top].mean()), 4) if (m & top).any() else None,
                        "recall_in_top10": round(float(top[pos].mean()), 4) if pos.any() else None})
    pd.DataFrame(grp).to_csv(OUT / "groups_extended.csv", index=False)

    summary = {"seed_check_max_score_difference": seed_gap, "leaky_upper_bound_auc": leaky_auc,
               "correlation_groups": corr, "seconds": round(time.time() - t0)}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    pd.set_option("display.width", 200)
    for f in sorted(OUT.glob("*.csv")):
        print(f"\n== {f.name}")
        print(pd.read_csv(f).to_string(index=False))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
