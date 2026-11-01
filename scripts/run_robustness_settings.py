"""Post-hoc: structured minus directions-only AUC on the test months for each
of the 30 shared tuning settings (same draw as run_robustness.py), and the
age-band check for the structured model without life stage."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

from dsp import data, models, reasoning, splits  # noqa: E402

OUT = Path("results/robustness")
rm = reasoning.load()
X, y, meta = data.prepare(data.rebuild_dates(data.load_raw()))
masks = splits.block_masks(meta, splits.load_blocks())
Xtr, ytr = X[masks["train"]].reset_index(drop=True), y[masks["train"]]
Xpi, ypi = X[masks["pilot"]].reset_index(drop=True), y[masks["pilot"]]
Xte, yte = X[masks["test"]].reset_index(drop=True), y[masks["test"]]
cols = list(X.columns)
rng = np.random.default_rng(7)
trials = [{"num_leaves": int(rng.choice([7, 15, 31, 63])),
           "learning_rate": float(np.exp(rng.uniform(np.log(0.02), np.log(0.2)))),
           "n_estimators": int(rng.choice([100, 200, 300, 500, 800])),
           "min_child_samples": int(rng.choice([10, 20, 50, 100, 200])),
           "reg_lambda": float(rng.choice([0.0, 1.0, 5.0, 10.0]))} for _ in range(30)]
old = dict(models.PARAMS)
rows = []
for i, t in enumerate(trials):
    models.PARAMS.update(t)
    res = {}
    for name, kw in {"unconstrained": {}, "directions": {"monotone": rm.monotone_vector(cols)},
                     "structured": {"interaction_groups": rm.interaction_groups(cols),
                                    "monotone": rm.monotone_vector(cols)}}.items():
        m = models.fit(Xtr, ytr, **kw)
        res[f"{name}_test"] = roc_auc_score(yte, m.predict(Xte, raw_score=True))
        res[f"{name}_pilot"] = roc_auc_score(ypi, m.predict(Xpi, raw_score=True))
    models.PARAMS.clear()
    models.PARAMS.update(old)
    rows.append({"setting": i, **t, **{k: round(v, 4) for k, v in res.items()}})
df = pd.DataFrame(rows)
df["structured_minus_directions_test"] = (df.structured_test - df.directions_test).round(4)
df["structured_minus_directions_pilot"] = (df.structured_pilot - df.directions_pilot).round(4)
df.to_csv(OUT / "settings_sweep.csv", index=False)
d = df.structured_minus_directions_test
print(df[["setting", "num_leaves", "learning_rate", "n_estimators", "min_child_samples", "reg_lambda",
          "directions_test", "structured_test", "structured_minus_directions_test",
          "structured_minus_directions_pilot"]].to_string(index=False))
print("test diff: median", d.median(), "min", d.min(), "max", d.max(), "share > 0:", (d > 0).mean(),
      "share < -0.02:", (d < -0.02).mean())
print("corr of pilot diff with test diff:", round(np.corrcoef(df.structured_minus_directions_pilot, d)[0, 1], 3))

# Age-band selection for the structured model without life stage.
life = rm.factors["life_stage"]["features"]
keep = [c for c in cols if c not in life]
r2 = rm.restrict(keep)
m = models.fit(Xtr[keep], ytr, r2.interaction_groups(keep), r2.monotone_vector(keep))
s = m.predict(Xte[keep], raw_score=True)
top = s >= np.sort(s)[::-1][max(1, round(0.1 * len(s))) - 1]
band = pd.cut(Xte["age"], [0, 29, 59, 200], labels=["under 30", "30 to 59", "60 and over"])
g = []
for b in band.cat.categories:
    mm = (band == b).to_numpy()
    g.append({"model": "lgbm_structured_no_life_stage", "age_band": b, "clients": int(mm.sum()),
              "selection_rate_top10": round(float(top[mm].mean()), 4),
              "precision_in_top10": round(float(yte[mm & top].mean()), 4) if (mm & top).any() else None,
              "recall_in_top10": round(float(top[mm & (yte == 1)].mean()), 4)})
pd.DataFrame(g).to_csv(OUT / "groups_no_life_stage.csv", index=False)
print(pd.DataFrame(g).to_string(index=False))
