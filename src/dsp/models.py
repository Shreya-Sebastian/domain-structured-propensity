"""Model ladder.

The three LightGBM models share every setting except their constraints, so
the gap between neighbours on the ladder is the cost of one constraint:

    lgbm_unconstrained -> lgbm_directions (monotone) -> lgbm_structured (+ factor groups)

Logistic regression, a rule baseline and two EBMs are reference points. No
model is tuned: all use fixed settings, written here before any comparison.
"""
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

PARAMS = dict(
    objective="binary",
    n_estimators=300,
    learning_rate=0.05,
    num_leaves=15,
    min_child_samples=50,
    random_state=0,
    deterministic=True,
    force_col_wise=True,
    n_jobs=1,
    verbose=-1,
)
MONOTONE_METHOD = "advanced"  # least restrictive LightGBM method; see README


def fit(X, y, interaction_groups=None, monotone=None) -> lgb.LGBMClassifier:
    params = dict(PARAMS)
    if interaction_groups is not None:
        params["interaction_constraints"] = interaction_groups
    if monotone is not None and any(monotone):
        params["monotone_constraints"] = monotone
        params["monotone_constraints_method"] = MONOTONE_METHOD
    model = lgb.LGBMClassifier(**params)
    model.fit(X, y)
    return model


class RuleBaseline:
    """Previous success first, then most recent contact, then everyone else."""

    def fit(self, X, y):
        return self

    def decision_function(self, X):
        days = X["days_since_contact"].fillna(1000).to_numpy()
        return 100 * X["prev_success"].to_numpy() + np.where(
            X["contacted_before"].to_numpy() == 1, 50 - np.minimum(days, 49), 0)


def _logistic(X):
    cat = [c for c in X.columns if isinstance(X[c].dtype, pd.CategoricalDtype)]
    num = [c for c in X.columns if c not in cat]
    pre = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat),
        ("num", StandardScaler(), num),
    ])
    return make_pipeline(pre, LogisticRegression(max_iter=5000, C=1.0))


def _logistic_frame(X):
    # Never contacted: days unknown; the flag contacted_before carries that.
    X = X.copy()
    X["days_since_contact"] = X["days_since_contact"].fillna(0)
    return X


class Logistic:
    def fit(self, X, y):
        self.model = _logistic(X).fit(_logistic_frame(X), y)
        return self

    def decision_function(self, X):
        return self.model.decision_function(_logistic_frame(X))


class LGBM:
    def __init__(self, interaction_groups=None, monotone=None):
        self.interaction_groups, self.monotone = interaction_groups, monotone

    def fit(self, X, y):
        self.model = fit(X, y, self.interaction_groups, self.monotone)
        return self

    def decision_function(self, X):
        return self.model.predict(X, raw_score=True)


class EBM:
    def __init__(self, interactions=None, directions=None):
        self.interactions, self.directions = interactions, directions or {}

    def fit(self, X, y):
        from interpret.glassbox import ExplainableBoostingClassifier
        kwargs = dict(random_state=0, n_jobs=-1)
        if self.interactions is not None:
            kwargs["interactions"] = self.interactions
        self.model = ExplainableBoostingClassifier(**kwargs).fit(X, y)
        # Directions applied after fitting, as the InterpretML docs recommend.
        for feature, direction in self.directions.items():
            self.model.monotonize(feature, increasing=(direction == "increasing"))
        return self

    def decision_function(self, X):
        return self.model.decision_function(X)


def ladder(rm, columns) -> dict:
    """Every model in the study, unfitted, in reporting order."""
    groups = rm.interaction_groups(columns)
    mono = rm.monotone_vector(columns)
    return {
        "rule_baseline": RuleBaseline(),
        "logistic": Logistic(),
        "lgbm_unconstrained": LGBM(),
        "lgbm_directions": LGBM(monotone=mono),
        "lgbm_structured": LGBM(interaction_groups=groups, monotone=mono),
        "ebm_unconstrained": EBM(),
        "ebm_structured": EBM(interactions=rm.factor_pairs(columns), directions=rm.directions),
    }


def tree_feature_sets(model) -> list:
    """The set of feature indices each tree splits on."""
    def walk(node, acc):
        if "split_feature" in node:
            acc.add(node["split_feature"])
            walk(node["left_child"], acc)
            walk(node["right_child"], acc)
        return acc

    return [walk(t["tree_structure"], set()) for t in model.booster_.dump_model()["tree_info"]]


def trees_mixing_factors(model, columns, feature_to_factor) -> int:
    """How many trees use features from more than one factor."""
    mixed = 0
    for features in tree_feature_sets(model):
        factors = {feature_to_factor[columns[i]] for i in features}
        mixed += len(factors) > 1
    return mixed


def precision_at_top(y, score, share=0.10) -> float:
    """Precision among the top `share` of clients. Ties at the cut-off are
    shared out at random (in expectation), so row order never matters."""
    y, score = np.asarray(y), np.asarray(score)
    k = max(1, int(round(share * len(score))))
    cut = np.sort(score)[::-1][k - 1]
    above = score > cut
    tied = score == cut
    take_tied = k - above.sum()
    hits = y[above].sum() + take_tied * y[tied].mean()
    return float(hits / k)
