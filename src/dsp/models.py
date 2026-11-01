"""Boosted-tree ladder. In the demo: unconstrained and factor-grouped.

Both models share every setting except the interaction constraint, so any gap
between them is the cost of that constraint.
"""
import lightgbm as lgb
import numpy as np

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


def fit(X, y, interaction_groups=None) -> lgb.LGBMClassifier:
    params = dict(PARAMS)
    if interaction_groups is not None:
        params["interaction_constraints"] = interaction_groups
    model = lgb.LGBMClassifier(**params)
    model.fit(X, y)
    return model


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
    k = max(1, int(round(share * len(score))))
    top = np.argsort(-score, kind="stable")[:k]
    return float(np.mean(y[top]))
