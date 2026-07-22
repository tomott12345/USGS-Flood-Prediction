"""Hyperparameter + feature-count tuning for the enriched (upstream+weather)
model.

The enriched feature set (50 columns) against ~600-650 training rows
overfits with the same fixed, unregularized params used for the 24-column
baseline (see xgboost_model/README.md). This searches regularization
strength and feature count together, using an *inner* validation split
carved out of the training window only -- the calibration and test slices
from evaluate_fixed_model are never touched by this search, so nothing here
can leak into the numbers that actually get reported.

Known limitation (see README.md): this helps a lot at short horizons (1-6h)
but was unstable/counterproductive at 12h+ in practice, most likely because
a *single* inner train/validation split -- not k-fold cross-validation --
gives a noisier read on which config is really best when there are fewer
effective training rows left after a larger horizon's lag/shift trimming.
"""
import itertools

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

PARAM_GRID = {
    "max_depth": [2, 3, 4],
    "reg_lambda": [1, 5, 20],
    "reg_alpha": [0, 1, 5],
    "min_child_weight": [1, 5, 10],
    "subsample": [0.7, 1.0],
    "colsample_bytree": [0.6, 0.8, 1.0],
}

FEATURE_COUNTS = [8, 12, 16, 20, 30, None]  # None = all available features


def _inner_split(X, y, val_fraction=0.2):
    """Chronological split -- no shuffling -- so the inner validation set is
    still a genuine future-holdout relative to the inner training rows."""
    n = len(X)
    split = int(n * (1 - val_fraction))
    return X.iloc[:split], y.iloc[:split], X.iloc[split:], y.iloc[split:]


def _fit_with_early_stopping(X_tr, y_tr, X_val, y_val, params):
    # n_estimators=500 is a generous ceiling, not a target -- early stopping
    # halts once X_val stops improving, so the *actual* tree count self-tunes
    # per param combination instead of being one more grid dimension to search.
    model = XGBRegressor(
        n_estimators=500,
        learning_rate=0.05,
        early_stopping_rounds=20,
        eval_metric="rmse",
        random_state=0,
        objective="reg:squarederror",
        **params,
    )
    model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
    return model


def _rank_features(X_tr, y_tr, X_val, y_val):
    """Fit one reasonably regularized model on all features, rank by
    importance -- used to build *nested* candidate feature subsets (top-8,
    top-12, top-16, ...) rather than searching feature subsets
    combinatorially, which would be intractable at 50 candidate columns."""
    baseline_params = dict(max_depth=3, reg_lambda=5, reg_alpha=1, min_child_weight=5, subsample=0.8, colsample_bytree=0.8)
    model = _fit_with_early_stopping(X_tr, y_tr, X_val, y_val, baseline_params)
    importances = pd.Series(model.feature_importances_, index=X_tr.columns).sort_values(ascending=False)
    return importances.index.tolist()


def tune(X_train, y_train, val_fraction=0.2, feature_counts=None, param_grid=None, max_combinations=24, random_state=0):
    """Search feature count x regularization params on an inner validation
    split of X_train/y_train. Returns the winning config, ready to retrain
    on the *full* X_train before calibration.
    """
    feature_counts = feature_counts or FEATURE_COUNTS
    param_grid = param_grid or PARAM_GRID

    X_tr, y_tr, X_val, y_val = _inner_split(X_train, y_train, val_fraction=val_fraction)
    ranked_features = _rank_features(X_tr, y_tr, X_val, y_val)

    # Full grid is 3*3*3*3*2*3 = 486 combinations -- random-sample a bounded
    # subset rather than an exhaustive search, since each combination gets
    # refit for every feature count too (486 x 6 candidate counts would be
    # slow for a search that's meant to run inside a single evaluation call).
    keys = list(param_grid.keys())
    all_combos = list(itertools.product(*param_grid.values()))
    rng = np.random.RandomState(random_state)
    if len(all_combos) > max_combinations:
        idx = rng.choice(len(all_combos), size=max_combinations, replace=False)
        combos = [all_combos[i] for i in idx]
    else:
        combos = all_combos

    best_val_rmse = np.inf
    best_params = None
    best_features = None

    for n_features in feature_counts:
        features_subset = ranked_features if n_features is None else ranked_features[:n_features]
        X_tr_sub = X_tr[features_subset]
        X_val_sub = X_val[features_subset]

        for combo in combos:
            params = dict(zip(keys, combo))
            try:
                model = _fit_with_early_stopping(X_tr_sub, y_tr, X_val_sub, y_val, params)
            except Exception:
                continue
            pred = model.predict(X_val_sub)
            rmse = float(np.sqrt(np.mean((y_val.values - pred) ** 2)))
            if rmse < best_val_rmse:
                best_val_rmse = rmse
                best_params = params
                best_features = features_subset

    return {
        "params": best_params,
        "features": best_features,
        "n_features": len(best_features),
        "val_rmse": best_val_rmse,
    }
