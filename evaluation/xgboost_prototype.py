"""Prototype: replace the AutoGluon Pompton models with plain XGBoost models.

Motivation: the production AutoGluon model failed to unpickle in backtest.py
(a numba-JIT-cache / Python-bytecode-version mismatch), and one horizon
couldn't even load due to an AutoGluon version incompatibility. XGBoost's
native serialization (JSON/UBJSON via save_model()/load_model(), not used
here yet but the eventual point) is specifically designed to be stable across
library and Python versions, unlike AutoGluon's deeply-nested pickle-of-
everything persistence.

This script is a pre-commit comparison only -- it fetches history once and
re-runs both engines (existing AutoGluon models via backtest.backtest_model,
and a freshly walk-forward-trained XGBoost model) over the *same* folds, so
the numbers are directly comparable. Nothing here touches models/ or the
production microservice.
"""
import time

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

from backtest import MODEL_REGISTRY, _nse, backtest_model
from usgs_data import fetch_site_history

LAG_HOURS = [1, 2, 3, 6, 12, 24]
POMPTON_SITE = "01388500"
HORIZONS = [1, 3, 6, 12, 24, 48]


def build_features(history_df):
    features = pd.DataFrame(index=history_df.index)
    for lag in LAG_HOURS:
        features[f"gage_lag_{lag}"] = history_df["Gage"].shift(lag)
        features[f"flow_lag_{lag}"] = history_df["Flow"].shift(lag)
        features[f"gage_roc_lag_{lag}"] = history_df["Gage_rate_of_change"].shift(lag)
        features[f"flow_roc_lag_{lag}"] = history_df["Flow_rate_of_change"].shift(lag)
    return features


def _fit_predict(X_train, y_train, x_pred, objective, quantile_alpha=None):
    kwargs = dict(n_estimators=200, max_depth=4, learning_rate=0.05, random_state=0, objective=objective)
    if quantile_alpha is not None:
        kwargs["quantile_alpha"] = quantile_alpha
    model = XGBRegressor(**kwargs)
    model.fit(X_train, y_train)
    return float(model.predict(x_pred)[0])


def backtest_xgboost(history_df, horizon_hours, stride_hours=24, min_context_hours=72, max_folds=10):
    features = build_features(history_df)
    target = history_df["Gage"].shift(-horizon_hours)
    feature_ready = features.notna().all(axis=1)

    timestamps = history_df.index
    if len(timestamps) < min_context_hours + horizon_hours + 1:
        return {"label": f"xgboost_h{horizon_hours}", "status": "insufficient_history", "horizon_hours": horizon_hours}

    origins = list(range(min_context_hours, len(timestamps) - horizon_hours, stride_hours))
    if max_folds:
        origins = origins[:max_folds]

    final_step_errors = []
    coverage_hits = 0
    coverage_total = 0

    for origin_idx in origins:
        if not feature_ready.iloc[origin_idx]:
            continue

        # Train only on rows whose target (row_idx + horizon) is already known
        # as of this origin -- i.e. no peeking into the future.
        last_trainable_row = origin_idx - horizon_hours
        if last_trainable_row < 0:
            continue
        train_mask = feature_ready.iloc[: last_trainable_row + 1] & target.iloc[: last_trainable_row + 1].notna()
        train_idx = train_mask[train_mask].index
        if len(train_idx) < 30:
            continue

        X_train = features.loc[train_idx]
        y_train = target.loc[train_idx]
        x_pred = features.iloc[[origin_idx]]

        future_idx = origin_idx + horizon_hours
        if future_idx >= len(history_df):
            continue
        actual = history_df["Gage"].iloc[future_idx]
        if pd.isna(actual):
            continue

        pred_mean = _fit_predict(X_train, y_train, x_pred, objective="reg:squarederror")
        pred_lower = _fit_predict(X_train, y_train, x_pred, objective="reg:quantileerror", quantile_alpha=0.1)
        pred_upper = _fit_predict(X_train, y_train, x_pred, objective="reg:quantileerror", quantile_alpha=0.9)

        final_step_errors.append((actual, pred_mean))
        coverage_total += 1
        if pred_lower <= actual <= pred_upper:
            coverage_hits += 1

    if not final_step_errors:
        return {"label": f"xgboost_h{horizon_hours}", "status": "no_valid_folds", "horizon_hours": horizon_hours}

    actuals, preds = zip(*final_step_errors)
    mae = float(np.mean(np.abs(np.array(actuals) - np.array(preds))))
    rmse = float(np.sqrt(np.mean((np.array(actuals) - np.array(preds)) ** 2)))
    nse = _nse(actuals, preds)
    coverage = coverage_hits / coverage_total if coverage_total else float("nan")

    return {
        "label": f"xgboost_h{horizon_hours}",
        "status": "ok",
        "horizon_hours": horizon_hours,
        "n_folds": len(final_step_errors),
        "mae_ft": round(mae, 4),
        "rmse_ft": round(rmse, 4),
        "nse": round(nse, 4) if not np.isnan(nse) else float("nan"),
        "ci80_coverage": round(coverage, 3) if not np.isnan(coverage) else float("nan"),
        "ci80_nominal": 0.8,
    }


if __name__ == "__main__":
    print("Fetching Pompton (01388500) history once, shared across both engines' backtests...")
    history_df = fetch_site_history(POMPTON_SITE, days=14)
    print(f"{len(history_df)} hourly rows available.\n")

    results = []

    pompton_entries = [e for e in MODEL_REGISTRY if e.site_code == POMPTON_SITE]
    for entry in pompton_entries:
        t0 = time.time()
        result = backtest_model(
            entry.site_code, entry.label, entry.path, history_df,
            stride_hours=24, min_context_hours=72, max_folds=10,
        )
        result["engine"] = "autogluon"
        result["seconds"] = round(time.time() - t0, 1)
        results.append(result)

    for horizon in HORIZONS:
        t0 = time.time()
        result = backtest_xgboost(history_df, horizon, stride_hours=24, min_context_hours=72, max_folds=10)
        result["site_code"] = POMPTON_SITE
        result["engine"] = "xgboost"
        result["seconds"] = round(time.time() - t0, 1)
        results.append(result)

    df = pd.DataFrame(results)
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 20)
    print(df.to_string(index=False))
    out_path = "xgboost_vs_autogluon_pompton.csv"
    df.to_csv(out_path, index=False)
    print(f"\nSaved to {out_path}")
