"""Lightweight test of whether precipitation as an exogenous feature would
likely improve accuracy -- without retraining (and risking) the production
AutoGluon models.

Trains a fast XGBoost regressor twice on the same time-ordered split: once
with only the features the production model already uses (gage/flow/rate-of-
change lags), once with those plus precipitation lag and rolling-sum
features. A meaningful drop in held-out error signals that retraining the
real models with precipitation is likely worth the cost; a negligible or
negative change means it probably isn't, for this site/horizon.
"""
import sys

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

from precipitation import fetch_precipitation_for_site
from usgs_data import fetch_site_history

LAG_HOURS = [1, 2, 3, 6, 12, 24]
PRECIP_ROLLING_WINDOWS = [3, 6, 12, 24]


def build_features(history_df, precip_df, horizon_hours):
    df = history_df.join(precip_df, how="left")
    df["precipitation_mm"] = df["precipitation_mm"].fillna(0.0)

    baseline_features = pd.DataFrame(index=df.index)
    for lag in LAG_HOURS:
        baseline_features[f"gage_lag_{lag}"] = df["Gage"].shift(lag)
        baseline_features[f"flow_lag_{lag}"] = df["Flow"].shift(lag)
        baseline_features[f"gage_roc_lag_{lag}"] = df["Gage_rate_of_change"].shift(lag)
        baseline_features[f"flow_roc_lag_{lag}"] = df["Flow_rate_of_change"].shift(lag)

    precip_features = pd.DataFrame(index=df.index)
    for window in PRECIP_ROLLING_WINDOWS:
        # shift(1) so the current hour's own precipitation isn't used to predict itself
        precip_features[f"precip_sum_{window}h"] = df["precipitation_mm"].rolling(window).sum().shift(1)

    target = df["Gage"].shift(-horizon_hours)
    return baseline_features, precip_features, target


def evaluate_variant(X, y, split_idx):
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    model = XGBRegressor(n_estimators=200, max_depth=4, learning_rate=0.05, random_state=0)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    mae = float(np.mean(np.abs(y_test.values - pred)))
    rmse = float(np.sqrt(np.mean((y_test.values - pred) ** 2)))
    return mae, rmse, len(X_test)


def run_precip_hypothesis_test(site_code, horizon_hours=6, days=21, test_fraction=0.2):
    history_df = fetch_site_history(site_code, days=days)
    precip_df = fetch_precipitation_for_site(site_code, past_days=min(days + 2, 92), forecast_days=1)

    baseline_features, precip_features, target = build_features(history_df, precip_df, horizon_hours)

    combined = pd.concat(
        [baseline_features, precip_features, target.rename("target")], axis=1
    ).dropna()

    if len(combined) < 30:
        return {"site_code": site_code, "status": "insufficient_data", "n_samples": len(combined)}

    baseline_X = combined[baseline_features.columns]
    with_precip_X = combined[list(baseline_features.columns) + list(precip_features.columns)]
    y = combined["target"]

    split_idx = int(len(combined) * (1 - test_fraction))

    baseline_mae, baseline_rmse, n_test = evaluate_variant(baseline_X, y, split_idx)
    precip_mae, precip_rmse, _ = evaluate_variant(with_precip_X, y, split_idx)

    return {
        "site_code": site_code,
        "status": "ok",
        "horizon_hours": horizon_hours,
        "n_train": split_idx,
        "n_test": n_test,
        "baseline_mae_ft": round(baseline_mae, 4),
        "baseline_rmse_ft": round(baseline_rmse, 4),
        "with_precip_mae_ft": round(precip_mae, 4),
        "with_precip_rmse_ft": round(precip_rmse, 4),
        "mae_improvement_pct": round(100 * (baseline_mae - precip_mae) / baseline_mae, 2) if baseline_mae else float("nan"),
        "rmse_improvement_pct": round(100 * (baseline_rmse - precip_rmse) / baseline_rmse, 2) if baseline_rmse else float("nan"),
    }


if __name__ == "__main__":
    site_codes = sys.argv[1:] or ["01388500", "01473730", "08393610"]
    results = [run_precip_hypothesis_test(site) for site in site_codes]
    df = pd.DataFrame(results)
    pd.set_option("display.width", 160)
    pd.set_option("display.max_columns", 20)
    print(df.to_string(index=False))
