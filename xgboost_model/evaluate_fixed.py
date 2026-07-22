"""Evaluate a *fixed*, once-trained model against a genuinely held-out future
test slice -- the realistic usage pattern (train periodically, serve the
same model for many predictions) rather than retraining before every single
prediction.

This replaces the walk-forward-retrain approach in backtest.py for judging
calibration quality: that approach recomputed a conformal margin from a tiny
per-fold calibration set (as few as ~15-50 rows) that's the *most recent*
slice of an already-small training window, which (a) starves the point model
of training data and (b) makes the calibration set unrepresentative whenever
the regime shifts between the calibration window and the prediction point --
split conformal's coverage guarantee assumes exchangeability, which a short,
recency-biased calibration slice does not give you on non-stationary
hydrological data. A larger, three-way chronological split (train /
calibrate / test) gives conformal a fair chance to actually work.
"""
import sys

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

import tuning
from conformal import conformal_margin
from data import fetch_site_history
from features import build_combined_features, build_features, build_target
from metrics import mae, nse, rmse
from train import train_model


def evaluate_fixed_model(
    site_code, horizon_hours, days=45, train_fraction=0.6, calib_fraction=0.2, alpha=0.2,
    return_predictions=False, feature_set="baseline", tune=False,
):
    """feature_set: "baseline" (site's own lags only) or "enriched" (also
    upstream gages + OpenMeteo weather, via build_combined_features).
    tune=True searches regularization + feature count on an inner split of
    the training window only (see tuning.py) before the final fit -- use for
    "enriched", where the default params overfit against the larger feature
    set."""
    history_df = fetch_site_history(site_code, days=days)

    if feature_set == "enriched":
        features = build_combined_features(site_code, history_df, days=days)
    else:
        features = build_features(history_df)
    target = build_target(history_df, horizon_hours)
    combined = pd.concat([features, target.rename("target")], axis=1).dropna()

    n = len(combined)
    if n < 60:
        return {"site_code": site_code, "horizon_hours": horizon_hours, "status": "insufficient_data", "n_samples": n}

    # Three-way chronological split -- train / calibrate / test, in that
    # time order. Test is the only slice that's *never* seen by anything:
    # not the point model's fit, not the conformal margin's calibration.
    # That's what makes its MAE/NSE/coverage numbers below a genuine
    # out-of-sample read rather than an optimistic in-sample one.
    train_end = int(n * train_fraction)
    calib_end = int(n * (train_fraction + calib_fraction))

    X = combined[features.columns]
    y = combined["target"]

    X_train, y_train = X.iloc[:train_end], y.iloc[:train_end]
    X_calib, y_calib = X.iloc[train_end:calib_end], y.iloc[train_end:calib_end]
    X_test, y_test = X.iloc[calib_end:], y.iloc[calib_end:]

    if len(X_test) < 10:
        return {"site_code": site_code, "horizon_hours": horizon_hours, "status": "insufficient_test_rows", "n_test": len(X_test)}

    tuned_params = None
    if tune:
        # tuning.tune() only ever sees X_train/y_train -- it carves its own
        # inner validation split out of that, so the winning params/feature
        # subset are chosen without any peeking at X_calib or X_test.
        tuning_result = tuning.tune(X_train, y_train)
        selected_features = tuning_result["features"]
        tuned_params = tuning_result["params"]
        # Refit on the *full* X_train (not just tuning's inner split) with
        # the winning config, so the final model gets all the training data
        # tuning itself had to hold some of back to validate against.
        model = XGBRegressor(
            n_estimators=300, learning_rate=0.05, random_state=0, objective="reg:squarederror", **tuned_params,
        )
        model.fit(X_train[selected_features], y_train)
    else:
        selected_features = list(X_train.columns)
        model = train_model(X_train, y_train)

    X_calib_used = X_calib[selected_features]
    X_test_used = X_test[selected_features]

    calib_pred = model.predict(X_calib_used)
    margin = conformal_margin(y_calib.values - calib_pred, alpha=alpha)

    test_pred = model.predict(X_test_used)
    actual = y_test.values

    lower = test_pred - margin
    upper = test_pred + margin
    coverage = float(np.mean((actual >= lower) & (actual <= upper)))

    result = {
        "site_code": site_code,
        "horizon_hours": horizon_hours,
        "feature_set": feature_set,
        "tuned": tune,
        "status": "ok",
        "n_features": len(selected_features),
        "n_train": len(X_train),
        "n_calib": len(X_calib),
        "n_test": len(X_test),
        "mae_ft": round(mae(actual, test_pred), 4),
        "rmse_ft": round(rmse(actual, test_pred), 4),
        "nse": round(nse(actual, test_pred), 4),
        "margin_ft": round(margin, 4),
        "ci80_coverage": round(coverage, 3),
        "ci80_nominal": 1 - alpha,
        "tuned_params": tuned_params,
    }

    if return_predictions:
        # The target timestamp is horizon_hours after each test row's own
        # timestamp -- shift the index forward so plots line up against the
        # actual calendar time each prediction is *for*.
        target_timestamps = X_test.index + pd.Timedelta(hours=horizon_hours)
        result["predictions"] = pd.DataFrame({
            "timestamp": target_timestamps,
            "actual": actual,
            "predicted": test_pred,
            "lower": lower,
            "upper": upper,
        })

    return result


if __name__ == "__main__":
    site_code = sys.argv[1] if len(sys.argv) > 1 else "01388500"
    # <=6h only by default -- see train.py's __main__ for why.
    horizons = [int(h) for h in sys.argv[2:]] or [1, 3, 6]

    results = [evaluate_fixed_model(site_code, h, days=45) for h in horizons]
    df = pd.DataFrame(results)
    pd.set_option("display.width", 160)
    print(df.to_string(index=False))
    df.to_csv("fixed_model_evaluation.csv", index=False)
