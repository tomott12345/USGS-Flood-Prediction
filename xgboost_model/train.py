"""Train XGBoost point-forecast models with conformal-calibrated intervals.

Saves via XGBoost's native JSON format (Booster.save_model()), not pickle --
that's the whole point of this replacement: native format is a stable,
documented schema designed to load across XGBoost and Python versions,
unlike the numba-JIT-cached pickle blobs that broke the AutoGluon production
model (see evaluation/README.md).
"""
import json
import os
import sys

import pandas as pd
from xgboost import XGBRegressor

from conformal import conformal_margin
from data import fetch_site_history
from features import build_features, build_target

ARTIFACTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "artifacts")


def _chronological_fit_calibration_split(X, y, calibration_fraction=0.2):
    # Split by position, not by shuffling: the calibration slice must be the
    # *most recent* rows, chronologically after everything the model fits
    # on, or conformal_margin's coverage guarantee doesn't hold for
    # autocorrelated time series data.
    n = len(X)
    split_idx = int(n * (1 - calibration_fraction))
    return X.iloc[:split_idx], y.iloc[:split_idx], X.iloc[split_idx:], y.iloc[split_idx:]


def train_model(X_fit, y_fit, **xgb_kwargs):
    """Fit a single XGBRegressor point-forecast model. xgb_kwargs overrides
    any of this function's own defaults, so callers with a larger/richer
    feature set (e.g. tuning.py's search) can supply their own regularization
    without needing a second training function.
    """
    # Deliberately modest defaults (shallow trees, low learning rate) given
    # how few rows are actually available (a few hundred to ~1500) -- see
    # tuning.py for a proper regularization search used on the larger
    # (upstream+weather) enriched feature set, where these defaults overfit.
    params = dict(n_estimators=200, max_depth=4, learning_rate=0.05, random_state=0, objective="reg:squarederror")
    params.update(xgb_kwargs)
    model = XGBRegressor(**params)
    model.fit(X_fit, y_fit)
    return model


def train_site_horizon(site_code, horizon_hours, history_df=None, days=60, alpha=0.2, calibration_fraction=0.2):
    """End-to-end for one (site, horizon) pair: fetch history (unless
    already supplied, so __main__ below can fetch once and reuse it across
    horizons), build baseline lag features, fit, calibrate a conformal
    margin, and save model.json + metadata.json under artifacts/. This is
    the plain-baseline building block auto_pipeline.py's own
    train_final_model wraps with feature-set auto-selection on top.
    """
    if history_df is None:
        history_df = fetch_site_history(site_code, days=days)

    features = build_features(history_df)
    target = build_target(history_df, horizon_hours)

    # dropna: build_features' lag columns are NaN for the first LAG_HOURS
    # rows (nothing to look back at yet) and build_target's shift(-horizon)
    # is NaN for the last horizon_hours rows (nothing to look forward to
    # yet) -- both ends get trimmed here, not silently zero-filled.
    combined = pd.concat([features, target.rename("target")], axis=1).dropna()
    if len(combined) < 30:
        raise ValueError(f"Not enough usable rows ({len(combined)}) to train site={site_code} horizon={horizon_hours}")

    X = combined[features.columns]
    y = combined["target"]

    X_fit, y_fit, X_calib, y_calib = _chronological_fit_calibration_split(X, y, calibration_fraction)

    model = train_model(X_fit, y_fit)

    # Residuals measured on the calibration slice the model never trained
    # on -- this is what makes the margin an honest estimate of future error
    # rather than an optimistic in-sample one.
    calib_pred = model.predict(X_calib)
    residuals = y_calib.values - calib_pred
    margin = conformal_margin(residuals, alpha=alpha)

    model_dir = os.path.join(ARTIFACTS_DIR, f"{site_code}_h{horizon_hours}")
    os.makedirs(model_dir, exist_ok=True)
    model.save_model(os.path.join(model_dir, "model.json"))

    metadata = {
        "site_code": site_code,
        "horizon_hours": horizon_hours,
        "feature_columns": list(features.columns),
        "conformal_margin": margin,
        "alpha": alpha,
        "nominal_coverage": 1 - alpha,
        "n_fit_rows": len(X_fit),
        "n_calibration_rows": len(X_calib),
        "trained_at": pd.Timestamp.utcnow().isoformat(),
    }
    with open(os.path.join(model_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    return model_dir, metadata


if __name__ == "__main__":
    site_code = sys.argv[1] if len(sys.argv) > 1 else "01388500"
    # Default scope is <=6h: backtesting showed NSE goes negative (worse than
    # predicting the mean) at 12h+ regardless of feature set or tuning -- see
    # README.md. Longer horizons can still be requested explicitly via argv.
    horizons = [int(h) for h in sys.argv[2:]] or [1, 3, 6]

    print(f"Fetching history for {site_code}...")
    history_df = fetch_site_history(site_code, days=60)
    print(f"{len(history_df)} hourly rows available.\n")

    for horizon in horizons:
        model_dir, metadata = train_site_horizon(site_code, horizon, history_df=history_df)
        print(
            f"horizon={horizon}h -> {model_dir} "
            f"(margin={metadata['conformal_margin']:.4f} ft, "
            f"n_fit={metadata['n_fit_rows']}, n_calib={metadata['n_calibration_rows']})"
        )
