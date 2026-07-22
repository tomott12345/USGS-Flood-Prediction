"""Walk-forward evaluation of the conformal-calibrated XGBoost model, using
the same fold structure as evaluation/backtest.py and
evaluation/xgboost_prototype.py so results are directly comparable.

Kept for reference/comparison, but evaluate_fixed.py is the one to trust for
calibration numbers: recomputing a conformal margin from a tiny per-fold
calibration set (as few as ~5-50 rows here) doesn't give conformal's coverage
guarantee a fair chance -- see README.md's calibration section for the full
story of why the fix was a bigger three-way split, not a different formula.
"""
import sys
import time

import numpy as np
import pandas as pd

from conformal import conformal_margin
from data import fetch_site_history
from features import build_features, build_target
from metrics import mae, nse, rmse
from train import _chronological_fit_calibration_split, train_model


def backtest_calibrated_xgboost(
    history_df, horizon_hours, stride_hours=24, min_context_hours=72,
    max_folds=15, alpha=0.2, calibration_fraction=0.2, min_train_rows=30,
):
    features = build_features(history_df)
    target = build_target(history_df, horizon_hours)
    feature_ready = features.notna().all(axis=1)

    timestamps = history_df.index
    if len(timestamps) < min_context_hours + horizon_hours + 1:
        return {"label": f"xgboost_calibrated_h{horizon_hours}", "status": "insufficient_history", "horizon_hours": horizon_hours}

    # An "origin" is one simulated prediction moment: everything up to and
    # including origin_idx is treated as known, everything after is the
    # future being forecast. Walking forward through many origins (instead
    # of one train/test split) is what makes this "walk-forward" rather than
    # a single backtest -- but note each origin below retrains a fresh model
    # from scratch, which is what makes this slow and small-sample compared
    # to evaluate_fixed.py's train-once approach.
    origins = list(range(min_context_hours, len(timestamps) - horizon_hours, stride_hours))
    if max_folds:
        origins = origins[:max_folds]

    final_step_errors = []
    coverage_hits = 0
    coverage_total = 0
    margins = []

    for origin_idx in origins:
        if not feature_ready.iloc[origin_idx]:
            continue

        # No-lookahead guard: a training row at position i has a target of
        # Gage[i + horizon_hours] (see build_target), so a row can only be
        # used for training if its own target has *already happened* by this
        # origin -- i.e. i + horizon_hours <= origin_idx. Rows past that
        # would let the model train on outcomes it hasn't "seen" yet at
        # prediction time.
        last_trainable_row = origin_idx - horizon_hours
        if last_trainable_row < 0:
            continue

        train_mask = feature_ready.iloc[: last_trainable_row + 1] & target.iloc[: last_trainable_row + 1].notna()
        train_idx = train_mask[train_mask].index
        if len(train_idx) < min_train_rows:
            continue

        X = features.loc[train_idx]
        y = target.loc[train_idx]
        X_fit, y_fit, X_calib, y_calib = _chronological_fit_calibration_split(X, y, calibration_fraction)
        if len(X_calib) < 5:
            continue

        model = train_model(X_fit, y_fit)
        calib_pred = model.predict(X_calib)
        residuals = y_calib.values - calib_pred
        margin = conformal_margin(residuals, alpha=alpha)
        margins.append(margin)

        # The actual prediction: features as of the origin itself (the
        # "now" of this simulated moment), forecasting horizon_hours ahead.
        x_pred = features.iloc[[origin_idx]]
        pred_mean = float(model.predict(x_pred)[0])

        future_idx = origin_idx + horizon_hours
        if future_idx >= len(history_df):
            continue
        actual = history_df["Gage"].iloc[future_idx]
        if pd.isna(actual):
            continue

        final_step_errors.append((actual, pred_mean))
        coverage_total += 1
        # Did this fold's actual outcome fall inside its own conformal
        # interval? Averaged across all folds below, this is the empirical
        # coverage compared against the 80% nominal target.
        if (pred_mean - margin) <= actual <= (pred_mean + margin):
            coverage_hits += 1

    if not final_step_errors:
        return {"label": f"xgboost_calibrated_h{horizon_hours}", "status": "no_valid_folds", "horizon_hours": horizon_hours}

    actuals, preds = zip(*final_step_errors)
    coverage = coverage_hits / coverage_total if coverage_total else float("nan")

    return {
        "label": f"xgboost_calibrated_h{horizon_hours}",
        "status": "ok",
        "horizon_hours": horizon_hours,
        "n_folds": len(final_step_errors),
        "mae_ft": round(mae(actuals, preds), 4),
        "rmse_ft": round(rmse(actuals, preds), 4),
        "nse": round(nse(actuals, preds), 4),
        "ci80_coverage": round(coverage, 3) if not np.isnan(coverage) else float("nan"),
        "ci80_nominal": 1 - alpha,
        "avg_margin_ft": round(float(np.mean(margins)), 4) if margins else float("nan"),
    }


if __name__ == "__main__":
    site_code = sys.argv[1] if len(sys.argv) > 1 else "01388500"
    # <=6h only by default -- see train.py's __main__ for why.
    horizons = [int(h) for h in sys.argv[2:]] or [1, 3, 6]

    print(f"Fetching history for {site_code}...")
    history_df = fetch_site_history(site_code, days=60)
    print(f"{len(history_df)} hourly rows available.\n")

    results = []
    for horizon in horizons:
        t0 = time.time()
        result = backtest_calibrated_xgboost(history_df, horizon, stride_hours=24, min_context_hours=72, max_folds=15)
        result["seconds"] = round(time.time() - t0, 1)
        results.append(result)

    df = pd.DataFrame(results)
    pd.set_option("display.width", 160)
    print(df.to_string(index=False))
    df.to_csv("backtest_calibrated_results.csv", index=False)
