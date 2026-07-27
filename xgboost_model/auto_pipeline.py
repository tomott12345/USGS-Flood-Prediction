"""End-to-end auto-training pipeline: streamgage ID in, deployable XGBoost
artifacts out.

Everything downstream of `train.py` in this directory has, until now,
required a human to decide two things by hand for a given site (see
README.md's "trial and error"): which upstream gages to use, and whether the
enriched (upstream + weather) feature set with tuning actually beats the
plain baseline at a given horizon -- the answer to the second question flips
depending on horizon and, per the README, doesn't generalize cleanly across
sites either. This module automates both decisions instead of hard-coding
Pompton's answer:

  1. Discover upstream gages for the given site via `upstream.find_upstream_gages`
     (no site-specific list to maintain).
  2. Per horizon, run the *same* held-out comparison documented in README.md
     -- baseline vs. enriched+tuned -- via `evaluate_fixed.evaluate_fixed_model`,
     and pick whichever wins on held-out MAE for *that* site and horizon.
  3. Refit the winning configuration on the full available training window
     and save it with XGBoost's native `save_model()` JSON format (the same
     format `train.py` already uses, and the whole reason this replacement
     exists -- see README.md's opening paragraph on AutoGluon's pickle
     breaking across Python/numba versions), plus a metadata.json rich
     enough for the microservice to reproduce the exact same feature
     pipeline live (feature_set, exact upstream site codes, tuned
     hyperparameters, conformal margin, feature column order).

Nothing here changes train.py, evaluate_fixed.py, or tuning.py -- this is a
new layer on top that automates the *choices* a human was making around
them, reusing their logic as-is.

Usage:
    python auto_pipeline.py 01388500
    python auto_pipeline.py 01388500 --horizons 1 3 6
    python auto_pipeline.py 01473730 --days 60 --alpha 0.2

Scope note: default horizons are 1/3/6h only, matching the scope decision in
README.md (NSE goes negative -- worse than predicting the mean -- at 12h+ for
every engine tried so far). Longer horizons can still be requested explicitly
via --horizons; the pipeline will train them, but treat the result with the
same skepticism the README does.
"""
import argparse
import json
import logging
import os
import sys
from datetime import datetime, timezone

import pandas as pd
import sklearn
import xgboost
from xgboost import XGBRegressor

import tuning
from conformal import conformal_margin
from data import fetch_site_history
from evaluate_fixed import evaluate_fixed_model
from features import build_combined_features, build_features, build_target
from train import train_model
from upstream import find_upstream_gages

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

ARTIFACTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "artifacts")

DEFAULT_HORIZONS = [1, 3, 6]

# The upstream lag/rolling-window features in features.py look back as far as
# 24h, so the training window has to be comfortably longer than that -- 60
# matches train.py's own default and stays well inside USGS's ~120-day
# instantaneous-values retention window.
DEFAULT_TRAIN_DAYS = 60


def _chronological_split(X, y, fraction=0.2):
    """Same chronological (not shuffled) fit/calibration split as
    train.py's _chronological_fit_calibration_split, duplicated locally
    (four lines) rather than importing a name train.py itself treats as
    private -- both must always agree that the calibration slice is the
    most recent rows, or conformal_margin's coverage guarantee doesn't hold.
    """
    n = len(X)
    split_idx = int(n * (1 - fraction))
    return X.iloc[:split_idx], y.iloc[:split_idx], X.iloc[split_idx:], y.iloc[split_idx:]


def select_feature_set(site_code, horizon_hours, days, alpha=0.2):
    """Run the held-out baseline vs. enriched+tuned comparison for one
    horizon and return the winner plus both sets of metrics, so the choice
    is auditable rather than a black box.
    """
    baseline = evaluate_fixed_model(site_code, horizon_hours, days=days, alpha=alpha, feature_set="baseline")
    enriched = evaluate_fixed_model(
        site_code, horizon_hours, days=days, alpha=alpha, feature_set="enriched", tune=True,
    )

    baseline_ok = baseline["status"] == "ok"
    enriched_ok = enriched["status"] == "ok"

    if not baseline_ok and not enriched_ok:
        winner = None
    elif baseline_ok and not enriched_ok:
        winner = "baseline"
    elif enriched_ok and not baseline_ok:
        winner = "enriched"
    else:
        winner = "baseline" if baseline["mae_ft"] <= enriched["mae_ft"] else "enriched"

    return {"winner": winner, "baseline": baseline, "enriched": enriched}


def train_final_model(site_code, horizon_hours, feature_set, days, upstream_site_codes=None, alpha=0.2):
    """Refit the winning configuration on the full available window (not the
    inner train/calib/test split evaluate_fixed_model uses just to compare
    options) and save it as a deployable artifact.
    """
    history_df = fetch_site_history(site_code, days=days)

    tuned_params = None
    if feature_set == "enriched":
        features = build_combined_features(site_code, history_df, days=days, upstream_site_codes=upstream_site_codes)
    else:
        features = build_features(history_df)

    target = build_target(history_df, horizon_hours)
    combined = pd.concat([features, target.rename("target")], axis=1).dropna()
    if len(combined) < 30:
        raise ValueError(
            f"Not enough usable rows ({len(combined)}) to train site={site_code} horizon={horizon_hours}"
        )

    X = combined[features.columns]
    y = combined["target"]
    X_fit, y_fit, X_calib, y_calib = _chronological_split(X, y, fraction=0.2)

    if feature_set == "enriched":
        # Same tune-on-inner-split-then-refit-on-all-of-X_fit pattern as
        # evaluate_fixed_model's tune=True branch, just against this
        # module's own X_fit/y_fit rather than evaluate_fixed_model's.
        tuning_result = tuning.tune(X_fit, y_fit)
        selected_features = tuning_result["features"]
        tuned_params = tuning_result["params"]
        model = XGBRegressor(
            n_estimators=300, learning_rate=0.05, random_state=0, objective="reg:squarederror", **tuned_params,
        )
        model.fit(X_fit[selected_features], y_fit)
    else:
        selected_features = list(X_fit.columns)
        model = train_model(X_fit, y_fit)

    calib_pred = model.predict(X_calib[selected_features])
    margin = conformal_margin(y_calib.values - calib_pred, alpha=alpha)

    model_dir = os.path.join(ARTIFACTS_DIR, f"{site_code}_h{horizon_hours}")
    os.makedirs(model_dir, exist_ok=True)
    model.save_model(os.path.join(model_dir, "model.json"))

    metadata = {
        "engine": "xgboost",
        "format": "xgboost_native_json",
        "site_code": site_code,
        "horizon_hours": horizon_hours,
        "feature_set": feature_set,
        "tuned": feature_set == "enriched",
        "tuned_params": tuned_params,
        "feature_columns": selected_features,
        "upstream_site_codes": upstream_site_codes if feature_set == "enriched" else [],
        "conformal_margin": margin,
        "alpha": alpha,
        "nominal_coverage": 1 - alpha,
        "n_fit_rows": len(X_fit),
        "n_calibration_rows": len(X_calib),
        "training_days": days,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "xgboost_version": xgboost.__version__,
        "sklearn_version": sklearn.__version__,
        "python_version": sys.version.split()[0],
    }
    with open(os.path.join(model_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    return model_dir, metadata


def run_pipeline(site_code, horizons=None, days=DEFAULT_TRAIN_DAYS, alpha=0.2):
    horizons = horizons or DEFAULT_HORIZONS

    logger.info(f"Discovering upstream gages for {site_code} (this scans USGS's NLDI network graph; "
                f"cached to artifacts/upstream_cache/ after the first run)...")
    upstream_gages = find_upstream_gages(site_code)
    upstream_site_codes = [g["site_code"] for g in upstream_gages]
    if upstream_site_codes:
        logger.info(f"Found {len(upstream_site_codes)} usable upstream gage(s): {upstream_site_codes}")
    else:
        logger.info("No usable upstream gages found -- enriched models will fall back to weather-only enrichment.")

    manifest = {
        "site_code": site_code,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "upstream_site_codes": upstream_site_codes,
        "training_days": days,
        "horizons": {},
    }

    for horizon in horizons:
        logger.info(f"--- horizon={horizon}h: evaluating baseline vs. enriched+tuned on held-out data ---")
        selection = select_feature_set(site_code, horizon, days=days, alpha=alpha)
        winner = selection["winner"]

        if winner is None:
            logger.warning(
                f"horizon={horizon}h: neither feature set had enough usable data "
                f"(baseline={selection['baseline'].get('status')}, enriched={selection['enriched'].get('status')}) "
                "-- skipping this horizon."
            )
            manifest["horizons"][str(horizon)] = {"status": "skipped", "reason": "insufficient_data"}
            continue

        baseline_mae = selection["baseline"].get("mae_ft")
        enriched_mae = selection["enriched"].get("mae_ft")
        logger.info(
            f"horizon={horizon}h: baseline_mae={baseline_mae} enriched+tuned_mae={enriched_mae} -> "
            f"chosen feature_set={winner}"
        )

        model_dir, metadata = train_final_model(
            site_code, horizon, feature_set=winner, days=days,
            upstream_site_codes=upstream_site_codes, alpha=alpha,
        )
        logger.info(
            f"horizon={horizon}h: saved {metadata['feature_set']} model -> {model_dir} "
            f"(margin={metadata['conformal_margin']:.4f} ft, n_fit={metadata['n_fit_rows']}, "
            f"n_calib={metadata['n_calibration_rows']})"
        )

        manifest["horizons"][str(horizon)] = {
            "status": "ok",
            "feature_set": winner,
            "model_dir": os.path.relpath(model_dir, ARTIFACTS_DIR),
            "selection": {
                "baseline_mae_ft": baseline_mae,
                "enriched_tuned_mae_ft": enriched_mae,
            },
            "conformal_margin_ft": metadata["conformal_margin"],
        }

    manifest_path = os.path.join(ARTIFACTS_DIR, f"{site_code}_manifest.json")
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    logger.info(f"Manifest written to {manifest_path}")

    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("site_code", help="USGS streamgage site code, e.g. 01388500")
    parser.add_argument(
        "--horizons", type=int, nargs="+", default=None,
        help=f"Forecast horizons in hours (default: {DEFAULT_HORIZONS}). "
             "Horizons past 6h have shown negative NSE in this repo's testing -- see README.md.",
    )
    parser.add_argument("--days", type=int, default=DEFAULT_TRAIN_DAYS, help="Training history window in days.")
    parser.add_argument("--alpha", type=float, default=0.2, help="Conformal miscoverage rate (0.2 = 80% CI).")
    args = parser.parse_args()

    run_pipeline(args.site_code, horizons=args.horizons, days=args.days, alpha=args.alpha)


if __name__ == "__main__":
    main()
