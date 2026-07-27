"""Additive XGBoost serving engine for the microservice.

This is deliberately a separate module from app.py's existing AutoGluon
path (fetch_latest_data / load_model / the /predict/{site_code}/{forecast_length}
route) -- nothing there is touched. This module serves whatever models
xgboost_model/auto_pipeline.py has trained and saved to
xgboost_model/artifacts/{site_code}_h{horizon}/ (XGBoost's native JSON
format, not pickle -- see xgboost_model/README.md for why).

Feature parity with training is the entire point of reusing
xgboost_model/features.py directly here rather than reimplementing the lag/
upstream/weather logic a second time: any drift between how a model was
trained and how it's fed at serve time would silently produce bad
predictions without erroring, so there is exactly one implementation of
"how are features for site X built," imported by both training and serving.
"""
import logging
import os
import sys
from datetime import datetime, timedelta

import xgboost as xgb

logger = logging.getLogger(__name__)

# Make xgboost_model's modules importable regardless of the microservice's
# cwd (the readme's documented run command is `uvicorn app:app` from inside
# microservice/, but sys.path should not depend on that).
_XGBOOST_MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "xgboost_model")
if _XGBOOST_MODEL_DIR not in sys.path:
    sys.path.append(_XGBOOST_MODEL_DIR)

from data import fetch_site_history  # noqa: E402
from features import build_combined_features, build_features  # noqa: E402

# Mirrors MODEL_DIRECTORY's convention (env override, cwd-relative default)
# for the existing AutoGluon models.
ARTIFACTS_DIR = os.environ.get("XGB_ARTIFACTS_DIR", "../xgboost_model/artifacts")

# Same staleness policy as app.py's fetch_latest_data -- refuse to predict on
# data that's too old to reflect current conditions.
STALE_DATA_THRESHOLD_HOURS = 3

# How much history to pull for a live prediction. The longest lookback used
# by any feature (features.py's LAG_HOURS/PRECIP_ROLLING_WINDOWS) is 24h;
# 5 days gives ample margin for that plus the occasional short data gap,
# while staying a fast, small request.
LIVE_FEATURE_WINDOW_DAYS = 5


class XGBoostModelError(Exception):
    """Raised for any XGBoost-serving failure that should map to an HTTP
    error in app.py -- keeps this module free of any FastAPI/HTTPException
    dependency so it can be unit-tested standalone."""

    def __init__(self, status_code, detail):
        self.status_code = status_code
        self.detail = detail
        super().__init__(detail)


def available_horizons(site_code):
    """Horizons with a trained artifact for this site, sorted ascending."""
    if not os.path.isdir(ARTIFACTS_DIR):
        return []
    prefix = f"{site_code}_h"
    horizons = []
    for name in os.listdir(ARTIFACTS_DIR):
        if name.startswith(prefix) and os.path.exists(os.path.join(ARTIFACTS_DIR, name, "metadata.json")):
            try:
                horizons.append(int(name[len(prefix):]))
            except ValueError:
                continue
    return sorted(horizons)


def load_artifact(site_code, horizon_hours):
    """Load metadata.json + the native-JSON XGBoost model for one
    site/horizon. Raises XGBoostModelError (404) if nothing was trained."""
    model_dir = os.path.join(ARTIFACTS_DIR, f"{site_code}_h{horizon_hours}")
    metadata_path = os.path.join(model_dir, "metadata.json")
    model_path = os.path.join(model_dir, "model.json")

    if not (os.path.exists(metadata_path) and os.path.exists(model_path)):
        raise XGBoostModelError(
            404,
            f"No XGBoost model found for site={site_code} horizon={horizon_hours}h. "
            f"Train one with: python xgboost_model/auto_pipeline.py {site_code} --horizons {horizon_hours}",
        )

    import json
    with open(metadata_path) as f:
        metadata = json.load(f)

    # Older artifacts trained directly by xgboost_model/train.py (before
    # auto_pipeline.py existed) only ever used the baseline feature set and
    # predate these keys entirely -- default them in rather than requiring
    # every artifact on disk to be retrained just to add a key.
    metadata.setdefault("feature_set", "baseline")
    metadata.setdefault("format", "xgboost_native_json")
    metadata.setdefault("upstream_site_codes", [])
    metadata.setdefault("engine", "xgboost")
    metadata.setdefault("trained_at", None)

    model = xgb.XGBRegressor()
    model.load_model(model_path)
    return model, metadata


def _build_live_feature_row(site_code, metadata):
    """Fetch a fresh, short history window and build the exact feature row
    the model expects (same columns, same order as metadata['feature_columns']),
    using the last fully-populated row as "now"."""
    history_df = fetch_site_history(site_code, days=LIVE_FEATURE_WINDOW_DAYS)
    if history_df.empty:
        raise XGBoostModelError(503, f"No recent gage data available for site {site_code}.")

    latest_reading_age = datetime.now() - history_df.index.max().to_pydatetime().replace(tzinfo=None)
    if latest_reading_age > timedelta(hours=STALE_DATA_THRESHOLD_HOURS):
        raise XGBoostModelError(
            503,
            f"Latest reading for site {site_code} is {latest_reading_age} old, exceeding the "
            f"{STALE_DATA_THRESHOLD_HOURS}h staleness threshold. Refusing to predict on stale data.",
        )

    if metadata["feature_set"] == "enriched":
        features = build_combined_features(
            site_code, history_df, days=LIVE_FEATURE_WINDOW_DAYS,
            upstream_site_codes=metadata.get("upstream_site_codes") or None,
        )
    else:
        features = build_features(history_df)

    feature_columns = metadata["feature_columns"]
    missing = [c for c in feature_columns if c not in features.columns]
    if missing:
        raise XGBoostModelError(
            500,
            f"Live feature build for site {site_code} is missing columns the model expects: {missing}. "
            "The model may have been trained with a different upstream/weather configuration.",
        )

    row = features[feature_columns].iloc[[-1]]
    if row.isna().any(axis=1).iloc[0]:
        nan_cols = row.columns[row.isna().iloc[0]].tolist()
        raise XGBoostModelError(
            503,
            f"Not enough recent data to compute all required features for site {site_code} "
            f"(missing: {nan_cols}). Upstream gage or weather feeds may be lagging.",
        )

    as_of = features.index[-1]
    return row, as_of


def predict(site_code, horizon_hours):
    """Full predict path: load artifact, build the live feature row, predict,
    apply the model's conformal margin. Returns a plain dict, ready to be
    returned as JSON by app.py's route."""
    model, metadata = load_artifact(site_code, horizon_hours)
    row, as_of = _build_live_feature_row(site_code, metadata)

    pred = float(model.predict(row)[0])
    margin = metadata["conformal_margin"]

    return {
        "site_code": site_code,
        "horizon_hours": horizon_hours,
        "as_of": as_of.isoformat(),
        "predicted_gage_height": round(pred, 3),
        "confidence_interval": {
            "lower_bound": round(pred - margin, 3),
            "upper_bound": round(pred + margin, 3),
            "nominal_coverage": metadata["nominal_coverage"],
        },
        "model": {
            "engine": "xgboost",
            "format": metadata["format"],
            "feature_set": metadata["feature_set"],
            "upstream_site_codes": metadata.get("upstream_site_codes", []),
            "trained_at": metadata["trained_at"],
        },
    }
