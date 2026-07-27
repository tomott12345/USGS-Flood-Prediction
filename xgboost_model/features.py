"""Lag-based feature engineering, shared by training, prediction, and backtesting.

Kept intentionally simple: lagged gage/flow levels and their rates of change,
matching the signal the AutoGluon models were already trained on (see
microservice/app.py). Nothing here should look ahead of the row it's computed
for -- every column is a backward-looking .shift(), which is what makes it
safe to reuse the same feature frame for both training and walk-forward
evaluation without leakage.
"""
import logging
from datetime import datetime, timedelta

import pandas as pd

from data import fetch_historical_series
from upstream import find_upstream_gages
from weather import fetch_weather_for_site

logger = logging.getLogger(__name__)

LAG_HOURS = [1, 2, 3, 6, 12, 24]

# Upstream sites and weather get a smaller lag set than the target site's own
# history -- with several upstream sites plus weather variables, using the
# full LAG_HOURS set everywhere would multiply feature count far faster than
# the ~1000-1500 training rows available can support without overfitting.
UPSTREAM_LAG_HOURS = [1, 3, 6]
WEATHER_LAG_HOURS = [0, 3, 6, 12]
PRECIP_ROLLING_WINDOWS = [6, 24]


def build_features(history_df):
    # Some sites (dam/impoundment gages) never report discharge at all --
    # fetch_site_history degrades to an all-NaN Flow column for those rather
    # than failing (see evaluation/usgs_data.py). Detect that here and skip
    # the flow-derived columns entirely rather than emitting all-NaN features
    # that would make every row fail the caller's dropna().
    has_flow = "Flow" in history_df.columns and history_df["Flow"].notna().any()

    features = pd.DataFrame(index=history_df.index)
    for lag in LAG_HOURS:
        features[f"gage_lag_{lag}"] = history_df["Gage"].shift(lag)
        if has_flow:
            features[f"flow_lag_{lag}"] = history_df["Flow"].shift(lag)
        features[f"gage_roc_lag_{lag}"] = history_df["Gage_rate_of_change"].shift(lag)
        if has_flow:
            features[f"flow_roc_lag_{lag}"] = history_df["Flow_rate_of_change"].shift(lag)
    return features


def build_target(history_df, horizon_hours):
    # Negative shift looks *forward*: the value at row i becomes "the gage
    # height horizon_hours in the future", which is what every model in this
    # directory is actually trained to predict from row i's features.
    return history_df["Gage"].shift(-horizon_hours)


def _fetch_upstream_gage_height(upstream_site_code, days):
    """Gage height alone, via fetch_historical_series directly -- unlike
    fetch_site_history, this doesn't also require discharge (00060) data,
    which some upstream sites don't report even though they report gage
    height. Upstream features only use gage height anyway.
    """
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    gage = fetch_historical_series(upstream_site_code, "00065", "_00065", start_date, end_date)
    gage = gage.rename(columns={"value": "Gage"}).resample("h").last()
    gage["Gage_rate_of_change"] = gage["Gage"].diff()
    return gage


def build_upstream_features(upstream_site_code, target_index, days=60):
    """Lag features for one upstream gage, reindexed onto the target site's
    hourly timestamps. A short forward-fill covers the occasional missing
    hour without carrying stale data far past a real gap.
    """
    try:
        upstream_history = _fetch_upstream_gage_height(upstream_site_code, days=days)
    except Exception as e:
        logger.warning(f"Skipping upstream site {upstream_site_code}: {type(e).__name__}: {e}")
        return pd.DataFrame(index=target_index)

    # reindex onto the target's own hourly timestamps (upstream sites can
    # have slightly different reporting timestamps/gaps than Pompton), then
    # forward-fill only short gaps (<=3h) -- past that, a stale reading is
    # worse than no reading, since XGBoost's own missing-value handling is
    # better than three-hour-old data pretending to be current.
    aligned = upstream_history.reindex(target_index).ffill(limit=3)

    features = pd.DataFrame(index=target_index)
    prefix = f"up_{upstream_site_code}"
    for lag in UPSTREAM_LAG_HOURS:
        features[f"{prefix}_gage_lag_{lag}"] = aligned["Gage"].shift(lag)
        features[f"{prefix}_gage_roc_lag_{lag}"] = aligned["Gage_rate_of_change"].shift(lag)
    return features


def build_upstream_aggregate_features(upstream_site_codes, target_index, days=60):
    """Cross-site aggregates (mean, max across all upstream gages) rather
    than one set of columns per site. Several upstream gages responding to
    the same regional event within the same hour (see README.md) makes
    per-site columns highly redundant with each other -- concatenating all
    of them turned out to hurt accuracy in practice (86 features against
    ~600-630 training rows overfit badly, see baseline_vs_enriched.csv).
    Aggregating keeps the cross-tributary signal in far fewer columns.
    """
    aligned_frames = []
    for site_code in upstream_site_codes:
        try:
            history = _fetch_upstream_gage_height(site_code, days=days)
        except Exception as e:
            logger.warning(f"Skipping upstream site {site_code}: {type(e).__name__}: {e}")
            continue
        aligned_frames.append(history.reindex(target_index).ffill(limit=3))

    features = pd.DataFrame(index=target_index)
    if not aligned_frames:
        return features

    # One column per upstream site, concatenated side by side so mean()/max()
    # below operate across sites (axis=1) at each shared timestamp.
    gage_stack = pd.concat([f["Gage"] for f in aligned_frames], axis=1)
    roc_stack = pd.concat([f["Gage_rate_of_change"] for f in aligned_frames], axis=1)

    for lag in UPSTREAM_LAG_HOURS:
        # mean = the watershed's overall/typical state; max = whichever
        # tributary is currently most active. Both are worth keeping since a
        # single flooding tributary can matter even when the basin average
        # looks unremarkable.
        features[f"upstream_gage_mean_lag_{lag}"] = gage_stack.mean(axis=1).shift(lag)
        features[f"upstream_gage_max_lag_{lag}"] = gage_stack.max(axis=1).shift(lag)
        features[f"upstream_roc_mean_lag_{lag}"] = roc_stack.mean(axis=1).shift(lag)
        features[f"upstream_roc_max_lag_{lag}"] = roc_stack.max(axis=1).shift(lag)

    return features


def build_weather_features(site_code, target_index, days=60):
    """Current + lagged weather, plus cumulative precipitation over a
    trailing window -- total accumulated rainfall predicts runoff better
    than a single hour's instantaneous rate.
    """
    try:
        weather = fetch_weather_for_site(site_code, days=days, forecast_days=0)
    except Exception:
        return pd.DataFrame(index=target_index)

    aligned = weather.reindex(target_index).ffill(limit=3)

    features = pd.DataFrame(index=target_index)
    for lag in WEATHER_LAG_HOURS:
        features[f"precip_lag_{lag}"] = aligned["precipitation"].shift(lag)
        features[f"soil_moisture_lag_{lag}"] = aligned["soil_moisture"].shift(lag)
        features[f"temperature_lag_{lag}"] = aligned["temperature_2m"].shift(lag)

    for window in PRECIP_ROLLING_WINDOWS:
        # .shift(1) after the rolling sum: pandas' rolling window is
        # inclusive of the current row, so without the shift this would
        # include the current hour's own (not-yet-fully-observed-at-
        # prediction-time) precipitation in its own feature.
        features[f"precip_sum_{window}h"] = aligned["precipitation"].rolling(window).sum().shift(1)

    return features


def build_combined_features(
    site_code, history_df, days=60, upstream_site_codes=None, include_weather=True, aggregate_upstream=True,
):
    """Own-site lag features + upstream-gage features + weather features,
    all aligned to history_df's hourly index. aggregate_upstream=True (the
    default) uses cross-site summary stats instead of per-site columns --
    see build_upstream_aggregate_features for why.
    """
    features = build_features(history_df)

    if upstream_site_codes is None:
        upstream_site_codes = [g["site_code"] for g in find_upstream_gages(site_code)]

    if aggregate_upstream:
        features = features.join(build_upstream_aggregate_features(upstream_site_codes, history_df.index, days=days))
    else:
        for upstream_code in upstream_site_codes:
            features = features.join(build_upstream_features(upstream_code, history_df.index, days=days))

    if include_weather:
        weather_features = build_weather_features(site_code, history_df.index, days=days)
        features = features.join(weather_features)

    return features
