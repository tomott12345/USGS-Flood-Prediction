"""Walk-forward backtesting across all sites/models, with confidence-interval
calibration checks.

For each site, freshly fetched USGS history is fed through the exact same
feature engineering as microservice/app.py (long-format gage/flow/rate-of-
change series), then each site's already-trained AutoGluon model is walked
forward through expanding-context origins, comparing its forecast against the
true future reading at each origin. This evaluates what's actually deployed,
not a re-trained stand-in.
"""
import logging
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor

from usgs_data import fetch_site_history

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@dataclass
class ModelEntry:
    site_code: str
    label: str
    path: str


MODEL_REGISTRY = [
    ModelEntry("01388500", "production_h1", "models/01388500_model_1"),
    ModelEntry("01388500", "pompton_h1", "usgs-streamgage-01388500/autogluon/models/pompton_gage_autogluon_1"),
    ModelEntry("01388500", "pompton_h3", "usgs-streamgage-01388500/autogluon/models/pompton_gage_autogluon_3"),
    ModelEntry("01388500", "pompton_h6", "usgs-streamgage-01388500/autogluon/models/pompton_gage_autogluon_6"),
    ModelEntry("01388500", "pompton_h12", "usgs-streamgage-01388500/autogluon/models/pompton_gage_autogluon_12"),
    ModelEntry("01388500", "pompton_h24", "usgs-streamgage-01388500/autogluon/models/pompton_gage_autogluon_24"),
    ModelEntry("01388500", "pompton_h48", "usgs-streamgage-01388500/autogluon/models/pompton_gage_autogluon_48"),
    ModelEntry("01473730", "schuylkill_h6", "usgs-streamgage-01473730/autogluon/schuylkill_gage_autogluon"),
    ModelEntry("08393610", "rio_hondo_h6", "usgs-streamgage-08393610/autogluon/rio_hondo_gage_autogluon_6"),
]


def _to_long_df(history_df):
    """Mirror microservice/app.py's fetch_latest_data melt step exactly, so the
    model sees the same shape of input it was trained/served on."""
    df = history_df.reset_index()
    long_df = df.melt(
        id_vars=["datetime"],
        value_vars=["Gage", "Flow", "Gage_rate_of_change", "Flow_rate_of_change"],
        var_name="item_id",
        value_name="series",
    )
    item_id_map = {
        "Gage": "gage",
        "Flow": "flow",
        "Gage_rate_of_change": "gage_rate_of_change",
        "Flow_rate_of_change": "flow_rate_of_change",
    }
    long_df["item_id"] = long_df["item_id"].map(item_id_map)
    long_df["datetime"] = pd.to_datetime(long_df["datetime"])
    return long_df


def _nse(actual, predicted):
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    denom = np.sum((actual - actual.mean()) ** 2)
    if denom == 0:
        return float("nan")
    return 1 - np.sum((actual - predicted) ** 2) / denom


def backtest_model(site_code, label, model_path, history_df, stride_hours=12, min_context_hours=72, max_folds=None):
    full_path = os.path.join(REPO_ROOT, model_path)
    try:
        predictor = TimeSeriesPredictor.load(full_path, require_version_match=False)
    except Exception as e:
        return {"site_code": site_code, "label": label, "status": f"load_failed: {type(e).__name__}: {e}"}

    horizon = predictor.prediction_length
    timestamps = history_df.index

    if len(timestamps) < min_context_hours + horizon + 1:
        return {"site_code": site_code, "label": label, "status": "insufficient_history", "horizon_hours": horizon}

    origins = list(range(min_context_hours, len(timestamps) - horizon, stride_hours))
    if max_folds:
        origins = origins[:max_folds]

    final_step_errors = []
    coverage_hits = 0
    coverage_total = 0

    for origin_idx in origins:
        origin_ts = timestamps[origin_idx]
        context = history_df.iloc[: origin_idx + 1]
        future_actual = history_df["Gage"].iloc[origin_idx + 1 : origin_idx + 1 + horizon]
        if len(future_actual) < horizon:
            continue

        long_df = _to_long_df(context)
        ts_df = TimeSeriesDataFrame.from_data_frame(long_df, id_column="item_id", timestamp_column="datetime")

        try:
            pred_df = predictor.predict(ts_df)
        except Exception as e:
            logger.warning(f"[{site_code}/{label}] predict failed at origin {origin_ts}: {e}")
            continue

        gage_pred = pred_df[pred_df.index.get_level_values("item_id") == "gage"]
        if gage_pred.empty:
            continue

        pred_mean = gage_pred["mean"].values
        lower = gage_pred["0.1"].values if "0.1" in gage_pred.columns else None
        upper = gage_pred["0.9"].values if "0.9" in gage_pred.columns else None
        actual_values = future_actual.values

        # Matches production: only the final step of the horizon is ever served.
        final_step_errors.append((actual_values[-1], pred_mean[-1]))

        if lower is not None and upper is not None:
            coverage_total += 1
            if lower[-1] <= actual_values[-1] <= upper[-1]:
                coverage_hits += 1

    if not final_step_errors:
        return {"site_code": site_code, "label": label, "status": "no_valid_folds", "horizon_hours": horizon}

    actuals, preds = zip(*final_step_errors)
    mae = float(np.mean(np.abs(np.array(actuals) - np.array(preds))))
    rmse = float(np.sqrt(np.mean((np.array(actuals) - np.array(preds)) ** 2)))
    nse = _nse(actuals, preds)
    coverage = coverage_hits / coverage_total if coverage_total else float("nan")

    return {
        "site_code": site_code,
        "label": label,
        "status": "ok",
        "horizon_hours": horizon,
        "n_folds": len(final_step_errors),
        "mae_ft": round(mae, 4),
        "rmse_ft": round(rmse, 4),
        "nse": round(nse, 4) if not np.isnan(nse) else float("nan"),
        "ci80_coverage": round(coverage, 3) if not np.isnan(coverage) else float("nan"),
        "ci80_nominal": 0.8,
    }


def run_backtest(days=21, stride_hours=12, min_context_hours=72, max_folds=None, registry=MODEL_REGISTRY):
    results = []
    history_cache = {}

    for entry in registry:
        if entry.site_code not in history_cache:
            try:
                history_cache[entry.site_code] = fetch_site_history(entry.site_code, days=days)
            except Exception as e:
                history_cache[entry.site_code] = e

        history = history_cache[entry.site_code]
        if isinstance(history, Exception):
            results.append({
                "site_code": entry.site_code,
                "label": entry.label,
                "status": f"data_fetch_failed: {type(history).__name__}: {history}",
            })
            continue

        logger.info(f"Backtesting {entry.site_code}/{entry.label} ({len(history)} hourly rows available)...")
        result = backtest_model(
            entry.site_code, entry.label, entry.path, history,
            stride_hours=stride_hours, min_context_hours=min_context_hours, max_folds=max_folds,
        )
        results.append(result)

    return pd.DataFrame(results)


if __name__ == "__main__":
    df = run_backtest()
    pd.set_option("display.width", 160)
    pd.set_option("display.max_columns", 20)
    print(df.to_string(index=False))
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "backtest_results.csv")
    df.to_csv(out_path, index=False)
    print(f"\nSaved to {out_path}")
