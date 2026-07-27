from fastapi import FastAPI, HTTPException
import pandas as pd
import numpy as np
import psutil
from datetime import datetime, timedelta
import os
import logging
import traceback
from autogluon.timeseries import TimeSeriesPredictor, TimeSeriesDataFrame
import pickle

import xgboost_engine

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI()

model_directory = os.environ.get("MODEL_DIRECTORY", "../models")

# USGS qualification codes that mean the reading itself shouldn't be trusted
# (as opposed to "P"/"A"/"e" which just mean provisional/approved/estimated).
# Deliberately does NOT include any check on the magnitude of a reading or its
# rate of change -- a fast-rising gage is the exact signal this service exists
# to catch, so a generic "reject big jumps" rule would suppress real flood events.
BAD_QUALITY_CODES = {"ice", "eqp", "mnt", "dis", "bkw", "rat"}

# Same staleness policy xgboost_engine.py's own _build_live_feature_row uses
# for the additive XGBoost route below. Both checks compare a naive
# datetime.now() against USGS's station-local (US Eastern) timestamps, so
# both only work correctly if the host's own clock is set to US Eastern --
# see microservice/readme.md's "Known limitation" section. The repo-root
# Dockerfile sets this for the containerized deployment; a bare
# `uvicorn app:app` on a non-Eastern host will hit false staleness errors.
STALE_DATA_THRESHOLD_HOURS = 3

def log_system_usage(stage=""):
    """Log CPU/memory usage at a named point in a request's lifecycle
    (called with "Start"/"Before Prediction"/"After Prediction" etc. from
    the predict routes below) -- diagnostic instrumentation for spotting
    memory growth or CPU spikes from loading large AutoGluon/XGBoost models,
    not something either predict route's behavior depends on.
    """
    cpu_percent = psutil.cpu_percent(interval=None)
    memory_info = psutil.virtual_memory()
    total_memory_mb = memory_info.total / (1024 ** 2)
    available_memory_mb = memory_info.available / (1024 ** 2)
    used_memory_mb = memory_info.used / (1024 ** 2)

    process = psutil.Process()
    process_memory_mb = process.memory_info().rss / (1024 ** 2)

    logger.info(
        f"[{stage}] CPU: {cpu_percent}% | "
        f"System memory total/available/used: {total_memory_mb:.2f}/{available_memory_mb:.2f}/{used_memory_mb:.2f} MB | "
        f"Process memory: {process_memory_mb:.2f} MB"
    )

def fetch_data_dynamically(url, column_suffix, skip_options=range(24, 30)):
    # USGS's RDB text format prefixes the real header/data rows with a
    # variable number of "#"-commented metadata lines (site name, agency
    # boilerplate, etc.) whose exact count isn't guaranteed to stay constant
    # across sites or over time -- so rather than hard-coding one skiprows
    # value, this just tries a small range of plausible values and takes the
    # first one that actually parses into a valid, non-empty column.
    for skip in skip_options:
        try:
            data = pd.read_csv(url, sep='\t', skiprows=skip, comment='#')
            data = data.dropna(how="all", axis=1)
            measurement_col = [col for col in data.columns if col.endswith(column_suffix)]
            if not measurement_col:
                raise ValueError(f"No column ending with {column_suffix} found.")
            measurement_col = measurement_col[0]

            quality_col = f"{measurement_col}_cd"
            if quality_col in data.columns:
                quality_values = data[quality_col].astype(str).str.strip().str.lower()
                bad_quality_mask = quality_values.str.contains(
                    "|".join(BAD_QUALITY_CODES), regex=True, na=False
                )
                if bad_quality_mask.any():
                    logger.warning(
                        f"Dropping {bad_quality_mask.sum()} row(s) from {url} flagged with "
                        f"quality codes {sorted(data.loc[bad_quality_mask, quality_col].astype(str).str.strip().unique())}."
                    )
                    data = data[~bad_quality_mask]

            data[measurement_col] = pd.to_numeric(data[measurement_col], errors='coerce')
            data = data.dropna(subset=['datetime', measurement_col])
            data = data[['datetime', measurement_col]].set_index('datetime')
            data.index = pd.to_datetime(data.index, errors='coerce')
            if data.index.isnull().any():
                raise ValueError("Datetime conversion failed.")
            return data, measurement_col
        except (ValueError, KeyError, IndexError, pd.errors.ParserError) as e:
            logger.warning(f"Failed with skiprows={skip} for {url}: {e}")
    raise ValueError(f"Failed to load data from {url} with any specified skiprows option.")

def fetch_latest_data(site_code, days=1):
    """Fetch and shape a recent window of live gage-height/discharge
    readings for the AutoGluon /predict route: pulls both parameters over
    the trailing `days`, merges them on shared timestamps, rejects the
    request if the newest reading is stale (see STALE_DATA_THRESHOLD_HOURS
    above), then reshapes into the long/melted item_id-series form
    TimeSeriesDataFrame.from_data_frame expects below.
    """
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    start_date_str = start_date.strftime('%Y-%m-%dT%H:%M:%S.000-05:00')
    end_date_str = end_date.strftime('%Y-%m-%dT%H:%M:%S.000-05:00')

    gage_url = f'https://waterservices.usgs.gov/nwis/iv/?sites={site_code}&parameterCd=00065&startDT={start_date_str}&endDT={end_date_str}&siteStatus=all&format=rdb'
    flow_url = f'https://waterservices.usgs.gov/nwis/iv/?sites={site_code}&parameterCd=00060&startDT={start_date_str}&endDT={end_date_str}&siteStatus=all&format=rdb'

    try:
        gage_data, gage_col_name = fetch_data_dynamically(gage_url, '_00065')
        flow_data, flow_col_name = fetch_data_dynamically(flow_url, '_00060')
    except ValueError as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch latest data: {e}")

    df = pd.merge(gage_data, flow_data, how='inner', left_index=True, right_index=True)
    df.columns = ['Gage', 'Flow']

    if df.empty:
        raise HTTPException(status_code=503, detail=f"No usable gage/flow readings found for site {site_code}.")

    latest_reading_age = end_date - df.index.max().to_pydatetime().replace(tzinfo=None)
    if latest_reading_age > timedelta(hours=STALE_DATA_THRESHOLD_HOURS):
        raise HTTPException(
            status_code=503,
            detail=(
                f"Latest reading for site {site_code} is {latest_reading_age} old, "
                f"exceeding the {STALE_DATA_THRESHOLD_HOURS}h staleness threshold. "
                "Refusing to predict on stale data."
            ),
        )

    df_resampled = df.resample('H').last()
    df_resampled['Gage_rate_of_change'] = df_resampled['Gage'].diff()
    df_resampled['Flow_rate_of_change'] = df_resampled['Flow'].diff()
    df_resampled.dropna(inplace=True)
    df_resampled = df_resampled.reset_index()

    long_df = df_resampled.melt(id_vars=['datetime'], 
                                value_vars=['Gage', 'Flow', 'Gage_rate_of_change', 'Flow_rate_of_change'],
                                var_name='item_id', value_name='series')
    item_id_map = {
        'Gage': 'gage', 
        'Flow': 'flow', 
        'Gage_rate_of_change': 'gage_rate_of_change', 
        'Flow_rate_of_change': 'flow_rate_of_change'
    }
    long_df['item_id'] = long_df['item_id'].map(item_id_map)
    long_df['datetime'] = pd.to_datetime(long_df['datetime'])
    return long_df

def load_model(site_code, forecast_length):
    # Two on-disk formats are supported because this repo's sites were
    # trained at different times with different AutoGluon export settings:
    # a TimeSeriesPredictor.load()-able directory (AutoGluon's own save
    # format, containing predictor.pkl plus its internal model files) for
    # some sites, and a single flat pickle file for others. Both are tried
    # here rather than picking one so existing models on disk keep working
    # regardless of which way they were originally saved.
    ag_model_dir = os.path.join(model_directory, f"{site_code}_model_{forecast_length}")
    pickle_model_path = os.path.join(model_directory, f"{site_code}_model_{forecast_length}.pkl")

    if os.path.isdir(ag_model_dir) and os.path.exists(os.path.join(ag_model_dir, "predictor.pkl")):
        return TimeSeriesPredictor.load(ag_model_dir, require_version_match=False)
    elif os.path.exists(pickle_model_path):
        with open(pickle_model_path, 'rb') as f:
            return pickle.load(f)
    else:
        raise FileNotFoundError(f"Model directory '{ag_model_dir}' or pickle file '{pickle_model_path}' not found.")

@app.get("/predict/{site_code}/{forecast_length}")
async def predict(site_code: str, forecast_length: int):
    try:
        log_system_usage("Start")
        model = load_model(site_code, forecast_length)
        long_df = fetch_latest_data(site_code)
        
        if long_df.empty or len(long_df) < forecast_length:
            raise HTTPException(status_code=400, detail="Not enough data for prediction.")
        
        feature_item_ids = ['gage', 'gage_rate_of_change', 'flow', 'flow_rate_of_change']
        feature_df = long_df[long_df['item_id'].isin(feature_item_ids)].copy()

        #print("Feature DataFrame:", feature_df)

        if feature_df.shape[0] < forecast_length * len(feature_item_ids):
            raise HTTPException(status_code=400, detail="Insufficient data for the required forecast length.")
        
        # Convert to TimeSeriesDataFrame for AutoGluon
        ts_df = TimeSeriesDataFrame.from_data_frame(feature_df, id_column="item_id", timestamp_column="datetime")
        
        #print("TimeSeriesDataFrame for prediction:", ts_df)

        log_system_usage("Before Prediction")
        
        # Make predictions
        pred_df = model.predict(ts_df)

        # Filter predictions specifically for 'gage'
        gage_pred_df = pred_df[pred_df.index.get_level_values("item_id") == "gage"]

        #print("Gage Prediction DataFrame:", gage_pred_df)
        
        # Extract the mean, lower, and upper bounds specifically for 'gage'.
        # "0.1"/"0.9" are the quantile levels this AutoGluon model was
        # trained to predict (an 80% interval), named as columns in its
        # output -- the positional iloc fallback below covers older saved
        # models whose predict() output didn't include named quantile
        # columns at all, only positional mean/lower/upper columns.
        if 'mean' in gage_pred_df.columns:
            pred_mean = gage_pred_df["mean"].values[-1]
            lower_bound = gage_pred_df["0.1"].values[-1]
            upper_bound = gage_pred_df["0.9"].values[-1]
        else:
            pred_mean = gage_pred_df.iloc[-1, 0]
            lower_bound = gage_pred_df.iloc[-1, 1]
            upper_bound = gage_pred_df.iloc[-1, 2]
        
        log_system_usage("After Prediction")
        
        return {
            "predicted_gage_height": round(pred_mean, 2),
            "confidence_interval": {
                "lower_bound": round(lower_bound, 2),
                "upper_bound": round(upper_bound, 2)
            }
        }

    except HTTPException as http_err:
        raise http_err
    except Exception:
        logger.error(f"Unexpected error predicting for site_code={site_code}, forecast_length={forecast_length}:\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail="An unexpected error occurred.")


@app.get("/predict/xgboost/{site_code}/{forecast_length}")
async def predict_xgboost(site_code: str, forecast_length: int):
    """Additive engine, separate from the AutoGluon /predict route above:
    serves models trained by xgboost_model/auto_pipeline.py and saved to
    xgboost_model/artifacts/ in XGBoost's native JSON format. Does not read
    or write anything the AutoGluon route touches.
    """
    try:
        log_system_usage("Start (xgboost)")
        result = xgboost_engine.predict(site_code, forecast_length)
        log_system_usage("After Prediction (xgboost)")
        return result
    except xgboost_engine.XGBoostModelError as e:
        raise HTTPException(status_code=e.status_code, detail=e.detail)
    except Exception:
        logger.error(
            f"Unexpected error in xgboost predict for site_code={site_code}, "
            f"forecast_length={forecast_length}:\n{traceback.format_exc()}"
        )
        raise HTTPException(status_code=500, detail="An unexpected error occurred.")


@app.get("/models/xgboost/{site_code}")
async def list_xgboost_models(site_code: str):
    """What horizons are actually available for a site -- lets a caller (or
    a human with curl) discover what auto_pipeline.py has trained without
    guessing forecast_length values against /predict/xgboost/.
    """
    horizons = xgboost_engine.available_horizons(site_code)
    if not horizons:
        raise HTTPException(
            status_code=404,
            detail=f"No XGBoost models found for site={site_code}. "
                   f"Train some with: python xgboost_model/auto_pipeline.py {site_code}",
        )
    return {"site_code": site_code, "available_horizons_hours": horizons}
