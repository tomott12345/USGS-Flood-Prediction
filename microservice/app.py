from fastapi import FastAPI, HTTPException
import pandas as pd
import numpy as np
import psutil
from datetime import datetime, timedelta
import os
from autogluon.timeseries import TimeSeriesPredictor, TimeSeriesDataFrame
import pickle

app = FastAPI()

model_directory = "../models"

def log_system_usage(stage=""):
    cpu_percent = psutil.cpu_percent(interval=None)
    memory_info = psutil.virtual_memory()
    total_memory_mb = memory_info.total / (1024 ** 2)  
    available_memory_mb = memory_info.available / (1024 ** 2)  
    used_memory_mb = memory_info.used / (1024 ** 2)  

    process = psutil.Process()
    process_memory_mb = process.memory_info().rss / (1024 ** 2)  

    print(f"[{stage}] CPU Usage: {cpu_percent}%")
    print(f"[{stage}] Total System Memory: {total_memory_mb:.2f} MB")
    print(f"[{stage}] Available Memory: {available_memory_mb:.2f} MB")
    print(f"[{stage}] System Used Memory: {used_memory_mb:.2f} MB")
    print(f"[{stage}] Process Memory Usage: {process_memory_mb:.2f} MB")

def fetch_data_dynamically(url, column_suffix, skip_options=range(24, 30)):
    for skip in skip_options:
        try:
            data = pd.read_csv(url, sep='\t', skiprows=skip, comment='#')
            data = data.dropna(how="all", axis=1)
            measurement_col = [col for col in data.columns if col.endswith(column_suffix)]
            if not measurement_col:
                raise ValueError(f"No column ending with {column_suffix} found.")
            data[measurement_col[0]] = pd.to_numeric(data[measurement_col[0]], errors='coerce')
            data = data.dropna(subset=['datetime', measurement_col[0]])
            data = data[['datetime', measurement_col[0]]].set_index('datetime')
            data.index = pd.to_datetime(data.index, errors='coerce')
            if data.index.isnull().any():
                raise ValueError("Datetime conversion failed.")
            return data, measurement_col[0]
        except (ValueError, KeyError, IndexError, pd.errors.ParserError) as e:
            print(f"Failed with skiprows={skip} for {url}: {e}")
    raise ValueError(f"Failed to load data from {url} with any specified skiprows option.")

def fetch_latest_data(site_code, days=1):
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
        
        # Extract the mean, lower, and upper bounds specifically for 'gage'
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
    except Exception as e:
        print(f"Unexpected server error: {e}")
        raise HTTPException(status_code=500, detail="An unexpected error occurred.")
