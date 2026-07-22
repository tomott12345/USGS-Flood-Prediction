"""Precipitation fetching for exogenous-feature experimentation.

Uses Open-Meteo's free forecast API (no API key required). Its `past_days`
parameter conveniently covers recent history in the same call that returns
the forecast horizon, which pairs well with USGS's own short retention window
for sub-daily readings.
"""
import pandas as pd
import requests

from usgs_data import get_site_coordinates

OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"
MAX_PAST_DAYS = 92  # Open-Meteo's limit for the `past_days` parameter


def fetch_precipitation(lat, lon, past_days=21, forecast_days=2):
    if past_days > MAX_PAST_DAYS:
        raise ValueError(f"past_days must be <= {MAX_PAST_DAYS} (Open-Meteo limit); got {past_days}")

    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": "precipitation",
        "past_days": past_days,
        "forecast_days": forecast_days,
        "timezone": "UTC",
    }
    response = requests.get(OPEN_METEO_URL, params=params, timeout=30)
    response.raise_for_status()
    hourly = response.json()["hourly"]

    return pd.DataFrame({
        "datetime": pd.to_datetime(hourly["time"]),
        "precipitation_mm": hourly["precipitation"],
    }).set_index("datetime")


def fetch_precipitation_for_site(site_code, past_days=21, forecast_days=2):
    lat, lon = get_site_coordinates(site_code)
    return fetch_precipitation(lat, lon, past_days=past_days, forecast_days=forecast_days)
