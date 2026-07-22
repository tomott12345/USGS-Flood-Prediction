"""OpenMeteo weather integration, ported from the sibling usgs-edge-app
repo's training/data_pipeline.py for consistency with that project's
approach (same cached/retried client, same variable set).

Adds soil moisture and soil temperature to the feature set, not just
precipitation -- soil saturation is a physically meaningful leading
indicator for flood risk (saturated ground sheds rain as runoff far faster
than dry ground absorbs it), which evaluation/precipitation.py (precipitation
only, built earlier in this project) doesn't capture.

IMPORTANT: for *historical* data, use fetch_weather_archive (archive-api.open-
meteo.com), not fetch_weather_forecast's past_days. A real storm on 2026-07-21
(0.48in at Pompton, confirmed against USGS gage response) was verified
present in archive-api but read back as *zero* through the forecast API's
past_days parameter and badly underreported (0.13in) through historical-
forecast-api -- both appear to serve forecast-model nowcasts for recent past
days rather than the gauge/satellite-corrected reanalysis archive-api
provides, and missed this localized convective event entirely. archive-api
also turned out to have no meaningful latency (same-day data available), so
there's no live-serving reason to prefer the less accurate endpoints.
"""
import os

import openmeteo_requests
import pandas as pd
import requests_cache
from retry_requests import retry

from data import get_site_coordinates

# Output column names (used everywhere downstream, e.g. features.py).
WEATHER_COLS = ["temperature_2m", "precipitation", "soil_temperature", "soil_moisture"]

# The archive (reanalysis) and forecast-model endpoints use different native
# variable names/depth bins for soil data -- requesting the wrong one for a
# given endpoint silently returns all-None, not an error. Map each endpoint's
# real variable name to the common output name above.
_ARCHIVE_SOURCE_COLS = {
    "temperature_2m": "temperature_2m",
    "precipitation": "precipitation",
    "soil_temperature": "soil_temperature_0_to_7cm",
    "soil_moisture": "soil_moisture_0_to_7cm",
}
_FORECAST_SOURCE_COLS = {
    "temperature_2m": "temperature_2m",
    "precipitation": "precipitation",
    "soil_temperature": "soil_temperature_0cm",
    "soil_moisture": "soil_moisture_0_to_1cm",
}

CACHE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".cache_openmeteo")


def _openmeteo_client():
    cache_session = requests_cache.CachedSession(CACHE_PATH, expire_after=3600)
    retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
    return openmeteo_requests.Client(session=retry_session)


def _parse_hourly_response(response, source_cols):
    # openmeteo_requests' response format: variables come back as parallel
    # arrays (Variables(0), Variables(1), ...) in the same order they were
    # requested in "hourly", with no column names attached -- that's why the
    # request and response both iterate source_cols.values() in the same
    # order, and why get the *output* names (source_cols.keys()) to label them.
    hourly = response.Hourly()
    output_names = list(source_cols.keys())
    vals = [hourly.Variables(i).ValuesAsNumpy() for i in range(len(output_names))]
    data = {
        "datetime": pd.date_range(
            start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
            end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
            freq=pd.Timedelta(seconds=hourly.Interval()),
            inclusive="left",
        )
    }
    for name, arr in zip(output_names, vals):
        data[name] = arr
    df = pd.DataFrame(data).set_index("datetime")
    df.index = df.index.tz_convert(None)  # tz-naive, matching fetch_site_history's index
    return df


def fetch_weather_archive(lat, lon, start_date, end_date):
    """The actual observation-corrected historical reanalysis (ERA5-Land-
    based) -- use this, not fetch_weather_forecast's past_days, for anything
    that needs to reflect what really happened. start_date/end_date:
    'YYYY-MM-DD' strings.
    """
    openmeteo = _openmeteo_client()
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": lat, "longitude": lon,
        "start_date": start_date, "end_date": end_date,
        "hourly": list(_ARCHIVE_SOURCE_COLS.values()),
        "temperature_unit": "fahrenheit", "wind_speed_unit": "mph",
        "precipitation_unit": "inch", "timezone": "auto",
    }
    responses = openmeteo.weather_api(url, params=params)
    return _parse_hourly_response(responses[0], _ARCHIVE_SOURCE_COLS)


def fetch_weather_forecast(lat, lon, forecast_days=2, past_days=0):
    """Recent history (past_days, max 92) + forecast (forecast_days) in one
    call. For genuine forecast_days only -- see the module docstring for why
    past_days shouldn't be trusted for anything historical.
    """
    openmeteo = _openmeteo_client()
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat, "longitude": lon,
        "hourly": list(_FORECAST_SOURCE_COLS.values()),
        "forecast_days": forecast_days, "past_days": past_days,
        "temperature_unit": "fahrenheit", "wind_speed_unit": "mph",
        "precipitation_unit": "inch", "timezone": "auto",
    }
    responses = openmeteo.weather_api(url, params=params)
    return _parse_hourly_response(responses[0], _FORECAST_SOURCE_COLS)


def fetch_weather_for_site(site_code, days=60, forecast_days=0):
    """Weather covering the same recent window as fetch_site_history(site_code,
    days=days), using the accurate archive API for everything up through
    today, and only falling back to the forecast endpoint for genuine future
    forecast_days (archive-api obviously can't provide those).
    """
    import datetime as dt

    lat, lon = get_site_coordinates(site_code)

    end_date = dt.date.today()
    start_date = end_date - dt.timedelta(days=days)
    historical = fetch_weather_archive(lat, lon, start_date.isoformat(), end_date.isoformat())

    if forecast_days <= 0:
        return historical

    forecast = fetch_weather_forecast(lat, lon, forecast_days=forecast_days, past_days=0)
    # The forecast call's own hourly range starts at today, overlapping the
    # tail of `historical` -- trim to strictly-after so the two don't
    # duplicate rows for today itself when concatenated.
    forecast = forecast[forecast.index > historical.index.max()]
    return pd.concat([historical, forecast])


if __name__ == "__main__":
    import sys
    site_code = sys.argv[1] if len(sys.argv) > 1 else "01388500"
    df = fetch_weather_for_site(site_code, days=14)
    print(df.shape)
    print(df.head())
    print(df.tail())
