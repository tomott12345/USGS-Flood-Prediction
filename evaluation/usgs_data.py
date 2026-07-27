"""Shared historical USGS data fetching for offline evaluation (backtesting,
precipitation hypothesis testing). Separate from microservice/app.py because
that module fetches only the *latest* window for live serving; this one
fetches arbitrary historical windows and skips the staleness check that only
makes sense for real-time predictions.
"""
from datetime import datetime, timedelta

import pandas as pd

# Same "untrustworthy reading" codes as microservice/app.py -- kept in sync
# manually since the two modules serve different purposes (live serving vs.
# offline evaluation) and don't share a runtime dependency.
BAD_QUALITY_CODES = {"ice", "eqp", "mnt", "dis", "bkw", "rat"}


def _read_rdb(url, skip_options=range(16, 35)):
    for skip in skip_options:
        try:
            data = pd.read_csv(url, sep="\t", skiprows=skip, comment="#")
            data = data.dropna(how="all", axis=1)
            if "agency_cd" not in data.columns:
                raise ValueError("agency_cd column missing; skiprows likely misaligned")
            return data
        except (ValueError, KeyError, IndexError, pd.errors.ParserError):
            continue
    raise ValueError(f"Failed to parse RDB response from {url}")


def get_site_coordinates(site_code):
    """Return (lat, lon) for a USGS site code, needed to look up precipitation."""
    url = f"https://waterservices.usgs.gov/nwis/site/?sites={site_code}&format=rdb"
    data = _read_rdb(url)
    # RDB includes a units-spec row (e.g. "16s") right after the header, which
    # isn't skipped by skiprows since it's not a comment line -- drop it here.
    data["dec_lat_va"] = pd.to_numeric(data["dec_lat_va"], errors="coerce")
    data["dec_long_va"] = pd.to_numeric(data["dec_long_va"], errors="coerce")
    data = data.dropna(subset=["dec_lat_va", "dec_long_va"])
    row = data.iloc[0]
    return float(row["dec_lat_va"]), float(row["dec_long_va"])


def fetch_historical_series(site_code, parameter_cd, column_suffix, start_date, end_date):
    """Fetch one parameter (gage height or flow) over an arbitrary historical window."""
    start_str = start_date.strftime("%Y-%m-%dT%H:%M:%S.000-05:00")
    end_str = end_date.strftime("%Y-%m-%dT%H:%M:%S.000-05:00")
    url = (
        f"https://waterservices.usgs.gov/nwis/iv/?sites={site_code}&parameterCd={parameter_cd}"
        f"&startDT={start_str}&endDT={end_str}&siteStatus=all&format=rdb"
    )

    for skip in range(24, 30):
        try:
            data = pd.read_csv(url, sep="\t", skiprows=skip, comment="#")
            data = data.dropna(how="all", axis=1)
            measurement_col = [c for c in data.columns if c.endswith(column_suffix)]
            if not measurement_col:
                raise ValueError(f"No column ending with {column_suffix} found.")
            measurement_col = measurement_col[0]

            quality_col = f"{measurement_col}_cd"
            if quality_col in data.columns:
                quality_values = data[quality_col].astype(str).str.strip().str.lower()
                bad_mask = quality_values.str.contains("|".join(BAD_QUALITY_CODES), regex=True, na=False)
                data = data[~bad_mask]

            data[measurement_col] = pd.to_numeric(data[measurement_col], errors="coerce")
            data = data.dropna(subset=["datetime", measurement_col])
            data = data[["datetime", measurement_col]].set_index("datetime")
            data.index = pd.to_datetime(data.index, errors="coerce")
            if data.index.isnull().any():
                raise ValueError("Datetime conversion failed.")
            return data.rename(columns={measurement_col: "value"})
        except (ValueError, KeyError, IndexError, pd.errors.ParserError):
            continue
    raise ValueError(f"Failed to load data from {url} with any specified skiprows option.")


def fetch_site_history(site_code, days=90):
    """Merged, resampled gage/flow history with rate-of-change features, mirroring
    the exact feature engineering in microservice/app.py's fetch_latest_data so
    backtest results reflect what the deployed model actually sees.

    Note: USGS's instantaneous-values (iv) service typically only retains
    sub-daily granularity for the last ~120 days; requesting a longer window
    will simply return less data than requested rather than erroring.
    """
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)

    gage = fetch_historical_series(site_code, "00065", "_00065", start_date, end_date)
    flow = fetch_historical_series(site_code, "00060", "_00060", start_date, end_date)

    df = pd.merge(gage, flow, how="inner", left_index=True, right_index=True, suffixes=("_gage", "_flow"))
    df.columns = ["Gage", "Flow"]
    df_resampled = df.resample("h").last()
    df_resampled["Gage_rate_of_change"] = df_resampled["Gage"].diff()
    df_resampled["Flow_rate_of_change"] = df_resampled["Flow"].diff()
    df_resampled.dropna(inplace=True)
    return df_resampled
