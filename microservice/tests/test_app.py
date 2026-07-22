import os
from datetime import datetime, timedelta

import pandas as pd
import pytest
from fastapi import HTTPException

import app

FIXTURES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures")
REPO_MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "..", "models")


def test_fetch_data_dynamically_parses_real_rdb_format():
    fixture_path = os.path.join(FIXTURES_DIR, "sample_gage_iv.rdb")

    data, col_name = app.fetch_data_dynamically(fixture_path, "_00065")

    assert col_name == "141166_00065"
    assert list(data[col_name]) == [2.34, 2.35, 2.36, 2.37]
    assert data.index.is_monotonic_increasing


def test_fetch_data_dynamically_raises_when_column_missing():
    fixture_path = os.path.join(FIXTURES_DIR, "sample_gage_iv.rdb")

    with pytest.raises(ValueError):
        app.fetch_data_dynamically(fixture_path, "_00060")


def test_fetch_data_dynamically_drops_bad_quality_rows():
    fixture_path = os.path.join(FIXTURES_DIR, "sample_gage_iv_with_bad_quality.rdb")

    data, col_name = app.fetch_data_dynamically(fixture_path, "_00065")

    # Rows flagged "P_Ice" (99.90) and "Eqp" (-999.0) must be dropped; only
    # the "P"/"A"-flagged readings should remain.
    assert list(data[col_name]) == [2.34, 2.36, 2.38]


def test_fetch_latest_data_raises_for_stale_data(monkeypatch):
    stale_index = pd.to_datetime(["2020-01-01 00:00", "2020-01-01 01:00"])

    def fake_fetch(url, column_suffix, skip_options=range(24, 30)):
        col = f"stale{column_suffix}"
        df = pd.DataFrame({col: [1.0, 2.0]}, index=stale_index)
        df.index.name = "datetime"
        return df, col

    monkeypatch.setattr(app, "fetch_data_dynamically", fake_fetch)

    with pytest.raises(HTTPException) as exc_info:
        app.fetch_latest_data("01388500")

    assert exc_info.value.status_code == 503


def test_fetch_latest_data_passes_for_fresh_data(monkeypatch):
    now = datetime.now()
    # Two distinct hourly buckets are needed so resample('H').diff() has a
    # non-null rate of change to survive the subsequent dropna().
    fresh_index = pd.to_datetime(
        [now - timedelta(hours=2), now - timedelta(hours=1), now - timedelta(minutes=15)]
    )

    def fake_fetch(url, column_suffix, skip_options=range(24, 30)):
        col = f"fresh{column_suffix}"
        df = pd.DataFrame({col: [1.0, 2.0, 3.0]}, index=fresh_index)
        df.index.name = "datetime"
        return df, col

    monkeypatch.setattr(app, "fetch_data_dynamically", fake_fetch)

    long_df = app.fetch_latest_data("01388500")

    assert set(long_df["item_id"].unique()) == {"gage", "flow", "gage_rate_of_change", "flow_rate_of_change"}


def test_load_model_fails_for_sample_autogluon_predictor_after_security_upgrade(monkeypatch):
    """This documents a real, known-accepted trade-off, not a bug to fix.

    The checked-in sample model was saved with AutoGluon 1.1.1. requirements.txt
    was bumped to autogluon.timeseries==1.5.0 to close CVEs (torch, transformers,
    scikit-learn, lightgbm, pytorch-lightning, ray -- see the Dependabot fix
    commit) that autogluon 1.1.1's own dependency pins made impossible to patch
    otherwise. That bump moved AutoGluon's internal module layout enough that
    this model's TimeSeriesPredictor.load() itself now fails (previously, under
    1.1.1, load() succeeded and only a *later* .predict() call failed -- see
    evaluation/README.md's numba/pickle finding). Either way the model was
    already non-functional for real predictions (0% success in
    evaluation/backtest.py); the security fix just moves the failure earlier
    and doesn't change the practical outcome for API callers (a 500 either way).
    See xgboost_model/ for the actual replacement.
    """
    monkeypatch.setattr(app, "model_directory", REPO_MODELS_DIR)

    with pytest.raises(Exception):
        app.load_model("01388500", 1)


def test_load_model_raises_for_unknown_site(monkeypatch):
    monkeypatch.setattr(app, "model_directory", REPO_MODELS_DIR)

    with pytest.raises(FileNotFoundError):
        app.load_model("99999999", 1)
