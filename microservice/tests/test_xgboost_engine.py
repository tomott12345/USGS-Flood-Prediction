import json
import os

import numpy as np
import pandas as pd
import pytest
from xgboost import XGBRegressor

import xgboost_engine


def _make_artifact(tmp_path, site_code, horizon_hours, feature_set="baseline", extra_metadata=None):
    """Train a tiny throwaway model against the exact feature columns
    build_features would produce, and save it via the same native-JSON
    format auto_pipeline.py uses -- so tests exercise the real load/predict
    path, not a mock of it.
    """
    from features import LAG_HOURS

    feature_columns = []
    for lag in LAG_HOURS:
        feature_columns += [f"gage_lag_{lag}", f"flow_lag_{lag}", f"gage_roc_lag_{lag}", f"flow_roc_lag_{lag}"]

    rng = np.random.RandomState(0)
    X = pd.DataFrame(rng.rand(50, len(feature_columns)), columns=feature_columns)
    y = rng.rand(50)
    model = XGBRegressor(n_estimators=5, max_depth=2)
    model.fit(X, y)

    model_dir = tmp_path / f"{site_code}_h{horizon_hours}"
    model_dir.mkdir(parents=True)
    model.save_model(str(model_dir / "model.json"))

    metadata = {
        "engine": "xgboost",
        "format": "xgboost_native_json",
        "site_code": site_code,
        "horizon_hours": horizon_hours,
        "feature_set": feature_set,
        "feature_columns": feature_columns,
        "upstream_site_codes": [],
        "conformal_margin": 0.05,
        "alpha": 0.2,
        "nominal_coverage": 0.8,
    }
    if extra_metadata:
        metadata.update(extra_metadata)
    with open(model_dir / "metadata.json", "w") as f:
        json.dump(metadata, f)

    return str(model_dir), feature_columns


def _synthetic_history(hours=80):
    """A plausible fresh hourly Gage/Flow history ending "now" -- fresh
    enough to pass the staleness check, long enough to fill every lag
    feature build_features computes.
    """
    now = pd.Timestamp.now().floor("h")
    index = pd.date_range(end=now, periods=hours, freq="h")
    gage = pd.Series(np.linspace(5.0, 6.0, hours), index=index)
    flow = pd.Series(np.linspace(100.0, 150.0, hours), index=index)
    df = pd.DataFrame({"Gage": gage, "Flow": flow})
    df["Gage_rate_of_change"] = df["Gage"].diff()
    df["Flow_rate_of_change"] = df["Flow"].diff()
    return df.dropna()


def test_available_horizons_lists_trained_sites(tmp_path, monkeypatch):
    _make_artifact(tmp_path, "01234567", 1)
    _make_artifact(tmp_path, "01234567", 6)
    _make_artifact(tmp_path, "01234567", 3)
    monkeypatch.setattr(xgboost_engine, "ARTIFACTS_DIR", str(tmp_path))

    assert xgboost_engine.available_horizons("01234567") == [1, 3, 6]


def test_available_horizons_empty_for_unknown_site(tmp_path, monkeypatch):
    monkeypatch.setattr(xgboost_engine, "ARTIFACTS_DIR", str(tmp_path))
    assert xgboost_engine.available_horizons("00000000") == []


def test_load_artifact_raises_404_for_missing_model(tmp_path, monkeypatch):
    monkeypatch.setattr(xgboost_engine, "ARTIFACTS_DIR", str(tmp_path))

    with pytest.raises(xgboost_engine.XGBoostModelError) as exc_info:
        xgboost_engine.load_artifact("00000000", 1)
    assert exc_info.value.status_code == 404


def test_load_artifact_defaults_missing_keys_for_legacy_train_py_metadata(tmp_path, monkeypatch):
    """train.py (pre-dating auto_pipeline.py) never wrote feature_set/format/
    upstream_site_codes -- older artifacts on disk shouldn't KeyError."""
    model_dir, _ = _make_artifact(tmp_path, "01234567", 1)
    legacy_metadata_path = os.path.join(model_dir, "metadata.json")
    with open(legacy_metadata_path) as f:
        metadata = json.load(f)
    for key in ("feature_set", "format", "upstream_site_codes", "engine"):
        metadata.pop(key, None)
    with open(legacy_metadata_path, "w") as f:
        json.dump(metadata, f)

    monkeypatch.setattr(xgboost_engine, "ARTIFACTS_DIR", str(tmp_path))

    model, loaded_metadata = xgboost_engine.load_artifact("01234567", 1)
    assert loaded_metadata["feature_set"] == "baseline"
    assert loaded_metadata["upstream_site_codes"] == []


def test_predict_returns_expected_shape_for_baseline_model(tmp_path, monkeypatch):
    _make_artifact(tmp_path, "01234567", 1, feature_set="baseline")
    monkeypatch.setattr(xgboost_engine, "ARTIFACTS_DIR", str(tmp_path))
    monkeypatch.setattr(xgboost_engine, "fetch_site_history", lambda site_code, days: _synthetic_history())

    result = xgboost_engine.predict("01234567", 1)

    assert result["site_code"] == "01234567"
    assert result["horizon_hours"] == 1
    assert isinstance(result["predicted_gage_height"], float)
    assert result["confidence_interval"]["lower_bound"] < result["predicted_gage_height"] < result["confidence_interval"]["upper_bound"]
    assert result["model"]["feature_set"] == "baseline"


def test_predict_raises_503_for_stale_data(tmp_path, monkeypatch):
    _make_artifact(tmp_path, "01234567", 1)
    monkeypatch.setattr(xgboost_engine, "ARTIFACTS_DIR", str(tmp_path))

    stale_history = _synthetic_history()
    stale_history.index = stale_history.index - pd.Timedelta(hours=6)
    monkeypatch.setattr(xgboost_engine, "fetch_site_history", lambda site_code, days: stale_history)

    with pytest.raises(xgboost_engine.XGBoostModelError) as exc_info:
        xgboost_engine.predict("01234567", 1)
    assert exc_info.value.status_code == 503


def test_predict_raises_503_when_lag_features_cannot_be_filled(tmp_path, monkeypatch):
    """Only a couple hours of history -- nowhere near enough for a 24h lag
    feature -- should fail clearly rather than predict on NaNs."""
    _make_artifact(tmp_path, "01234567", 1)
    monkeypatch.setattr(xgboost_engine, "ARTIFACTS_DIR", str(tmp_path))
    monkeypatch.setattr(xgboost_engine, "fetch_site_history", lambda site_code, days: _synthetic_history(hours=3))

    with pytest.raises(xgboost_engine.XGBoostModelError) as exc_info:
        xgboost_engine.predict("01234567", 1)
    assert exc_info.value.status_code == 503


def test_predict_uses_enriched_feature_builder_and_upstream_codes(tmp_path, monkeypatch):
    """Doesn't hit the network -- just confirms an "enriched" model routes
    through build_combined_features with the exact upstream_site_codes
    metadata stored, rather than baseline's build_features."""
    from features import LAG_HOURS

    baseline_columns = []
    for lag in LAG_HOURS:
        baseline_columns += [f"gage_lag_{lag}", f"flow_lag_{lag}", f"gage_roc_lag_{lag}", f"flow_roc_lag_{lag}"]
    feature_columns = baseline_columns + ["upstream_gage_mean_lag_1"]

    rng = np.random.RandomState(0)
    X = pd.DataFrame(rng.rand(50, len(feature_columns)), columns=feature_columns)
    y = rng.rand(50)
    model = XGBRegressor(n_estimators=5, max_depth=2)
    model.fit(X, y)

    model_dir = tmp_path / "01234567_h1"
    model_dir.mkdir(parents=True)
    model.save_model(str(model_dir / "model.json"))
    metadata = {
        "feature_set": "enriched",
        "feature_columns": feature_columns,
        "upstream_site_codes": ["01111111", "02222222"],
        "conformal_margin": 0.05,
        "nominal_coverage": 0.8,
        "format": "xgboost_native_json",
        "trained_at": "2026-01-01T00:00:00+00:00",
    }
    with open(model_dir / "metadata.json", "w") as f:
        json.dump(metadata, f)

    monkeypatch.setattr(xgboost_engine, "ARTIFACTS_DIR", str(tmp_path))
    monkeypatch.setattr(xgboost_engine, "fetch_site_history", lambda site_code, days: _synthetic_history())

    captured = {}

    def fake_build_combined_features(site_code, history_df, days, upstream_site_codes=None):
        captured["upstream_site_codes"] = upstream_site_codes
        features = pd.DataFrame(index=history_df.index)
        for col in feature_columns:
            features[col] = 1.0
        return features

    monkeypatch.setattr(xgboost_engine, "build_combined_features", fake_build_combined_features)

    result = xgboost_engine.predict("01234567", 1)

    assert captured["upstream_site_codes"] == ["01111111", "02222222"]
    assert result["model"]["feature_set"] == "enriched"
    assert result["model"]["upstream_site_codes"] == ["01111111", "02222222"]
