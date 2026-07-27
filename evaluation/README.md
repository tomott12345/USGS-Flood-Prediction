# Evaluation tooling

Offline scripts for checking model accuracy and testing feature ideas without
touching the production microservice or its models. All scripts fetch fresh
data from USGS/Open-Meteo at run time (nothing is cached locally).

## `usgs_data.py`
Shared historical data fetching. `fetch_site_history(site_code, days=90)`
returns hourly gage/flow + rate-of-change, built with the exact same feature
engineering as `microservice/app.py`'s `fetch_latest_data`, so results reflect
what the deployed model actually sees. `get_site_coordinates(site_code)` looks
up lat/lon from the USGS site service.

Note: USGS's instantaneous-values service generally only retains sub-daily
granularity for roughly the last 120 days.

## `backtest.py`
Walk-forward evaluation of every model currently checked into the repo
(`MODEL_REGISTRY`), against freshly-pulled recent history. For each origin
point, it predicts forward and compares against the true reading, matching
production's behavior of only ever serving the final step of the horizon.
Reports MAE/RMSE/NSE plus 80% confidence-interval coverage (calibration)
against the nominal 0.8.

```
python backtest.py
```

Defaults to a small number of folds (`max_folds`) per model to keep runtime
reasonable — increase it for a more statistically confident read, at the cost
of runtime (each fold re-runs `predictor.predict()`).

**Latest run turned up a real production bug**, not just a benchmarking
artifact: `models/01388500_model_1` (the model the microservice actually
serves) fails on every single fold. Its best-performing submodel
(`DirectTabular`) throws `TypeError: code() argument 13 must be str, not int`
while unpickling a numba-JIT-cached object — a Python-bytecode-version
mismatch between whatever Python trained the model and whatever Python loads
it (the repo's tracked `.pyc` was compiled for Python 3.10; this evaluation
ran under 3.11). Any deployment of this exact model on a different Python
minor version than it was trained with is at risk of the same failure.

## `precipitation.py`
`fetch_precipitation_for_site(site_code, past_days=21, forecast_days=2)` pulls
hourly precipitation at the gage's coordinates from Open-Meteo's free
forecast API (no key needed).

## `precip_hypothesis.py`
Before retraining any production model with precipitation as a covariate
(slow, and changes what's actually deployed), this trains a quick XGBoost
regressor twice on the same time-ordered split — once with only the features
the production models already use (gage/flow/rate-of-change lags), once with
those plus precipitation lag/rolling-sum features — and compares held-out
error.

```
python precip_hypothesis.py 01388500 01473730 08393610
```

**Result on the last ~3 weeks of data: precipitation did not help.** MAE was
flat-to-worse at all three sites (-27%, -3%, -6% "improvement", i.e. all
regressions) at a 6-hour horizon. Two likely reasons, not mutually exclusive:

- The test window was mostly dry (~8mm total over 10 days at Pompton), so
  there was little rainfall signal to learn from, while the extra features
  added enough dimensionality to a fairly small sample (~370 rows) to hurt
  more than help.
- Precipitation *at the gage's own coordinates* isn't necessarily what's
  driving the gage's rise — the relevant rainfall usually falls upstream in
  the contributing watershed, arriving at the gage after a travel-time lag
  this quick test doesn't model.

If this is worth revisiting: test over a window that actually includes a
storm/rise event, and source precipitation averaged over the upstream
watershed polygon (not a single point at the gage) with a lag matched to that
watershed's time of concentration.
