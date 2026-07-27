# XGBoost replacement (prototype, Pompton only)

Why this exists: the production AutoGluon model (`models/01388500_model_1`)
fails to unpickle in this environment (see `evaluation/README.md` for the
traceback) -- a Python-bytecode/numba version mismatch baked into AutoGluon's
persistence format. This directory is a from-scratch replacement candidate
built on plain XGBoost, whose native `save_model()`/`load_model()` format
(JSON) is specifically designed to be stable across library and Python
versions.

Status: prototype for one site (Pompton, 01388500), not wired into
`microservice/`. Nothing here touches `models/` or the production service.

**Scope is now <=6h.** Every horizon past 6h showed negative NSE (worse than
just predicting the mean) for every engine tried -- AutoGluon, plain XGBoost,
tuned+enriched XGBoost -- regardless of feature set or tuning (see the
tuning section below). Default horizon lists across `train.py`,
`evaluate_fixed.py`, `backtest.py`, and `charts.py` are `[1, 3, 6]`. The 12h/
24h/48h numbers earlier in this README are kept as-is since they're the
evidence for that decision, not because they're still an active target.

## Modules

- `data.py` -- thin re-export of `evaluation/usgs_data.py`'s `fetch_site_history`.
- `features.py` -- lag-based feature engineering (gage/flow/rate-of-change lags).
- `conformal.py` -- split conformal prediction for the confidence intervals.
- `train.py` -- trains + saves a model per horizon to `artifacts/{site}_h{horizon}/`
  (native XGBoost JSON + a `metadata.json` with the conformal margin and feature list).
- `evaluate_fixed.py` -- the calibration-correctness check: train once /
  calibrate once / evaluate on a genuinely held-out future test slice.
- `backtest.py` -- an earlier walk-forward-retrain variant, kept for
  reference; see the calibration note below for why `evaluate_fixed.py`
  is the one to trust for CI coverage.
- `charts.py` -- forecast-with-band charts and cross-engine comparison charts
  (saved to `charts/`).

## The confidence interval fix

The first attempt (two independently-trained `reg:quantileerror` models for
the 0.1/0.9 bounds, see `evaluation/xgboost_prototype.py`) badly under-covered
-- as low as 11% empirical coverage against an 80% nominal target. Splitting
a **large enough, chronologically-separate** calibration slice and using
**split conformal prediction** (`conformal.py`) fixes this: `pred +/- margin`,
where `margin` is the finite-sample-corrected empirical quantile of absolute
residuals measured on held-out data. Verified coverage on Pompton, train
once / calibrate once / evaluate on a later held-out slice
(`evaluate_fixed.py`, `days=45`):

| horizon | MAE (ft) | NSE | CI coverage (nominal 80%) |
|---|---|---|---|
| 1h | 0.029 | 0.97 | 98% |
| 3h | 0.044 | 0.92 | 98% |
| 6h | 0.076 | 0.78 | 97% |
| 12h | 0.132 | 0.44 | 94% |
| 24h | 0.243 | -0.21 | 89% |
| 48h | 0.259 | -0.35 | 95% |

Coverage now sits at or above nominal everywhere -- the safe direction for a
flood alert system -- though it runs conservative (wider than strictly
necessary) rather than tight. An early walk-forward-retrain variant
(`backtest.py`) that recomputed a *tiny* calibration set at every fold still
under-covered; the fix was giving conformal a properly-sized, representative
calibration set, not the conformal method itself.

Point-forecast accuracy past 12h is weak (NSE goes negative at 24h/48h,
meaning the model does worse than just predicting the mean) -- see
`charts/forecast_h24.png` for what that looks like against a real rise event:
the interval still bounds the actual value, but the point forecast misses the
rise entirely. AutoGluon's ensembling is modestly more accurate at every
horizon where it actually loads (see `charts/error_by_horizon.png`), but it
can't load at all at 48h, and the production model fails at every horizon in
this environment. XGBoost trades some accuracy for actually running.

## Upstream gages + weather (`upstream.py`, `weather.py`) -- built, but doesn't help yet

Prompted by a real gap found while reading `charts/forecast_h6.png`: the model's
6h-ahead forecast doesn't anticipate a rise at all, it only starts moving
*after* the rise is already visible in its own lag features -- because every
feature is a lag of the target site's own past readings, there's nothing that
could give genuine advance warning.

**`upstream.py`** finds genuinely upstream gages via USGS's NLDI navigation
API in "UT" (upstream-with-tributaries) mode, which correctly crosses river
boundaries -- for Pompton (formed by the confluence of the Ramapo, Pequannock,
and Wanaque Rivers), it returns gages on all three, not just one mainstem.
Results are cached to `artifacts/upstream_cache/` since the scan takes ~90s.

**`weather.py`** ports the OpenMeteo integration pattern from the sibling
`usgs-edge-app` repo (same cached/retried `openmeteo_requests` client, same
endpoints, same variable set: temperature, precipitation, soil temperature,
soil moisture -- soil moisture matters because saturated ground sheds rain as
runoff much faster than dry ground absorbs it).

**Empirical result, `feature_set="enriched"` in `evaluate_fixed_model`
(`baseline_vs_enriched.csv`): worse at every horizon**, not better -- e.g. at
6h, MAE goes from 0.076ft to 0.173ft and NSE drops from 0.78 to 0.55.

**Correction:** an earlier version of this README claimed the July 21 rise
was rain-free and likely a reservoir release. That was wrong, and traced to a
real bug, not the physical event: `weather.py` originally fetched "historical"
weather via `api.open-meteo.com/v1/forecast`'s `past_days` parameter, which
turns out to serve forecast-*model* nowcasts for recent past days, not
observation-corrected data -- and that model simply missed this storm. Cross-
checking against `archive-api.open-meteo.com` (the real ERA5-Land-based
reanalysis, ingests actual observations, and turned out to have no meaningful
serving latency -- same-day data is available) shows **0.48in of rain at
Pompton on July 21, peaking at 4pm, starting around 8am -- before the gage
even began rising at 9-10am.** The rain was real, and it did precede the
rise. `weather.py` now fetches from `archive-api.open-meteo.com` by default
(see `fetch_weather_archive`); the two endpoints also use different soil
variable names for the same quantities (`soil_moisture_0_to_7cm` vs
`soil_moisture_0_to_1cm`), which silently produced all-NaN soil columns until
that was fixed too. Separately, checking the Ramapo gage just above the
Pompton Lakes dam (`01387998`) shows a smooth, gradual rise (10.18ft ->
10.75ft over the day) tracking the rainfall's timing, not the sudden step a
gate-opening release would produce -- reinforcing that this was a real,
natural, rain-driven flood, not a managed release.

So there *was* genuine leading signal available, and the earlier "nothing
could have helped" conclusion was wrong. But re-running the comparison with
corrected data still doesn't flip the result on its own -- enriched (untuned)
remains worse at every horizon.

## Tuning (`tuning.py`) -- helps a lot at short horizons, not at long ones

`tuning.py` searches XGBoost regularization (`max_depth`, `reg_lambda`,
`reg_alpha`, `min_child_weight`, `subsample`, `colsample_bytree`) and feature
count together, using an *inner* validation split carved out of the training
window only -- `evaluate_fixed_model`'s calibration and test slices are never
touched by the search. `evaluate_fixed_model(..., tune=True)` wires it in.
Full numbers in `baseline_vs_enriched_tuned.csv`:

| horizon | baseline MAE | enriched MAE | enriched+tuned MAE | tuned n_features |
|---|---|---|---|---|
| 1h | 0.030 | 0.057 | **0.055** | 16 |
| 3h | 0.045 | 0.106 | **0.072** | 16 |
| 6h | 0.076 | 0.173 | **0.112** | 20 |
| 12h | 0.134 | 0.145 | 0.189 (worse) | 20 |
| 24h | 0.242 | 0.261 | 0.272 (worse) | 20 |
| 48h | 0.266 | 0.307 | 0.387 (worse) | 30 |

Tuning substantially narrows the gap at 1h/3h/6h (roughly halving the
untuned-enriched error toward baseline, though still not quite matching it),
cutting feature count down to 16-20 of the 50 available. At 12h/24h/48h it
makes things *worse* than even the untuned enriched model. Two likely
reasons, not mutually exclusive: (1) the search uses a single inner
train/validation split rather than cross-validation, so at longer horizons
-- where fewer effective training examples remain after the larger lag/target
shift -- the winning hyperparameters are more a fit to that one validation
slice's noise than a genuinely better config; (2) baseline itself has
negative NSE at 24h/48h (worse than predicting the mean), meaning there's
very little real signal left to find at those horizons regardless of feature
set or tuning -- no amount of regularization fixes a horizon where the
underlying autoregressive signal has already decayed.

**Bottom line:** the upstream+weather data is now correct and does contain
real predictive value, particularly at short horizons where a storm's
leading edge is still informative. Realizing that value took real
regularization/feature-selection tuning, not just concatenating everything
in -- and even then it narrows the gap at 1-6h rather than closing it, while
actively hurting at 12h+. Further gains would likely need cross-validated
(not single-split) hyperparameter search, and probably a fundamentally
different approach for 24h+ given how little signal even the baseline
finds there.

## Regenerating

```
python train.py 01388500 1 3 6 12 24 48
python evaluate_fixed.py 01388500 1 3 6 12 24 48
python charts.py
```

## Next steps, if this gets extended

- Wire up other sites (Schuylkill, Rio Hondo) -- only Pompton has been done.
- The 24h/48h point forecasts need real feature work (this is still the
  original lag-only feature set from `evaluation/precip_hypothesis.py`) --
  precipitation didn't help in that test, but wasn't tried at the longer
  horizons or with upstream/watershed-averaged rainfall.
- Tighten the conformal margins if the current conservatism proves overly
  wide in practice (e.g. adaptive conformal inference, which adjusts alpha
  online based on recent coverage, instead of a fixed margin).
- Wire into `microservice/app.py` as an alternate/fallback engine once
  accuracy at longer horizons is addressed.
