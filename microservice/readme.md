# Working Microservice

This code is a simple FastAPI microservice that can be used to serve the autogluon models trained on the various streamgage nodes. You will need to install all the Python libraries (see requirements.txt) and place all the autogluon models in a ../models folder. A sample model has been trained and saved to the ../models folder for your testing. 

Note: the microservice does some pretty intense data munging to get the data shaped into the series format that Autogluon likes. Inspect the sample streamgage notebooks to understand how the data is shaped. 

To run the mircorservice simply pass the following command via the command line `uvicorn app:app` and then point your browser to http://localhost:8000/predict/{site_code}/{forecast_length}, where the example model site_code = 01388500, and forecast_length = 1 ; http://localhost:8000/predict/01388500/1

You can also use curl as well such as `curl -X GET "http://localhost:8000/predict/01388500/1"`

## XGBoost engine (additive, separate models)

`app.py` also serves models trained by `xgboost_model/auto_pipeline.py` --
train one for any USGS streamgage with a single command:

```
python ../xgboost_model/auto_pipeline.py 01388500
```

That discovers upstream gages, auto-picks baseline vs. upstream+weather-
enriched features per horizon based on held-out accuracy, and saves the
result to `../xgboost_model/artifacts/{site_code}_h{horizon}/` as XGBoost's
native JSON format (not pickle -- see `xgboost_model/README.md` for why
that matters). Then:

```
curl "http://localhost:8000/predict/xgboost/01388500/1"
curl "http://localhost:8000/models/xgboost/01388500"   # what horizons are trained
```

This is a completely separate route and model store from the AutoGluon
`/predict/{site_code}/{forecast_length}` endpoint above -- neither reads nor
writes anything the other touches. `XGB_ARTIFACTS_DIR` (default
`../xgboost_model/artifacts`) overrides where it looks for trained models,
mirroring `MODEL_DIRECTORY` for the AutoGluon models.

Currently AutoGluon and this additive XGBoost engine are the only model types supported.

## Known limitation: the server's clock must be set to US Eastern time

Both routes' staleness check (`fetch_latest_data` in `app.py`, `_build_live_feature_row`
in `xgboost_engine.py`) compares `datetime.now()` against the timestamps USGS returns,
which reflect each site's actual local time (these sites are all NJ/PA, i.e. US Eastern) --
neither call is timezone-aware. This works as long as the machine running the service has
its system clock set to US Eastern, and silently breaks otherwise: on a server set to UTC
(the default for most base Docker images, including `python:3.11-slim`), every live reading
looks ~4-5 hours old and every `/predict` request fails with a false "stale data" 503, even
though the underlying USGS data is current. The repo-root `Dockerfile` fixes this for the
containerized deployment by setting the container's timezone to `America/New_York`; if
you're running `uvicorn app:app` directly on a host, make sure that host's clock is set the
same way, or these routes will fail in exactly this way.