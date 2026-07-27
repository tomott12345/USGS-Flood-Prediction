# Data Driven Flood Prediction at a USGS streamgage nodes

Flood prediction is a very important task for emergency response officials, engineers, research scientists, and hydrologists. Timely forecasts are of the utmost importance in protecting life and property. Several [simulations](https://www.nssl.noaa.gov/projects/flash/#:~:text=The%20Flooded%20Locations%20And%20Simulated,saving%20lives%20and%20protecting%20infrastructure.) exist that can [forecast several hours in advance](https://water.noaa.gov/about/nwm) for a particular streamgage or node in a watershed. These simulations are a physics driven model approach as opposed to a data driven model, which we analyze here.

The purpose for a data driven approach is to evaluate whether or not flood prediction models can trained and deployed at the edge (on the streamgage) that will allow researchers to remotely turn on additional monitoring tools if the machine learning model inferences a large rise in streamgage height will occur over a predefined time window.

There are two subset folders in this repository. All of them use a combination of open source algorithms ranging from NeuralProphet, XGBoost, and Autogluon. The subset folders are broken out into two locations in the NJ and PA area: The Pompton River crossing in Pompton Plans NJ (site #01388500) and the Schuylkill River at Conshohocken, PA (site #01473730). Two additional sites exist for development purposes. 

The inspiration for this analysis comes from the paper [Data-Driven Flood Alert System (FAS) Using Extreme Gradient Boosting (XGBoost) to Forecast Flood Stages](https://www.researchgate.net/publication/358910939_Data-Driven_Flood_Alert_System_FAS_Using_Extreme_Gradient_Boosting_XGBoost_to_Forecast_Flood_Stages). 

## Repository layout

- `usgs-streamgage-*/` -- the original per-site notebooks (AutoGluon, XGBoost prototypes, NeuralProphet) that this project started from.
- `models/` -- the production AutoGluon model(s) served by `microservice/`.
- `evaluation/` -- offline scripts for checking model accuracy and testing feature ideas (backtesting, precipitation hypothesis testing) against fresh USGS data, without touching production. See `evaluation/README.md`; it also documents the AutoGluon pickle/numba version-mismatch failure that motivated the XGBoost replacement below.
- `xgboost_model/` -- the XGBoost replacement for AutoGluon: conformal-calibrated point forecasts, upstream-gage and weather enrichment, hyperparameter tuning, and (new) `auto_pipeline.py`, a single command that trains and exports a deployable model for any USGS streamgage. See `xgboost_model/README.md` for the full write-up, including why native XGBoost JSON was chosen over AutoGluon's pickle format.
- `microservice/` -- the FastAPI service. Serves the original AutoGluon models via `/predict/{site_code}/{forecast_length}`, and (new) also serves `auto_pipeline.py`'s XGBoost models via an additive `/predict/xgboost/{site_code}/{forecast_length}` route. See `microservice/readme.md`.
- `webapp/` -- (new) a browser front end for the whole flow: enter a USGS site code, watch `auto_pipeline.py` train and `charts.py` render live over Server-Sent Events, then click "verify live scoring" to prove the microservice is serving the new model. Go standard library only, no JS framework, no CDN. See `webapp/README.md`.
- `Dockerfile` / `docker-entrypoint.sh` -- (new) bundles `webapp/`, the Python training pipeline, and the microservice into one image, following the pattern in the sibling `usgs-edge-app` repo's `dockerfile.optimized`.

## Training and deploying a new site

Previously, adding a new streamgage meant hand-picking upstream gages and manually comparing feature sets per horizon (see `xgboost_model/README.md`'s "trial and error"). That's now one command:

```
python xgboost_model/auto_pipeline.py <site_code>
```

Given a USGS site code (e.g. `01388500`), this discovers genuinely upstream gages (crossing tributary boundaries, via USGS's NLDI navigation API), auto-selects between plain lag features and upstream+weather-enriched, tuned features per forecast horizon based on which one actually performs better on held-out data for that site, and saves the winning model per horizon to `xgboost_model/artifacts/{site_code}_h{horizon}/` in XGBoost's native JSON format (portable across library/Python versions, unlike AutoGluon's pickle -- see `evaluation/README.md`). A `{site_code}_manifest.json` records which feature set was chosen per horizon and why.

Once trained, the model is immediately servable:

```
curl "http://localhost:8000/predict/xgboost/01388500/1"
curl "http://localhost:8000/models/xgboost/01388500"   # what horizons are trained
```

This has been run end-to-end against the Pompton River (01388500) and the Susquehanna River above the dam at Sunbury, PA (01553990) -- the latter also surfaced and fixed a real gap where sites that never report discharge (impoundment/dam gages) previously failed outright instead of falling back to gage-height-only features.

### Or do all of this from a browser

`webapp/` wraps the same command line flow in a web UI: enter a site code, watch training and chart rendering happen live, then click a button to prove the microservice is actually serving the new model.

```
cd webapp && go run .          # http://localhost:8080
```

See `webapp/README.md` for configuration and the Docker-packaged version.

