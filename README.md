# Data Driven Flood Prediction at a USGS streamgage nodes

Flood prediction is a very important task for emergency response officials, engineers, research scientists, and hydrologists. Timely forecasts are of the utmost importance in protecting life and property. Several [simulations](https://www.nssl.noaa.gov/projects/flash/#:~:text=The%20Flooded%20Locations%20And%20Simulated,saving%20lives%20and%20protecting%20infrastructure.) exist that can [forecast several hours in advance](https://water.noaa.gov/about/nwm) for a particular streamgage or node in a watershed. These simulations are a physics driven model approach as opposed to a data driven model, which we analyze here.

The purpose for a data driven approach is to evaluate whether or not flood prediction models can trained and deployed at the edge (on the streamgage) that will allow researchers to remotely turn on additional monitoring tools if the machine learning model inferences a large rise in streamgage height will occur over a predefined time window.

There are two subset folders in this repository. All of them use a combination of open source algorithms ranging from NeuralProphet, XGBoost, and Autogluon. The subset folders are broken out into two locations in the NJ and PA area: The Pompton River crossing in Pompton Plans NJ (site #01388500) and the Schuylkill River at Conshohocken, PA (site #01473730). Two additional sites exist for development purposes.

The inspiration for this analysis comes from the paper [Data-Driven Flood Alert System (FAS) Using Extreme Gradient Boosting (XGBoost) to Forecast Flood Stages](https://www.researchgate.net/publication/358910939_Data-Driven_Flood_Alert_System_FAS_Using_Extreme_Gradient_Boosting_XGBoost_to_Forecast_Flood_Stages).

## What's new: an XGBoost replacement, auto-trainable for any site, with a browser front end

The original production model (AutoGluon, served by `microservice/`) turned out to fail to unpickle across Python/numba versions (see `evaluation/README.md`). That prompted three rounds of work, all now in this repo:

1. **A native-format XGBoost replacement** (`xgboost_model/`) with conformal-calibrated confidence intervals, optional upstream-gage and weather enrichment, and hyperparameter tuning -- saved via XGBoost's own stable JSON format instead of pickle. Full write-up, including the "what did and didn't help" evidence, in `xgboost_model/README.md`.
2. **`auto_pipeline.py`**, which turns what used to be per-site manual trial and error into one command for *any* USGS streamgage: discover upstream gages, auto-pick the best feature set per forecast horizon, train, save. Wired into the microservice as an additive route and into a **new Go web app** (`webapp/`) for doing all of that from a browser instead of the command line.
3. **CI and a Docker image** so both of those are exercised automatically rather than only manually: GitHub Actions builds the Docker image and smoke-tests it, and runs the Python and Go test suites, on every push and pull request.

## Quickstart

### Prerequisites

- **Python 3.11+** -- `microservice/requirements.txt` pins `numpy`/`scipy`/`contourpy` versions that require it; Python 3.10 cannot resolve these dependencies at all (this is exactly what `.github/workflows/microservice-tests.yml` checks on every PR).
- **Go 1.22+** -- only needed to run `webapp/` outside Docker; skip if you're only using the CLI or the Docker image.
- **git**

### Install

```
git clone https://github.com/tomott12345/USGS-Flood-Prediction.git
cd USGS-Flood-Prediction
python3 -m venv .venv && source .venv/bin/activate
pip install -r microservice/requirements.txt
```

`microservice/requirements.txt` is the one dependency file for the whole repo -- it covers the microservice itself, `xgboost_model/`'s training/evaluation scripts, and everything `webapp/` shells out to.

### Train a model for any streamgage

```
python xgboost_model/auto_pipeline.py 01388500
```

Discovers upstream gages via USGS's NLDI navigation API, auto-selects the best feature set per forecast horizon by comparing held-out accuracy, and saves the result to `xgboost_model/artifacts/{site_code}_h{horizon}/` in XGBoost's native JSON format. See "Training and deploying a new site" below for what this actually does and the options it takes.

### Serve it

```
cd microservice
uvicorn app:app
```

```
curl "http://localhost:8000/predict/xgboost/01388500/1"
curl "http://localhost:8000/models/xgboost/01388500"   # what horizons are trained
```

### Or do all of the above from a browser

```
cd webapp
go run .
```

Open `http://localhost:8080`, enter a site code, watch it train live, then click "Verify" to confirm the microservice is actually serving the new model. See `webapp/README.md` for configuration.

### Or run everything in one container

```
docker build -t usgs-flood-webapp .
docker run -p 8080:8080 -p 8000:8000 usgs-flood-webapp
```

See "Running everything in Docker" below.

## What's in this repo

- `usgs-streamgage-*/` -- the original per-site notebooks (AutoGluon, XGBoost prototypes, NeuralProphet) that this project started from.
- `models/` -- the production AutoGluon model(s) served by `microservice/`.
- `evaluation/` -- offline scripts for checking model accuracy and testing feature ideas (backtesting, precipitation hypothesis testing) against fresh USGS data, without touching production. See `evaluation/README.md`; it also documents the AutoGluon pickle/numba version-mismatch failure that motivated the XGBoost replacement below.
- `xgboost_model/` -- the XGBoost replacement for AutoGluon: conformal-calibrated point forecasts, upstream-gage and weather enrichment, hyperparameter tuning, and `auto_pipeline.py`, a single command that trains and exports a deployable model for any USGS streamgage. See `xgboost_model/README.md` for the full write-up, including why native XGBoost JSON was chosen over AutoGluon's pickle format.
- `microservice/` -- the FastAPI service. Serves the original AutoGluon models via `/predict/{site_code}/{forecast_length}`, and also serves `auto_pipeline.py`'s XGBoost models via an additive `/predict/xgboost/{site_code}/{forecast_length}` route. See `microservice/readme.md`.
- `webapp/` -- a browser front end for the whole flow: enter a USGS site code, watch `auto_pipeline.py` train and `charts.py` render live over Server-Sent Events, then click "verify live scoring" to prove the microservice is serving the new model. Go standard library only, no JS framework, no CDN. See `webapp/README.md`.
- `Dockerfile` / `docker-entrypoint.sh` -- bundles `webapp/`, the Python training pipeline, and the microservice into one image, following the pattern in the sibling `usgs-edge-app` repo's `dockerfile.optimized`.
- `.github/workflows/` -- CI: see "Continuous integration" below.

## Training and deploying a new site

Previously, adding a new streamgage meant hand-picking upstream gages and manually comparing feature sets per horizon (see `xgboost_model/README.md`'s "trial and error"). That's now one command:

```
python xgboost_model/auto_pipeline.py <site_code>
```

Given a USGS site code (e.g. `01388500`), this discovers genuinely upstream gages (crossing tributary boundaries, via USGS's NLDI navigation API), auto-selects between plain lag features and upstream+weather-enriched, tuned features per forecast horizon based on which one actually performs better on held-out data for that site, and saves the winning model per horizon to `xgboost_model/artifacts/{site_code}_h{horizon}/` in XGBoost's native JSON format (portable across library/Python versions, unlike AutoGluon's pickle -- see `evaluation/README.md`). A `{site_code}_manifest.json` records which feature set was chosen per horizon and why.

Once trained, the model is immediately servable -- the microservice reads from `xgboost_model/artifacts/` on every request, no caching or restart needed:

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

## Running everything in Docker

The repo-root `Dockerfile` builds a single image containing:

- the Go `webapp/` binary,
- a Python virtualenv with `microservice/requirements.txt` installed (this covers `xgboost_model/` too), and
- `xgboost_model/`, `microservice/`, `evaluation/`, and `models/` copied in as-is.

```
docker build -t usgs-flood-webapp .
docker run -p 8080:8080 -p 8000:8000 usgs-flood-webapp
```

`docker-entrypoint.sh` starts the FastAPI microservice (port 8000) and the Go webapp (port 8080) together inside the container and stops both if either exits. Once it's up:

- `http://localhost:8080` -- the web app (train a new site, view charts, verify live scoring).
- `http://localhost:8000/predict/xgboost/01388500/1` -- the microservice directly.

Models trained through the running container are written inside the container's filesystem (`/app/xgboost_model/artifacts/`) and are not persisted once the container is removed -- mount a volume over that path (`-v ./xgboost_model/artifacts:/app/xgboost_model/artifacts`) if you want trained models to survive a container restart or to seed the image with models trained outside it.

This image is not optimized for size (it installs the same AutoGluon+torch dependency stack the rest of the repo already uses, so the AutoGluon route keeps working unchanged) -- see the Dockerfile's own header comment for the tradeoff, and the sibling `usgs-edge-app` repo's `dockerfile.optimized` for a leaner reference if that matters for your deployment.

## Continuous integration

Four GitHub Actions workflows run on every push to `main` and every pull request (`.github/workflows/`):

| Workflow | What it checks |
|---|---|
| `microservice-tests.yml` | Installs `microservice/requirements.txt` under Python 3.11 and runs the pytest suite (`microservice/tests/`) -- covers the AutoGluon route's data parsing/staleness handling and the XGBoost route's serving logic, both against mocked USGS/network calls. |
| `webapp-tests.yml` | `gofmt`, `go vet`, `go build`, and `go test ./... -race` for `webapp/` -- covers input validation, manifest parsing, the microservice HTTP client, job subscribe/replay semantics, and subprocess line-buffering, entirely offline. |
| `docker-build.yml` | Builds the repo-root `Dockerfile`, starts the resulting image, and smoke-tests both services inside it -- the webapp's `/health` route, and the microservice's `/predict/xgboost/01388500/1` route against the models already committed under `xgboost_model/artifacts/`. A green run means the image doesn't just build, it actually serves a real prediction. |
| `auto-pipeline-tests.yml` | Runs `python xgboost_model/auto_pipeline.py 01388500` for real -- unlike the other three workflows, this one makes live calls to USGS/NLDI/OpenMeteo rather than mocking them, then verifies the manifest reports a successfully trained horizon and that its model/metadata files were actually written. Because it depends on external services, it's the one workflow here that can fail for reasons outside this repo's control; see the workflow file's own header comment before assuming a failure is a real regression. |

## Troubleshooting

A few gotchas that have come up running this repo locally, in case you hit the same thing:

- **`docker build` says `Dockerfile: no such file or directory`.** The `Dockerfile` lives at the repo root, not inside `webapp/`. If you `cd`'d into `webapp/` to run `go run .` earlier in the same session, `cd ..` back to the repo root before `docker build -t usgs-flood-webapp .`.
- **`webapp`'s `/train` page fails with `ModuleNotFoundError: No module named 'sklearn'` (or similar) when run outside Docker.** `go run .`/the compiled `webapp-server` binary shells out to whatever Python `PYTHON_BIN` points to (default: `python3` on `PATH`) -- it doesn't install anything for you. Create a venv and `pip install -r microservice/requirements.txt` into it, then point `PYTHON_BIN` at that venv's `python3` (e.g. `PYTHON_BIN=$(pwd)/.venv/bin/python3 go run ./webapp`) rather than relying on an ambient system Python.
- **A `/predict` or `/verify` call fails with a "stale data" 503 even though USGS shows current readings.** Both prediction routes compare the current time against USGS's station-local (US Eastern) timestamps using a timezone-naive clock check, so this only works correctly if the machine's own clock is set to US Eastern. The Docker image sets this for you (`ENV TZ=America/New_York` in the `Dockerfile`); running `uvicorn app:app` directly on a non-Eastern host will need the same fix. See `microservice/readme.md`'s "Known limitation" section for the full explanation.
