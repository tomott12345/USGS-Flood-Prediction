# USGS Flood Forecasting — web app

A browser front end for training and deploying per-site XGBoost flood
forecast models. It wraps three things that already exist in this repo and
work on their own:

- `xgboost_model/auto_pipeline.py` — discovers upstream gages, auto-selects
  a feature set per horizon, trains, and saves a model for any USGS
  streamgage site code.
- `xgboost_model/charts.py` — renders forecast/lead-time charts for a
  trained site.
- `microservice/app.py`'s additive XGBoost route — serves whatever
  `auto_pipeline.py` has saved, reading straight from disk on every request.

This app does not reimplement any of that. It runs the first two as
subprocesses, streaming their output to the browser live over
Server-Sent Events while they run (a cold run — no cached upstream-gage
scan yet — can take several minutes), and calls the third over plain HTTP
to prove a newly trained model is really being served. There is no separate
"deploy" step to perform: `auto_pipeline.py` already writes straight into
`xgboost_model/artifacts/`, exactly where the microservice's XGBoost route
already looks by default, and it re-reads from disk on every request — no
caching, no restart required.

Zero external dependencies by design: Go's standard library only
(`net/http`, `html/template`, `embed`, `os/exec`) — no router, no ORM —
and vanilla JavaScript in the browser (native `EventSource`/`fetch`, no
framework, no CDN). Nothing here needs a build step or an internet
connection to run once the two `pip install`s below are done.

## Running in development

Requires: Go 1.22+, and this repo's existing Python setup (Python 3.11+;
`pip install -r ../microservice/requirements.txt` covers everything both
`xgboost_model/` and `microservice/` need, including the extra weather/
upstream-gage packages `xgboost_model/requirements.txt` lists).

```
cd webapp
go run .
```

By default it: listens on `:8080`, auto-detects the repo root by walking
up from the working directory looking for `xgboost_model/` and
`microservice/`, shells out to `python3` on your `PATH`, and expects the
microservice at `http://localhost:8000` (start it separately: `cd
../microservice && uvicorn app:app`).

Then open `http://localhost:8080`, click "Train a new site," enter a USGS
site code (e.g. `01388500`), and watch it train live.

If you're running the microservice standalone in dev (not via the
Dockerfile, which sets this for you) rather than in a container, make sure
its host's system clock is set to US Eastern time -- see
`microservice/readme.md`'s "Known limitation" section. Otherwise the
"Verify" button will fail with a false "stale data" error even though the
model trained fine.

### Configuration (all optional — every env var has the default above)

| Env var | Default | Purpose |
|---|---|---|
| `WEBAPP_PORT` | `8080` | Port this app listens on. |
| `REPO_ROOT` | auto-detected | Override if auto-detection picks the wrong directory. |
| `PYTHON_BIN` | `python3` | Interpreter used to run `auto_pipeline.py`/`charts.py`. Point this at a venv's `python3` if you're not using an ambient install. |
| `MICROSERVICE_URL` | `http://localhost:8000` | Base URL of the FastAPI microservice, for the "verify live scoring" button. |
| `MICROSERVICE_TIMEOUT_SECONDS` | `30` | HTTP timeout for calls to the microservice. |

## Running via Docker

The repo-root `Dockerfile` bundles this app, a Python virtualenv with
`microservice/requirements.txt` installed, and both services into one
image, following the pattern in the sibling `usgs-edge-app` repo's
`dockerfile.optimized`:

```
docker build -t usgs-flood-webapp -f ../Dockerfile ..
docker run -p 8080:8080 -p 8000:8000 usgs-flood-webapp
```

(or from the repo root: `docker build -t usgs-flood-webapp .`)

`docker-entrypoint.sh` starts the microservice and this app together and
stops both if either exits. This isn't optimized for image size — see the
Dockerfile's own header comment for why, and where to look if that matters
for your deployment.

## Routes

| Route | Purpose |
|---|---|
| `GET /` | Dashboard — every site trained so far, read from `xgboost_model/artifacts/*_manifest.json`. |
| `GET /train` | Form: site code, forecast horizons, training window. |
| `POST /api/train` | Starts a training job, returns `{"id": "..."}`. |
| `GET /jobs/{id}` | Live progress page for one job. |
| `GET /jobs/{id}/stream` | Server-Sent Events: log lines as they're produced, then a final status event. |
| `GET /sites/{site}` | Permanent page for a previously trained site: per-horizon results table, chart gallery, live-verify buttons. |
| `POST /api/sites/{site}/verify?horizon=N` | Calls the microservice's `/predict/xgboost/{site}/{N}` and returns the result — the "deploy" proof. |
| `GET /charts/{site}/{file}` | Serves one chart PNG from `xgboost_model/charts/`. |
| `GET /health` | Health check. |

## Tests

```
cd webapp
go test ./...
```

Covers input validation, manifest parsing, the microservice HTTP client
(against an `httptest.Server`, no real network), the job manager's
subscribe/replay semantics, and the subprocess line-buffering logic
(against a throwaway shell script, so these don't require Python or this
repo's ML dependencies to run).
