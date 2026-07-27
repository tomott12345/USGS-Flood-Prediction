# Bundles three things that otherwise require separate manual setup into one
# image: the Go web front end (webapp/), the Python training pipeline
# (xgboost_model/), and the FastAPI scoring microservice (microservice/).
# Pattern follows the sibling usgs-edge-app repo's dockerfile.optimized (a
# Go-builder stage + a Python-deps stage + a slim runtime stage), adapted to
# this repo's layout.
#
# Not optimized for minimal image size: microservice/requirements.txt is the
# single source of truth for Python dependencies used everywhere else in
# this repo (evaluation/, xgboost_model/, microservice/), including
# AutoGluon/torch for the existing AutoGluon route, so this installs that
# file as-is rather than maintaining a second, XGBoost-only requirements set
# that could drift from it. If image size matters more than that for your
# deployment, usgs-edge-app's dockerfile.optimized is a good reference for
# trimming it (CPU-only torch wheels, multi-stage layer pruning, etc).
#
# Build:  docker build -t usgs-flood-webapp .
# Run:    docker run -p 8080:8080 -p 8000:8000 usgs-flood-webapp
# Then:   open http://localhost:8080

# ---- Stage 1: Go binary --------------------------------------------------
FROM golang:1.22-alpine AS go-builder
WORKDIR /src
COPY webapp/go.mod ./webapp/go.mod
COPY webapp/*.go ./webapp/
COPY webapp/templates ./webapp/templates
COPY webapp/static ./webapp/static
WORKDIR /src/webapp
RUN CGO_ENABLED=0 go build -trimpath -o /out/webapp-server .

# ---- Stage 2: Python virtualenv with this repo's pinned dependencies -----
FROM python:3.11-slim AS python-deps
WORKDIR /build
COPY microservice/requirements.txt ./requirements.txt
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:${PATH}"
# requirements.txt pins numpy==2.3.5/scipy==1.16.3/contourpy==1.3.3, which
# require Python >=3.11 -- see .github/workflows/microservice-tests.yml for
# the same constraint hit (and fixed) in CI.
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

# ---- Stage 3: runtime ------------------------------------------------------
FROM python:3.11-slim
RUN useradd --create-home --uid 1000 appuser

WORKDIR /app
COPY --from=python-deps /opt/venv /opt/venv
COPY --from=go-builder /out/webapp-server /app/webapp-server

# Only what's actually needed at runtime -- not the whole repo (notebooks,
# .git, other sites' exploratory work).
COPY evaluation ./evaluation
COPY xgboost_model ./xgboost_model
COPY microservice ./microservice
COPY models ./models
COPY docker-entrypoint.sh /app/docker-entrypoint.sh

RUN chmod +x /app/docker-entrypoint.sh /app/webapp-server \
 && chown -R appuser:appuser /app

ENV PATH="/opt/venv/bin:${PATH}" \
    PYTHON_BIN=/opt/venv/bin/python3 \
    REPO_ROOT=/app \
    WEBAPP_PORT=8080 \
    MICROSERVICE_URL=http://localhost:8000

EXPOSE 8080 8000
USER appuser

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s \
  CMD python3 -c "import urllib.request; urllib.request.urlopen('http://localhost:8080/health', timeout=3)" || exit 1

ENTRYPOINT ["/app/docker-entrypoint.sh"]
