#!/bin/bash
# Starts the FastAPI microservice (background) and the Go webapp (foreground
# under this script's control), and stops both together on shutdown. This is
# a small step up from the sibling usgs-edge-app's `./usgs_app & uvicorn ...`
# (fire-and-forget, no shutdown coordination): here, either process exiting
# stops the container, and a container stop signal is propagated to both.
#
# A real process supervisor (tini + s6, supervisord, etc.) would be a more
# robust choice for production -- e.g. automatic restart of just the
# process that died -- this is intentionally simple rather than pulling in
# another dependency for a two-process container.
set -euo pipefail

cleanup() {
  echo "shutting down..."
  kill "${MICROSERVICE_PID:-}" "${WEBAPP_PID:-}" 2>/dev/null || true
}
trap cleanup TERM INT

(cd /app/microservice && exec uvicorn app:app --host 0.0.0.0 --port 8000) &
MICROSERVICE_PID=$!
echo "microservice started (pid $MICROSERVICE_PID)"

/app/webapp-server &
WEBAPP_PID=$!
echo "webapp started (pid $WEBAPP_PID)"

# Exit as soon as either process does, propagating its exit code -- a
# training/scoring container where the scoring half silently died but the
# web front end kept serving stale "everything's fine" pages is worse than
# the whole container visibly stopping.
wait -n "$MICROSERVICE_PID" "$WEBAPP_PID"
EXIT_CODE=$?
cleanup
exit "$EXIT_CODE"
