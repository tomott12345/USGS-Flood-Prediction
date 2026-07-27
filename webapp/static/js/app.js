// Vanilla JS only -- deliberately no framework, no bundler, no CDN fetch.
// Everything this app needs (SSE via the browser's native EventSource,
// fetch, template literals) has been in every evergreen browser for years,
// so there is nothing here to vendor or keep in sync with a build step.

function wireTrainForm() {
  var form = document.getElementById("train-form");
  var errorEl = document.getElementById("train-error");
  if (!form) return;

  form.addEventListener("submit", function (event) {
    event.preventDefault();
    errorEl.hidden = true;

    var formData = new FormData(form);
    fetch("/api/train", { method: "POST", body: formData })
      .then(function (resp) {
        if (!resp.ok) {
          return resp.json().then(function (body) {
            throw new Error(body.error || ("request failed: " + resp.status));
          });
        }
        return resp.json();
      })
      .then(function (body) {
        window.location.href = "/jobs/" + encodeURIComponent(body.id);
      })
      .catch(function (err) {
        errorEl.textContent = err.message;
        errorEl.hidden = false;
      });
  });
}

function watchJob(jobId, siteCode) {
  var logEl = document.getElementById("log");
  var badgeEl = document.getElementById("status-badge");
  var errorEl = document.getElementById("error-panel");
  var resultsEl = document.getElementById("results-panel");

  var source = new EventSource("/jobs/" + encodeURIComponent(jobId) + "/stream");

  source.addEventListener("log", function (event) {
    logEl.textContent += event.data + "\n";
    logEl.scrollTop = logEl.scrollHeight;
  });

  source.addEventListener("status", function (event) {
    var status = JSON.parse(event.data);
    setStatusBadge(badgeEl, status.status);
    source.close();

    if (status.status === "failed") {
      errorEl.textContent = status.error || "Training failed -- see the log above for details.";
      errorEl.hidden = false;
      return;
    }

    if (status.status === "succeeded") {
      fetchSiteSummary(siteCode)
        .then(function (summary) {
          resultsEl.innerHTML = renderSiteResultsHTML(siteCode, summary);
          wireVerifyButtons(resultsEl);
        })
        .catch(function (err) {
          errorEl.textContent = "Training finished but results couldn't be loaded: " + err.message;
          errorEl.hidden = false;
        });
    }
  });

  source.onerror = function () {
    // EventSource retries connections on its own; this just guards against
    // silently spinning forever if the server has genuinely gone away.
  };
}

function setStatusBadge(el, status) {
  el.innerHTML = '<span class="badge badge-' + status + '">' + status + "</span>";
}

function fetchSiteSummary(siteCode) {
  return fetch("/api/sites/" + encodeURIComponent(siteCode) + "/summary").then(function (resp) {
    if (!resp.ok) {
      return resp.json().then(function (body) {
        throw new Error(body.error || ("status " + resp.status));
      });
    }
    return resp.json();
  });
}

function renderSiteResultsHTML(siteCode, summary) {
  var manifest = summary.manifest;
  var chartFiles = summary.chart_files || [];
  var horizons = Object.keys(manifest.horizons)
    .map(Number)
    .sort(function (a, b) { return a - b; });

  var rows = horizons
    .map(function (h) {
      var r = manifest.horizons[String(h)];
      if (r.status !== "ok") {
        return "<tr><td>" + h + "h</td><td>" + r.status + "</td><td>&mdash;</td><td>&mdash;</td><td>&mdash;</td><td>&mdash;</td><td>&mdash;</td></tr>";
      }
      return (
        "<tr>" +
        "<td>" + h + "h</td>" +
        "<td>ok</td>" +
        "<td>" + escapeHTML(r.feature_set) + "</td>" +
        "<td>" + r.selection.baseline_mae_ft.toFixed(4) + "</td>" +
        "<td>" + r.selection.enriched_tuned_mae_ft.toFixed(4) + "</td>" +
        "<td>" + r.conformal_margin_ft.toFixed(4) + "</td>" +
        "<td>" +
        '<button class="button button-small verify-btn" data-site="' + escapeHTML(siteCode) + '" data-horizon="' + h + '">Verify</button>' +
        '<div class="verify-result" data-horizon-result="' + h + '"></div>' +
        "</td>" +
        "</tr>"
      );
    })
    .join("");

  var chartsHTML = chartFiles.length
    ? '<div class="chart-grid">' +
      chartFiles
        .map(function (f) {
          var url = "/charts/" + encodeURIComponent(siteCode) + "/" + encodeURIComponent(f);
          return (
            '<a href="' + url + '" target="_blank" rel="noopener">' +
            '<img src="' + url + '" alt="' + escapeHTML(f) + '" loading="lazy">' +
            '<div class="chart-caption">' + escapeHTML(f) + "</div>" +
            "</a>"
          );
        })
        .join("") +
      "</div>"
    : '<p class="muted">No charts rendered for this site yet.</p>';

  return (
    "<h2>Results</h2>" +
    '<table class="results-table"><thead><tr>' +
    "<th>Horizon</th><th>Status</th><th>Feature set</th><th>Baseline MAE (ft)</th>" +
    "<th>Enriched+tuned MAE (ft)</th><th>Conformal margin (ft)</th><th>Live scoring</th>" +
    "</tr></thead><tbody>" +
    rows +
    "</tbody></table>" +
    "<h2>Charts</h2>" +
    chartsHTML +
    '<p><a href="/sites/' + encodeURIComponent(siteCode) + '">View this site’s permanent page &rarr;</a></p>'
  );
}

function wireVerifyButtons(scopeEl) {
  var buttons = scopeEl.querySelectorAll(".verify-btn");
  buttons.forEach(function (btn) {
    btn.addEventListener("click", function () {
      var site = btn.getAttribute("data-site");
      var horizon = btn.getAttribute("data-horizon");
      var resultEl = scopeEl.querySelector('.verify-result[data-horizon-result="' + horizon + '"]');
      btn.disabled = true;
      resultEl.className = "verify-result";
      resultEl.textContent = "Checking live scoring…";

      fetch("/api/sites/" + encodeURIComponent(site) + "/verify?horizon=" + encodeURIComponent(horizon), {
        method: "POST",
      })
        .then(function (resp) {
          return resp.json().then(function (body) {
            if (!resp.ok) throw new Error(body.error || ("status " + resp.status));
            return body;
          });
        })
        .then(function (result) {
          resultEl.className = "verify-result ok";
          resultEl.textContent =
            "Live: " + result.predicted_gage_height + " ft " +
            "(" + result.confidence_interval.lower_bound + "–" + result.confidence_interval.upper_bound + " ft, " +
            Math.round(result.confidence_interval.nominal_coverage * 100) + "% CI) " +
            "as of " + result.as_of + " — engine: " + result.model.feature_set;
        })
        .catch(function (err) {
          resultEl.className = "verify-result error";
          resultEl.textContent = "Verify failed: " + err.message;
        })
        .finally(function () {
          btn.disabled = false;
        });
    });
  });
}

function escapeHTML(s) {
  var div = document.createElement("div");
  div.textContent = s;
  return div.innerHTML;
}
