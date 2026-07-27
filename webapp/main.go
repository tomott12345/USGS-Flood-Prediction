// Command usgs-flood-webapp is a browser front end for training and
// deploying per-site XGBoost flood-forecast models in this repo.
//
// It wraps three things that already exist and work on their own:
//   - xgboost_model/auto_pipeline.py -- discovers upstream gages, trains,
//     and saves a model for any USGS streamgage site code.
//   - xgboost_model/charts.py -- renders forecast/lead-time charts for a
//     trained site.
//   - microservice/app.py's additive XGBoost route -- serves whatever
//     auto_pipeline.py has saved, reading from disk on every request.
//
// This app doesn't reimplement any of that logic; it runs the first two as
// subprocesses (streaming their output live to the browser over
// Server-Sent Events) and calls the third over plain HTTP to prove a newly
// trained model is really being served. See webapp/README.md for how to
// run it, in dev and via the repo-root Dockerfile.
package main

import (
	"context"
	"embed"
	"errors"
	"html/template"
	"io/fs"
	"log"
	"net/http"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"time"
)

//go:embed templates/*.html
var templateFS embed.FS

//go:embed static
var staticFS embed.FS

type App struct {
	cfg  Config
	tmpl *template.Template
	jobs *JobManager
	msvc *MicroserviceClient
}

func main() {
	cfg, err := loadConfig()
	if err != nil {
		log.Fatalf("config error: %v", err)
	}
	log.Printf("repo root: %s", cfg.RepoRoot)
	log.Printf("python bin: %s", cfg.PythonBin)
	log.Printf("microservice url: %s", cfg.MicroserviceURL)

	tmpl, err := template.New("").Funcs(templateFuncs).ParseFS(templateFS, "templates/*.html")
	if err != nil {
		log.Fatalf("parsing templates: %v", err)
	}

	app := &App{
		cfg:  cfg,
		tmpl: tmpl,
		jobs: newJobManager(),
		msvc: newMicroserviceClient(cfg.MicroserviceURL, cfg.MicroserviceTimeout),
	}

	mux := http.NewServeMux()
	app.routes(mux)

	addr := ":" + cfg.Port
	log.Printf("listening on %s", addr)
	if err := http.ListenAndServe(addr, mux); err != nil {
		log.Fatal(err)
	}
}

func (a *App) routes(mux *http.ServeMux) {
	mux.HandleFunc("GET /{$}", a.handleDashboard)
	mux.HandleFunc("GET /train", a.handleTrainForm)
	mux.HandleFunc("POST /api/train", a.handleAPITrain)
	mux.HandleFunc("GET /jobs/{id}", a.handleJobPage)
	mux.HandleFunc("GET /jobs/{id}/stream", a.handleJobStream)
	mux.HandleFunc("GET /jobs/{id}/status", a.handleJobStatus)
	mux.HandleFunc("GET /sites/{site}", a.handleSiteDetail)
	mux.HandleFunc("GET /api/sites/{site}/summary", a.handleSiteSummary)
	mux.HandleFunc("POST /api/sites/{site}/verify", a.handleVerify)
	mux.HandleFunc("GET /charts/{site}/{file}", a.handleChartFile)
	mux.HandleFunc("GET /health", a.handleHealth)

	staticSub, err := fs.Sub(staticFS, "static")
	if err != nil {
		log.Fatalf("static assets: %v", err)
	}
	mux.Handle("GET /static/", http.StripPrefix("/static/", http.FileServer(http.FS(staticSub))))
}

var templateFuncs = template.FuncMap{
	"formatFloat": func(v float64) string {
		return strconv.FormatFloat(v, 'f', 4, 64)
	},
}

func (a *App) render(w http.ResponseWriter, name string, data interface{}) {
	w.Header().Set("Content-Type", "text/html; charset=utf-8")
	if err := a.tmpl.ExecuteTemplate(w, name, data); err != nil {
		log.Printf("template %s: %v", name, err)
		http.Error(w, "internal error rendering page", http.StatusInternalServerError)
	}
}

// ---- dashboard ----

type dashboardData struct {
	Sites []Manifest
}

func (a *App) handleDashboard(w http.ResponseWriter, r *http.Request) {
	sites, err := listManifests(a.cfg.artifactsDir())
	if err != nil {
		log.Printf("listManifests: %v", err)
	}
	a.render(w, "index.html", dashboardData{Sites: sites})
}

// ---- train form + job creation ----

func (a *App) handleTrainForm(w http.ResponseWriter, r *http.Request) {
	a.render(w, "train.html", nil)
}

type apiError struct {
	Error string `json:"error"`
}

func (a *App) handleAPITrain(w http.ResponseWriter, r *http.Request) {
	// ParseMultipartForm, not ParseForm: the browser posts a FormData body
	// (app.js's wireTrainForm), which is multipart/form-data, not
	// application/x-www-form-urlencoded -- ParseForm never reads multipart
	// bodies at all, and calling it here would additionally block
	// FormValue's own lazy multipart-parsing fallback later (it only
	// triggers when r.Form is still nil). ParseMultipartForm handles both:
	// it parses the multipart body directly, and falls back to ParseForm
	// internally for a plain urlencoded body (returning ErrNotMultipart,
	// which isn't a real failure -- r.Form is still populated correctly in
	// that case, exactly as net/http's own FormValue implementation treats it).
	if err := r.ParseMultipartForm(32 << 20); err != nil && err != http.ErrNotMultipart {
		writeJSONError(w, http.StatusBadRequest, "could not parse form: "+err.Error())
		return
	}

	siteCode := strings.TrimSpace(r.FormValue("site_code"))
	if err := validateSiteCode(siteCode); err != nil {
		writeJSONError(w, http.StatusBadRequest, err.Error())
		return
	}

	var horizons []int
	for _, v := range r.Form["horizons"] {
		h, err := strconv.Atoi(v)
		if err != nil {
			writeJSONError(w, http.StatusBadRequest, "invalid horizon value: "+v)
			return
		}
		horizons = append(horizons, h)
	}
	if err := validateHorizons(horizons); err != nil {
		writeJSONError(w, http.StatusBadRequest, err.Error())
		return
	}

	days := 60
	if v := r.FormValue("days"); v != "" {
		parsed, err := strconv.Atoi(v)
		if err != nil {
			writeJSONError(w, http.StatusBadRequest, "invalid days value: "+v)
			return
		}
		days = parsed
	}
	if err := validateDays(days); err != nil {
		writeJSONError(w, http.StatusBadRequest, err.Error())
		return
	}

	job := newJob(siteCode, horizons, days)
	a.jobs.add(job)

	// A single site's auto_pipeline.py run + charts.py run is capped at 20
	// minutes: generous next to the cold-cache NLDI scan (a few minutes)
	// plus per-horizon tuning, but bounded so a stuck subprocess can't pin
	// this server's resources forever.
	ctx, cancel := context.WithTimeout(context.Background(), 20*time.Minute)
	go func() {
		defer cancel()
		runTrainingJob(ctx, job, a.cfg)
	}()

	writeJSON(w, http.StatusAccepted, map[string]string{"id": job.ID})
}

func writeJSONError(w http.ResponseWriter, status int, msg string) {
	writeJSON(w, status, apiError{Error: msg})
}

// ---- job page / status / SSE stream ----

type jobPageData struct {
	Job JobSnapshot
}

func (a *App) handleJobPage(w http.ResponseWriter, r *http.Request) {
	job, ok := a.jobs.get(r.PathValue("id"))
	if !ok {
		http.NotFound(w, r)
		return
	}
	a.render(w, "job.html", jobPageData{Job: job.snapshot()})
}

func (a *App) handleJobStatus(w http.ResponseWriter, r *http.Request) {
	job, ok := a.jobs.get(r.PathValue("id"))
	if !ok {
		http.NotFound(w, r)
		return
	}
	writeJSON(w, http.StatusOK, job.snapshot())
}

// handleJobStream streams log lines as Server-Sent Events so the browser
// shows training progress live instead of polling or staring at a blank
// page for however long auto_pipeline.py takes. Replays everything logged
// so far immediately (so a page load/reload never misses history), then
// streams new lines as they're written, then a final "status" event once
// the job reaches a terminal state.
func (a *App) handleJobStream(w http.ResponseWriter, r *http.Request) {
	job, ok := a.jobs.get(r.PathValue("id"))
	if !ok {
		http.NotFound(w, r)
		return
	}

	flusher, ok := w.(http.Flusher)
	if !ok {
		http.Error(w, "streaming unsupported", http.StatusInternalServerError)
		return
	}

	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")

	snapshot, ch, unsubscribe := job.subscribe()
	defer unsubscribe()

	writeSSE := func(event, data string) bool {
		if event != "" {
			if _, err := w.Write([]byte("event: " + event + "\n")); err != nil {
				return false
			}
		}
		for _, line := range strings.Split(data, "\n") {
			if _, err := w.Write([]byte("data: " + line + "\n")); err != nil {
				return false
			}
		}
		if _, err := w.Write([]byte("\n")); err != nil {
			return false
		}
		flusher.Flush()
		return true
	}

	for _, line := range snapshot {
		if !writeSSE("log", line) {
			return
		}
	}

	ctx := r.Context()
	for {
		select {
		case <-ctx.Done():
			return
		case <-job.done:
			s := job.snapshot()
			payload, _ := jsonString(s)
			writeSSE("status", payload)
			return
		case line, ok := <-ch:
			if !ok {
				return
			}
			if !writeSSE("log", line) {
				return
			}
		}
	}
}

// ---- site detail ----

type siteDetailData struct {
	SiteCode   string
	Manifest   *Manifest
	ChartFiles []string
}

func (a *App) handleSiteDetail(w http.ResponseWriter, r *http.Request) {
	siteCode := r.PathValue("site")
	if err := validateSiteCode(siteCode); err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}

	manifest, err := loadManifest(a.cfg.artifactsDir(), siteCode)
	if errors.Is(err, os.ErrNotExist) {
		a.render(w, "site_not_found.html", siteCode)
		return
	}
	if err != nil {
		log.Printf("loadManifest(%s): %v", siteCode, err)
		http.Error(w, "could not read manifest for this site", http.StatusInternalServerError)
		return
	}

	chartFiles, err := chartFilesForSite(a.cfg.chartsDir(), siteCode)
	if err != nil {
		log.Printf("chartFilesForSite(%s): %v", siteCode, err)
	}

	a.render(w, "site.html", siteDetailData{SiteCode: siteCode, Manifest: manifest, ChartFiles: chartFiles})
}

// handleSiteSummary is the JSON equivalent of handleSiteDetail's data, used
// by job.html's client-side JS to render the "results" panel in place once
// a training job finishes, without a full page navigation.
func (a *App) handleSiteSummary(w http.ResponseWriter, r *http.Request) {
	siteCode := r.PathValue("site")
	if err := validateSiteCode(siteCode); err != nil {
		writeJSONError(w, http.StatusBadRequest, err.Error())
		return
	}

	manifest, err := loadManifest(a.cfg.artifactsDir(), siteCode)
	if errors.Is(err, os.ErrNotExist) {
		writeJSONError(w, http.StatusNotFound, "no manifest found for this site yet")
		return
	}
	if err != nil {
		writeJSONError(w, http.StatusInternalServerError, err.Error())
		return
	}

	chartFiles, _ := chartFilesForSite(a.cfg.chartsDir(), siteCode)

	writeJSON(w, http.StatusOK, map[string]interface{}{
		"manifest":    manifest,
		"chart_files": chartFiles,
	})
}

// ---- live-verify (the "deploy" step) ----

func (a *App) handleVerify(w http.ResponseWriter, r *http.Request) {
	siteCode := r.PathValue("site")
	if err := validateSiteCode(siteCode); err != nil {
		writeJSONError(w, http.StatusBadRequest, err.Error())
		return
	}

	horizonStr := r.URL.Query().Get("horizon")
	horizon, err := strconv.Atoi(horizonStr)
	if err != nil {
		writeJSONError(w, http.StatusBadRequest, "horizon query parameter must be an integer")
		return
	}

	result, err := a.msvc.Predict(r.Context(), siteCode, horizon)
	if err != nil {
		var msErr *microserviceError
		if errors.As(err, &msErr) {
			writeJSONError(w, msErr.StatusCode, msErr.Detail)
			return
		}
		writeJSONError(w, http.StatusBadGateway, err.Error())
		return
	}

	writeJSON(w, http.StatusOK, result)
}

// ---- chart files ----

func (a *App) handleChartFile(w http.ResponseWriter, r *http.Request) {
	siteCode := r.PathValue("site")
	file := r.PathValue("file")

	if err := validateSiteCode(siteCode); err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}
	// filepath.Base strips any directory components a crafted request
	// might smuggle in (e.g. "../../etc/passwd"), so the file actually
	// opened is always a bare filename directly inside chartsDir.
	safeName := filepath.Base(file)
	if safeName != file || safeName == "." || safeName == string(filepath.Separator) {
		http.Error(w, "invalid file name", http.StatusBadRequest)
		return
	}

	full := filepath.Join(a.cfg.chartsDir(), safeName)
	http.ServeFile(w, r, full)
}

// ---- health ----

func (a *App) handleHealth(w http.ResponseWriter, r *http.Request) {
	writeJSON(w, http.StatusOK, map[string]string{"status": "ok"})
}
