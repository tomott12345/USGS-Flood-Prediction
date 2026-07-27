package main

import (
	"bytes"
	"context"
	"fmt"
	"os/exec"
	"path/filepath"
	"strconv"
	"sync"
)

// lineWriter is an io.Writer that buffers partial lines and emits complete
// ones via onLine as they arrive. Used to turn a subprocess's byte-stream
// stdout/stderr into the line-at-a-time log auto_pipeline.py and charts.py
// already produce (they use Python's line-buffered logging/print), so the
// browser sees output as it happens rather than only once the whole
// process exits.
type lineWriter struct {
	mu     sync.Mutex
	buf    bytes.Buffer
	onLine func(string)
}

func (w *lineWriter) Write(p []byte) (int, error) {
	w.mu.Lock()
	defer w.mu.Unlock()
	w.buf.Write(p)
	for {
		line, err := w.buf.ReadString('\n')
		if err != nil {
			// No newline yet -- `line` is whatever was left unread; put it
			// back since bytes.Buffer.ReadString drains the buffer either way.
			w.buf.Reset()
			w.buf.WriteString(line)
			break
		}
		w.onLine(trimNewline(line))
	}
	return len(p), nil
}

// flush emits any trailing partial line once the process has exited (a
// script that doesn't end its last line of output with \n shouldn't have
// that line silently swallowed).
func (w *lineWriter) flush() {
	w.mu.Lock()
	defer w.mu.Unlock()
	if w.buf.Len() > 0 {
		w.onLine(trimNewline(w.buf.String()))
		w.buf.Reset()
	}
}

func trimNewline(s string) string {
	for len(s) > 0 && (s[len(s)-1] == '\n' || s[len(s)-1] == '\r') {
		s = s[:len(s)-1]
	}
	return s
}

// runPython runs one Python script to completion, streaming every line of
// its combined stdout+stderr into job.appendLine. name/args are passed to
// exec.Command as a proper argv slice (never through a shell), so there is
// no shell-injection surface even though siteCode ends up in the argument
// list -- unlike string-concatenating into `bash -c "... " + siteCode`,
// which is what the sibling usgs-edge-app's main.go does.
func runPython(ctx context.Context, job *Job, pythonBin, dir string, scriptArgs ...string) error {
	cmd := exec.CommandContext(ctx, pythonBin, scriptArgs...)
	cmd.Dir = dir

	lw := &lineWriter{onLine: job.appendLine}
	cmd.Stdout = lw
	cmd.Stderr = lw

	err := cmd.Run()
	lw.flush()
	return err
}

// runAutoPipeline invokes xgboost_model/auto_pipeline.py for one site,
// exactly as documented in xgboost_model/README.md's "auto_pipeline.py --
// one command, any site" section: discover upstream gages, auto-select
// baseline vs. enriched+tuned features per horizon, save native-JSON
// artifacts + a manifest.
func runAutoPipeline(ctx context.Context, job *Job, cfg Config, siteCode string, horizons []int, days int) error {
	scriptPath := filepath.Join(cfg.xgboostModelDir(), "auto_pipeline.py")
	args := []string{scriptPath, siteCode}
	if len(horizons) > 0 {
		args = append(args, "--horizons")
		for _, h := range horizons {
			args = append(args, strconv.Itoa(h))
		}
	}
	if days > 0 {
		args = append(args, "--days", strconv.Itoa(days))
	}
	return runPython(ctx, job, cfg.PythonBin, cfg.xgboostModelDir(), args...)
}

// runCharts invokes xgboost_model/charts.py for one site (it accepts an
// optional site-code CLI argument -- see the __main__ block), producing
// the forecast/lead-time/comparison PNGs under xgboost_model/charts/,
// namespaced by site code so a new site never overwrites another site's
// saved charts.
func runCharts(ctx context.Context, job *Job, cfg Config, siteCode string) error {
	scriptPath := filepath.Join(cfg.xgboostModelDir(), "charts.py")
	return runPython(ctx, job, cfg.PythonBin, cfg.xgboostModelDir(), scriptPath, siteCode)
}

// runTrainingJob is the full pipeline behind one job: train, then chart,
// then mark terminal. Meant to be launched in its own goroutine by the
// HTTP handler that creates the job, so the request returns immediately.
func runTrainingJob(ctx context.Context, job *Job, cfg Config) {
	job.appendLine(fmt.Sprintf("=== training %s (horizons=%v, days=%d) ===", job.SiteCode, job.Horizons, job.Days))
	if err := runAutoPipeline(ctx, job, cfg, job.SiteCode, job.Horizons, job.Days); err != nil {
		job.appendLine(fmt.Sprintf("auto_pipeline.py failed: %v", err))
		job.finish(JobFailed, err.Error())
		return
	}

	job.setStage("charts")
	job.appendLine("=== rendering charts ===")
	if err := runCharts(ctx, job, cfg, job.SiteCode); err != nil {
		// Charts are a nice-to-have on top of a successfully trained model,
		// not the thing that defines success -- a chart-rendering failure
		// (e.g. a transient Open-Meteo hiccup while building the
		// cross-engine comparison) shouldn't hide that the model itself
		// trained and is already servable by the microservice.
		job.appendLine(fmt.Sprintf("charts.py failed (model was still trained successfully): %v", err))
		job.finish(JobSucceeded, "")
		return
	}

	job.appendLine("=== done ===")
	job.finish(JobSucceeded, "")
}
