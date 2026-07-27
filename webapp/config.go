package main

import (
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"time"
)

// Config holds everything the app needs to find the rest of the repo (the
// Python training pipeline and the FastAPI microservice) and to talk to
// them. Every value has a sane default so `go run .` works out of the box
// against this repo's existing layout; every value is also overridable via
// environment variable so the same binary works unmodified inside the
// Docker image (see ../Dockerfile), where the working directory and the
// microservice's network address differ from local dev.
type Config struct {
	Port                string
	RepoRoot            string
	PythonBin           string
	MicroserviceURL     string
	MicroserviceTimeout time.Duration
}

func loadConfig() (Config, error) {
	cfg := Config{
		Port:                envOr("WEBAPP_PORT", "8080"),
		PythonBin:           envOr("PYTHON_BIN", "python3"),
		MicroserviceURL:     envOr("MICROSERVICE_URL", "http://localhost:8000"),
		MicroserviceTimeout: 30 * time.Second,
	}

	if v := os.Getenv("MICROSERVICE_TIMEOUT_SECONDS"); v != "" {
		secs, err := strconv.Atoi(v)
		if err != nil {
			return cfg, fmt.Errorf("MICROSERVICE_TIMEOUT_SECONDS must be an integer: %w", err)
		}
		cfg.MicroserviceTimeout = time.Duration(secs) * time.Second
	}

	if v := os.Getenv("REPO_ROOT"); v != "" {
		abs, err := filepath.Abs(v)
		if err != nil {
			return cfg, fmt.Errorf("REPO_ROOT %q: %w", v, err)
		}
		cfg.RepoRoot = abs
		return cfg, nil
	}

	root, err := findRepoRoot()
	if err != nil {
		return cfg, err
	}
	cfg.RepoRoot = root
	return cfg, nil
}

func envOr(key, fallback string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return fallback
}

// findRepoRoot locates the USGS-Flood-Prediction checkout by walking upward
// from both the current working directory and the running binary's own
// directory, looking for a directory that has both xgboost_model/ and
// microservice/ as children. This lets the same compiled binary work whether
// it's run via `go run .` from webapp/ (cwd = webapp/, repo root is one
// level up) or from an image where the whole repo is copied to one
// directory and the binary runs with that as its working directory.
func findRepoRoot() (string, error) {
	var starts []string
	if cwd, err := os.Getwd(); err == nil {
		starts = append(starts, cwd)
	}
	if exe, err := os.Executable(); err == nil {
		if resolved, err := filepath.EvalSymlinks(exe); err == nil {
			starts = append(starts, filepath.Dir(resolved))
		} else {
			starts = append(starts, filepath.Dir(exe))
		}
	}

	const maxDepth = 6
	for _, start := range starts {
		dir := start
		for i := 0; i < maxDepth; i++ {
			if looksLikeRepoRoot(dir) {
				return dir, nil
			}
			parent := filepath.Dir(dir)
			if parent == dir {
				break
			}
			dir = parent
		}
	}
	return "", fmt.Errorf(
		"could not locate the repo root (looked for xgboost_model/ and microservice/ near the working " +
			"directory and the running binary); set REPO_ROOT explicitly to override",
	)
}

func looksLikeRepoRoot(dir string) bool {
	xg, err1 := os.Stat(filepath.Join(dir, "xgboost_model"))
	ms, err2 := os.Stat(filepath.Join(dir, "microservice"))
	return err1 == nil && xg.IsDir() && err2 == nil && ms.IsDir()
}

func (c Config) xgboostModelDir() string {
	return filepath.Join(c.RepoRoot, "xgboost_model")
}

func (c Config) artifactsDir() string {
	return filepath.Join(c.xgboostModelDir(), "artifacts")
}

func (c Config) chartsDir() string {
	return filepath.Join(c.xgboostModelDir(), "charts")
}
