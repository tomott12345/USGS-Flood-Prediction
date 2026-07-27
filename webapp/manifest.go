package main

import (
	"encoding/json"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
)

// HorizonResult mirrors one entry of auto_pipeline.py's manifest.json
// "horizons" map (see xgboost_model/auto_pipeline.py's run_pipeline). Field
// names/JSON tags match exactly what that script writes.
type HorizonResult struct {
	Status     string `json:"status"`
	Reason     string `json:"reason,omitempty"`
	FeatureSet string `json:"feature_set,omitempty"`
	ModelDir   string `json:"model_dir,omitempty"`
	Selection  struct {
		BaselineMAEFt      float64 `json:"baseline_mae_ft"`
		EnrichedTunedMAEFt float64 `json:"enriched_tuned_mae_ft"`
	} `json:"selection"`
	ConformalMarginFt float64 `json:"conformal_margin_ft"`
}

// Manifest mirrors {site_code}_manifest.json exactly.
type Manifest struct {
	SiteCode          string                   `json:"site_code"`
	TrainedAt         string                   `json:"trained_at"`
	UpstreamSiteCodes []string                 `json:"upstream_site_codes"`
	TrainingDays      int                      `json:"training_days"`
	Horizons          map[string]HorizonResult `json:"horizons"`
}

// SortedHorizons returns the horizon keys in ascending numeric order --
// map iteration order in Go/JSON is unspecified, and "1h, 12h, 3h, 6h"
// (lexical) would look wrong in the UI next to "1h, 3h, 6h, 12h" (numeric).
func (m Manifest) SortedHorizons() []int {
	out := make([]int, 0, len(m.Horizons))
	for k := range m.Horizons {
		if n, err := strconv.Atoi(k); err == nil {
			out = append(out, n)
		}
	}
	sort.Ints(out)
	return out
}

func (m Manifest) Horizon(h int) (HorizonResult, bool) {
	r, ok := m.Horizons[strconv.Itoa(h)]
	return r, ok
}

func manifestPath(artifactsDir, siteCode string) string {
	return filepath.Join(artifactsDir, siteCode+"_manifest.json")
}

func loadManifest(artifactsDir, siteCode string) (*Manifest, error) {
	data, err := os.ReadFile(manifestPath(artifactsDir, siteCode))
	if err != nil {
		return nil, err
	}
	var m Manifest
	if err := json.Unmarshal(data, &m); err != nil {
		return nil, err
	}
	return &m, nil
}

// listManifests scans artifactsDir for every *_manifest.json auto_pipeline.py
// has ever written, newest-trained first, for the dashboard's "sites
// trained so far" list. A missing/unreadable artifacts directory (nothing
// trained yet) is not an error -- it just means an empty dashboard.
func listManifests(artifactsDir string) ([]Manifest, error) {
	entries, err := os.ReadDir(artifactsDir)
	if os.IsNotExist(err) {
		return nil, nil
	}
	if err != nil {
		return nil, err
	}

	var manifests []Manifest
	for _, e := range entries {
		if e.IsDir() || !strings.HasSuffix(e.Name(), "_manifest.json") {
			continue
		}
		siteCode := strings.TrimSuffix(e.Name(), "_manifest.json")
		m, err := loadManifest(artifactsDir, siteCode)
		if err != nil {
			continue // skip anything unreadable/corrupt rather than fail the whole dashboard
		}
		manifests = append(manifests, *m)
	}

	sort.Slice(manifests, func(i, j int) bool {
		return manifests[i].TrainedAt > manifests[j].TrainedAt
	})
	return manifests, nil
}

// chartFilesForSite lists the PNG/CSV files charts.py saved for one site,
// namespaced by the "{site_code}_" prefix charts.py's __main__ block uses
// for every site except the original default (01388500, which predates
// that namespacing and has no prefix).
func chartFilesForSite(chartsDir, siteCode string) ([]string, error) {
	entries, err := os.ReadDir(chartsDir)
	if os.IsNotExist(err) {
		return nil, nil
	}
	if err != nil {
		return nil, err
	}

	prefix := siteCode + "_"
	var files []string
	for _, e := range entries {
		if e.IsDir() {
			continue
		}
		name := e.Name()
		if !strings.HasSuffix(name, ".png") {
			continue
		}
		if strings.HasPrefix(name, prefix) || (siteCode == "01388500" && isDefaultSiteChart(name)) {
			files = append(files, name)
		}
	}
	sort.Strings(files)
	return files, nil
}

// isDefaultSiteChart recognizes the original, un-prefixed Pompton (01388500)
// chart filenames (forecast_h1.png, lead_time_h3.png, calibration_reliability.png,
// error_by_horizon.png) that predate charts.py's per-site filename prefix.
func isDefaultSiteChart(name string) bool {
	for _, known := range []string{"forecast_h", "lead_time_h", "calibration_reliability", "error_by_horizon"} {
		if strings.HasPrefix(name, known) {
			return true
		}
	}
	return false
}
