package main

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
)

func writeManifestFixture(t *testing.T, dir, siteCode, trainedAt string) {
	t.Helper()
	m := map[string]interface{}{
		"site_code":           siteCode,
		"trained_at":          trainedAt,
		"upstream_site_codes": []string{"01111111", "02222222"},
		"training_days":       60,
		"horizons": map[string]interface{}{
			"6": map[string]interface{}{
				"status":      "ok",
				"feature_set": "baseline",
				"model_dir":   siteCode + "_h6",
				"selection": map[string]interface{}{
					"baseline_mae_ft":       0.06,
					"enriched_tuned_mae_ft": 0.08,
				},
				"conformal_margin_ft": 0.07,
			},
			"1": map[string]interface{}{
				"status":      "ok",
				"feature_set": "enriched",
				"model_dir":   siteCode + "_h1",
				"selection": map[string]interface{}{
					"baseline_mae_ft":       0.02,
					"enriched_tuned_mae_ft": 0.03,
				},
				"conformal_margin_ft": 0.01,
			},
			"12": map[string]interface{}{
				"status": "skipped",
				"reason": "insufficient_data",
			},
		},
	}
	data, err := json.Marshal(m)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(manifestPath(dir, siteCode), data, 0o644); err != nil {
		t.Fatal(err)
	}
}

func TestLoadManifestAndSortedHorizons(t *testing.T) {
	dir := t.TempDir()
	writeManifestFixture(t, dir, "01388500", "2026-07-27T10:00:00Z")

	m, err := loadManifest(dir, "01388500")
	if err != nil {
		t.Fatalf("loadManifest: %v", err)
	}

	got := m.SortedHorizons()
	want := []int{1, 6, 12}
	if len(got) != len(want) {
		t.Fatalf("SortedHorizons() = %v, want %v", got, want)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("SortedHorizons() = %v, want %v", got, want)
		}
	}

	r, ok := m.Horizon(1)
	if !ok || r.FeatureSet != "enriched" || r.Status != "ok" {
		t.Fatalf("Horizon(1) = %+v, ok=%v", r, ok)
	}

	skipped, ok := m.Horizon(12)
	if !ok || skipped.Status != "skipped" || skipped.Reason != "insufficient_data" {
		t.Fatalf("Horizon(12) = %+v, ok=%v", skipped, ok)
	}

	if _, ok := m.Horizon(48); ok {
		t.Fatalf("Horizon(48) should not exist in this fixture")
	}
}

func TestLoadManifestMissingFileReturnsError(t *testing.T) {
	dir := t.TempDir()
	if _, err := loadManifest(dir, "00000000"); err == nil {
		t.Fatal("expected an error for a missing manifest file")
	}
}

func TestListManifestsSortsNewestFirstAndSkipsUnreadable(t *testing.T) {
	dir := t.TempDir()
	writeManifestFixture(t, dir, "01388500", "2026-07-20T00:00:00Z")
	writeManifestFixture(t, dir, "01553990", "2026-07-27T00:00:00Z")

	// Corrupt/unreadable manifest -- must be skipped, not fail the whole call.
	if err := os.WriteFile(manifestPath(dir, "09999999"), []byte("not json"), 0o644); err != nil {
		t.Fatal(err)
	}

	manifests, err := listManifests(dir)
	if err != nil {
		t.Fatalf("listManifests: %v", err)
	}
	if len(manifests) != 2 {
		t.Fatalf("expected 2 valid manifests, got %d", len(manifests))
	}
	if manifests[0].SiteCode != "01553990" {
		t.Fatalf("expected 01553990 (trained later) first, got %s", manifests[0].SiteCode)
	}
}

func TestListManifestsEmptyDirIsNotAnError(t *testing.T) {
	dir := t.TempDir()
	manifests, err := listManifests(filepath.Join(dir, "does-not-exist"))
	if err != nil {
		t.Fatalf("expected no error for a nonexistent artifacts dir, got %v", err)
	}
	if manifests != nil {
		t.Fatalf("expected nil/empty result, got %v", manifests)
	}
}

func TestChartFilesForSite(t *testing.T) {
	dir := t.TempDir()
	files := []string{
		"01553990_forecast_h1.png",
		"01553990_forecast_h3.png",
		"01553990_combined_comparison.csv", // not .png -- must be excluded
		"01473730_forecast_h1.png",         // different site -- must be excluded
		"forecast_h1.png",                  // unprefixed default-site (01388500) chart
	}
	for _, f := range files {
		if err := os.WriteFile(filepath.Join(dir, f), []byte("x"), 0o644); err != nil {
			t.Fatal(err)
		}
	}

	got, err := chartFilesForSite(dir, "01553990")
	if err != nil {
		t.Fatalf("chartFilesForSite: %v", err)
	}
	want := []string{"01553990_forecast_h1.png", "01553990_forecast_h3.png"}
	if len(got) != len(want) {
		t.Fatalf("chartFilesForSite(01553990) = %v, want %v", got, want)
	}

	defaultSite, err := chartFilesForSite(dir, "01388500")
	if err != nil {
		t.Fatalf("chartFilesForSite: %v", err)
	}
	if len(defaultSite) != 1 || defaultSite[0] != "forecast_h1.png" {
		t.Fatalf("chartFilesForSite(01388500) = %v, want [forecast_h1.png]", defaultSite)
	}
}
