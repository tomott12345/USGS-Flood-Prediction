package main

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"
)

func TestMicroserviceClientAvailableHorizons(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/models/xgboost/01388500" {
			t.Errorf("unexpected path: %s", r.URL.Path)
		}
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]interface{}{
			"site_code":                "01388500",
			"available_horizons_hours": []int{1, 3, 6},
		})
	}))
	defer srv.Close()

	client := newMicroserviceClient(srv.URL, 5*time.Second)
	resp, err := client.AvailableHorizons(context.Background(), "01388500")
	if err != nil {
		t.Fatalf("AvailableHorizons: %v", err)
	}
	if resp.SiteCode != "01388500" || len(resp.AvailableHorizonsHrs) != 3 {
		t.Fatalf("unexpected response: %+v", resp)
	}
}

func TestMicroserviceClientPredictSuccess(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/predict/xgboost/01388500/1" {
			t.Errorf("unexpected path: %s", r.URL.Path)
		}
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]interface{}{
			"site_code":             "01388500",
			"horizon_hours":         1,
			"as_of":                 "2026-07-27T10:00:00",
			"predicted_gage_height": 7.74,
			"confidence_interval": map[string]interface{}{
				"lower_bound":      7.71,
				"upper_bound":      7.77,
				"nominal_coverage": 0.8,
			},
			"model": map[string]interface{}{
				"engine":              "xgboost",
				"format":              "xgboost_native_json",
				"feature_set":         "baseline",
				"upstream_site_codes": []string{},
				"trained_at":          "2026-07-27T14:11:59Z",
			},
		})
	}))
	defer srv.Close()

	client := newMicroserviceClient(srv.URL, 5*time.Second)
	resp, err := client.Predict(context.Background(), "01388500", 1)
	if err != nil {
		t.Fatalf("Predict: %v", err)
	}
	if resp.PredictedGageHeight != 7.74 || resp.Model.FeatureSet != "baseline" {
		t.Fatalf("unexpected response: %+v", resp)
	}
}

func TestMicroserviceClientPredictPropagatesErrorStatusAndDetail(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusNotFound)
		_ = json.NewEncoder(w).Encode(map[string]string{"detail": "no model found"})
	}))
	defer srv.Close()

	client := newMicroserviceClient(srv.URL, 5*time.Second)
	_, err := client.Predict(context.Background(), "01388500", 48)
	if err == nil {
		t.Fatal("expected an error")
	}
	var msErr *microserviceError
	if !errors.As(err, &msErr) {
		t.Fatalf("expected a *microserviceError, got %T: %v", err, err)
	}
	if msErr.StatusCode != http.StatusNotFound || msErr.Detail != "no model found" {
		t.Fatalf("unexpected microserviceError: %+v", msErr)
	}
}

func TestMicroserviceClientUnreachableServer(t *testing.T) {
	client := newMicroserviceClient("http://127.0.0.1:1", 200*time.Millisecond)
	_, err := client.Predict(context.Background(), "01388500", 1)
	if err == nil {
		t.Fatal("expected an error for an unreachable server")
	}
}
