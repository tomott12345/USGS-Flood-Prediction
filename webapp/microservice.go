package main

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"time"
)

// MicroserviceClient talks to the existing FastAPI microservice (see
// microservice/app.py and microservice/xgboost_engine.py) over plain HTTP.
// This is deliberately the only integration point between this Go app and
// the Python service: auto_pipeline.py already writes trained models
// straight into xgboost_model/artifacts/, exactly where the microservice's
// XGB_ARTIFACTS_DIR already looks by default, and the microservice reads
// from disk on every request with no caching -- so there is no separate
// "deploy" step to perform. "Deploying" a model here means proving it's
// really being served: calling the microservice's own routes and showing
// the result, not copying any files.
type MicroserviceClient struct {
	BaseURL string
	HTTP    *http.Client
}

func newMicroserviceClient(baseURL string, timeout time.Duration) *MicroserviceClient {
	return &MicroserviceClient{
		BaseURL: baseURL,
		HTTP:    &http.Client{Timeout: timeout},
	}
}

// AvailableHorizonsResponse mirrors GET /models/xgboost/{site_code}.
type AvailableHorizonsResponse struct {
	SiteCode             string `json:"site_code"`
	AvailableHorizonsHrs []int  `json:"available_horizons_hours"`
}

func (c *MicroserviceClient) AvailableHorizons(ctx context.Context, siteCode string) (*AvailableHorizonsResponse, error) {
	var out AvailableHorizonsResponse
	err := c.getJSON(ctx, fmt.Sprintf("/models/xgboost/%s", url.PathEscape(siteCode)), &out)
	if err != nil {
		return nil, err
	}
	return &out, nil
}

// PredictResponse mirrors xgboost_engine.py's predict() return value, as
// returned by GET /predict/xgboost/{site_code}/{forecast_length}.
type PredictResponse struct {
	SiteCode            string  `json:"site_code"`
	HorizonHours        int     `json:"horizon_hours"`
	AsOf                string  `json:"as_of"`
	PredictedGageHeight float64 `json:"predicted_gage_height"`
	ConfidenceInterval  struct {
		LowerBound      float64 `json:"lower_bound"`
		UpperBound      float64 `json:"upper_bound"`
		NominalCoverage float64 `json:"nominal_coverage"`
	} `json:"confidence_interval"`
	Model struct {
		Engine            string   `json:"engine"`
		Format            string   `json:"format"`
		FeatureSet        string   `json:"feature_set"`
		UpstreamSiteCodes []string `json:"upstream_site_codes"`
		TrainedAt         string   `json:"trained_at"`
	} `json:"model"`
}

func (c *MicroserviceClient) Predict(ctx context.Context, siteCode string, horizonHours int) (*PredictResponse, error) {
	var out PredictResponse
	path := fmt.Sprintf("/predict/xgboost/%s/%d", url.PathEscape(siteCode), horizonHours)
	if err := c.getJSON(ctx, path, &out); err != nil {
		return nil, err
	}
	return &out, nil
}

// microserviceError carries the microservice's own HTTP status + detail
// message through, rather than flattening every failure into "request
// failed" -- a 404 ("no model trained for this horizon") and a 503
// ("stale/insufficient live data") mean very different things to a user
// clicking "verify live scoring."
type microserviceError struct {
	StatusCode int
	Detail     string
}

func (e *microserviceError) Error() string {
	return fmt.Sprintf("microservice returned %d: %s", e.StatusCode, e.Detail)
}

func (c *MicroserviceClient) getJSON(ctx context.Context, path string, out interface{}) error {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, c.BaseURL+path, nil)
	if err != nil {
		return err
	}
	resp, err := c.HTTP.Do(req)
	if err != nil {
		return fmt.Errorf("could not reach microservice at %s%s: %w", c.BaseURL, path, err)
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return err
	}

	if resp.StatusCode != http.StatusOK {
		var detail struct {
			Detail string `json:"detail"`
		}
		_ = json.Unmarshal(body, &detail)
		if detail.Detail == "" {
			detail.Detail = string(body)
		}
		return &microserviceError{StatusCode: resp.StatusCode, Detail: detail.Detail}
	}

	return json.Unmarshal(body, out)
}
