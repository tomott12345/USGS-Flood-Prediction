package main

import (
	"fmt"
	"regexp"
)

// USGS site codes are 8-15 digits (8 is standard; some sites, mostly
// groundwater wells, run longer). Matches the same pattern the sibling
// usgs-edge-app's main.go validates against.
var siteCodeRe = regexp.MustCompile(`^[0-9]{8,15}$`)

func validateSiteCode(s string) error {
	if !siteCodeRe.MatchString(s) {
		return fmt.Errorf("site code must be 8-15 digits (got %q)", s)
	}
	return nil
}

// Horizons are validated against a generous but bounded range rather than
// the repo's own <=6h default scope -- auto_pipeline.py itself is the thing
// that decides what's trainable, and README.md documents that horizons past
// 6h have shown negative NSE (worse than predicting the mean) so far. The
// web form defaults to 1/3/6 and warns before letting a user opt into more,
// but doesn't hard-block it here: someone re-testing that finding on a new
// site is a legitimate use of this tool.
const (
	minHorizonHours = 1
	maxHorizonHours = 168 // one week; matches the sibling app's forecast-length upper bound
	maxHorizonCount = 12
)

func validateHorizons(horizons []int) error {
	if len(horizons) == 0 {
		return fmt.Errorf("at least one forecast horizon is required")
	}
	if len(horizons) > maxHorizonCount {
		return fmt.Errorf("too many horizons requested (max %d)", maxHorizonCount)
	}
	seen := make(map[int]bool, len(horizons))
	for _, h := range horizons {
		if h < minHorizonHours || h > maxHorizonHours {
			return fmt.Errorf("horizon %dh out of range (%d-%d)", h, minHorizonHours, maxHorizonHours)
		}
		if seen[h] {
			return fmt.Errorf("duplicate horizon %dh", h)
		}
		seen[h] = true
	}
	return nil
}

// USGS's instantaneous-values service generally only retains sub-daily
// granularity for roughly the last 120 days (see xgboost_model's data.py /
// evaluation/usgs_data.py) -- days above that silently returns less data
// than requested rather than erroring, so cap it here with a clear message
// instead of letting a user request a window that can't do what it implies.
const (
	minTrainingDays = 14
	maxTrainingDays = 120
)

func validateDays(days int) error {
	if days < minTrainingDays || days > maxTrainingDays {
		return fmt.Errorf("training window must be %d-%d days (USGS's instantaneous-values service "+
			"generally only retains ~120 days of sub-daily history)", minTrainingDays, maxTrainingDays)
	}
	return nil
}
