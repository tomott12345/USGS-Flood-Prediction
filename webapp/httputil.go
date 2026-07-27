package main

import (
	"encoding/json"
	"log"
	"net/http"
)

// writeJSON is the one place every JSON API response in this package goes
// through, so the Content-Type header and status-code-then-body ordering
// stay consistent across handlers instead of being repeated at each call site.
func writeJSON(w http.ResponseWriter, status int, v interface{}) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	if err := json.NewEncoder(w).Encode(v); err != nil {
		log.Printf("writeJSON: %v", err)
	}
}

// jsonString marshals v to a JSON string rather than writing it to an
// http.ResponseWriter -- used by handleJobStream to embed a JSON payload as
// the "data:" field of a Server-Sent Event, where the caller controls the
// "event: status\n" framing around it.
func jsonString(v interface{}) (string, error) {
	b, err := json.Marshal(v)
	if err != nil {
		return "", err
	}
	return string(b), nil
}
