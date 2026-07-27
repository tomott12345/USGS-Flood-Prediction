package main

import (
	"crypto/rand"
	"encoding/hex"
	"sync"
	"time"
)

type JobStatus string

const (
	JobRunning   JobStatus = "running"
	JobSucceeded JobStatus = "succeeded"
	JobFailed    JobStatus = "failed"
)

// Job tracks one training run (auto_pipeline.py, then charts.py) end to
// end: its parameters, live log output, and terminal status. Training a
// site can take anywhere from under a minute (warm upstream-gage cache) to
// several minutes (cold NLDI scan, see xgboost_model/upstream.py), so this
// is deliberately async: starting a job returns immediately with an ID, and
// /jobs/{id}/stream lets the browser watch it happen live via
// Server-Sent Events instead of blocking one HTTP request for the whole
// run.
type Job struct {
	ID        string
	SiteCode  string
	Horizons  []int
	Days      int
	StartedAt time.Time
	EndedAt   time.Time

	mu     sync.Mutex
	status JobStatus
	stage  string // "training" | "charts" | "" (done)
	lines  []string
	errMsg string
	subs   map[chan string]struct{}
	done   chan struct{}
}

func newJob(siteCode string, horizons []int, days int) *Job {
	return &Job{
		ID:        newJobID(),
		SiteCode:  siteCode,
		Horizons:  horizons,
		Days:      days,
		StartedAt: time.Now(),
		status:    JobRunning,
		stage:     "training",
		subs:      make(map[chan string]struct{}),
		done:      make(chan struct{}),
	}
}

func newJobID() string {
	b := make([]byte, 8)
	_, _ = rand.Read(b)
	return hex.EncodeToString(b)
}

// appendLine records one log line and fans it out to every active
// subscriber. Subscriber channels are buffered and sends are non-blocking:
// a slow or stalled browser tab must never be able to stall the training
// subprocess whose output is being piped through this.
func (j *Job) appendLine(line string) {
	j.mu.Lock()
	defer j.mu.Unlock()
	j.lines = append(j.lines, line)
	for ch := range j.subs {
		select {
		case ch <- line:
		default:
			// Subscriber isn't keeping up; drop this line for them. They
			// still got everything up to now via the replay in subscribe,
			// and the final status is always fetchable via snapshot().
		}
	}
}

func (j *Job) setStage(stage string) {
	j.mu.Lock()
	j.stage = stage
	j.mu.Unlock()
}

// finish marks the job terminal. Safe to call exactly once per job.
func (j *Job) finish(status JobStatus, errMsg string) {
	j.mu.Lock()
	j.status = status
	j.stage = ""
	j.errMsg = errMsg
	j.EndedAt = time.Now()
	j.mu.Unlock()
	close(j.done)
}

// subscribe returns every line logged so far, plus a channel that will
// receive every line logged from this point on. Registration happens under
// the same lock appendLine uses, so no line can be lost or duplicated
// between the snapshot and the channel starting to receive.
func (j *Job) subscribe() (snapshot []string, ch chan string, unsubscribe func()) {
	j.mu.Lock()
	snapshot = append([]string(nil), j.lines...)
	ch = make(chan string, 256)
	j.subs[ch] = struct{}{}
	j.mu.Unlock()

	return snapshot, ch, func() {
		j.mu.Lock()
		delete(j.subs, ch)
		j.mu.Unlock()
	}
}

type JobSnapshot struct {
	ID        string    `json:"id"`
	SiteCode  string    `json:"site_code"`
	Horizons  []int     `json:"horizons"`
	Days      int       `json:"days"`
	Status    JobStatus `json:"status"`
	Stage     string    `json:"stage"`
	Error     string    `json:"error,omitempty"`
	StartedAt time.Time `json:"started_at"`
	EndedAt   time.Time `json:"ended_at,omitempty"`
}

func (j *Job) snapshot() JobSnapshot {
	j.mu.Lock()
	defer j.mu.Unlock()
	s := JobSnapshot{
		ID:        j.ID,
		SiteCode:  j.SiteCode,
		Horizons:  j.Horizons,
		Days:      j.Days,
		Status:    j.status,
		Stage:     j.stage,
		Error:     j.errMsg,
		StartedAt: j.StartedAt,
	}
	if !j.EndedAt.IsZero() {
		s.EndedAt = j.EndedAt
	}
	return s
}

// JobManager is a process-lifetime, in-memory registry of jobs. There's no
// persistence across restarts by design -- a job is a live subprocess run,
// not a durable record; the durable record of what got trained is the
// manifest.json / model.json files auto_pipeline.py writes to
// xgboost_model/artifacts/, which survive restarts on their own.
type JobManager struct {
	mu   sync.RWMutex
	jobs map[string]*Job
}

func newJobManager() *JobManager {
	return &JobManager{jobs: make(map[string]*Job)}
}

func (m *JobManager) add(j *Job) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.jobs[j.ID] = j
}

func (m *JobManager) get(id string) (*Job, bool) {
	m.mu.RLock()
	defer m.mu.RUnlock()
	j, ok := m.jobs[id]
	return j, ok
}
