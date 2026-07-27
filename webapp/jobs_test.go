package main

import (
	"testing"
	"time"
)

func TestJobSubscribeReplaysHistoryThenStreamsNewLines(t *testing.T) {
	job := newJob("01388500", []int{1, 3, 6}, 60)
	job.appendLine("line 1")
	job.appendLine("line 2")

	snapshot, ch, unsubscribe := job.subscribe()
	defer unsubscribe()

	if len(snapshot) != 2 || snapshot[0] != "line 1" || snapshot[1] != "line 2" {
		t.Fatalf("unexpected snapshot: %v", snapshot)
	}

	job.appendLine("line 3")
	select {
	case line := <-ch:
		if line != "line 3" {
			t.Fatalf("expected 'line 3', got %q", line)
		}
	case <-time.After(time.Second):
		t.Fatal("timed out waiting for streamed line")
	}
}

func TestJobFinishClosesDoneChannelAndSetsStatus(t *testing.T) {
	job := newJob("01388500", []int{1}, 60)
	job.finish(JobSucceeded, "")

	select {
	case <-job.done:
	default:
		t.Fatal("done channel should be closed after finish()")
	}

	snap := job.snapshot()
	if snap.Status != JobSucceeded {
		t.Fatalf("expected status %q, got %q", JobSucceeded, snap.Status)
	}
	if snap.EndedAt.IsZero() {
		t.Fatal("expected EndedAt to be set")
	}
}

func TestJobFinishWithError(t *testing.T) {
	job := newJob("01388500", []int{1}, 60)
	job.finish(JobFailed, "auto_pipeline.py exited with status 1")

	snap := job.snapshot()
	if snap.Status != JobFailed || snap.Error == "" {
		t.Fatalf("expected failed status with error message, got %+v", snap)
	}
}

func TestJobManagerAddAndGet(t *testing.T) {
	mgr := newJobManager()
	job := newJob("01388500", []int{1}, 60)
	mgr.add(job)

	got, ok := mgr.get(job.ID)
	if !ok || got.ID != job.ID {
		t.Fatalf("expected to retrieve job %s, got ok=%v got=%+v", job.ID, ok, got)
	}

	_, ok = mgr.get("does-not-exist")
	if ok {
		t.Fatal("expected ok=false for an unknown job ID")
	}
}

func TestNewJobIDsAreUnique(t *testing.T) {
	seen := make(map[string]bool)
	for i := 0; i < 1000; i++ {
		id := newJobID()
		if seen[id] {
			t.Fatalf("duplicate job ID generated: %s", id)
		}
		seen[id] = true
	}
}
