package main

import (
	"context"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

func TestLineWriterSplitsOnNewlinesAcrossWrites(t *testing.T) {
	var lines []string
	lw := &lineWriter{onLine: func(s string) { lines = append(lines, s) }}

	// Deliberately split a single logical line across two Write calls, the
	// way a subprocess's stdout can arrive in arbitrary chunks.
	_, _ = lw.Write([]byte("hello wor"))
	_, _ = lw.Write([]byte("ld\nsecond line\nthird li"))
	lw.flush()

	want := []string{"hello world", "second line", "third li"}
	if len(lines) != len(want) {
		t.Fatalf("got %v, want %v", lines, want)
	}
	for i := range want {
		if lines[i] != want[i] {
			t.Fatalf("got %v, want %v", lines, want)
		}
	}
}

func TestLineWriterFlushIsNoOpWhenBufferEmpty(t *testing.T) {
	var lines []string
	lw := &lineWriter{onLine: func(s string) { lines = append(lines, s) }}
	_, _ = lw.Write([]byte("complete line\n"))
	lw.flush()
	if len(lines) != 1 || lines[0] != "complete line" {
		t.Fatalf("got %v", lines)
	}
	lw.flush() // should not append an empty line
	if len(lines) != 1 {
		t.Fatalf("flush() with an empty buffer should be a no-op, got %v", lines)
	}
}

func TestRunPythonStreamsOutputIntoJob(t *testing.T) {
	// Exercise runPython's plumbing (exec.CommandContext, combined
	// stdout/stderr capture, streaming into job.appendLine) against a
	// throwaway shell script rather than requiring xgboost_model's real
	// Python dependencies to be installed just to run this test.
	dir := t.TempDir()
	script := filepath.Join(dir, "fake_pipeline.sh")
	contents := "#!/bin/sh\necho line one\necho line two 1>&2\necho line three\n"
	if err := os.WriteFile(script, []byte(contents), 0o755); err != nil {
		t.Fatal(err)
	}

	job := newJob("01388500", []int{1}, 60)
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	err := runPython(ctx, job, "sh", dir, script)
	if err != nil {
		t.Fatalf("runPython: %v", err)
	}

	joined := strings.Join(job.lines, "\n")
	for _, want := range []string{"line one", "line two", "line three"} {
		if !strings.Contains(joined, want) {
			t.Fatalf("expected log output to contain %q, got: %q", want, joined)
		}
	}
}

func TestRunPythonPropagatesNonZeroExit(t *testing.T) {
	dir := t.TempDir()
	script := filepath.Join(dir, "fails.sh")
	if err := os.WriteFile(script, []byte("#!/bin/sh\necho about to fail\nexit 7\n"), 0o755); err != nil {
		t.Fatal(err)
	}

	job := newJob("01388500", []int{1}, 60)
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	err := runPython(ctx, job, "sh", dir, script)
	if err == nil {
		t.Fatal("expected a non-nil error for a script that exits non-zero")
	}
}
