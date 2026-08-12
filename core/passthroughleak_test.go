package bifrost

import (
	"fmt"
	"net/http"
	"net/http/httptest"
	"runtime"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	schemas "github.com/maximhq/bifrost/core/schemas"
)

// TestReviewerNilContextStreamTerminatesOnShutdownR7 is the shutdown regression.
// A nil-ctx passthrough stream runs on a context.WithoutCancel child (Done()
// stripped), so before the fix Shutdown() — which cancels only bifrost.ctx — left
// the provider stream goroutine and its upstream conn alive until upstream EOF or
// the 120s idle timeout. The fix stashes bifrost.ctx.Done() on the child so
// SetupStreamCancellation's bounded goroutine also selects on it. The upstream here
// never closes on its own; the stream channel MUST close promptly after Shutdown().
func TestReviewerNilContextStreamTerminatesOnShutdownR7(t *testing.T) {
	release := make(chan struct{})
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
		fl, _ := w.(http.Flusher)
		fmt.Fprint(w, "data: FIRST\n\n")
		if fl != nil {
			fl.Flush()
		}
		<-release // hold the stream open until the test tears down
	}))
	defer server.Close()
	defer close(release)

	account := NewMockAccount()
	account.AddProviderWithBaseURL(schemas.OpenAI, 1, 1, server.URL)
	account.AddProviderWithBaseURL(schemas.Anthropic, 1, 1, server.URL)
	account.SetKeysForProvider(schemas.Anthropic, []schemas.Key{
		{ID: "fallback-key", Value: *schemas.NewSecretVar("sk-fallback"), Models: schemas.WhiteList{"*"}, Weight: 100},
	})
	client := newPassthroughTestClient(t, account)

	stream, bifrostErr := client.PassthroughStream(nil, schemas.OpenAI, &schemas.BifrostPassthroughRequest{
		Method: http.MethodPost, Path: "/v1/messages", Body: []byte(`{"model":"m"}`),
		Fallbacks: []schemas.Fallback{{Provider: schemas.Anthropic, Model: "claude-haiku"}},
	})
	if bifrostErr != nil {
		t.Fatalf("PassthroughStream(nil) failed: %s", bifrostErr.Error.Message)
	}

	drained := make(chan struct{})
	go func() {
		for range stream {
		}
		close(drained)
	}()

	client.Shutdown()

	select {
	case <-drained:
	case <-time.After(2 * time.Second):
		t.Fatal("stream still live 2s after Shutdown() — nil-ctx passthrough stream not torn down")
	}
}

// settleGoroutines forces GC and lets any exiting goroutines unwind, then returns
// the resident goroutine count. It samples until the count stabilizes so a slow
// watcher shutdown isn't mistaken for a leak.
func settleGoroutines() int {
	last := runtime.NumGoroutine()
	for i := 0; i < 20; i++ {
		runtime.GC()
		time.Sleep(20 * time.Millisecond)
		n := runtime.NumGoroutine()
		if n == last {
			return n
		}
		last = n
	}
	return last
}

// TestPassthroughNilCtx_NoGoroutineLeak is the leak guard for the nil-ctx
// passthrough path. It drives the real public entry client.Passthrough(nil, …) N
// times against a live server; the nil ctx makes handleRequest substitute
// bifrost.ctx (built via NewBifrostContextWithCancel — cancellable, production
// shape) as the isolation parent. Before the fix each call spawned a
// watchCancellation goroutine blocked on bifrost.ctx.Done() with nothing to
// Cancel() it, so the resident count grew by ~N. The test asserts the settled
// delta stays near zero and LOGS the actual value.
func TestPassthroughNilCtx_NoGoroutineLeak(t *testing.T) {
	var hits atomic.Int32
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		hits.Add(1)
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte(`{"ok":true}`))
	}))
	defer server.Close()

	account := NewMockAccount()
	account.AddProviderWithBaseURL(schemas.OpenAI, 1, 1, server.URL)
	account.AddProviderWithBaseURL(schemas.Anthropic, 1, 1, server.URL)
	// Fallback key with wildcard models so armPassthroughFailover arms the
	// isolation path this leak lives on (a nil-ctx passthrough with fallbacks).
	account.SetKeysForProvider(schemas.Anthropic, []schemas.Key{
		{ID: "fallback-key", Value: *schemas.NewSecretVar("sk-fallback"), Models: schemas.WhiteList{"*"}, Weight: 100},
	})
	client := newPassthroughTestClient(t, account)

	const n = 100
	// Warm up one call so lazily-started worker goroutines exist before baseline.
	if _, err := client.Passthrough(nil, schemas.OpenAI, &schemas.BifrostPassthroughRequest{
		Method: http.MethodPost, Path: "/v1/messages", Body: []byte(`{"model":"m"}`),
		Fallbacks: []schemas.Fallback{{Provider: schemas.Anthropic, Model: "claude-haiku"}},
	}); err != nil {
		t.Fatalf("warmup Passthrough(nil) failed: %s", err.Error.Message)
	}
	before := settleGoroutines()

	for i := 0; i < n; i++ {
		resp, err := client.Passthrough(nil, schemas.OpenAI, &schemas.BifrostPassthroughRequest{
			Method: http.MethodPost, Path: "/v1/messages", Body: []byte(`{"model":"m"}`),
			Fallbacks: []schemas.Fallback{{Provider: schemas.Anthropic, Model: "claude-haiku"}},
		})
		if err != nil {
			t.Fatalf("Passthrough(nil) #%d failed: %s", i, err.Error.Message)
		}
		if resp.StatusCode != http.StatusOK {
			t.Fatalf("Passthrough(nil) #%d status = %d, want 200", i, resp.StatusCode)
		}
	}

	after := settleGoroutines()
	delta := after - before
	t.Logf("goroutine delta over %d nil-ctx passthrough calls: before=%d after=%d delta=%d", n, before, after, delta)
	// Allow a small slack for transient runtime goroutines; a real leak is ~N.
	if delta > n/10 {
		t.Fatalf("LEAK: %d resident goroutines after %d nil-ctx passthrough calls (want ~0)", delta, n)
	}
}

// TestPassthroughStreamNilCtx_FullBodyAfterReturn guards against a direction-(a)
// mistake: a naive defer cancel() on the isolated context would kill an in-flight
// stream. The nil-ctx PassthroughStream returns a channel the caller drains AFTER
// the call returns; this asserts the full multi-chunk body still arrives.
func TestPassthroughStreamNilCtx_FullBodyAfterReturn(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
		fl, _ := w.(http.Flusher)
		for _, part := range []string{"CHUNK-A", "CHUNK-B", "CHUNK-C"} {
			fmt.Fprintf(w, "data: %s\n\n", part)
			if fl != nil {
				fl.Flush()
			}
		}
	}))
	defer server.Close()

	account := NewMockAccount()
	account.AddProviderWithBaseURL(schemas.OpenAI, 1, 1, server.URL)
	account.AddProviderWithBaseURL(schemas.Anthropic, 1, 1, server.URL)
	account.SetKeysForProvider(schemas.Anthropic, []schemas.Key{
		{ID: "fallback-key", Value: *schemas.NewSecretVar("sk-fallback"), Models: schemas.WhiteList{"*"}, Weight: 100},
	})
	client := newPassthroughTestClient(t, account)

	stream, bifrostErr := client.PassthroughStream(nil, schemas.OpenAI, &schemas.BifrostPassthroughRequest{
		Method: http.MethodPost, Path: "/v1/messages", Body: []byte(`{"model":"m"}`),
		Fallbacks: []schemas.Fallback{{Provider: schemas.Anthropic, Model: "claude-haiku"}},
	})
	if bifrostErr != nil {
		t.Fatalf("PassthroughStream(nil) failed: %s", bifrostErr.Error.Message)
	}

	body, status, errs := drainPassthroughStream(stream)
	if len(errs) > 0 {
		t.Fatalf("stream emitted error chunks: %v", errs)
	}
	if status != http.StatusOK {
		t.Fatalf("final status = %d, want 200", status)
	}
	for _, part := range []string{"CHUNK-A", "CHUNK-B", "CHUNK-C"} {
		if !strings.Contains(body, part) {
			t.Fatalf("stream body %q missing %q — a premature cancel truncated the stream", body, part)
		}
	}
}
