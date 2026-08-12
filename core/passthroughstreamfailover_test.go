package bifrost

import (
	"context"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	schemas "github.com/maximhq/bifrost/core/schemas"
)

// drainPassthroughStream collects the forwarded passthrough body, the last status
// seen on a data chunk, and any error-chunk messages.
func drainPassthroughStream(ch chan *schemas.BifrostStreamChunk) (body string, lastStatus int, errs []string) {
	var b strings.Builder
	for chunk := range ch {
		if chunk.BifrostError != nil && chunk.BifrostError.Error != nil {
			errs = append(errs, chunk.BifrostError.Error.Message)
			continue
		}
		if chunk.BifrostPassthroughResponse != nil {
			lastStatus = chunk.BifrostPassthroughResponse.StatusCode
			b.Write(chunk.BifrostPassthroughResponse.Body)
		}
	}
	return b.String(), lastStatus, errs
}

func doPassthroughStream(t *testing.T, client *Bifrost, provider schemas.ModelProvider, fallbacks []schemas.Fallback) (chan *schemas.BifrostStreamChunk, *schemas.BifrostError) {
	t.Helper()
	ctx := schemas.NewBifrostContext(context.Background(), time.Now().Add(30*time.Second))
	return client.PassthroughStream(ctx, provider, &schemas.BifrostPassthroughRequest{
		Method:    http.MethodPost,
		Path:      "/v1/messages",
		Body:      []byte(`{"model":"m"}`),
		Fallbacks: fallbacks,
	})
}

// STREAM-1: a primary passthrough stream whose FIRST chunk is a real 5xx must
// escalate to the fallback, and the client must receive the FALLBACK's stream
// content — never the primary's 500 body. maxRetries=0 forces immediate
// escalation (no same-provider retry to observe here).
func TestPassthroughStreamFailover_5xxFirstChunkEscalates(t *testing.T) {
	var primaryHits atomic.Int32
	// Primary emits two flushed chunks so the peek re-injects the first and the
	// background drain must consume the rest — a leak or deadlock would time out.
	primarySrv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		primaryHits.Add(1)
		w.WriteHeader(http.StatusInternalServerError)
		fl, _ := w.(http.Flusher)
		fmt.Fprint(w, "data: PRIMARY-500-A\n\n")
		if fl != nil {
			fl.Flush()
		}
		fmt.Fprint(w, "data: PRIMARY-500-B\n\n")
		if fl != nil {
			fl.Flush()
		}
	}))
	defer primarySrv.Close()

	var fallbackHits atomic.Int32
	fallbackSrv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		fallbackHits.Add(1)
		w.WriteHeader(http.StatusOK)
		fl, _ := w.(http.Flusher)
		fmt.Fprint(w, "data: FALLBACK-STREAM\n\n")
		if fl != nil {
			fl.Flush()
		}
	}))
	defer fallbackSrv.Close()

	account := NewMockAccount()
	account.AddProviderWithBaseURL(schemas.OpenAI, 1, 1, primarySrv.URL)
	account.AddProviderWithBaseURL(schemas.Anthropic, 1, 1, fallbackSrv.URL)
	account.configs[schemas.OpenAI].NetworkConfig.MaxRetries = 0
	account.configs[schemas.Anthropic].NetworkConfig.MaxRetries = 0
	account.SetKeysForProvider(schemas.OpenAI, []schemas.Key{
		{ID: "primary-key", Value: *schemas.NewSecretVar("sk-primary"), Models: schemas.WhiteList{"*"}, Weight: 100},
	})
	account.SetKeysForProvider(schemas.Anthropic, []schemas.Key{
		{ID: "fallback-key", Value: *schemas.NewSecretVar("sk-fallback"), Models: schemas.WhiteList{"*"}, Weight: 100},
	})
	client := newPassthroughTestClient(t, account)

	stream, bifrostErr := doPassthroughStream(t, client, schemas.OpenAI,
		[]schemas.Fallback{{Provider: schemas.Anthropic, Model: "claude-haiku"}})
	if bifrostErr != nil {
		t.Fatalf("passthrough stream failed (primary=%d fallback=%d): %s", primaryHits.Load(), fallbackHits.Load(), bifrostErr.Error.Message)
	}
	body, status, errs := drainPassthroughStream(stream)
	if fallbackHits.Load() != 1 {
		t.Fatalf("fallback hits = %d, want 1 (5xx first chunk must escalate exactly once)", fallbackHits.Load())
	}
	if len(errs) > 0 {
		t.Fatalf("fallback stream emitted error chunks: %v", errs)
	}
	if status != http.StatusOK {
		t.Fatalf("final status = %d, want 200 (fallback served the stream)", status)
	}
	if !strings.Contains(body, "FALLBACK-STREAM") {
		t.Fatalf("client body = %q, want the FALLBACK stream content", body)
	}
	if strings.Contains(body, "PRIMARY-500") {
		t.Fatalf("client body = %q leaked the abandoned primary 500 stream", body)
	}
}

// STREAM-2 (MONEY RULE): a transient 5xx first chunk on the primary that RECOVERS
// on the same-provider retry must NEVER reach the fallback. maxRetries=1 gives one
// retry-same; the primary returns 200 on its second hit, so the pricier fallback
// must record zero hits.
func TestPassthroughStreamFailover_5xxRecoversSameNoEscalate(t *testing.T) {
	// countingHandler: 500 on hit 1, then 200. Reused from passthroughfailover_test.go.
	primary := &countingHandler{failStatus: http.StatusInternalServerError, failN: 1}
	primarySrv := httptest.NewServer(primary)
	defer primarySrv.Close()

	var fallbackHits atomic.Int32
	fallbackSrv := httptest.NewServer(statusHandler(http.StatusOK, &fallbackHits))
	defer fallbackSrv.Close()

	account := NewMockAccount()
	account.AddProviderWithBaseURL(schemas.OpenAI, 1, 1, primarySrv.URL)
	account.AddProviderWithBaseURL(schemas.Anthropic, 1, 1, fallbackSrv.URL)
	account.configs[schemas.OpenAI].NetworkConfig.MaxRetries = 1
	account.configs[schemas.OpenAI].NetworkConfig.RetryBackoffInitial = time.Millisecond
	account.SetKeysForProvider(schemas.OpenAI, []schemas.Key{
		{ID: "primary-key", Value: *schemas.NewSecretVar("sk-primary"), Models: schemas.WhiteList{"*"}, Weight: 100},
	})
	account.SetKeysForProvider(schemas.Anthropic, []schemas.Key{
		{ID: "fallback-key", Value: *schemas.NewSecretVar("sk-fallback"), Models: schemas.WhiteList{"*"}, Weight: 100},
	})
	client := newPassthroughTestClient(t, account)

	stream, bifrostErr := doPassthroughStream(t, client, schemas.OpenAI,
		[]schemas.Fallback{{Provider: schemas.Anthropic, Model: "claude-haiku"}})
	if bifrostErr != nil {
		t.Fatalf("passthrough stream failed (primary=%d): %s", primary.hits.Load(), bifrostErr.Error.Message)
	}
	_, status, errs := drainPassthroughStream(stream)
	if primary.hits.Load() != 2 {
		t.Fatalf("primary hits = %d, want 2 (500 must retry the SAME provider, then recover)", primary.hits.Load())
	}
	if fallbackHits.Load() != 0 {
		t.Fatalf("fallback called %d time(s) — a transient 5xx escalated to a pricier provider (MONEY RULE VIOLATION)", fallbackHits.Load())
	}
	if len(errs) > 0 {
		t.Fatalf("recovered stream emitted error chunks: %v", errs)
	}
	if status != http.StatusOK {
		t.Fatalf("final status = %d, want 200 (primary recovered on retry-same)", status)
	}
}

// STREAM-3: a 2xx passthrough stream is untouched even with the classifier active
// (fallbacks configured). First chunk re-emitted in order, full body delivered,
// no escalation.
func TestPassthroughStreamFailover_2xxUntouched(t *testing.T) {
	payloads := []string{"CHUNK-1", "CHUNK-2", "CHUNK-3"}
	var primaryHits atomic.Int32
	primarySrv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		primaryHits.Add(1)
		w.WriteHeader(http.StatusOK)
		fl, _ := w.(http.Flusher)
		for _, p := range payloads {
			fmt.Fprintf(w, "data: %s\n\n", p)
			if fl != nil {
				fl.Flush()
			}
		}
	}))
	defer primarySrv.Close()

	var fallbackHits atomic.Int32
	fallbackSrv := httptest.NewServer(statusHandler(http.StatusOK, &fallbackHits))
	defer fallbackSrv.Close()

	account := NewMockAccount()
	account.AddProviderWithBaseURL(schemas.OpenAI, 1, 1, primarySrv.URL)
	account.AddProviderWithBaseURL(schemas.Anthropic, 1, 1, fallbackSrv.URL)
	account.SetKeysForProvider(schemas.OpenAI, []schemas.Key{
		{ID: "primary-key", Value: *schemas.NewSecretVar("sk-primary"), Models: schemas.WhiteList{"*"}, Weight: 100},
	})
	account.SetKeysForProvider(schemas.Anthropic, []schemas.Key{
		{ID: "fallback-key", Value: *schemas.NewSecretVar("sk-fallback"), Models: schemas.WhiteList{"*"}, Weight: 100},
	})
	client := newPassthroughTestClient(t, account)

	stream, bifrostErr := doPassthroughStream(t, client, schemas.OpenAI,
		[]schemas.Fallback{{Provider: schemas.Anthropic, Model: "claude-haiku"}})
	if bifrostErr != nil {
		t.Fatalf("2xx passthrough stream failed: %s", bifrostErr.Error.Message)
	}
	body, status, errs := drainPassthroughStream(stream)
	if primaryHits.Load() != 1 {
		t.Fatalf("primary hits = %d, want 1 (2xx must not retry or escalate)", primaryHits.Load())
	}
	if fallbackHits.Load() != 0 {
		t.Fatalf("fallback hits = %d, want 0 (2xx must not escalate)", fallbackHits.Load())
	}
	if len(errs) > 0 {
		t.Fatalf("2xx stream emitted error chunks: %v", errs)
	}
	if status != http.StatusOK {
		t.Fatalf("final status = %d, want 200", status)
	}
	// Full body delivered, in order (first chunk re-emitted, not dropped by the peek).
	want := "data: CHUNK-1\n\ndata: CHUNK-2\n\ndata: CHUNK-3\n\n"
	if body != want {
		t.Fatalf("stream body = %q, want %q (order/boundaries not preserved)", body, want)
	}
}

// STREAM-4: with NO fallbacks configured, a non-2xx passthrough stream behaves
// exactly as before — the classifier is not installed, the 5xx forwards verbatim
// as stream data, and there is exactly ONE upstream call (no synthesized error,
// no retry-same, no escalation path).
func TestPassthroughStreamFailover_NoFallbacks5xxSingleCall(t *testing.T) {
	var hits atomic.Int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		hits.Add(1)
		w.WriteHeader(http.StatusInternalServerError)
		fl, _ := w.(http.Flusher)
		fmt.Fprint(w, "data: UPSTREAM-500\n\n")
		if fl != nil {
			fl.Flush()
		}
	}))
	defer srv.Close()

	account := NewMockAccount()
	account.AddProviderWithBaseURL(schemas.OpenAI, 1, 1, srv.URL)
	account.SetKeysForProvider(schemas.OpenAI, []schemas.Key{
		{ID: "primary-key", Value: *schemas.NewSecretVar("sk-primary"), Models: schemas.WhiteList{"*"}, Weight: 100},
	})
	client := newPassthroughTestClient(t, account)

	stream, bifrostErr := doPassthroughStream(t, client, schemas.OpenAI, nil)
	if bifrostErr != nil {
		t.Fatalf("no-fallback passthrough stream returned error: %s", bifrostErr.Error.Message)
	}
	body, status, errs := drainPassthroughStream(stream)
	if hits.Load() != 1 {
		t.Fatalf("upstream hits = %d, want exactly 1 (no fallbacks: 5xx must not retry or escalate)", hits.Load())
	}
	if len(errs) > 0 {
		t.Fatalf("no-fallback 5xx stream synthesized error chunks %v — must forward verbatim", errs)
	}
	if status != http.StatusInternalServerError {
		t.Fatalf("final status = %d, want 500 forwarded verbatim", status)
	}
	if !strings.Contains(body, "UPSTREAM-500") {
		t.Fatalf("client body = %q, want the verbatim upstream 500 body", body)
	}
}

// BRIDGE GUARD. The race fix only works if the dial-site registry actually hands a
// *headerDeadlineConn to StreamPassthrough: closers unblock a parked read via that handle
// instead of closing the stream from a third goroutine (read-after-release). The dedicated
// unit test injects its own handle, so it CANNOT see a broken bridge — with
// lookupStreamConn stubbed to nil the whole suite stayed green while the original repro
// went back to 3 data races. This test closes that hole by asserting the handle is present
// on a REAL passthrough stream. Discriminator (revert-proof): handle nil => FAIL.
func TestPassthroughStream_ConnHandleReachesCancellation(t *testing.T) {
	// The handler parks after flushing its first chunk so the stream is GUARANTEED live
	// while the handle is sampled. Returning immediately makes this test flaky: the reader
	// can finish and clear the handle in its defer before the sample runs, reporting a
	// broken bridge for the wrong reason.
	hold := make(chan struct{})
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
		fl, _ := w.(http.Flusher)
		fmt.Fprint(w, "data: CHUNK\n\n")
		if fl != nil {
			fl.Flush()
		}
		<-hold
	}))
	defer srv.Close()

	account := NewMockAccount()
	account.AddProviderWithBaseURL(schemas.OpenAI, 1, 1, srv.URL)
	account.SetKeysForProvider(schemas.OpenAI, []schemas.Key{
		{ID: "k", Value: *schemas.NewSecretVar("sk-k"), Models: schemas.WhiteList{"*"}, Weight: 100},
	})
	client := newPassthroughTestClient(t, account)

	ctx := schemas.NewBifrostContext(context.Background(), time.Now().Add(30*time.Second))
	ch, bifrostErr := client.PassthroughStream(ctx, schemas.OpenAI, &schemas.BifrostPassthroughRequest{
		Method: http.MethodPost,
		Path:   "/v1/messages",
		Body:   []byte(`{"model":"m"}`),
	})
	if bifrostErr != nil {
		t.Fatalf("passthrough stream failed: %s", bifrostErr.Error.Message)
	}

	// Sample while the stream is live: StreamPassthrough clears the handle in its reader
	// defer, so draining first would race the teardown and read nil for the wrong reason.
	var got any
	deadline := time.Now().Add(2 * time.Second)
	for time.Now().Before(deadline) {
		if v := ctx.Value(schemas.BifrostContextKeyStreamConnHandle); v != nil {
			got = v
			break
		}
		time.Sleep(2 * time.Millisecond)
	}
	close(hold)
	body, status, _ := drainPassthroughStream(ch)

	if got == nil {
		t.Fatalf("no %v on the context during a live passthrough stream (status=%d body=%q) — the dial->stream registry bridge is broken, so every closer silently falls back to closing the body stream from a third goroutine (read-after-release UAF)",
			schemas.BifrostContextKeyStreamConnHandle, status, body)
	}
	if _, ok := got.(interface{ SetReadDeadline(time.Time) error }); !ok {
		t.Fatalf("stashed handle is %T, which cannot SetReadDeadline — closers cannot unblock a parked read with it", got)
	}
	if status != http.StatusOK || !strings.Contains(body, "CHUNK") {
		t.Fatalf("stream did not deliver its body (status=%d body=%q) — test vacuous", status, body)
	}
}
