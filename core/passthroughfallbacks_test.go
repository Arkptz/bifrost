package bifrost

import (
	"context"
	"net/http"
	"net/http/httptest"
	"sync/atomic"
	"testing"
	"time"

	schemas "github.com/maximhq/bifrost/core/schemas"
)

// TestGetRequestFields_PassthroughFallbacks verifies that fallbacks set on a
// BifrostPassthroughRequest are surfaced by GetRequestFields — the field must be
// reachable for the fallback machinery to see it.
func TestGetRequestFields_PassthroughFallbacks(t *testing.T) {
	want := []schemas.Fallback{
		{Provider: schemas.Anthropic, Model: "claude-haiku"},
		{Provider: schemas.OpenAI, Model: "gpt-4o"},
	}
	req := &schemas.BifrostRequest{
		RequestType: schemas.PassthroughRequest,
		PassthroughRequest: &schemas.BifrostPassthroughRequest{
			Provider:  schemas.Anthropic,
			Model:     "claude-sonnet",
			Fallbacks: want,
		},
	}

	provider, model, fallbacks := req.GetRequestFields()
	if provider != schemas.Anthropic {
		t.Errorf("provider = %q, want %q", provider, schemas.Anthropic)
	}
	if model != "claude-sonnet" {
		t.Errorf("model = %q, want %q", model, "claude-sonnet")
	}
	if len(fallbacks) != len(want) {
		t.Fatalf("fallbacks length = %d, want %d", len(fallbacks), len(want))
	}
	for i := range want {
		if fallbacks[i] != want[i] {
			t.Errorf("fallbacks[%d] = %+v, want %+v", i, fallbacks[i], want[i])
		}
	}
}

// TestClassifyPassthroughStatus is the money-critical centerpiece: it pins the
// per-status/condition failover action for every case in the design. A wrong
// verdict here can jump a cheap provider to an expensive one on a single error,
// so every branch is asserted explicitly.
func TestClassifyPassthroughStatus(t *testing.T) {
	// A "normal" upstream duration — the request actually reached the provider.
	const normalMs int64 = 250
	// A local fast-fail duration — fasthttp returned before the upstream was called.
	const fastFailMs int64 = 2

	tests := []struct {
		name       string
		statusCode int
		durationMs int64
		want       passthroughFailoverAction
	}{
		// 2xx — not a failure, no failover.
		{"200 ok", 200, normalMs, passthroughNotFailure},
		{"201 created", 201, normalMs, passthroughNotFailure},
		{"204 no content", 204, normalMs, passthroughNotFailure},

		// Fail fast — genuinely request-bound 4xx, retrying/escalating cannot help.
		{"400 bad request", 400, normalMs, passthroughFailFast},
		{"404 not found", 404, normalMs, passthroughFailFast},
		{"409 conflict", 409, normalMs, passthroughFailFast},
		{"422 unprocessable", 422, normalMs, passthroughFailFast},

		// 401/403 — reseller/gateway pool-account symptom: retry SAME provider
		// with a cooled key, never escalate (NEW REQ). NOT fail-fast.
		{"401 unauthorized pooled retry", 401, normalMs, passthroughRetrySamePooled},
		{"403 forbidden pooled retry", 403, normalMs, passthroughRetrySamePooled},

		// Retry same provider — transient request-timing 4xx.
		{"408 request timeout", 408, normalMs, passthroughRetrySame},
		{"425 too early", 425, normalMs, passthroughRetrySame},

		// 429 — retry SAME, NEVER escalate (explicit money-safety requirement).
		{"429 rate limited normal duration", 429, normalMs, passthroughRetrySame},
		{"429 rate limited fast duration", 429, fastFailMs, passthroughRetrySame},

		// 5xx with normal duration — real upstream failure, escalate after retries.
		{"500 normal", 500, normalMs, passthroughEscalate},
		{"502 normal", 502, normalMs, passthroughEscalate},
		{"503 normal", 503, normalMs, passthroughEscalate},
		{"504 normal", 504, normalMs, passthroughEscalate},
		{"529 overloaded", 529, normalMs, passthroughEscalate},

		// 502/503 with ~1-3ms duration — local fast-fail, upstream never called.
		// Retry same immediately; do NOT charge the escalation budget.
		{"502 local fast-fail 1ms", 502, 1, passthroughRetrySame},
		{"502 local fast-fail 3ms", 502, 3, passthroughRetrySame},
		{"503 local fast-fail 2ms", 503, 2, passthroughRetrySame},

		// 500/504 fast duration are NOT treated as local fast-fail — only 502/503
		// are the fasthttp local-failure surface, so these still escalate.
		{"500 fast duration still escalates", 500, fastFailMs, passthroughEscalate},
		{"504 fast duration still escalates", 504, fastFailMs, passthroughEscalate},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := classifyPassthroughStatus(tt.statusCode, tt.durationMs)
			if got != tt.want {
				t.Errorf("classifyPassthroughStatus(%d, %dms) = %v, want %v",
					tt.statusCode, tt.durationMs, got, tt.want)
			}
		})
	}
}

// TestClassifyPassthroughStatus_429NeverEscalates is a standalone guard for the
// single most dangerous cost regression: a 429 from a cheap provider must retry
// the SAME provider and must never escalate to a more expensive fallback.
func TestClassifyPassthroughStatus_429NeverEscalates(t *testing.T) {
	for _, durationMs := range []int64{1, 3, 50, 250, 5000} {
		got := classifyPassthroughStatus(429, durationMs)
		if got == passthroughEscalate {
			t.Fatalf("429 at %dms escalated — must retry same provider, never escalate", durationMs)
		}
		if got != passthroughRetrySame {
			t.Errorf("classifyPassthroughStatus(429, %dms) = %v, want passthroughRetrySame", durationMs, got)
		}
	}
}

// TestPrepareFallbackRequest_PassthroughDoesNotMutateOriginal proves the clone
// branch: preparing a passthrough fallback must not corrupt the primary request's
// PassthroughRequest, which is aliased through the shallow struct copy.
func TestPrepareFallbackRequest_PassthroughDoesNotMutateOriginal(t *testing.T) {
	account := NewMockAccount()
	account.AddProvider(schemas.OpenAI, 1, 1)
	bifrost := &Bifrost{account: account, logger: NewDefaultLogger(schemas.LogLevelError)}

	original := &schemas.BifrostPassthroughRequest{
		Provider: schemas.Anthropic,
		Model:    "claude-sonnet",
		Path:     "/v1/messages",
		Body:     []byte(`{"model":"claude-sonnet"}`),
	}
	req := &schemas.BifrostRequest{
		RequestType:        schemas.PassthroughRequest,
		PassthroughRequest: original,
	}

	fallbackReq := bifrost.prepareFallbackRequest(req, schemas.Fallback{Provider: schemas.OpenAI, Model: "gpt-4o"})
	if fallbackReq == nil {
		t.Fatal("prepareFallbackRequest returned nil for a configured provider")
	}

	if original.Provider != schemas.Anthropic {
		t.Errorf("original provider mutated to %q, want %q", original.Provider, schemas.Anthropic)
	}
	if original.Model != "claude-sonnet" {
		t.Errorf("original model mutated to %q, want %q", original.Model, "claude-sonnet")
	}
	if fallbackReq.PassthroughRequest == original {
		t.Error("fallback PassthroughRequest still aliases the original pointer")
	}
	if fallbackReq.PassthroughRequest.Provider != schemas.OpenAI {
		t.Errorf("fallback provider = %q, want %q", fallbackReq.PassthroughRequest.Provider, schemas.OpenAI)
	}
	if fallbackReq.PassthroughRequest.Model != "gpt-4o" {
		t.Errorf("fallback model = %q, want %q", fallbackReq.PassthroughRequest.Model, "gpt-4o")
	}
}

func statusHandler(status int, hits *atomic.Int32) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		hits.Add(1)
		w.WriteHeader(status)
		_, _ = w.Write([]byte(`{"ok":false}`))
	}
}

func newPassthroughTestClient(t *testing.T, account *MockAccount) *Bifrost {
	t.Helper()
	client, err := Init(context.Background(), schemas.BifrostConfig{
		Account: account,
		Logger:  NewDefaultLogger(schemas.LogLevelError),
	})
	if err != nil {
		t.Fatalf("failed to initialize bifrost: %v", err)
	}
	t.Cleanup(client.Shutdown)
	return client
}

// TestPassthroughFailover_5xxEscalates verifies a real upstream 5xx on the cheap
// primary escalates to the next (more expensive) fallback in the caller-supplied
// order, which succeeds.
func TestPassthroughFailover_5xxEscalates(t *testing.T) {
	var primaryHits, fallbackHits atomic.Int32
	primary := httptest.NewServer(statusHandler(http.StatusInternalServerError, &primaryHits))
	defer primary.Close()
	fallback := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		fallbackHits.Add(1)
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte(`{"ok":true}`))
	}))
	defer fallback.Close()

	account := NewMockAccount()
	account.AddProviderWithBaseURL(schemas.OpenAI, 1, 1, primary.URL)
	account.AddProviderWithBaseURL(schemas.Anthropic, 1, 1, fallback.URL)
	// The fallback carries an explicit model, so key selection runs the model
	// filter; default mock keys have an empty Models list (deny-all). Grant the
	// fallback key wildcard model support so escalation can select it (matches
	// streamfallback_test.go).
	account.SetKeysForProvider(schemas.Anthropic, []schemas.Key{
		{ID: "fallback-key", Value: *schemas.NewSecretVar("sk-fallback"), Models: schemas.WhiteList{"*"}, Weight: 100},
	})
	client := newPassthroughTestClient(t, account)

	ctx := schemas.NewBifrostContext(context.Background(), time.Now().Add(30*time.Second))
	resp, bifrostErr := client.Passthrough(ctx, schemas.OpenAI, &schemas.BifrostPassthroughRequest{
		Method:    http.MethodPost,
		Path:      "/v1/messages",
		Body:      []byte(`{"model":"m"}`),
		Fallbacks: []schemas.Fallback{{Provider: schemas.Anthropic, Model: "claude-haiku"}},
	})
	if bifrostErr != nil {
		t.Fatalf("Passthrough returned error: %s", bifrostErr.Error.Message)
	}
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("final status = %d, want 200 (should escalate to fallback)", resp.StatusCode)
	}
	if fallbackHits.Load() == 0 {
		t.Fatal("fallback provider was never called — 5xx did not escalate")
	}
}

// TestPassthroughFailover_429DoesNotEscalate is the money-safety integration
// guard: a 429 from the cheap primary must NOT jump to the expensive fallback.
// The upstream 429 body must reach the client verbatim and the fallback provider
// must never be called.
func TestPassthroughFailover_429DoesNotEscalate(t *testing.T) {
	var primaryHits, fallbackHits atomic.Int32
	primary := httptest.NewServer(statusHandler(http.StatusTooManyRequests, &primaryHits))
	defer primary.Close()
	fallback := httptest.NewServer(statusHandler(http.StatusOK, &fallbackHits))
	defer fallback.Close()

	account := NewMockAccount()
	account.AddProviderWithBaseURL(schemas.OpenAI, 1, 1, primary.URL)
	account.AddProviderWithBaseURL(schemas.Anthropic, 1, 1, fallback.URL)
	client := newPassthroughTestClient(t, account)

	ctx := schemas.NewBifrostContext(context.Background(), time.Now().Add(30*time.Second))
	resp, bifrostErr := client.Passthrough(ctx, schemas.OpenAI, &schemas.BifrostPassthroughRequest{
		Method:    http.MethodPost,
		Path:      "/v1/messages",
		Body:      []byte(`{"model":"m"}`),
		Fallbacks: []schemas.Fallback{{Provider: schemas.Anthropic, Model: "claude-haiku"}},
	})
	if bifrostErr != nil {
		t.Fatalf("Passthrough returned error: %s", bifrostErr.Error.Message)
	}
	if resp.StatusCode != http.StatusTooManyRequests {
		t.Fatalf("final status = %d, want 429 verbatim", resp.StatusCode)
	}
	if fallbackHits.Load() != 0 {
		t.Fatalf("fallback provider was called %d time(s) — 429 must NEVER escalate", fallbackHits.Load())
	}
}

// TestPassthroughFailover_NoFallbacksUnchanged is the byte-identity regression
// guard: with no fallbacks set, a non-2xx passthrough status is returned verbatim
// exactly as before this feature, with no failover machinery engaged.
func TestPassthroughFailover_NoFallbacksUnchanged(t *testing.T) {
	var hits atomic.Int32
	server := httptest.NewServer(statusHandler(http.StatusInternalServerError, &hits))
	defer server.Close()

	account := NewMockAccount()
	account.AddProviderWithBaseURL(schemas.OpenAI, 1, 1, server.URL)
	client := newPassthroughTestClient(t, account)

	ctx := schemas.NewBifrostContext(context.Background(), time.Now().Add(30*time.Second))
	resp, bifrostErr := client.Passthrough(ctx, schemas.OpenAI, &schemas.BifrostPassthroughRequest{
		Method: http.MethodPost,
		Path:   "/v1/messages",
		Body:   []byte(`{"model":"m"}`),
	})
	if bifrostErr != nil {
		t.Fatalf("Passthrough returned error: %s", bifrostErr.Error.Message)
	}
	if resp.StatusCode != http.StatusInternalServerError {
		t.Fatalf("final status = %d, want 500 verbatim (no failover without fallbacks)", resp.StatusCode)
	}
	if hits.Load() != 1 {
		t.Fatalf("server hits = %d, want exactly 1 (no retry/escalation without fallbacks)", hits.Load())
	}
}

func TestParsePassthroughRetryAfter(t *testing.T) {
	future := time.Now().Add(3 * time.Second).UTC().Format(http.TimeFormat)
	past := time.Now().Add(-time.Hour).UTC().Format(http.TimeFormat)

	tests := []struct {
		name    string
		headers map[string]string
		wantMin time.Duration
		wantMax time.Duration
	}{
		{"absent", map[string]string{}, 0, 0},
		{"delay seconds", map[string]string{"Retry-After": "5"}, 5 * time.Second, 5 * time.Second},
		{"delay seconds capped", map[string]string{"Retry-After": "3600"}, passthroughRetryAfterCap, passthroughRetryAfterCap},
		{"max int64 capped not zero", map[string]string{"Retry-After": "9223372036854775807"}, passthroughRetryAfterCap, passthroughRetryAfterCap},
		{"negative", map[string]string{"Retry-After": "-5"}, 0, 0},
		{"http date future honored", map[string]string{"Retry-After": future}, time.Second, 4 * time.Second},
		{"http date past", map[string]string{"Retry-After": past}, 0, 0},
		{"garbage", map[string]string{"Retry-After": "soon"}, 0, 0},
		{"case insensitive header", map[string]string{"retry-after": "5"}, 5 * time.Second, 5 * time.Second},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := parsePassthroughRetryAfter(tt.headers)
			if got < tt.wantMin || got > tt.wantMax {
				t.Fatalf("parsePassthroughRetryAfter(%v) = %v, want within [%v, %v]", tt.headers, got, tt.wantMin, tt.wantMax)
			}
			if got < 0 || got > passthroughRetryAfterCap {
				t.Fatalf("result %v escaped bounds [0, %v]", got, passthroughRetryAfterCap)
			}
		})
	}
}
