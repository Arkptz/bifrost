package bifrost

import (
	"testing"

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

		// Fail fast — request-bound 4xx, retrying/escalating cannot help.
		{"400 bad request", 400, normalMs, passthroughFailFast},
		{"401 unauthorized", 401, normalMs, passthroughFailFast},
		{"403 forbidden", 403, normalMs, passthroughFailFast},
		{"404 not found", 404, normalMs, passthroughFailFast},
		{"409 conflict", 409, normalMs, passthroughFailFast},
		{"422 unprocessable", 422, normalMs, passthroughFailFast},

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
