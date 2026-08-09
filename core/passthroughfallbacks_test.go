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
