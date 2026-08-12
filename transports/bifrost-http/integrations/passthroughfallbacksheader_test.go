package integrations

import (
	"testing"

	"github.com/maximhq/bifrost/core/schemas"
	"github.com/valyala/fasthttp"
)

// parsePassthroughFallbacks is the ENTRY POINT of the cost-aware failover feature: it turns
// the public x-bf-fallbacks header into the ladder the core loop walks. It shipped with no
// transport-level test, so neutralizing it (returning nil) left every suite green while the
// feature silently became a no-op for real HTTP callers — the core tests build their ladders
// in Go and never exercise the header.
//
// Ordering is load-bearing: the caller sorts cheap->expensive and the ladder is walked in
// order, so a parser that reorders entries would spend money in the wrong sequence.
func TestParsePassthroughFallbacks(t *testing.T) {
	for _, tc := range []struct {
		name   string
		header string
		want   []schemas.Fallback
	}{
		{
			name:   "absent header yields no ladder",
			header: "",
			want:   nil,
		},
		{
			name:   "ordered pairs preserve caller order",
			header: "anthropic/claude-3-5-haiku-latest,openai/gpt-4o-mini",
			want: []schemas.Fallback{
				{Provider: schemas.Anthropic, Model: "claude-3-5-haiku-latest"},
				{Provider: schemas.OpenAI, Model: "gpt-4o-mini"},
			},
		},
		{
			name:   "surrounding whitespace is tolerated",
			header: " anthropic/claude-haiku , openai/gpt-4o-mini ",
			want: []schemas.Fallback{
				{Provider: schemas.Anthropic, Model: "claude-haiku"},
				{Provider: schemas.OpenAI, Model: "gpt-4o-mini"},
			},
		},
		{
			name:   "malformed entries are skipped, valid ones survive",
			header: "no-slash,/missing-provider,anthropic/,anthropic/claude-haiku",
			want: []schemas.Fallback{
				{Provider: schemas.Anthropic, Model: "claude-haiku"},
			},
		},
		{
			name:   "a model containing a slash keeps its full name",
			header: "vertex/publishers/google/models/gemini-2.5-flash",
			want: []schemas.Fallback{
				{Provider: schemas.Vertex, Model: "publishers/google/models/gemini-2.5-flash"},
			},
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			ctx := &fasthttp.RequestCtx{}
			if tc.header != "" {
				ctx.Request.Header.Set("x-bf-fallbacks", tc.header)
			}
			got := parsePassthroughFallbacks(ctx)
			if len(got) != len(tc.want) {
				t.Fatalf("parsed %d fallback(s) %+v from %q, want %d %+v", len(got), got, tc.header, len(tc.want), tc.want)
			}
			for i := range tc.want {
				if got[i].Provider != tc.want[i].Provider || got[i].Model != tc.want[i].Model {
					t.Fatalf("fallback[%d] = %s/%s, want %s/%s (order is cost-ordered and load-bearing)",
						i, got[i].Provider, got[i].Model, tc.want[i].Provider, tc.want[i].Model)
				}
			}
		})
	}
}
