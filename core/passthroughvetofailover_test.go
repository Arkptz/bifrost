package bifrost

import (
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	schemas "github.com/maximhq/bifrost/core/schemas"
)

const passthroughVetoMessage = "VETO-AUTH-DENIED"

// vetoPlugin short-circuits every PreLLMHook for vetoProvider with an
// AllowFallbacks=false error — the shape a governance/auth/policy plugin uses to
// forbid fallbacks (a veto carries no HTTP status). It passes other providers
// through untouched, so an (incorrectly) escalated leg lands on the live fallback
// and is observable.
type vetoPlugin struct {
	vetoProvider schemas.ModelProvider
	preHits      atomic.Int32
}

func (p *vetoPlugin) GetName() string { return "vetoPlugin" }
func (p *vetoPlugin) Cleanup() error  { return nil }
func (p *vetoPlugin) PreRequestHook(ctx *schemas.BifrostContext, req *schemas.BifrostRequest) error {
	return nil
}
func (p *vetoPlugin) PreLLMHook(ctx *schemas.BifrostContext, req *schemas.BifrostRequest) (*schemas.BifrostRequest, *schemas.LLMPluginShortCircuit, error) {
	provider, _, _ := req.GetRequestFields()
	if provider != p.vetoProvider {
		return req, nil, nil
	}
	p.preHits.Add(1)
	return req, &schemas.LLMPluginShortCircuit{
		Error: &schemas.BifrostError{
			IsBifrostError: false,
			AllowFallbacks: schemas.Ptr(false),
			Error: &schemas.ErrorField{
				Message: passthroughVetoMessage,
			},
		},
	}, nil
}
func (p *vetoPlugin) PostLLMHook(ctx *schemas.BifrostContext, resp *schemas.BifrostResponse, bifrostErr *schemas.BifrostError) (*schemas.BifrostResponse, *schemas.BifrostError, error) {
	return resp, bifrostErr, nil
}

func newVetoClient(t *testing.T, account *MockAccount, plugin schemas.LLMPlugin) *Bifrost {
	t.Helper()
	client, err := Init(t.Context(), schemas.BifrostConfig{
		Account:    account,
		Logger:     NewDefaultLogger(schemas.LogLevelError),
		LLMPlugins: []schemas.LLMPlugin{plugin},
	})
	if err != nil {
		t.Fatalf("failed to initialize bifrost: %v", err)
	}
	t.Cleanup(client.Shutdown)
	return client
}

// F4 (CWE-863): a plugin veto (AllowFallbacks=false) on a STREAMING passthrough WITH
// fallbacks carries no HTTP status, so streamFailoverStatus reports status 0 —
// which classifies as passthroughEscalate. The fix gates on shouldTryFallbacks
// BEFORE classifying, so the veto is returned verbatim and the pricier fallback is
// NEVER called. Without the guard the veto would be retried then escalated
// cheap->expensive (a policy/auth denial silently promoted to a fallback provider).
func TestPassthroughStreamFailover_PluginVetoNoEscalate(t *testing.T) {
	var primaryHits atomic.Int32
	primarySrv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		primaryHits.Add(1) // must stay 0: the plugin short-circuits before any HTTP call
		w.WriteHeader(http.StatusOK)
		fmt.Fprint(w, "data: PRIMARY\n\n")
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
	account.SetKeysForProvider(schemas.OpenAI, []schemas.Key{
		{ID: "primary-key", Value: *schemas.NewSecretVar("sk-primary"), Models: schemas.WhiteList{"*"}, Weight: 100},
	})
	account.SetKeysForProvider(schemas.Anthropic, []schemas.Key{
		{ID: "fallback-key", Value: *schemas.NewSecretVar("sk-fallback"), Models: schemas.WhiteList{"*"}, Weight: 100},
	})
	veto := &vetoPlugin{vetoProvider: schemas.OpenAI}
	client := newVetoClient(t, account, veto)

	stream, bifrostErr := doPassthroughStream(t, client, schemas.OpenAI,
		[]schemas.Fallback{{Provider: schemas.Anthropic, Model: "claude-haiku"}})

	// The veto must reach the client as the terminal error, never a live fallback stream.
	if stream != nil {
		body, status, errs := drainPassthroughStream(stream)
		t.Fatalf("veto returned a live stream (body=%q status=%d errs=%v) — a plugin AllowFallbacks=false was overridden by escalation (CWE-863)", body, status, errs)
	}
	if bifrostErr == nil {
		t.Fatalf("veto returned no error — the AllowFallbacks=false short-circuit was swallowed")
	}
	if bifrostErr.Error == nil || bifrostErr.Error.Message != passthroughVetoMessage {
		t.Fatalf("client error = %+v, want the veto message %q reaching the client verbatim", bifrostErr, passthroughVetoMessage)
	}
	if fallbackHits.Load() != 0 {
		t.Fatalf("fallback called %d time(s) — a plugin veto (AllowFallbacks=false) escalated cheap->expensive (CWE-863)", fallbackHits.Load())
	}
	if primaryHits.Load() != 0 {
		t.Fatalf("primary HTTP server hit %d time(s) — a PreLLMHook short-circuit skips the provider call entirely", primaryHits.Load())
	}
	// Anti-vacuous: the veto plugin must actually have fired.
	if veto.preHits.Load() < 1 {
		t.Fatalf("veto plugin never fired — test is vacuous")
	}
}

// F3: an upstream 503 whose first chunk arrives WELL past the 3ms local-fast-fail
// window is a genuine upstream failure (escalate-class), not a fasthttp local
// fast-fail. The forwarded chunk must carry its real elapsed Latency so the
// first-chunk classifier sees durationMs > 3 and escalates after ONE same-provider
// retry (=> 2 primary attempts). If forwarded chunks carried Latency==0 the 503
// would misclassify as a local fast-fail and burn its off-budget retries
// (=> 3 primary attempts) before escalating. The discriminator is the physical
// primary call count, not the final status (both paths eventually escalate).
func TestPassthroughStreamFailover_SlowFirstChunk503Escalates(t *testing.T) {
	var primaryHits atomic.Int32
	primarySrv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		primaryHits.Add(1)
		time.Sleep(30 * time.Millisecond) // push first-chunk latency far past the 3ms window
		w.WriteHeader(http.StatusServiceUnavailable)
		fl, _ := w.(http.Flusher)
		fmt.Fprint(w, "data: UPSTREAM-503\n\n")
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
		t.Fatalf("slow-503 passthrough stream failed (primary=%d fallback=%d): %s", primaryHits.Load(), fallbackHits.Load(), bifrostErr.Error.Message)
	}
	body, status, errs := drainPassthroughStream(stream)

	if primaryHits.Load() != 2 {
		t.Fatalf("primary hits = %d, want 2 (a >3ms upstream 503 must escalate after one same-provider retry; a Latency==0 misclassification burns the local-fast-fail budget => 3 primary attempts)", primaryHits.Load())
	}
	if fallbackHits.Load() != 1 {
		t.Fatalf("fallback hits = %d, want 1 (a genuine upstream 503 must escalate to the fallback exactly once)", fallbackHits.Load())
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
}

// C1 (CWE-863): a plugin veto (AllowFallbacks=false) surfacing on a FALLBACK LEG
// carries no HTTP status, so streamFailoverStatus reports 0 → passthroughEscalate.
// Without the guard inside runPassthroughStreamFallbackAttempt the veto is REISSUED
// on that same fallback (its PreLLMHook fires twice) before the caller finally halts —
// a policy/auth denial retried cheap->expensive. The fix gates on the fallback-leg
// twin predicate (shouldContinueWithFallbacks) BEFORE classifying, so the veto leg is
// returned verbatim, NOT reissued, and the ladder does NOT advance to a pricier third
// provider. Discriminator (revert-proof): veto.preHits — 1 with the guard, 2 without.
func TestPassthroughStreamFailover_FallbackVetoNotReissued(t *testing.T) {
	// Primary always 5xx first chunk → escalates to the fallback ladder.
	var primaryHits atomic.Int32
	primarySrv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		primaryHits.Add(1)
		w.WriteHeader(http.StatusInternalServerError)
		fl, _ := w.(http.Flusher)
		fmt.Fprint(w, "data: PRIMARY-500\n\n")
		if fl != nil {
			fl.Flush()
		}
	}))
	defer primarySrv.Close()

	// Fallback 1 (Anthropic) is vetoed by the plugin. Its HTTP server must never be
	// reached (PreLLMHook short-circuits), so hits here stay 0 regardless.
	var anthropicHits atomic.Int32
	anthropicSrv := httptest.NewServer(statusHandler(http.StatusOK, &anthropicHits))
	defer anthropicSrv.Close()

	// Fallback 2 (Gemini) is a live 200 — the pricier rung the veto must NOT reach.
	var geminiHits atomic.Int32
	geminiSrv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		geminiHits.Add(1)
		w.WriteHeader(http.StatusOK)
		fl, _ := w.(http.Flusher)
		fmt.Fprint(w, "data: FALLBACK-STREAM\n\n")
		if fl != nil {
			fl.Flush()
		}
	}))
	defer geminiSrv.Close()

	account := NewMockAccount()
	account.AddProviderWithBaseURL(schemas.OpenAI, 1, 1, primarySrv.URL)
	account.AddProviderWithBaseURL(schemas.Anthropic, 1, 1, anthropicSrv.URL)
	account.AddProviderWithBaseURL(schemas.Gemini, 1, 1, geminiSrv.URL)
	account.SetKeysForProvider(schemas.OpenAI, []schemas.Key{
		{ID: "primary-key", Value: *schemas.NewSecretVar("sk-primary"), Models: schemas.WhiteList{"*"}, Weight: 100},
	})
	account.SetKeysForProvider(schemas.Anthropic, []schemas.Key{
		{ID: "anthropic-key", Value: *schemas.NewSecretVar("sk-anthropic"), Models: schemas.WhiteList{"*"}, Weight: 100},
	})
	account.SetKeysForProvider(schemas.Gemini, []schemas.Key{
		{ID: "gemini-key", Value: *schemas.NewSecretVar("sk-gemini"), Models: schemas.WhiteList{"*"}, Weight: 100},
	})
	// The veto fires on the FALLBACK provider (Anthropic), not the primary.
	veto := &vetoPlugin{vetoProvider: schemas.Anthropic}
	client := newVetoClient(t, account, veto)

	stream, bifrostErr := doPassthroughStream(t, client, schemas.OpenAI, []schemas.Fallback{
		{Provider: schemas.Anthropic, Model: "claude-haiku"},
		{Provider: schemas.Gemini, Model: "gemini-flash"},
	})

	if stream != nil {
		body, status, errs := drainPassthroughStream(stream)
		t.Fatalf("fallback-leg veto returned a live stream (body=%q status=%d errs=%v) — the veto was overridden by escalation (CWE-863)", body, status, errs)
	}
	if bifrostErr == nil {
		t.Fatalf("fallback-leg veto returned no error — the AllowFallbacks=false short-circuit was swallowed")
	}
	if bifrostErr.Error == nil || bifrostErr.Error.Message != passthroughVetoMessage {
		t.Fatalf("client error = %+v, want the veto message %q reaching the client verbatim", bifrostErr, passthroughVetoMessage)
	}
	// The veto leg must be issued EXACTLY ONCE — not reissued on the same fallback.
	if got := veto.preHits.Load(); got != 1 {
		t.Fatalf("veto PreLLMHook fired %d time(s) on the fallback leg, want exactly 1 (a veto reissued on its own fallback leg is CWE-863 cheap->expensive)", got)
	}
	// The ladder must NOT advance to the pricier third provider.
	if geminiHits.Load() != 0 {
		t.Fatalf("pricier fallback (Gemini) hit %d time(s) — a plugin veto advanced the ladder past its own leg (CWE-863)", geminiHits.Load())
	}
	// Anti-vacuous: the primary must actually have escalated (>=2 attempts), or the
	// fallback ladder was never entered and the veto-leg assertion is trivially true.
	if primaryHits.Load() < 2 {
		t.Fatalf("primary hits = %d, want >=2 — the primary never escalated, so the fallback veto leg was never exercised (test vacuous)", primaryHits.Load())
	}
}

// C1 CONTROL (money rule not broken by the veto guard): a GENUINE status-0 transport
// error on a fallback leg (no veto, nil AllowFallbacks) must STILL advance the ladder —
// the guard must suppress ONLY a real veto, never a legitimate failover. A down OpenAI
// surfaces as a bare transport error (StatusCode nil ⇒ status 0 ⇒ passthroughEscalate),
// the exact status-0 shape the veto also carries; the guard distinguishes them by
// AllowFallbacks alone. Fallback 1 (OpenAI) is down; the ladder must move on to the
// healthy Fallback 2 (Gemini). This is the non-regression twin of the veto test: it
// must stay GREEN whether or not the attempt-level guard is present (ladder-advance is
// the CALLER's escalate check, not the attempt loop), so it proves the veto guard does
// not suppress a legitimate status-0 failover. Its sensitivity is to ladder-advance
// itself: breaking the caller's escalate-continue strands the request on the dead
// fallback and Gemini is never reached — verified by revert-proof. (A 502-mapping
// provider like Anthropic would classify as a ≤3ms local-fast-fail retry-same and
// legitimately stop the ladder, so the down fallback here MUST be a status-0 provider.)
func TestPassthroughStreamFailover_FallbackTransportStillEscalates(t *testing.T) {
	// Primary (Anthropic) always 5xx first chunk → escalates to the fallback ladder.
	var primaryHits atomic.Int32
	primarySrv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		primaryHits.Add(1)
		w.WriteHeader(http.StatusInternalServerError)
		fl, _ := w.(http.Flusher)
		fmt.Fprint(w, "data: PRIMARY-500\n\n")
		if fl != nil {
			fl.Flush()
		}
	}))
	defer primarySrv.Close()

	// Fallback 1 (OpenAI): a server closed before use → every dial is a genuine
	// status-0 transport error (connection refused, no HTTP status), NOT a veto.
	downSrv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {}))
	downURL := downSrv.URL
	downSrv.Close()

	// Fallback 2 (Gemini): live 200 — the rung the transport-error fallback must reach.
	var geminiHits atomic.Int32
	geminiSrv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		geminiHits.Add(1)
		w.WriteHeader(http.StatusOK)
		fl, _ := w.(http.Flusher)
		fmt.Fprint(w, "data: FALLBACK-STREAM\n\n")
		if fl != nil {
			fl.Flush()
		}
	}))
	defer geminiSrv.Close()

	counter := &primaryCallCounter{}
	account := NewMockAccount()
	account.AddProviderWithBaseURL(schemas.Anthropic, 1, 1, primarySrv.URL)
	account.AddProviderWithBaseURL(schemas.OpenAI, 1, 1, downURL)
	account.AddProviderWithBaseURL(schemas.Gemini, 1, 1, geminiSrv.URL)
	account.SetKeysForProvider(schemas.Anthropic, []schemas.Key{
		{ID: "primary-key", Value: *schemas.NewSecretVar("sk-primary"), Models: schemas.WhiteList{"*"}, Weight: 100},
	})
	account.SetKeysForProvider(schemas.OpenAI, []schemas.Key{
		{ID: "openai-key", Value: *schemas.NewSecretVar("sk-openai"), Models: schemas.WhiteList{"*"}, Weight: 100},
	})
	account.SetKeysForProvider(schemas.Gemini, []schemas.Key{
		{ID: "gemini-key", Value: *schemas.NewSecretVar("sk-gemini"), Models: schemas.WhiteList{"*"}, Weight: 100},
	})
	client := newCounterClient(t, account, counter)

	stream, bifrostErr := doPassthroughStream(t, client, schemas.Anthropic, []schemas.Fallback{
		{Provider: schemas.OpenAI, Model: "gpt-4o"},
		{Provider: schemas.Gemini, Model: "gemini-flash"},
	})
	if bifrostErr != nil {
		t.Fatalf("transport-error fallback leg failed to escalate (primary=%d openai=%d gemini=%d): %s",
			primaryHits.Load(), counter.count(schemas.OpenAI), geminiHits.Load(), bifrostErr.Error.Message)
	}
	body, status, errs := drainPassthroughStream(stream)
	// Anti-vacuous: the down fallback must actually have been attempted.
	if counter.count(schemas.OpenAI) < 1 {
		t.Fatalf("down fallback (OpenAI) never attempted — the transport-error leg was never exercised (test vacuous)")
	}
	// The ladder MUST advance past the transport-error fallback to the healthy one.
	if geminiHits.Load() != 1 {
		t.Fatalf("healthy fallback (Gemini) hit %d time(s), want 1 — the veto guard over-suppressed a GENUINE transport error on a fallback leg", geminiHits.Load())
	}
	if len(errs) > 0 {
		t.Fatalf("healthy fallback stream emitted error chunks: %v", errs)
	}
	if status != http.StatusOK {
		t.Fatalf("final status = %d, want 200 (healthy fallback served the stream)", status)
	}
	if !strings.Contains(body, "FALLBACK-STREAM") {
		t.Fatalf("client body = %q, want the healthy FALLBACK stream content", body)
	}
}

// F4 (CWE-693): the UNARY primary veto guard had no test of its own — deleting it
// left the whole TestPassthrough* suite green. A veto carries no HTTP status, so it
// reaches the dedicated loop as status 0 (escalate-class); only the shouldTryFallbacks
// gate keeps it terminal. Asserts the pricier fallback is never called.
func TestPassthroughFailover_UnaryPrimaryVetoNotEscalated(t *testing.T) {
	var fallbackHits atomic.Int32
	fallbackSrv := httptest.NewServer(statusHandler(http.StatusOK, &fallbackHits))
	defer fallbackSrv.Close()

	primarySrv := httptest.NewServer(statusHandler(http.StatusOK, new(atomic.Int32)))
	defer primarySrv.Close()

	account := NewMockAccount()
	account.AddProviderWithBaseURL(schemas.OpenAI, 1, 1, primarySrv.URL)
	account.AddProviderWithBaseURL(schemas.Anthropic, 1, 1, fallbackSrv.URL)
	account.SetKeysForProvider(schemas.OpenAI, []schemas.Key{
		{ID: "p", Value: *schemas.NewSecretVar("sk-p"), Models: schemas.WhiteList{"*"}, Weight: 100},
	})
	account.SetKeysForProvider(schemas.Anthropic, []schemas.Key{
		{ID: "fb", Value: *schemas.NewSecretVar("sk-fb"), Models: schemas.WhiteList{"*"}, Weight: 100},
	})

	plugin := &vetoPlugin{vetoProvider: schemas.OpenAI}
	client := newVetoClient(t, account, plugin)

	resp, bifrostErr := doPassthrough(t, client, schemas.OpenAI,
		[]schemas.Fallback{{Provider: schemas.Anthropic, Model: "claude-haiku"}})

	if plugin.preHits.Load() == 0 {
		t.Fatal("veto plugin never fired — test is vacuous")
	}
	if fallbackHits.Load() != 0 {
		t.Fatalf("fallback called %d time(s) — a plugin veto escalated cheap→expensive (CWE-863)", fallbackHits.Load())
	}
	if bifrostErr == nil {
		t.Fatalf("veto did not surface: resp=%v", resp)
	}
	if !strings.Contains(bifrostErr.Error.Message, passthroughVetoMessage) {
		t.Fatalf("error = %q, want the veto message %q", bifrostErr.Error.Message, passthroughVetoMessage)
	}
}
