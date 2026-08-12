package bifrost

import (
	"context"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"sync/atomic"
	"testing"

	schemas "github.com/maximhq/bifrost/core/schemas"
)

// F2 (MONEY RULE): a transport error on a CHEAP fallback leg must spend that
// fallback's same-provider budget BEFORE the ladder advances to the pricier one —
// exactly like the primary leg and the streaming fallback leg. Primary 500 escalates
// to the cheap fallback (Anthropic, down => status-0 transport error); the cheap leg
// must be attempted >=2 times (first + one same-provider retry) before the expensive
// fallback (Gemini) serves the request. Discriminator (revert-proof): the cheap leg's
// physical attempt count — >=2 with the fix, exactly 1 if the unary fallback loop
// returns early on any error before classifying.
func TestPassthroughFailover_FallbackTransportSpendsBudgetBeforeEscalate(t *testing.T) {
	var primaryHits atomic.Int32
	primarySrv := httptest.NewServer(statusHandler(http.StatusInternalServerError, &primaryHits))
	defer primarySrv.Close()

	// Cheap fallback (Anthropic): a server closed before use => every dial is a genuine
	// status-0 transport error (connection refused, no HTTP status), NOT a status. Its
	// physical attempts are counted via PreLLMHook (a down server has no counter).
	downSrv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {}))
	downURL := downSrv.URL
	downSrv.Close()

	// Expensive fallback (Gemini): live 200 — the pricier rung the budget must reach
	// only AFTER the cheap leg's same-provider retries are exhausted.
	var geminiHits atomic.Int32
	geminiSrv := httptest.NewServer(statusHandler(http.StatusOK, &geminiHits))
	defer geminiSrv.Close()

	counter := &primaryCallCounter{}
	account := NewMockAccount()
	account.AddProviderWithBaseURL(schemas.OpenAI, 1, 1, primarySrv.URL)
	account.AddProviderWithBaseURL(schemas.Anthropic, 1, 1, downURL)
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
	client := newCounterClient(t, account, counter)

	resp, bifrostErr := doPassthrough(t, client, schemas.OpenAI, []schemas.Fallback{
		{Provider: schemas.Anthropic, Model: "claude-haiku"},
		{Provider: schemas.Gemini, Model: "gemini-flash"},
	})
	if bifrostErr != nil {
		t.Fatalf("passthrough failed (primary=%d cheap=%d gemini=%d): %s",
			primaryHits.Load(), counter.count(schemas.Anthropic), geminiHits.Load(), bifrostErr.Error.Message)
	}
	// Anti-vacuous: the primary must actually have escalated to the fallback ladder.
	if primaryHits.Load() < 2 {
		t.Fatalf("primary hits = %d, want >=2 — the primary never escalated, so the cheap fallback leg was never exercised (test vacuous)", primaryHits.Load())
	}
	// THE MONEY RULE: the cheap fallback must spend its same-provider budget.
	if got := counter.count(schemas.Anthropic); got < 2 {
		t.Fatalf("cheap fallback attempted %d time(s), want >=2 — a transport error on the cheap fallback escalated to the pricier one on its FIRST failure (MONEY RULE VIOLATION)", got)
	}
	// The pricier fallback is reached exactly once, AFTER the cheap budget is spent.
	if geminiHits.Load() != 1 {
		t.Fatalf("expensive fallback hits = %d, want 1 (reached exactly once after the cheap leg exhausts its budget)", geminiHits.Load())
	}
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("final status = %d, want 200 (expensive fallback served the request)", resp.StatusCode)
	}
}

// F2 REVERT-PROOF (the escalate arm). The test above reaches the cheap leg through the
// passthroughRetrySame arm, NOT the escalate arm: a connection-refused dial surfaces as a
// sub-millisecond 502, and classifyPassthroughStatus exempts a 502/503 faster than
// passthroughLocalFastFailMaxMs (3ms) into retry-same. Measured with a probe:
// "PROBE-CLASSIFY prov=anthropic status=502 action=2" (2 == passthroughRetrySame). That
// left the passthroughEscalate arm's budget block — the actual F2 fix — as dead code under
// the whole suite: replacing that arm with panic() kept every test green.
//
// A 500 carries no fast-fail exemption (the carve-out at classifyPassthroughStatus is
// 502/503 only), so it classifies as passthroughEscalate at any duration and drives the
// cheap leg through that arm. Discriminator (revert-proof): cheapHits — 2 with the fix,
// 1 if the escalate arm returns early instead of spending its same-provider budget.
func TestPassthroughFailover_FallbackEscalateArmSpendsBudget(t *testing.T) {
	var primaryHits, cheapHits, geminiHits atomic.Int32

	primarySrv := httptest.NewServer(statusHandler(http.StatusInternalServerError, &primaryHits))
	defer primarySrv.Close()

	// Cheap fallback (Anthropic): a live 500 => passthroughEscalate, no fast-fail exemption.
	cheapSrv := httptest.NewServer(statusHandler(http.StatusInternalServerError, &cheapHits))
	defer cheapSrv.Close()

	// Expensive fallback (Gemini): the pricier rung, reachable only after the cheap
	// leg's same-provider budget is spent.
	geminiSrv := httptest.NewServer(statusHandler(http.StatusOK, &geminiHits))
	defer geminiSrv.Close()

	account := NewMockAccount()
	account.AddProviderWithBaseURL(schemas.OpenAI, 1, 1, primarySrv.URL)
	account.AddProviderWithBaseURL(schemas.Anthropic, 1, 1, cheapSrv.URL)
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
	client := newPassthroughTestClient(t, account)

	resp, bifrostErr := doPassthrough(t, client, schemas.OpenAI, []schemas.Fallback{
		{Provider: schemas.Anthropic, Model: "claude-haiku"},
		{Provider: schemas.Gemini, Model: "gemini-flash"},
	})
	if bifrostErr != nil {
		t.Fatalf("passthrough failed (primary=%d cheap=%d gemini=%d): %s",
			primaryHits.Load(), cheapHits.Load(), geminiHits.Load(), bifrostErr.Error.Message)
	}
	t.Logf("escalate-arm budget: primary=%d cheap=%d gemini=%d", primaryHits.Load(), cheapHits.Load(), geminiHits.Load())

	// Anti-vacuous: the primary must actually have escalated into the fallback ladder.
	if primaryHits.Load() < 2 {
		t.Fatalf("primary hits = %d, want >=2 — the primary never escalated, so the cheap fallback leg was never exercised (test vacuous)", primaryHits.Load())
	}
	// THE MONEY RULE, on the escalate arm specifically.
	if got := cheapHits.Load(); got < 2 {
		t.Fatalf("cheap fallback attempted %d time(s), want >=2 — an escalate-class 500 on the cheap fallback advanced the ladder to the pricier provider on its FIRST failure (MONEY RULE VIOLATION on the passthroughEscalate arm)", got)
	}
	// The pricier fallback is reached exactly once, only after the cheap budget is spent.
	if geminiHits.Load() != 1 {
		t.Fatalf("expensive fallback hits = %d, want 1 (reached exactly once after the cheap leg exhausts its budget)", geminiHits.Load())
	}
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("final status = %d, want 200 (expensive fallback served the request)", resp.StatusCode)
	}
}

// F2 CONTROL (CWE-863, unary twin of TestPassthroughStreamFailover_FallbackVetoNotReissued):
// a plugin veto (AllowFallbacks=false) surfacing on a CHEAP fallback leg carries no HTTP
// status, so unaryFailoverStatus reports 0 => passthroughEscalate. The veto guard inside
// runPassthroughFallbackAttempt must return it verbatim, NOT reissue it on the same leg,
// and the ladder must NOT advance to the pricier third provider. Discriminator
// (revert-proof): veto.preHits — 1 with the guard, 2 without.
func TestPassthroughFailover_FallbackVetoNotReissued(t *testing.T) {
	var primaryHits atomic.Int32
	primarySrv := httptest.NewServer(statusHandler(http.StatusInternalServerError, &primaryHits))
	defer primarySrv.Close()

	// Cheap fallback (Anthropic) is vetoed by the plugin; its HTTP server must never be
	// reached (PreLLMHook short-circuits), so hits stay 0 regardless.
	var anthropicHits atomic.Int32
	anthropicSrv := httptest.NewServer(statusHandler(http.StatusOK, &anthropicHits))
	defer anthropicSrv.Close()

	// Expensive fallback (Gemini): live 200 — the pricier rung the veto must NOT reach.
	var geminiHits atomic.Int32
	geminiSrv := httptest.NewServer(statusHandler(http.StatusOK, &geminiHits))
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
	// The veto fires on the cheap FALLBACK provider (Anthropic), not the primary.
	veto := &vetoPlugin{vetoProvider: schemas.Anthropic}
	client := newVetoClient(t, account, veto)

	resp, bifrostErr := doPassthrough(t, client, schemas.OpenAI, []schemas.Fallback{
		{Provider: schemas.Anthropic, Model: "claude-haiku"},
		{Provider: schemas.Gemini, Model: "gemini-flash"},
	})

	if resp != nil {
		t.Fatalf("fallback-leg veto returned a live response (status=%d) — the veto was overridden by escalation (CWE-863)", resp.StatusCode)
	}
	if bifrostErr == nil {
		t.Fatalf("fallback-leg veto returned no error — the AllowFallbacks=false short-circuit was swallowed")
	}
	if bifrostErr.Error == nil || bifrostErr.Error.Message != passthroughVetoMessage {
		t.Fatalf("client error = %+v, want the veto message %q reaching the client verbatim", bifrostErr, passthroughVetoMessage)
	}
	// The veto leg must be issued EXACTLY ONCE — not reissued on the same fallback.
	if got := veto.preHits.Load(); got != 1 {
		t.Fatalf("veto PreLLMHook fired %d time(s) on the cheap fallback leg, want exactly 1 (a veto reissued on its own leg is CWE-863 cheap->expensive)", got)
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

// F4 (CWE-693): a UNARY plugin veto (AllowFallbacks=false) on the PRIMARY carries no
// HTTP status, so unaryFailoverStatus reports 0 => passthroughEscalate. The unary
// shouldTryFallbacks guard in runPassthroughStatusFailover must return the veto verbatim
// so the fallback is NEVER called. Without the guard the veto is retried then escalated
// cheap->expensive (a policy/auth denial silently promoted to a fallback provider).
// Discriminator (revert-proof): fallbackHits — 0 with the guard, >=1 without.
func TestPassthroughFailover_PrimaryVetoNoEscalate(t *testing.T) {
	var primaryHits atomic.Int32
	primarySrv := httptest.NewServer(statusHandler(http.StatusOK, &primaryHits))
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
	veto := &vetoPlugin{vetoProvider: schemas.OpenAI}
	client := newVetoClient(t, account, veto)

	resp, bifrostErr := doPassthrough(t, client, schemas.OpenAI,
		[]schemas.Fallback{{Provider: schemas.Anthropic, Model: "claude-haiku"}})

	if resp != nil {
		t.Fatalf("primary veto returned a live response (status=%d) — a plugin AllowFallbacks=false was overridden by escalation (CWE-863)", resp.StatusCode)
	}
	if bifrostErr == nil {
		t.Fatalf("primary veto returned no error — the AllowFallbacks=false short-circuit was swallowed")
	}
	if bifrostErr.Error == nil || bifrostErr.Error.Message != passthroughVetoMessage {
		t.Fatalf("client error = %+v, want the veto message %q reaching the client verbatim", bifrostErr, passthroughVetoMessage)
	}
	// THE GUARD: the fallback must NEVER be called on a primary veto.
	if fallbackHits.Load() != 0 {
		t.Fatalf("fallback called %d time(s) — a unary plugin veto (AllowFallbacks=false) escalated cheap->expensive (CWE-693)", fallbackHits.Load())
	}
	if primaryHits.Load() != 0 {
		t.Fatalf("primary HTTP server hit %d time(s) — a PreLLMHook short-circuit skips the provider call entirely", primaryHits.Load())
	}
	// Anti-vacuous: the veto plugin must actually have fired.
	if veto.preHits.Load() < 1 {
		t.Fatalf("veto plugin never fired — test is vacuous")
	}
}

// F5 (STREAMING TWIN of the money rule). Round 9 closed the UNARY fallback escalate
// arm; the streaming leg's identical arm stayed unguarded. Independently confirmed by
// two reviewer lineages and reproduced here: replacing the streaming fallback escalate
// arm's budget block with `return current` left all 46 tests green, so nothing observed
// the cheap streaming leg skipping its same-provider retry.
//
// A live 500 carries no passthroughLocalFastFailMaxMs exemption (that carve-out is
// 502/503 only), so it classifies as passthroughEscalate and drives the leg through
// that arm. Discriminator (revert-proof): cheapHits — 2 with the fix, 1 without.
func TestPassthroughStreamFailover_FallbackEscalateArmSpendsBudget(t *testing.T) {
	var primaryHits, cheapHits, geminiHits atomic.Int32

	primarySrv := httptest.NewServer(streamStatusHandler(http.StatusInternalServerError, &primaryHits, ""))
	defer primarySrv.Close()
	cheapSrv := httptest.NewServer(streamStatusHandler(http.StatusInternalServerError, &cheapHits, ""))
	defer cheapSrv.Close()
	geminiSrv := httptest.NewServer(streamStatusHandler(http.StatusOK, &geminiHits, ""))
	defer geminiSrv.Close()

	account := NewMockAccount()
	account.AddProviderWithBaseURL(schemas.OpenAI, 1, 1, primarySrv.URL)
	account.AddProviderWithBaseURL(schemas.Anthropic, 1, 1, cheapSrv.URL)
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
	client := newPassthroughTestClient(t, account)

	ch, bifrostErr := doPassthroughStream(t, client, schemas.OpenAI, []schemas.Fallback{
		{Provider: schemas.Anthropic, Model: "claude-haiku"},
		{Provider: schemas.Gemini, Model: "gemini-flash"},
	})
	if bifrostErr != nil {
		t.Fatalf("passthrough stream failed (primary=%d cheap=%d gemini=%d): %s",
			primaryHits.Load(), cheapHits.Load(), geminiHits.Load(), bifrostErr.Error.Message)
	}
	_, lastStatus, _ := drainPassthroughStream(ch)
	t.Logf("stream escalate-arm budget: primary=%d cheap=%d gemini=%d lastStatus=%d",
		primaryHits.Load(), cheapHits.Load(), geminiHits.Load(), lastStatus)

	if primaryHits.Load() < 2 {
		t.Fatalf("primary hits = %d, want >=2 — the primary never escalated, so the cheap streaming leg was never exercised (test vacuous)", primaryHits.Load())
	}
	if got := cheapHits.Load(); got < 2 {
		t.Fatalf("cheap streaming fallback attempted %d time(s), want >=2 — an escalate-class 500 advanced the ladder to the pricier provider on its FIRST failure (MONEY RULE VIOLATION on the streaming passthroughEscalate arm)", got)
	}
	if geminiHits.Load() != 1 {
		t.Fatalf("expensive streaming fallback hits = %d, want 1 (reached exactly once after the cheap leg exhausts its budget)", geminiHits.Load())
	}
	if lastStatus != http.StatusOK {
		t.Fatalf("final stream status = %d, want 200 (expensive fallback served the stream)", lastStatus)
	}
}

// F3 (POOL RULE, streaming). TestPassthroughStreamFailover_401RetriesSameNoEscalate
// uses a server that recovers on hit 2 (failN:1), so the pooled arm and an
// escalate-after-one-retry arm produce the SAME observable outcome: retry, get 200,
// fallback untouched. Mutating 401/403 to retry-once-then-escalate left that test —
// and the whole suite — green, so nothing guarded the pool rule on the streaming leg.
//
// A PERMANENT 401 separates them: the pooled arm exhausts passthroughMaxRetryPooled
// against the same provider and returns the 401 verbatim, never touching the pricier
// fallback. Under escalate semantics the fallback IS called. Discriminator
// (revert-proof): fallbackHits — 0 with the pooled arm, >=1 without.
func TestPassthroughStreamFailover_PermanentPoolErrorNeverEscalates(t *testing.T) {
	var primaryHits, fallbackHits atomic.Int32

	primarySrv := httptest.NewServer(streamStatusHandler(http.StatusUnauthorized, &primaryHits, ""))
	defer primarySrv.Close()
	fallbackSrv := httptest.NewServer(streamStatusHandler(http.StatusOK, &fallbackHits, ""))
	defer fallbackSrv.Close()

	account := NewMockAccount()
	account.AddProviderWithBaseURL(schemas.OpenAI, 1, 1, primarySrv.URL)
	account.SetKeysForProvider(schemas.OpenAI, []schemas.Key{
		{ID: "k1", Value: *schemas.NewSecretVar("sk-1"), Models: schemas.WhiteList{"*"}, Weight: 100},
		{ID: "k2", Value: *schemas.NewSecretVar("sk-2"), Models: schemas.WhiteList{"*"}, Weight: 100},
	})
	account.AddProviderWithBaseURL(schemas.Anthropic, 1, 1, fallbackSrv.URL)
	account.SetKeysForProvider(schemas.Anthropic, []schemas.Key{
		{ID: "fb", Value: *schemas.NewSecretVar("sk-fb"), Models: schemas.WhiteList{"*"}, Weight: 100},
	})
	client := newPassthroughTestClient(t, account)

	// A permanently-401 pool exhausts its budget and surfaces the denial rather than a
	// stream, so an error return here is the CORRECT outcome — the assertions below,
	// not the error, decide the verdict.
	ch, bifrostErr := doPassthroughStream(t, client, schemas.OpenAI,
		[]schemas.Fallback{{Provider: schemas.Anthropic, Model: "claude-haiku"}})
	lastStatus := 0
	if ch != nil {
		_, lastStatus, _ = drainPassthroughStream(ch)
	}
	t.Logf("permanent pool 401: primary=%d fallback=%d lastStatus=%d err=%v (maxRetryPooled=%d)",
		primaryHits.Load(), fallbackHits.Load(), lastStatus, bifrostErr != nil, passthroughMaxRetryPooled)

	// THE POOL RULE: a pool-account symptom must never reach a pricier provider.
	if fallbackHits.Load() != 0 {
		t.Fatalf("pricier fallback called %d time(s) — a permanent pool 401 escalated cheap->expensive (POOL RULE VIOLATION on the streaming pooled arm)", fallbackHits.Load())
	}
	// Bounded same-provider retry actually happened (not fail-fast on first 401).
	if got := primaryHits.Load(); got < 2 {
		t.Fatalf("primary hits = %d, want >=2 — a pool 401 must retry the SAME provider, not fail fast", got)
	}
	// And it stayed bounded rather than spinning.
	if got := primaryHits.Load(); got > int32(passthroughMaxRetryPooled)+1 {
		t.Fatalf("primary hits = %d, exceeds 1+passthroughMaxRetryPooled=%d — the pooled retry budget is unbounded", got, passthroughMaxRetryPooled+1)
	}
	// The denial reaches the client verbatim on one path or the other.
	if bifrostErr == nil && lastStatus != http.StatusUnauthorized {
		t.Fatalf("no error and final stream status = %d — the 401 was neither surfaced verbatim nor returned as an error", lastStatus)
	}
}

// LFF (streaming local-fast-fail). A 502/503 returned faster than
// passthroughLocalFastFailMaxMs (3ms) is a LOCAL failure — a dead socket or a proxy
// refusing instantly — not a real upstream 502, so it gets its own off-budget retry
// allowance (passthroughLocalFastFailMaxRetry) before the ladder advances. A panic
// probe in this branch left the entire suite green: no test reached it, so the whole
// streaming fast-fail machinery was unguarded.
//
// A local httptest server answering 502 immediately is sub-millisecond, which is what
// the carve-out keys on. Discriminator (revert-proof): primaryHits — 1+maxRetry with
// the branch, 2 if the carve-out is removed (plain retry-same budget of 1).
func TestPassthroughStreamFailover_LocalFastFailRetriesBeforeEscalate(t *testing.T) {
	var primaryHits, fallbackHits atomic.Int32

	primarySrv := httptest.NewServer(streamStatusHandler(http.StatusBadGateway, &primaryHits, ""))
	defer primarySrv.Close()
	fallbackSrv := httptest.NewServer(streamStatusHandler(http.StatusOK, &fallbackHits, ""))
	defer fallbackSrv.Close()

	account := NewMockAccount()
	account.AddProviderWithBaseURL(schemas.OpenAI, 1, 1, primarySrv.URL)
	account.AddProviderWithBaseURL(schemas.Anthropic, 1, 1, fallbackSrv.URL)
	account.SetKeysForProvider(schemas.OpenAI, []schemas.Key{
		{ID: "primary-key", Value: *schemas.NewSecretVar("sk-primary"), Models: schemas.WhiteList{"*"}, Weight: 100},
	})
	account.SetKeysForProvider(schemas.Anthropic, []schemas.Key{
		{ID: "fb", Value: *schemas.NewSecretVar("sk-fb"), Models: schemas.WhiteList{"*"}, Weight: 100},
	})
	client := newPassthroughTestClient(t, account)

	ch, bifrostErr := doPassthroughStream(t, client, schemas.OpenAI,
		[]schemas.Fallback{{Provider: schemas.Anthropic, Model: "claude-haiku"}})
	if bifrostErr != nil {
		t.Fatalf("passthrough stream failed (primary=%d fallback=%d): %s",
			primaryHits.Load(), fallbackHits.Load(), bifrostErr.Error.Message)
	}
	_, lastStatus, _ := drainPassthroughStream(ch)
	t.Logf("stream local-fast-fail: primary=%d fallback=%d lastStatus=%d (maxRetry=%d, maxMs=%d)",
		primaryHits.Load(), fallbackHits.Load(), lastStatus,
		passthroughLocalFastFailMaxRetry, passthroughLocalFastFailMaxMs)

	// The fast-fail allowance is spent on the SAME provider before escalating.
	if got := primaryHits.Load(); got != int32(passthroughLocalFastFailMaxRetry)+1 {
		t.Fatalf("primary hits = %d, want %d (1 + passthroughLocalFastFailMaxRetry) — the sub-3ms 502 fast-fail retry allowance was not spent on the same provider before escalating", got, passthroughLocalFastFailMaxRetry+1)
	}
	if fallbackHits.Load() != 1 {
		t.Fatalf("fallback hits = %d, want 1 (reached exactly once after the fast-fail allowance is exhausted)", fallbackHits.Load())
	}
	if lastStatus != http.StatusOK {
		t.Fatalf("final stream status = %d, want 200 (fallback served the stream)", lastStatus)
	}
}

// CAP (global attempt bound). passthroughMaxTotalAttempts is defense-in-depth: each leg
// spends 2 logical attempts (first + one same-provider retry), so without a global bound
// N configured fallbacks fan out unboundedly. Raising the constant 8 -> 100000 left the
// pre-existing suite green: the older cap tests assert `logical <= cap` but only ever
// produce 2-4 logical attempts, so the per-class budgets always bind first.
//
// Building a LONG ladder from distinct providers does not work: only anthropic, azure,
// gemini, openai and vertex implement Passthrough at all (the rest return
// NewUnsupportedOperationError), and azure/vertex are not routable in MockAccount — a
// distinct-provider ladder stalls at 2 dialed legs (total 6 < cap 8), so the cap never
// binds and the test is vacuous. Repeating ONE real provider across several fallback
// entries makes the ladder long enough: primary 2 + 4 legs x 2 = 10 > cap 8.
// Discriminator (revert-proof): total hits — exactly 8 (the cap) with the bound in
// place, 10 once the cap is raised.
func TestPassthroughFailover_GlobalCapCutsLongLadder(t *testing.T) {
	var primaryHits, cheapHits atomic.Int32
	primarySrv := httptest.NewServer(statusHandler(http.StatusInternalServerError, &primaryHits))
	defer primarySrv.Close()
	cheapSrv := httptest.NewServer(statusHandler(http.StatusInternalServerError, &cheapHits))
	defer cheapSrv.Close()

	account := NewMockAccount()
	account.AddProviderWithBaseURL(schemas.OpenAI, 1, 1, primarySrv.URL)
	account.AddProviderWithBaseURL(schemas.Anthropic, 1, 1, cheapSrv.URL)
	account.SetKeysForProvider(schemas.OpenAI, []schemas.Key{
		{ID: "primary-key", Value: *schemas.NewSecretVar("sk-primary"), Models: schemas.WhiteList{"*"}, Weight: 100},
	})
	account.SetKeysForProvider(schemas.Anthropic, []schemas.Key{
		{ID: "anthropic-key", Value: *schemas.NewSecretVar("sk-anthropic"), Models: schemas.WhiteList{"*"}, Weight: 100},
	})
	client := newPassthroughTestClient(t, account)

	fallbacks := []schemas.Fallback{
		{Provider: schemas.Anthropic, Model: "claude-a"},
		{Provider: schemas.Anthropic, Model: "claude-b"},
		{Provider: schemas.Anthropic, Model: "claude-c"},
		{Provider: schemas.Anthropic, Model: "claude-d"},
	}
	_, _ = doPassthrough(t, client, schemas.OpenAI, fallbacks)

	total := primaryHits.Load() + cheapHits.Load()
	uncapped := int32(2 + 2*len(fallbacks))
	t.Logf("long ladder: cap=%d total=%d (primary=%d ladder=%d) uncapped_would_be=%d",
		passthroughMaxTotalAttempts, total, primaryHits.Load(), cheapHits.Load(), uncapped)

	if total > int32(passthroughMaxTotalAttempts) {
		t.Fatalf("total upstream attempts = %d, exceeds passthroughMaxTotalAttempts=%d — the global bound does not cut a long fallback ladder", total, passthroughMaxTotalAttempts)
	}
	// ANTI-VACUOUS: the cap must actually BIND, i.e. the ladder was cut short of what it
	// would have spent unbounded. Without this, raising the cap changes nothing observable.
	if total >= uncapped {
		t.Fatalf("total = %d reached the uncapped fan-out of %d — the global cap never bound, so this test cannot detect an unbounded ladder", total, uncapped)
	}
	if primaryHits.Load() < 2 || cheapHits.Load() < 2 {
		t.Fatalf("primary=%d ladder=%d — the ladder was not exercised, test vacuous", primaryHits.Load(), cheapHits.Load())
	}
}

// newOrderedKeyTestClient builds a client whose key selector is DETERMINISTIC: it always
// returns the lowest key ID still eligible. The default WeightedRandom picks at random, which
// makes any "was the rejected key re-picked?" assertion pass ~half the time by luck — a
// coin-flip guard, not a test. Pinning the order makes the rotation observable every run.
func newOrderedKeyTestClient(t *testing.T, account *MockAccount) *Bifrost {
	t.Helper()
	client, err := Init(context.Background(), schemas.BifrostConfig{
		Account: account,
		Logger:  NewDefaultLogger(schemas.LogLevelError),
		KeySelector: func(_ *schemas.BifrostContext, keys []schemas.Key, _ schemas.ModelProvider, _ string) (schemas.Key, error) {
			if len(keys) == 0 {
				return schemas.Key{}, fmt.Errorf("no keys available")
			}
			best := keys[0]
			for _, k := range keys[1:] {
				if k.ID < best.ID {
					best = k
				}
			}
			return best, nil
		},
	})
	if err != nil {
		t.Fatalf("failed to initialize bifrost: %v", err)
	}
	t.Cleanup(client.Shutdown)
	return client
}

// F2 (cross-leg key rotation memory). Passthrough pins maxRetries=0, so every failover leg
// re-enters executeRequestWithRetries fresh and the retry loop's key-state maps were rebuilt
// empty each time: a key rejected 401/402/403 on one leg was forgotten and re-picked on the
// next, while a healthy pool key sat unused.
//
// The rejected key seeds usedKeyIDs (a SOFT skip the key provider RESETS once no key is left),
// never deadKeyIDs. A first attempt used deadKeyIDs and broke the money rule: dead entries
// shrink the eligible pool, so a permanent pool 401 exhausted it and escalated cheap->expensive.
// TestPassthroughStreamFailover_PermanentPoolErrorNeverEscalates and ..._NilCtxPool401NoEscalate
// are the guards that caught that and must stay green alongside this one.
//
// Key selection is pinned to lowest-ID-first so the rotation is deterministic; k1 always goes
// first and is the one the upstream rejects. Discriminator (revert-proof): k1's hit count — 1
// with the cross-leg memory, 2 without.
func TestPassthroughFailover_DeadKeyNotReusedAcrossLegs(t *testing.T) {
	var mu sync.Mutex
	seen := []string{}

	cheapSrv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		k := r.Header.Get("authorization") + r.Header.Get("x-api-key")
		mu.Lock()
		seen = append(seen, k)
		mu.Unlock()
		if strings.Contains(k, "sk-k1") {
			w.WriteHeader(http.StatusUnauthorized)
			_, _ = w.Write([]byte(`{"error":"invalid key"}`))
			return
		}
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte(`{"ok":true}`))
	}))
	defer cheapSrv.Close()

	var primaryHits atomic.Int32
	primarySrv := httptest.NewServer(statusHandler(http.StatusInternalServerError, &primaryHits))
	defer primarySrv.Close()

	account := NewMockAccount()
	account.AddProviderWithBaseURL(schemas.OpenAI, 1, 1, primarySrv.URL)
	account.AddProviderWithBaseURL(schemas.Anthropic, 1, 1, cheapSrv.URL)
	account.SetKeysForProvider(schemas.OpenAI, []schemas.Key{
		{ID: "primary-key", Value: *schemas.NewSecretVar("sk-primary"), Models: schemas.WhiteList{"*"}, Weight: 100},
	})
	account.SetKeysForProvider(schemas.Anthropic, []schemas.Key{
		{ID: "k1", Value: *schemas.NewSecretVar("sk-k1"), Models: schemas.WhiteList{"*"}, Weight: 100},
		{ID: "k2", Value: *schemas.NewSecretVar("sk-k2"), Models: schemas.WhiteList{"*"}, Weight: 100},
	})
	client := newOrderedKeyTestClient(t, account)

	resp, bifrostErr := doPassthrough(t, client, schemas.OpenAI,
		[]schemas.Fallback{{Provider: schemas.Anthropic, Model: "claude-haiku"}})

	mu.Lock()
	defer mu.Unlock()
	deadHits := 0
	for _, s := range seen {
		if strings.Contains(s, "sk-k1") {
			deadHits++
		}
	}
	t.Logf("cross-leg key rotation: seen=%v deadHits(k1)=%d err=%v", seen, deadHits, bifrostErr != nil)

	// Anti-vacuous: the rejected key must actually have been tried, and the leg exercised.
	if deadHits == 0 || len(seen) < 2 {
		t.Fatalf("seen=%v — k1 was never tried or the fallback leg was not exercised; test vacuous", seen)
	}
	if deadHits > 1 {
		t.Fatalf("rejected key k1 was selected %d times (seen=%v) — a 401-rejected credential was re-picked on a later failover leg while healthy key k2 was available", deadHits, seen)
	}
	if bifrostErr != nil || resp == nil || resp.StatusCode != http.StatusOK {
		t.Fatalf("final resp/err = %v/%v (seen=%v) — rotation to the healthy key did not recover the request", resp, bifrostErr, seen)
	}
}

// STREAM FB-LEG POOLED ARM. runPassthroughStreamFallbackAttempt has its own
// passthroughRetrySamePooled arm (core/passthroughfallbacks.go:447), distinct from the
// primary loop's. A panic probe in it left the whole ./core/ suite green: no test ever
// drove a pool error onto a streaming FALLBACK leg — every existing pool test hits the
// primary. The arm is correct but was unguarded.
//
// Primary 500 escalates to the cheap fallback, which then answers a PERMANENT 401 so the
// pooled budget is spent on that leg. Discriminator (revert-proof): cheap-leg hits —
// 1+passthroughMaxRetryPooled with the arm, 1 without.
func TestPassthroughStreamFailover_FallbackLegPooledArmSpendsBudget(t *testing.T) {
	var primaryHits, cheapHits, pricierHits atomic.Int32

	primarySrv := httptest.NewServer(streamStatusHandler(http.StatusInternalServerError, &primaryHits, ""))
	defer primarySrv.Close()
	cheapSrv := httptest.NewServer(streamStatusHandler(http.StatusUnauthorized, &cheapHits, ""))
	defer cheapSrv.Close()
	pricierSrv := httptest.NewServer(streamStatusHandler(http.StatusOK, &pricierHits, ""))
	defer pricierSrv.Close()

	account := NewMockAccount()
	account.AddProviderWithBaseURL(schemas.OpenAI, 1, 1, primarySrv.URL)
	account.AddProviderWithBaseURL(schemas.Anthropic, 1, 1, cheapSrv.URL)
	account.AddProviderWithBaseURL(schemas.Gemini, 1, 1, pricierSrv.URL)
	for p, k := range map[schemas.ModelProvider]string{
		schemas.OpenAI: "sk-p", schemas.Anthropic: "sk-a", schemas.Gemini: "sk-g",
	} {
		account.SetKeysForProvider(p, []schemas.Key{
			{ID: string(p), Value: *schemas.NewSecretVar(k), Models: schemas.WhiteList{"*"}, Weight: 100},
		})
	}
	client := newPassthroughTestClient(t, account)

	ch, bifrostErr := doPassthroughStream(t, client, schemas.OpenAI, []schemas.Fallback{
		{Provider: schemas.Anthropic, Model: "claude-haiku"},
		{Provider: schemas.Gemini, Model: "gemini-flash"},
	})
	lastStatus := 0
	if ch != nil {
		_, lastStatus, _ = drainPassthroughStream(ch)
	}
	t.Logf("stream fb-leg pooled: primary=%d cheap=%d pricier=%d lastStatus=%d err=%v (maxRetryPooled=%d)",
		primaryHits.Load(), cheapHits.Load(), pricierHits.Load(), lastStatus, bifrostErr != nil, passthroughMaxRetryPooled)

	if primaryHits.Load() < 2 {
		t.Fatalf("primary hits = %d, want >=2 — the primary never escalated, so the streaming fallback leg was never entered (test vacuous)", primaryHits.Load())
	}
	// THE POOL RULE on the fallback leg: the 401 must be retried on that SAME fallback.
	if got := cheapHits.Load(); got < 2 {
		t.Fatalf("cheap streaming fallback attempted %d time(s), want >=2 — a pool 401 on the fallback leg advanced the ladder instead of spending its pooled budget", got)
	}
	// ...and stay bounded rather than spinning.
	if got := cheapHits.Load(); got > int32(passthroughMaxRetryPooled)+1 {
		t.Fatalf("cheap streaming fallback attempted %d time(s), exceeds 1+passthroughMaxRetryPooled=%d — the pooled budget is unbounded on the fallback leg", got, passthroughMaxRetryPooled+1)
	}
}

// STREAM FB-LEG RETRY-SAME ARM. Twin of the test above for the
// passthroughRetrySame arm on the same leg (429/408). A panic probe there was likewise
// never reached. A permanent 429 on the cheap fallback must spend its same-provider budget
// on that leg. Discriminator (revert-proof): cheap-leg hits — 1+passthroughMaxRetrySame
// with the arm, 1 without.
func TestPassthroughStreamFailover_FallbackLegRetrySameArmSpendsBudget(t *testing.T) {
	var primaryHits, cheapHits, pricierHits atomic.Int32

	primarySrv := httptest.NewServer(streamStatusHandler(http.StatusInternalServerError, &primaryHits, ""))
	defer primarySrv.Close()
	cheapSrv := httptest.NewServer(streamStatusHandler(http.StatusTooManyRequests, &cheapHits, ""))
	defer cheapSrv.Close()
	pricierSrv := httptest.NewServer(streamStatusHandler(http.StatusOK, &pricierHits, ""))
	defer pricierSrv.Close()

	account := NewMockAccount()
	account.AddProviderWithBaseURL(schemas.OpenAI, 1, 1, primarySrv.URL)
	account.AddProviderWithBaseURL(schemas.Anthropic, 1, 1, cheapSrv.URL)
	account.AddProviderWithBaseURL(schemas.Gemini, 1, 1, pricierSrv.URL)
	for p, k := range map[schemas.ModelProvider]string{
		schemas.OpenAI: "sk-p", schemas.Anthropic: "sk-a", schemas.Gemini: "sk-g",
	} {
		account.SetKeysForProvider(p, []schemas.Key{
			{ID: string(p), Value: *schemas.NewSecretVar(k), Models: schemas.WhiteList{"*"}, Weight: 100},
		})
	}
	client := newPassthroughTestClient(t, account)

	ch, bifrostErr := doPassthroughStream(t, client, schemas.OpenAI, []schemas.Fallback{
		{Provider: schemas.Anthropic, Model: "claude-haiku"},
		{Provider: schemas.Gemini, Model: "gemini-flash"},
	})
	lastStatus := 0
	if ch != nil {
		_, lastStatus, _ = drainPassthroughStream(ch)
	}
	t.Logf("stream fb-leg retry-same: primary=%d cheap=%d pricier=%d lastStatus=%d err=%v (maxRetrySame=%d)",
		primaryHits.Load(), cheapHits.Load(), pricierHits.Load(), lastStatus, bifrostErr != nil, passthroughMaxRetrySame)

	if primaryHits.Load() < 2 {
		t.Fatalf("primary hits = %d, want >=2 — the primary never escalated, so the streaming fallback leg was never entered (test vacuous)", primaryHits.Load())
	}
	// A 429 must NEVER escalate: it is retried on the same fallback within its budget.
	if got := cheapHits.Load(); got < 2 {
		t.Fatalf("cheap streaming fallback attempted %d time(s), want >=2 — a 429 on the fallback leg advanced the ladder instead of retrying the same provider (MONEY RULE)", got)
	}
	if got := cheapHits.Load(); got > int32(passthroughMaxRetrySame)+1 {
		t.Fatalf("cheap streaming fallback attempted %d time(s), exceeds 1+passthroughMaxRetrySame=%d — the retry-same budget is unbounded on the fallback leg", got, passthroughMaxRetrySame+1)
	}
}
