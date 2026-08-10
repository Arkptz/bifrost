package bifrost

import (
	"context"
	"net/http"
	"net/http/httptest"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	schemas "github.com/maximhq/bifrost/core/schemas"
)

// countingHandler serves a fixed status for the first `failN` hits, then 200.
// It records every hit and the api key seen, so tests can assert same-provider
// retries and per-key cooling.
type countingHandler struct {
	failStatus int
	failN      int32
	hits       atomic.Int32
	keysSeen   sync.Map // key: x-api-key value seen -> count (atomic.Int32)
	retryAfter string
}

func (h *countingHandler) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	n := h.hits.Add(1)
	if k := strings.TrimPrefix(r.Header.Get("Authorization"), "Bearer "); k != "" {
		cnt, _ := h.keysSeen.LoadOrStore(k, new(atomic.Int32))
		cnt.(*atomic.Int32).Add(1)
	}
	if n <= h.failN {
		if h.retryAfter != "" {
			w.Header().Set("Retry-After", h.retryAfter)
		}
		w.WriteHeader(h.failStatus)
		_, _ = w.Write([]byte(`{"ok":false}`))
		return
	}
	w.WriteHeader(http.StatusOK)
	_, _ = w.Write([]byte(`{"ok":true}`))
}

func doPassthrough(t *testing.T, client *Bifrost, provider schemas.ModelProvider, fallbacks []schemas.Fallback) (*schemas.BifrostPassthroughResponse, *schemas.BifrostError) {
	t.Helper()
	ctx := schemas.NewBifrostContext(context.Background(), time.Now().Add(30*time.Second))
	return client.Passthrough(ctx, provider, &schemas.BifrostPassthroughRequest{
		Method:    http.MethodPost,
		Path:      "/v1/messages",
		Body:      []byte(`{"model":"m"}`),
		Fallbacks: fallbacks,
	})
}

// FIX-1: a single normal-duration 500 must retry the SAME provider before any
// escalation. The primary recovers on its second attempt, so the fallback must
// never be called.
func TestPassthroughFailover_5xxRetriesSameBeforeEscalate(t *testing.T) {
	primary := &countingHandler{failStatus: http.StatusInternalServerError, failN: 1}
	primarySrv := httptest.NewServer(primary)
	defer primarySrv.Close()
	var fallbackHits atomic.Int32
	fallbackSrv := httptest.NewServer(statusHandler(http.StatusOK, &fallbackHits))
	defer fallbackSrv.Close()

	account := NewMockAccount()
	account.AddProviderWithBaseURL(schemas.OpenAI, 1, 1, primarySrv.URL)
	account.AddProviderWithBaseURL(schemas.Anthropic, 1, 1, fallbackSrv.URL)
	client := newPassthroughTestClient(t, account)

	resp, bifrostErr := doPassthrough(t, client, schemas.OpenAI,
		[]schemas.Fallback{{Provider: schemas.Anthropic, Model: "claude-haiku"}})
	if bifrostErr != nil {
		t.Fatalf("Passthrough returned error: %s", bifrostErr.Error.Message)
	}
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("final status = %d, want 200 (primary recovers on retry-same)", resp.StatusCode)
	}
	if primary.hits.Load() < 2 {
		t.Fatalf("primary hits = %d, want >=2 (5xx must retry same before escalate)", primary.hits.Load())
	}
	if fallbackHits.Load() != 0 {
		t.Fatalf("fallback called %d time(s) — 5xx escalated before exhausting same-provider retries", fallbackHits.Load())
	}
}

// FIX-2: with NO fallbacks configured, a 429 must produce exactly ONE upstream
// call and the first body returned verbatim — byte-identical to upstream.
func TestPassthroughFailover_NoFallbacks429SingleCall(t *testing.T) {
	var hits atomic.Int32
	srv := httptest.NewServer(statusHandler(http.StatusTooManyRequests, &hits))
	defer srv.Close()

	account := NewMockAccount()
	account.AddProviderWithBaseURL(schemas.OpenAI, 1, 1, srv.URL)
	client := newPassthroughTestClient(t, account)

	resp, bifrostErr := doPassthrough(t, client, schemas.OpenAI, nil)
	if bifrostErr != nil {
		t.Fatalf("Passthrough returned error: %s", bifrostErr.Error.Message)
	}
	if resp.StatusCode != http.StatusTooManyRequests {
		t.Fatalf("final status = %d, want 429 verbatim", resp.StatusCode)
	}
	if hits.Load() != 1 {
		t.Fatalf("server hits = %d, want exactly 1 (no retry without fallbacks)", hits.Load())
	}
}

// FIX-4: Retry-After is honored and bounded. A 429 with a small Retry-After
// makes the loop wait ~that long before the same-provider retry; a huge
// Retry-After is capped so the request cannot hang past the client timeout.
func TestPassthroughFailover_RetryAfterHonoredAndBounded(t *testing.T) {
	// Small Retry-After: 1s. The retry-same must wait at least ~1s.
	primary := &countingHandler{failStatus: http.StatusTooManyRequests, failN: 1, retryAfter: "1"}
	primarySrv := httptest.NewServer(primary)
	defer primarySrv.Close()

	account := NewMockAccount()
	account.AddProviderWithBaseURL(schemas.OpenAI, 1, 1, primarySrv.URL)
	account.AddProviderWithBaseURL(schemas.Anthropic, 1, 1, primarySrv.URL)
	client := newPassthroughTestClient(t, account)

	start := time.Now()
	resp, bifrostErr := doPassthrough(t, client, schemas.OpenAI,
		[]schemas.Fallback{{Provider: schemas.Anthropic, Model: "claude-haiku"}})
	elapsed := time.Since(start)
	if bifrostErr != nil {
		t.Fatalf("Passthrough returned error: %s", bifrostErr.Error.Message)
	}
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("final status = %d, want 200 (recovers after honoring Retry-After)", resp.StatusCode)
	}
	if elapsed < 900*time.Millisecond {
		t.Fatalf("elapsed = %v, want >= ~1s (Retry-After honored)", elapsed)
	}
	// Bounded: a 1s Retry-After must not push us anywhere near the huge cap.
	if elapsed > passthroughRetryAfterCap+5*time.Second {
		t.Fatalf("elapsed = %v exceeded bound", elapsed)
	}
}

// FIX-4b: a huge Retry-After is capped at passthroughRetryAfterCap.
func TestPassthroughFailover_RetryAfterCapped(t *testing.T) {
	got := boundedRetryAfter(3600 * time.Second)
	if got > passthroughRetryAfterCap {
		t.Fatalf("boundedRetryAfter(1h) = %v, want <= cap %v", got, passthroughRetryAfterCap)
	}
	if got != passthroughRetryAfterCap {
		t.Fatalf("boundedRetryAfter(1h) = %v, want exactly cap %v", got, passthroughRetryAfterCap)
	}
	if boundedRetryAfter(-5*time.Second) != 0 {
		t.Fatalf("negative Retry-After must be treated as 0")
	}
}

// FIX-5: a persistent ≤3ms 502 (local fast-fail, upstream never contacted)
// retries the same provider up to its budget then ESCALATES to the fallback.
// Local fast-fail httptest servers respond in well under 3ms.
func TestPassthroughFailover_LocalFastFailEventuallyEscalates(t *testing.T) {
	var primaryHits, fallbackHits atomic.Int32
	// Always 502 with a tiny body — httptest loopback replies in <3ms.
	primarySrv := httptest.NewServer(statusHandler(http.StatusBadGateway, &primaryHits))
	defer primarySrv.Close()
	fallbackSrv := httptest.NewServer(statusHandler(http.StatusOK, &fallbackHits))
	defer fallbackSrv.Close()

	account := NewMockAccount()
	account.AddProviderWithBaseURL(schemas.OpenAI, 1, 1, primarySrv.URL)
	account.AddProviderWithBaseURL(schemas.Anthropic, 1, 1, fallbackSrv.URL)
	account.SetKeysForProvider(schemas.Anthropic, []schemas.Key{
		{ID: "fallback-key", Value: *schemas.NewSecretVar("sk-fallback"), Models: schemas.WhiteList{"*"}, Weight: 100},
	})
	client := newPassthroughTestClient(t, account)

	resp, bifrostErr := doPassthrough(t, client, schemas.OpenAI,
		[]schemas.Fallback{{Provider: schemas.Anthropic, Model: "claude-haiku"}})
	if bifrostErr != nil {
		t.Fatalf("Passthrough returned error: %s", bifrostErr.Error.Message)
	}
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("final status = %d, want 200 (persistent local fast-fail must escalate)", resp.StatusCode)
	}
	if fallbackHits.Load() == 0 {
		t.Fatal("fallback never called — persistent local fast-fail did not escalate")
	}
}

// FIX-6: a fallback provider that returns 429 is retried on THAT fallback (the
// same classification/retry policy applies to fallback attempts).
func TestPassthroughFailover_FallbackRetriedOn429(t *testing.T) {
	var primaryHits atomic.Int32
	primarySrv := httptest.NewServer(statusHandler(http.StatusInternalServerError, &primaryHits))
	defer primarySrv.Close()
	// Fallback: 429 once, then 200. Must be retried on the fallback itself.
	fallback := &countingHandler{failStatus: http.StatusTooManyRequests, failN: 1}
	fallbackSrv := httptest.NewServer(fallback)
	defer fallbackSrv.Close()

	account := NewMockAccount()
	account.AddProviderWithBaseURL(schemas.OpenAI, 1, 1, primarySrv.URL)
	account.AddProviderWithBaseURL(schemas.Anthropic, 1, 1, fallbackSrv.URL)
	account.SetKeysForProvider(schemas.Anthropic, []schemas.Key{
		{ID: "fallback-key", Value: *schemas.NewSecretVar("sk-fallback"), Models: schemas.WhiteList{"*"}, Weight: 100},
	})
	client := newPassthroughTestClient(t, account)

	resp, bifrostErr := doPassthrough(t, client, schemas.OpenAI,
		[]schemas.Fallback{{Provider: schemas.Anthropic, Model: "claude-haiku"}})
	if bifrostErr != nil {
		t.Fatalf("Passthrough returned error: %s", bifrostErr.Error.Message)
	}
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("final status = %d, want 200 (fallback 429 must be retried on fallback)", resp.StatusCode)
	}
	if fallback.hits.Load() < 2 {
		t.Fatalf("fallback hits = %d, want >=2 (429 on fallback must retry same fallback)", fallback.hits.Load())
	}
}

// NEW REQ: 401 then success on the SAME provider (pool-account case). The 401
// must NOT be fail-fast and must NOT escalate to the pricier fallback.
func TestPassthroughFailover_401RetriesSameNoEscalate(t *testing.T) {
	primary := &countingHandler{failStatus: http.StatusUnauthorized, failN: 1}
	primarySrv := httptest.NewServer(primary)
	defer primarySrv.Close()
	var fallbackHits atomic.Int32
	fallbackSrv := httptest.NewServer(statusHandler(http.StatusOK, &fallbackHits))
	defer fallbackSrv.Close()

	account := NewMockAccount()
	// Two keys so a different pool account can be selected on retry.
	account.AddProviderWithBaseURL(schemas.OpenAI, 1, 1, primarySrv.URL)
	account.SetKeysForProvider(schemas.OpenAI, []schemas.Key{
		{ID: "k1", Value: *schemas.NewSecretVar("sk-1"), Weight: 100},
		{ID: "k2", Value: *schemas.NewSecretVar("sk-2"), Weight: 100},
	})
	account.AddProviderWithBaseURL(schemas.Anthropic, 1, 1, fallbackSrv.URL)
	client := newPassthroughTestClient(t, account)

	resp, bifrostErr := doPassthrough(t, client, schemas.OpenAI,
		[]schemas.Fallback{{Provider: schemas.Anthropic, Model: "claude-haiku"}})
	if bifrostErr != nil {
		t.Fatalf("Passthrough returned error: %s", bifrostErr.Error.Message)
	}
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("final status = %d, want 200 (401 retries same pool, recovers)", resp.StatusCode)
	}
	if primary.hits.Load() < 2 {
		t.Fatalf("primary hits = %d, want >=2 (401 must retry same provider)", primary.hits.Load())
	}
	if fallbackHits.Load() != 0 {
		t.Fatalf("fallback called %d time(s) — 401 must NOT escalate", fallbackHits.Load())
	}
}

// NEW REQ: 403 then success on the SAME provider — same policy as 401.
func TestPassthroughFailover_403RetriesSameNoEscalate(t *testing.T) {
	primary := &countingHandler{failStatus: http.StatusForbidden, failN: 1}
	primarySrv := httptest.NewServer(primary)
	defer primarySrv.Close()
	var fallbackHits atomic.Int32
	fallbackSrv := httptest.NewServer(statusHandler(http.StatusOK, &fallbackHits))
	defer fallbackSrv.Close()

	account := NewMockAccount()
	account.AddProviderWithBaseURL(schemas.OpenAI, 1, 1, primarySrv.URL)
	account.SetKeysForProvider(schemas.OpenAI, []schemas.Key{
		{ID: "k1", Value: *schemas.NewSecretVar("sk-1"), Weight: 100},
		{ID: "k2", Value: *schemas.NewSecretVar("sk-2"), Weight: 100},
	})
	account.AddProviderWithBaseURL(schemas.Anthropic, 1, 1, fallbackSrv.URL)
	client := newPassthroughTestClient(t, account)

	resp, bifrostErr := doPassthrough(t, client, schemas.OpenAI,
		[]schemas.Fallback{{Provider: schemas.Anthropic, Model: "claude-haiku"}})
	if bifrostErr != nil {
		t.Fatalf("Passthrough returned error: %s", bifrostErr.Error.Message)
	}
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("final status = %d, want 200 (403 retries same pool, recovers)", resp.StatusCode)
	}
	if primary.hits.Load() < 2 {
		t.Fatalf("primary hits = %d, want >=2 (403 must retry same provider)", primary.hits.Load())
	}
	if fallbackHits.Load() != 0 {
		t.Fatalf("fallback called %d time(s) — 403 must NOT escalate", fallbackHits.Load())
	}
}

// NEW REQ: a key that returned 401/403/429 is cooled — the immediate next
// same-provider retry must select a DIFFERENT key. The handler fails only the
// first key it sees; if cooling steers to the other key, the request recovers
// and both keys are used exactly once.
func TestPassthroughFailover_CoolsFailedKey(t *testing.T) {
	h := &poolAccountHandler{}
	srv := httptest.NewServer(h)
	defer srv.Close()

	account := NewMockAccount()
	account.AddProviderWithBaseURL(schemas.OpenAI, 1, 1, srv.URL)
	account.SetKeysForProvider(schemas.OpenAI, []schemas.Key{
		{ID: "k1", Value: *schemas.NewSecretVar("sk-1"), Weight: 100},
		{ID: "k2", Value: *schemas.NewSecretVar("sk-2"), Weight: 100},
	})
	// Fallback exists but must not be needed.
	account.AddProviderWithBaseURL(schemas.Anthropic, 1, 1, srv.URL)
	client := newPassthroughTestClient(t, account)

	resp, bifrostErr := doPassthrough(t, client, schemas.OpenAI,
		[]schemas.Fallback{{Provider: schemas.Anthropic, Model: "claude-haiku"}})
	if bifrostErr != nil {
		t.Fatalf("Passthrough returned error: %s", bifrostErr.Error.Message)
	}
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("final status = %d, want 200 (cooling steers to healthy key)", resp.StatusCode)
	}
	first := h.firstKey.Load()
	if first == nil {
		t.Fatal("no key was seen")
	}
	// The cooled (first, failing) key must not be the one that served the retry.
	if h.retryKey.Load() == nil {
		t.Fatal("no retry attempt observed — cooling not exercised")
	}
	if h.retryKey.Load().(string) == first.(string) {
		t.Fatalf("retry reused cooled key %q — cooling did not steer away", first)
	}
}

// poolAccountHandler simulates a reseller pool: the FIRST distinct api key it
// sees returns 401 (that account is dead); any other key returns 200. Records
// the first key and the key that served the successful retry.
type poolAccountHandler struct {
	firstKey atomic.Value // string
	retryKey atomic.Value // string
	hits     atomic.Int32
}

func (h *poolAccountHandler) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	h.hits.Add(1)
	key := strings.TrimPrefix(r.Header.Get("Authorization"), "Bearer ")
	if h.firstKey.Load() == nil {
		h.firstKey.Store(key)
		w.WriteHeader(http.StatusUnauthorized)
		_, _ = w.Write([]byte(`{"ok":false}`))
		return
	}
	if key != h.firstKey.Load().(string) {
		h.retryKey.Store(key)
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte(`{"ok":true}`))
		return
	}
	// Same dead key again — should not happen if cooling works.
	h.retryKey.Store(key)
	w.WriteHeader(http.StatusUnauthorized)
	_, _ = w.Write([]byte(`{"ok":false}`))
}

// NEW REQ (bound): under a permanently-failing provider, total attempts and
// wall-clock stay bounded — the loop must terminate and return the last status.
func TestPassthroughFailover_BoundedUnderPermanentFailure(t *testing.T) {
	var hits atomic.Int32
	// Always 500; fallback also always 500. Loop must terminate.
	primarySrv := httptest.NewServer(statusHandler(http.StatusInternalServerError, &hits))
	defer primarySrv.Close()
	var fbHits atomic.Int32
	fallbackSrv := httptest.NewServer(statusHandler(http.StatusInternalServerError, &fbHits))
	defer fallbackSrv.Close()

	account := NewMockAccount()
	account.AddProviderWithBaseURL(schemas.OpenAI, 1, 1, primarySrv.URL)
	account.AddProviderWithBaseURL(schemas.Anthropic, 1, 1, fallbackSrv.URL)
	account.SetKeysForProvider(schemas.Anthropic, []schemas.Key{
		{ID: "fallback-key", Value: *schemas.NewSecretVar("sk-fallback"), Models: schemas.WhiteList{"*"}, Weight: 100},
	})
	client := newPassthroughTestClient(t, account)

	start := time.Now()
	resp, bifrostErr := doPassthrough(t, client, schemas.OpenAI,
		[]schemas.Fallback{{Provider: schemas.Anthropic, Model: "claude-haiku"}})
	elapsed := time.Since(start)
	if bifrostErr != nil {
		t.Fatalf("Passthrough returned error: %s", bifrostErr.Error.Message)
	}
	if resp.StatusCode != http.StatusInternalServerError {
		t.Fatalf("final status = %d, want 500 (all providers down)", resp.StatusCode)
	}
	total := hits.Load() + fbHits.Load()
	if total > int32(passthroughMaxTotalAttempts) {
		t.Fatalf("total attempts = %d, exceeds bound %d", total, passthroughMaxTotalAttempts)
	}
	if elapsed > 25*time.Second {
		t.Fatalf("elapsed = %v — unbounded under permanent failure", elapsed)
	}
	_ = strconv.Itoa
}
