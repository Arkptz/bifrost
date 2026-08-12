package bifrost

import (
	"context"
	"io"
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

// poolKeyAwareHandler keys its behaviour off the ACTUAL credential: the designated
// bad key ALWAYS returns failStatus (401/403), every other key returns 200. This makes
// a test sensitive to whether the pooled retry actually rotated OFF the rejected key —
// reusing the dead key keeps failing, so a 200 genuinely requires a healthy-key pick.
type poolKeyAwareHandler struct {
	badKeyValue string
	failStatus  int
	hits        atomic.Int32
	keysSeen    sync.Map // key value -> *atomic.Int32
}

func (h *poolKeyAwareHandler) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	h.hits.Add(1)
	key := strings.TrimPrefix(r.Header.Get("Authorization"), "Bearer ")
	cnt, _ := h.keysSeen.LoadOrStore(key, new(atomic.Int32))
	cnt.(*atomic.Int32).Add(1)
	if key == h.badKeyValue {
		w.WriteHeader(h.failStatus)
		_, _ = w.Write([]byte(`{"ok":false}`))
		return
	}
	w.WriteHeader(http.StatusOK)
	_, _ = w.Write([]byte(`{"ok":true}`))
}

// runKeyAwarePoolFailover drives one passthrough over a key-aware two-key pool whose
// bad key ALWAYS returns failStatus (401/403) and whose good key returns 200, with a
// live fallback behind it. It returns the final status, the fallback hit count, and how
// many times each pool key value was actually presented to the upstream (via keysSeen).
func runKeyAwarePoolFailover(t *testing.T, failStatus int) (finalStatus int, fbHits int32, badSeen, goodSeen int32) {
	t.Helper()
	h := &poolKeyAwareHandler{badKeyValue: "sk-bad", failStatus: failStatus}
	srv := httptest.NewServer(h)
	defer srv.Close()
	var fb atomic.Int32
	fbSrv := httptest.NewServer(statusHandler(http.StatusOK, &fb))
	defer fbSrv.Close()

	account := NewMockAccount()
	account.AddProviderWithBaseURL(schemas.OpenAI, 1, 1, srv.URL)
	account.SetKeysForProvider(schemas.OpenAI, []schemas.Key{
		{ID: "bad", Value: *schemas.NewSecretVar("sk-bad"), Weight: 100},
		{ID: "good", Value: *schemas.NewSecretVar("sk-good"), Weight: 100},
	})
	account.AddProviderWithBaseURL(schemas.Anthropic, 1, 1, fbSrv.URL)
	// The fallback MUST carry a wildcard key or the model check silently skips it —
	// then a (wrongly) escalated 401/403 never reaches the fallback and the money-rule
	// assertion passes vacuously (AGENTS.md fallback-key trap).
	account.SetKeysForProvider(schemas.Anthropic, []schemas.Key{
		{ID: "fallback-key", Value: *schemas.NewSecretVar("sk-fallback"), Models: schemas.WhiteList{"*"}, Weight: 100},
	})
	client := newPassthroughTestClient(t, account)

	resp, bifrostErr := doPassthrough(t, client, schemas.OpenAI,
		[]schemas.Fallback{{Provider: schemas.Anthropic, Model: "claude-haiku"}})
	if bifrostErr != nil {
		t.Fatalf("unexpected error over key-aware pool: %s", bifrostErr.Error.Message)
	}
	if v, ok := h.keysSeen.Load("sk-bad"); ok {
		badSeen = v.(*atomic.Int32).Load()
	}
	if v, ok := h.keysSeen.Load("sk-good"); ok {
		goodSeen = v.(*atomic.Int32).Load()
	}
	return resp.StatusCode, fb.Load(), badSeen, goodSeen
}

// assertKeyAwarePool runs the key-aware pool scenario many times and asserts the
// properties that ARE deterministic on this path:
//   - MONEY RULE (revert-proof): a pooled 401/403 NEVER escalates to the pricier
//     fallback, on EVERY run — reverting the pooled classification makes it escalate.
//   - KEY-AWARE CORRECTNESS: a 200 outcome happens IFF the good key actually served a
//     request; a fail-status outcome means only the bad key was ever seen. This is what
//     makes the server non-vacuous — a same-dead-key retry genuinely keeps failing.
//   - ROTATION IS POSSIBLE (non-vacuous): across the runs the good key is selected at
//     least once AND the pooled retry budget is spent (aggregate upstream hits exceed
//     the run count), proving the two-key pool CAN rotate and the retry actually fires.
//
// It deliberately does NOT assert per-run "the dead key is not reused": rotation on this
// path is weighted-random with no cross-leg memory (single-attempt pin ⇒ empty deadKeyIDs
// each leg; per-key cooling was deleted), so the dead key IS sometimes re-picked. That is
// the REAL FINDING, reported separately — not something to lock a flaky test around.
func assertKeyAwarePool(t *testing.T, failStatus int) {
	t.Helper()
	const runs = 40
	recovered, failed := 0, 0
	var totalUpstream, totalGood int32
	for i := 0; i < runs; i++ {
		status, fbHits, badSeen, goodSeen := runKeyAwarePoolFailover(t, failStatus)
		if fbHits != 0 {
			t.Fatalf("run %d: pooled %d escalated to the fallback (%d hits) — MONEY RULE VIOLATION", i, failStatus, fbHits)
		}
		switch status {
		case http.StatusOK:
			if goodSeen == 0 {
				t.Fatalf("run %d: 200 with the good key never presented (bad=%d good=%d) — server not key-aware", i, badSeen, goodSeen)
			}
			recovered++
		case failStatus:
			if goodSeen != 0 {
				t.Fatalf("run %d: final %d yet the good key was presented (bad=%d good=%d) — a good-key request should have returned 200", i, failStatus, badSeen, goodSeen)
			}
			failed++
		default:
			t.Fatalf("run %d: unexpected final status %d, want 200 or %d", i, status, failStatus)
		}
		totalUpstream += badSeen + goodSeen
		totalGood += goodSeen
	}
	if totalGood == 0 {
		t.Fatalf("good key never selected across %d runs — the two-key pool never rotated (test vacuous)", runs)
	}
	if int(totalUpstream) <= runs {
		t.Fatalf("aggregate upstream hits = %d over %d runs — the pooled retry budget was never spent (a %d must retry the same provider)", totalUpstream, runs, failStatus)
	}
	t.Logf("key-aware pool over %d runs (status %d): recovered=%d failed=%d upstreamHits=%d goodKeyHits=%d — recovery is NOT deterministic (REAL FINDING: weighted-random re-pick, no cooling)", runs, failStatus, recovered, failed, totalUpstream, totalGood)
}

// NEW REQ (C2): 401 on a key-aware two-key pool. The bad key ALWAYS 401s, the good key
// 200s, so — unlike the old vacuous version whose server returned 200 on the 2nd call
// regardless of key — recovery genuinely requires the pooled retry to land on the good
// key. Asserts the deterministic money rule (never escalate) and that rotation is at
// least possible. Rotation is NOT guaranteed per-run (REAL FINDING, see deliverable).
func TestPassthroughFailover_401RetriesSameNoEscalate(t *testing.T) {
	assertKeyAwarePool(t, http.StatusUnauthorized)
}

// NEW REQ (C2): 403 on a key-aware two-key pool — same policy and same assertions as 401.
func TestPassthroughFailover_403RetriesSameNoEscalate(t *testing.T) {
	assertKeyAwarePool(t, http.StatusForbidden)
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

// A transient 5xx on a CHEAP fallback must not advance the ladder to a pricier
// one: the fallback gets the same retry-same budget the primary gets. fb1
// recovers on its second attempt, so fb2 must never be called.
func TestPassthroughFailover_FallbackRetriesSameBeforeAdvancing(t *testing.T) {
	primary := &countingHandler{failStatus: http.StatusInternalServerError, failN: 1000}
	primarySrv := httptest.NewServer(primary)
	defer primarySrv.Close()

	fb1 := &countingHandler{failStatus: http.StatusInternalServerError, failN: 1}
	fb1Srv := httptest.NewServer(fb1)
	defer fb1Srv.Close()

	var fb2Hits atomic.Int32
	fb2Srv := httptest.NewServer(statusHandler(http.StatusOK, &fb2Hits))
	defer fb2Srv.Close()

	account := NewMockAccount()
	account.AddProviderWithBaseURL(schemas.OpenAI, 1, 1, primarySrv.URL)
	account.AddProviderWithBaseURL(schemas.Anthropic, 1, 1, fb1Srv.URL)
	account.AddProviderWithBaseURL(schemas.Gemini, 1, 1, fb2Srv.URL)
	account.SetKeysForProvider(schemas.Anthropic, []schemas.Key{
		{ID: "fb1-key", Value: *schemas.NewSecretVar("sk-fb1"), Models: schemas.WhiteList{"*"}, Weight: 100},
	})
	account.SetKeysForProvider(schemas.Gemini, []schemas.Key{
		{ID: "fb2-key", Value: *schemas.NewSecretVar("sk-fb2"), Models: schemas.WhiteList{"*"}, Weight: 100},
	})
	client := newPassthroughTestClient(t, account)

	resp, bifrostErr := doPassthrough(t, client, schemas.OpenAI, []schemas.Fallback{
		{Provider: schemas.Anthropic, Model: "claude-haiku"},
		{Provider: schemas.Gemini, Model: "gemini-flash"},
	})
	if bifrostErr != nil {
		t.Fatalf("Passthrough returned error: %s", bifrostErr.Error.Message)
	}
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("final status = %d, want 200 (fb1 recovers on its retry-same)", resp.StatusCode)
	}
	if fb1.hits.Load() < 2 {
		t.Fatalf("fb1 hits = %d, want >=2 (transient 5xx must retry the same fallback)", fb1.hits.Load())
	}
	if fb2Hits.Load() != 0 {
		t.Fatalf("fb2 called %d time(s) — a single transient 5xx escalated to a pricier provider", fb2Hits.Load())
	}
}

// billingRow is one PostLLMHook observation: the RequestID:AttemptNumber identity
// governance dedups billing on (it reads AttemptNumber from NumberOfRetries), plus
// the fallback id and observed status for diagnostics.
type billingRow struct {
	numberOfRetries int
	fallbackID      string
	requestID       string
	status          int
	hasResp         bool
}

// billingRecorder is a faithful stand-in for governance's PostLLMHook: it reads the
// exact same context key (BifrostContextKeyNumberOfRetries) that governance's tracker
// uses as the AttemptNumber half of its RequestID:AttemptNumber dedup identity.
type billingRecorder struct {
	mu   sync.Mutex
	rows []billingRow
}

func (r *billingRecorder) GetName() string { return "billingRecorder" }
func (r *billingRecorder) Cleanup() error  { return nil }
func (r *billingRecorder) PreRequestHook(ctx *schemas.BifrostContext, req *schemas.BifrostRequest) error {
	return nil
}
func (r *billingRecorder) PreLLMHook(ctx *schemas.BifrostContext, req *schemas.BifrostRequest) (*schemas.BifrostRequest, *schemas.LLMPluginShortCircuit, error) {
	return req, nil, nil
}
func (r *billingRecorder) PostLLMHook(ctx *schemas.BifrostContext, resp *schemas.BifrostResponse, bifrostErr *schemas.BifrostError) (*schemas.BifrostResponse, *schemas.BifrostError, error) {
	n, _ := ctx.Value(schemas.BifrostContextKeyNumberOfRetries).(int)
	fid, _ := ctx.Value(schemas.BifrostContextKeyFallbackRequestID).(string)
	rid, _ := ctx.Value(schemas.BifrostContextKeyRequestID).(string)
	row := billingRow{numberOfRetries: n, fallbackID: fid, requestID: rid}
	if resp != nil && resp.PassthroughResponse != nil {
		row.hasResp = true
		row.status = resp.PassthroughResponse.StatusCode
	}
	r.mu.Lock()
	r.rows = append(r.rows, row)
	r.mu.Unlock()
	return resp, bifrostErr, nil
}
func (r *billingRecorder) snapshot() []billingRow {
	r.mu.Lock()
	defer r.mu.Unlock()
	out := make([]billingRow, len(r.rows))
	copy(out, r.rows)
	return out
}
func (r *billingRecorder) reset() {
	r.mu.Lock()
	r.rows = nil
	r.mu.Unlock()
}

func newRecorderClient(t *testing.T, account *MockAccount, rp *billingRecorder) *Bifrost {
	t.Helper()
	client, err := Init(context.Background(), schemas.BifrostConfig{
		Account:    account,
		Logger:     NewDefaultLogger(schemas.LogLevelError),
		LLMPlugins: []schemas.LLMPlugin{rp},
	})
	if err != nil {
		t.Fatalf("failed to initialize bifrost: %v", err)
	}
	t.Cleanup(client.Shutdown)
	return client
}

// readCloseHandler counts each physical hit then closes the connection without a
// response, forcing the provider HTTP client into a transport error (a retryable
// *BifrostError) — the only passthrough failure mode that drives the inner retry
// loop's fan-out. Reproduces Defect 2's amplification.
type readCloseHandler struct{ hits atomic.Int32 }

func (h *readCloseHandler) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	h.hits.Add(1)
	_, _ = io.Copy(io.Discard, r.Body)
	if hj, ok := w.(http.Hijacker); ok {
		if conn, _, err := hj.Hijack(); err == nil {
			_ = conn.Close()
		}
	}
}

// DEFECT 1: governance bills on RequestID:AttemptNumber, reading AttemptNumber from
// BifrostContextKeyNumberOfRetries. Every distinct physical leg of one passthrough
// request must therefore carry a DISTINCT NumberOfRetries, or the failed cheap leg
// and the successful escalation collide on one billing slot. Sweep max_retries so the
// property holds regardless of the per-provider retry knob. Anti-vacuous: assert
// several physical legs actually occurred (failover was exercised, not a single call).
func TestPassthroughFailover_BillingIdentityDistinctAcrossLegs(t *testing.T) {
	for _, mr := range []int{0, 1, 2, 3, 5} {
		var pHits, fHits atomic.Int32
		primarySrv := httptest.NewServer(statusHandler(http.StatusInternalServerError, &pHits))
		fallbackSrv := httptest.NewServer(statusHandler(http.StatusInternalServerError, &fHits))

		account := NewMockAccount()
		account.AddProviderWithBaseURL(schemas.OpenAI, 1, 1, primarySrv.URL)
		account.AddProviderWithBaseURL(schemas.Anthropic, 1, 1, fallbackSrv.URL)
		account.configs[schemas.OpenAI].NetworkConfig.MaxRetries = mr
		account.configs[schemas.Anthropic].NetworkConfig.MaxRetries = mr
		account.SetKeysForProvider(schemas.Anthropic, []schemas.Key{
			{ID: "fb", Value: *schemas.NewSecretVar("sk-fb"), Models: schemas.WhiteList{"*"}, Weight: 100},
		})

		rp := &billingRecorder{}
		client := newRecorderClient(t, account, rp)

		ctx := schemas.NewBifrostContext(context.Background(), time.Now().Add(30*time.Second))
		_, _ = client.Passthrough(ctx, schemas.OpenAI, &schemas.BifrostPassthroughRequest{
			Method:    http.MethodPost,
			Path:      "/v1/messages",
			Body:      []byte(`{"model":"m"}`),
			Fallbacks: []schemas.Fallback{{Provider: schemas.Anthropic, Model: "claude-haiku"}},
		})

		rows := rp.snapshot()
		physicalHits := pHits.Load() + fHits.Load()

		legs := 0
		seen := map[string]int{}
		for _, row := range rows {
			if !row.hasResp {
				continue
			}
			legs++
			id := row.requestID + ":" + strconv.Itoa(row.numberOfRetries)
			if _, dup := seen[id]; dup {
				t.Fatalf("MR=%d: two distinct physical legs share billing identity %q (rows=%+v) — governance would double-count one and under-bill the other", mr, id, rows)
			}
			seen[id] = row.numberOfRetries
		}

		// Anti-vacuous: the primary retry-same plus the fallback ladder must have
		// produced multiple physical legs, and each provider hit must surface exactly
		// one post-hook — otherwise the distinctness assertion is trivially true.
		if legs < 2 {
			t.Fatalf("MR=%d: only %d physical legs observed — failover not exercised, distinctness test vacuous", mr, legs)
		}
		if int(physicalHits) != len(rows) {
			t.Fatalf("MR=%d: physical upstream hits=%d but post-hooks=%d — physical/billing accounting mismatch", mr, physicalHits, len(rows))
		}
		t.Logf("MR=%d: physicalLegs=%d hits=%d identities=%v", mr, legs, physicalHits, seen)
	}
}

// DEFECT 2 (TWO-TIER BOUND). passthroughMaxTotalAttempts caps LOGICAL upstream
// attempts — the money/billing bound: one attemptFn invocation, one distinct
// AttemptBase/billing identity, one PostLLMHook observation. It does NOT, and must
// not be read as, a cap on physical sockets.
//
// StaleConnectionRetryIfErr (core/network/http.go:265, const maxStaleConnRetries = 3
// at :263) is installed as fasthttp's RetryIfErr. On io.EOF from a hijack-and-close
// server it redials ONE LAYER BELOW passthroughAttemptCounter, so a single logical
// attempt can open up to maxStaleConnRetries+1 physical sockets. A redial is NOT a
// billable attempt: the server returned zero bytes, no tokens, no LLM work — nothing
// to bill for. Counting sockets against the money cap conflates the two tiers and is
// the false assertion this test previously carried.
//
// The honest claim is a bound per tier:
//  1. LOGICAL attempts (PostLLMHook observations, i.e. billing identities) must stay
//     <= passthroughMaxTotalAttempts. This is the money bound — counted by billing
//     identity, NEVER by sockets.
//  2. PHYSICAL sockets must stay <= passthroughMaxTotalAttempts*(maxStaleConnRetries+1),
//     the transport ceiling. This proves redials still exist but remain BOUNDED — no
//     infinite redial storm — without pretending they are billable.
//
// Name keeps the "PhysicalAttemptsHonorCap" prefix so the physical-ceiling tier stays
// documented; the suffix records the logical money bound the old assertion mangled.
func TestPassthroughFailover_PhysicalAttemptsHonorCap_LogicalIsMoneyBound(t *testing.T) {
	// maxStaleConnRetries mirrors the unexported const in package network
	// (core/network/http.go:263). It is not reachable from package bifrost and must
	// not be exported, so tier 2 is derived from named constants here — never a
	// hardcoded 32.
	const maxStaleConnRetries = 3

	var pHits atomic.Int32
	primarySrv := httptest.NewServer(statusHandler(http.StatusInternalServerError, &pHits))
	defer primarySrv.Close()
	fb := &readCloseHandler{}
	fallbackSrv := httptest.NewServer(fb)
	defer fallbackSrv.Close()

	account := NewMockAccount()
	account.AddProviderWithBaseURL(schemas.OpenAI, 1, 1, primarySrv.URL)
	account.AddProviderWithBaseURL(schemas.Anthropic, 1, 1, fallbackSrv.URL)
	account.configs[schemas.OpenAI].NetworkConfig.MaxRetries = 5
	account.configs[schemas.Anthropic].NetworkConfig.MaxRetries = 5
	account.SetKeysForProvider(schemas.Anthropic, []schemas.Key{
		{ID: "fb", Value: *schemas.NewSecretVar("sk-fb"), Models: schemas.WhiteList{"*"}, Weight: 100},
	})

	// billingRecorder's PostLLMHook fires exactly once per LOGICAL attempt (each with a
	// distinct billing identity), regardless of how many sockets the inner transport
	// redials underneath it — so len(rows) is the tier-1 (money) count.
	rp := &billingRecorder{}
	client := newRecorderClient(t, account, rp)

	ctx := schemas.NewBifrostContext(context.Background(), time.Now().Add(30*time.Second))
	_, _ = client.Passthrough(ctx, schemas.OpenAI, &schemas.BifrostPassthroughRequest{
		Method:    http.MethodPost,
		Path:      "/v1/messages",
		Body:      []byte(`{"model":"m"}`),
		Fallbacks: []schemas.Fallback{{Provider: schemas.Anthropic, Model: "claude-haiku"}},
	})

	logical := int32(len(rp.snapshot()))      // tier 1: billable attempts
	physical := pHits.Load() + fb.hits.Load() // tier 2: sockets opened
	physicalCap := int32(passthroughMaxTotalAttempts * (maxStaleConnRetries + 1))

	t.Logf("max_retries=5 permanent failure: LOGICAL attempts=%d (money cap=%d) | PHYSICAL sockets=%d (primary=%d fallback=%d, transport ceiling=%d)",
		logical, passthroughMaxTotalAttempts, physical, pHits.Load(), fb.hits.Load(), physicalCap)

	// Anti-vacuous: failover must actually have run. If only the primary's legs were
	// billed, the cap assertion would be trivially satisfied and prove nothing.
	if logical < 2 {
		t.Fatalf("only %d logical attempts observed — failover not exercised, money-cap assertion vacuous", logical)
	}
	// Tier 1 — money bound. This is the assertion that MUST be able to fail if the
	// billable-attempt counter regresses.
	if logical > int32(passthroughMaxTotalAttempts) {
		t.Fatalf("LOGICAL attempts = %d, exceeds money cap passthroughMaxTotalAttempts=%d — billable-attempt counter is not bounded", logical, passthroughMaxTotalAttempts)
	}
	// Tier 2 — transport ceiling. Redials are admitted but must stay bounded; a blown
	// ceiling means a stale-conn redial storm, not a billing bug.
	if physical > physicalCap {
		t.Fatalf("PHYSICAL sockets = %d, exceeds transport ceiling %d (=passthroughMaxTotalAttempts*(maxStaleConnRetries+1)) — stale-conn redials are unbounded", physical, physicalCap)
	}
}

// LEAK: BifrostContextKeyAttemptBase lives on the context. If a passthrough failover
// leaves it stamped, a LATER request sharing that context (new RequestID, no
// fallbacks) inherits the base and its inner loop stamps base+0 instead of 0,
// mis-billing the fresh request. The base must be cleared on failover exit.
func TestPassthroughFailover_AttemptBaseDoesNotLeak(t *testing.T) {
	var pHits, fHits atomic.Int32
	primarySrv := httptest.NewServer(statusHandler(http.StatusInternalServerError, &pHits))
	defer primarySrv.Close()
	fallbackSrv := httptest.NewServer(statusHandler(http.StatusInternalServerError, &fHits))
	defer fallbackSrv.Close()

	account := NewMockAccount()
	account.AddProviderWithBaseURL(schemas.OpenAI, 1, 1, primarySrv.URL)
	account.AddProviderWithBaseURL(schemas.Anthropic, 1, 1, fallbackSrv.URL)
	account.SetKeysForProvider(schemas.Anthropic, []schemas.Key{
		{ID: "fb", Value: *schemas.NewSecretVar("sk-fb"), Models: schemas.WhiteList{"*"}, Weight: 100},
	})

	rp := &billingRecorder{}
	client := newRecorderClient(t, account, rp)

	// Request 1: passthrough WITH fallbacks over an all-500 ladder advances the base
	// past 0 (primary + retry-same + fallback + retry-same).
	ctx := schemas.NewBifrostContext(context.Background(), time.Now().Add(30*time.Second))
	_, _ = client.Passthrough(ctx, schemas.OpenAI, &schemas.BifrostPassthroughRequest{
		Method:    http.MethodPost,
		Path:      "/v1/messages",
		Body:      []byte(`{"model":"m"}`),
		Fallbacks: []schemas.Fallback{{Provider: schemas.Anthropic, Model: "claude-haiku"}},
	})
	first := rp.snapshot()
	maxBase := 0
	for _, row := range first {
		if row.numberOfRetries > maxBase {
			maxBase = row.numberOfRetries
		}
	}
	if maxBase == 0 {
		t.Fatalf("first request did not advance the base past 0 (rows=%+v) — leak test cannot distinguish a leak", first)
	}

	// Request 2: SAME context, a fresh RequestID and NO fallbacks. It never enters
	// passthrough failover, so it must observe a pristine base of 0.
	rp.reset()
	ctx.SetValue(schemas.BifrostContextKeyRequestID, "second-request-shares-context")
	_, _ = client.Passthrough(ctx, schemas.OpenAI, &schemas.BifrostPassthroughRequest{
		Method: http.MethodPost,
		Path:   "/v1/messages",
		Body:   []byte(`{"model":"m"}`),
	})
	second := rp.snapshot()
	if len(second) != 1 {
		t.Fatalf("second request produced %d post-hooks, want exactly 1 (no fallbacks, single call): %+v", len(second), second)
	}
	if second[0].numberOfRetries != 0 {
		t.Fatalf("AttemptBase leaked: second request observed NumberOfRetries=%d, want 0 (first request advanced base to %d and it survived)", second[0].numberOfRetries, maxBase)
	}
}
