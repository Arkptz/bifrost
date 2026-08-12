package bifrost

import (
	"context"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strconv"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	schemas "github.com/maximhq/bifrost/core/schemas"
)

// streamStatusHandler serves the given status as a passthrough SSE stream (a
// flushed data chunk so the first-chunk peek fires), recording every physical hit.
// An optional Retry-After header is stamped on each response.
func streamStatusHandler(status int, hits *atomic.Int32, retryAfter string) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		hits.Add(1)
		if retryAfter != "" {
			w.Header().Set("Retry-After", retryAfter)
		}
		w.WriteHeader(status)
		fl, _ := w.(http.Flusher)
		fmt.Fprintf(w, "data: STATUS-%d\n\n", status)
		if fl != nil {
			fl.Flush()
		}
	}
}

// streamCountingHandler serves failStatus for the first failN hits then 200,
// each as a flushed passthrough SSE stream. Records physical hits.
type streamCountingHandler struct {
	failStatus int
	failN      int32
	hits       atomic.Int32
	retryAfter string
}

func (h *streamCountingHandler) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	n := h.hits.Add(1)
	fl, _ := w.(http.Flusher)
	if n <= h.failN {
		if h.retryAfter != "" {
			w.Header().Set("Retry-After", h.retryAfter)
		}
		w.WriteHeader(h.failStatus)
		fmt.Fprintf(w, "data: FAIL-%d\n\n", h.failStatus)
		if fl != nil {
			fl.Flush()
		}
		return
	}
	w.WriteHeader(http.StatusOK)
	fmt.Fprint(w, "data: RECOVERED\n\n")
	if fl != nil {
		fl.Flush()
	}
}

func doStreamRecorder(t *testing.T, client *Bifrost, provider schemas.ModelProvider, ctx *schemas.BifrostContext, fallbacks []schemas.Fallback) (chan *schemas.BifrostStreamChunk, *schemas.BifrostError) {
	t.Helper()
	return client.PassthroughStream(ctx, provider, &schemas.BifrostPassthroughRequest{
		Method:    http.MethodPost,
		Path:      "/v1/messages",
		Body:      []byte(`{"model":"m"}`),
		Fallbacks: fallbacks,
	})
}

// B1+B2: every PHYSICAL streaming leg of one passthrough request must carry a
// DISTINCT NumberOfRetries so the RequestID:AttemptNumber billing identity never
// collides, across the full sweep of the operator max_retries knob. The pinned
// inner loop (PassthroughSingleAttempt) makes each provider run exactly one physical
// call per logical leg, so physical hits == post-hooks == distinct identities.
// Anti-vacuous: assert >=2 physical legs actually occurred and log them.
func TestPassthroughStreamFailover_BillingIdentityDistinctAcrossLegs(t *testing.T) {
	for _, mr := range []int{0, 1, 2, 3, 5} {
		var pHits, fHits atomic.Int32
		primarySrv := httptest.NewServer(streamStatusHandler(http.StatusInternalServerError, &pHits, ""))
		fallbackSrv := httptest.NewServer(streamStatusHandler(http.StatusInternalServerError, &fHits, ""))

		account := NewMockAccount()
		account.AddProviderWithBaseURL(schemas.OpenAI, 1, 1, primarySrv.URL)
		account.AddProviderWithBaseURL(schemas.Anthropic, 1, 1, fallbackSrv.URL)
		account.configs[schemas.OpenAI].NetworkConfig.MaxRetries = mr
		account.configs[schemas.Anthropic].NetworkConfig.MaxRetries = mr
		account.SetKeysForProvider(schemas.OpenAI, []schemas.Key{
			{ID: "primary-key", Value: *schemas.NewSecretVar("sk-primary"), Models: schemas.WhiteList{"*"}, Weight: 100},
		})
		account.SetKeysForProvider(schemas.Anthropic, []schemas.Key{
			{ID: "fb", Value: *schemas.NewSecretVar("sk-fb"), Models: schemas.WhiteList{"*"}, Weight: 100},
		})

		rp := &billingRecorder{}
		client := newRecorderClient(t, account, rp)

		ctx := schemas.NewBifrostContext(context.Background(), time.Now().Add(30*time.Second))
		stream, _ := doStreamRecorder(t, client, schemas.OpenAI, ctx,
			[]schemas.Fallback{{Provider: schemas.Anthropic, Model: "claude-haiku"}})
		if stream != nil {
			drainPassthroughStream(stream)
		}

		rows := rp.snapshot()
		physicalHits := pHits.Load() + fHits.Load()

		// Streaming emits several PostLLMHooks per physical leg (one per forwarded SSE
		// chunk plus the finalize hook), and every hook of ONE leg legitimately shares
		// that leg's RequestID:AttemptNumber — it is one billable attempt. The billing
		// defect is two DISTINCT physical legs colliding on one identity, so group the
		// hooks by identity and assert the number of distinct identities equals the
		// number of physical upstream calls: one billing slot per physical leg, no more.
		identities := map[string]bool{}
		for _, row := range rows {
			if !row.hasResp {
				continue
			}
			identities[row.requestID+":"+strconv.Itoa(row.numberOfRetries)] = true
		}
		distinctLegs := len(identities)

		if distinctLegs < 2 {
			t.Fatalf("MR=%d: only %d distinct billing legs observed — stream failover not exercised, distinctness test vacuous", mr, distinctLegs)
		}
		if int32(distinctLegs) != physicalHits {
			t.Fatalf("MR=%d: %d distinct billing identities but %d physical upstream calls — two physical legs collided on one billing slot (governance double-counts one, under-bills the other). identities=%v rows=%+v", mr, distinctLegs, physicalHits, identities, rows)
		}
		t.Logf("MR=%d: distinctBillingLegs=%d physicalHits=%d identities=%v", mr, distinctLegs, physicalHits, identities)

		primarySrv.Close()
		fallbackSrv.Close()
	}
}

// B2: with a provider configured max_retries=5 and a permanently failing upstream,
// the total PHYSICAL stream calls across primary+fallback must stay within
// passthroughMaxTotalAttempts. Without the single-attempt pin each logical leg
// fans out to max_retries+1 physical calls and blows the cap.
func TestPassthroughStreamFailover_PhysicalAttemptsHonorCap(t *testing.T) {
	var pHits, fHits atomic.Int32
	primarySrv := httptest.NewServer(streamStatusHandler(http.StatusInternalServerError, &pHits, ""))
	defer primarySrv.Close()
	fallbackSrv := httptest.NewServer(streamStatusHandler(http.StatusInternalServerError, &fHits, ""))
	defer fallbackSrv.Close()

	account := NewMockAccount()
	account.AddProviderWithBaseURL(schemas.OpenAI, 1, 1, primarySrv.URL)
	account.AddProviderWithBaseURL(schemas.Anthropic, 1, 1, fallbackSrv.URL)
	account.configs[schemas.OpenAI].NetworkConfig.MaxRetries = 5
	account.configs[schemas.Anthropic].NetworkConfig.MaxRetries = 5
	account.SetKeysForProvider(schemas.OpenAI, []schemas.Key{
		{ID: "primary-key", Value: *schemas.NewSecretVar("sk-primary"), Models: schemas.WhiteList{"*"}, Weight: 100},
	})
	account.SetKeysForProvider(schemas.Anthropic, []schemas.Key{
		{ID: "fb", Value: *schemas.NewSecretVar("sk-fb"), Models: schemas.WhiteList{"*"}, Weight: 100},
	})
	client := newPassthroughTestClient(t, account)

	ctx := schemas.NewBifrostContext(context.Background(), time.Now().Add(30*time.Second))
	stream, _ := doStreamRecorder(t, client, schemas.OpenAI, ctx,
		[]schemas.Fallback{{Provider: schemas.Anthropic, Model: "claude-haiku"}})
	if stream != nil {
		drainPassthroughStream(stream)
	}

	total := pHits.Load() + fHits.Load()
	t.Logf("max_retries=5 permanent failure: physical stream calls = %d (primary=%d fallback=%d), cap=%d", total, pHits.Load(), fHits.Load(), passthroughMaxTotalAttempts)
	if total > int32(passthroughMaxTotalAttempts) {
		t.Fatalf("physical stream calls = %d, exceeds cap passthroughMaxTotalAttempts=%d — inner loop fan-out is not bounded by the outer counter", total, passthroughMaxTotalAttempts)
	}
	if total < 2 {
		t.Fatalf("physical stream calls = %d — failover not exercised, cap test vacuous", total)
	}
}

// B3 (POOL RULE): a first-chunk 401 must retry the SAME provider within its budget
// and NEVER escalate to the pricier fallback. Two pool keys let a rotation recover.
func TestPassthroughStreamFailover_401RetriesSameNoEscalate(t *testing.T) {
	primary := &streamCountingHandler{failStatus: http.StatusUnauthorized, failN: 1}
	primarySrv := httptest.NewServer(primary)
	defer primarySrv.Close()
	var fallbackHits atomic.Int32
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

	ctx := schemas.NewBifrostContext(context.Background(), time.Now().Add(30*time.Second))
	stream, bifrostErr := doStreamRecorder(t, client, schemas.OpenAI, ctx,
		[]schemas.Fallback{{Provider: schemas.Anthropic, Model: "claude-haiku"}})
	if bifrostErr != nil {
		t.Fatalf("passthrough stream failed (primary=%d): %s", primary.hits.Load(), bifrostErr.Error.Message)
	}
	_, status, _ := drainPassthroughStream(stream)

	if primary.hits.Load() < 2 {
		t.Fatalf("primary hits = %d, want >=2 (401 must retry the SAME provider)", primary.hits.Load())
	}
	if fallbackHits.Load() != 0 {
		t.Fatalf("fallback called %d time(s) — a pool 401 escalated to a pricier provider (POOL/MONEY RULE VIOLATION)", fallbackHits.Load())
	}
	if status != http.StatusOK {
		t.Fatalf("final status = %d, want 200 (401 retries same pool, recovers)", status)
	}
}

// B4: a first-chunk 429 carrying Retry-After: 1 must make the same-provider retry
// WAIT ~1s before re-hitting, bounded by the existing cap. The upstream recovers on
// its second hit so no escalation is involved — this isolates the Retry-After wait.
func TestPassthroughStreamFailover_RetryAfterHonoredAndBounded(t *testing.T) {
	primary := &streamCountingHandler{failStatus: http.StatusTooManyRequests, failN: 1, retryAfter: "1"}
	primarySrv := httptest.NewServer(primary)
	defer primarySrv.Close()
	var fallbackHits atomic.Int32
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

	ctx := schemas.NewBifrostContext(context.Background(), time.Now().Add(30*time.Second))
	start := time.Now()
	stream, bifrostErr := doStreamRecorder(t, client, schemas.OpenAI, ctx,
		[]schemas.Fallback{{Provider: schemas.Anthropic, Model: "claude-haiku"}})
	if bifrostErr != nil {
		t.Fatalf("passthrough stream failed: %s", bifrostErr.Error.Message)
	}
	_, status, _ := drainPassthroughStream(stream)
	elapsed := time.Since(start)

	if primary.hits.Load() < 2 {
		t.Fatalf("primary hits = %d, want >=2 (429 must retry same provider)", primary.hits.Load())
	}
	if fallbackHits.Load() != 0 {
		t.Fatalf("fallback called %d time(s) — 429 must NOT escalate", fallbackHits.Load())
	}
	if status != http.StatusOK {
		t.Fatalf("final status = %d, want 200 (recovers after honoring Retry-After)", status)
	}
	if elapsed < 900*time.Millisecond {
		t.Fatalf("elapsed = %v, want >= ~1s (Retry-After not honored on streaming)", elapsed)
	}
	if elapsed > passthroughRetryAfterCap+5*time.Second {
		t.Fatalf("elapsed = %v exceeded bound (Retry-After not clamped)", elapsed)
	}
}

// B5 (CONCURRENCY): a nil-ctx passthrough stream (which substitutes the shared
// bifrost.ctx) running CONCURRENTLY with a transforming request must not change the
// transforming request's retry behaviour. The transforming provider fails twice then
// recovers; with MaxRetries=3 its inner loop must make exactly 3 physical calls. If
// PassthroughSingleAttempt leaked onto the shared context the transforming loop would
// be pinned to 1 physical call and never recover.
func TestPassthroughStreamFailover_NilCtxDoesNotLeakToTransforming(t *testing.T) {
	// The B5 leak vector is the SHARED instance context: a nil-ctx public call
	// substitutes bifrost.ctx, so the only way a concurrent request observes a key set
	// by another is if it TOO runs on bifrost.ctx. This test runs BOTH on ctx == nil and
	// makes the window deterministic: the nil-ctx passthrough primary blocks on
	// Retry-After: 1 between physical legs, so PassthroughSingleAttempt (if leaked onto
	// bifrost.ctx) is live on the shared context while the concurrent request reads it.
	//
	// The concurrent request is UNARY (ChatCompletionRequest), not streaming, on purpose:
	// two nil-ctx STREAMING requests would also collide on stream-teardown keys
	// (ConnectionClosed / StreamEndIndicator) that have nothing to do with this defect,
	// masking the signal. Unary shares the exact same inner loop (executeRequestWithRetries)
	// and reads the exact same PassthroughSingleAttempt key, but carries no stream-teardown
	// state — so a failure here is attributable solely to the single-attempt pin leaking.
	// The transforming upstream fails twice then recovers; with MaxRetries=3 the inner loop
	// must make exactly 3 physical calls. A leaked pin forces maxRetries=0 → one call → no
	// recovery → error.
	passPrimary := &streamCountingHandler{failStatus: http.StatusInternalServerError, failN: 1000, retryAfter: "1"}
	passSrv := httptest.NewServer(passPrimary)
	defer passSrv.Close()

	var xHits atomic.Int32
	transformSrv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		n := xHits.Add(1)
		if n <= 2 {
			w.WriteHeader(http.StatusInternalServerError)
			_, _ = w.Write([]byte(`{"error":{"message":"transient","type":"server_error"}}`))
			return
		}
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte(`{"id":"c1","object":"chat.completion","choices":[{"index":0,"message":{"role":"assistant","content":"hello"},"finish_reason":"stop"}],"usage":{"prompt_tokens":10,"completion_tokens":2,"total_tokens":12}}`))
	}))
	defer transformSrv.Close()

	account := NewMockAccount()
	account.AddProviderWithBaseURL(schemas.Anthropic, 1, 1, passSrv.URL)
	account.AddProviderWithBaseURL(schemas.OpenAI, 1, 1, transformSrv.URL)
	// Transforming provider keeps NewMockAccount's MaxRetries=3.
	account.configs[schemas.OpenAI].NetworkConfig.RetryBackoffInitial = time.Millisecond
	account.SetKeysForProvider(schemas.Anthropic, []schemas.Key{
		{ID: "p", Value: *schemas.NewSecretVar("sk-p"), Models: schemas.WhiteList{"*"}, Weight: 100},
	})
	account.SetKeysForProvider(schemas.OpenAI, []schemas.Key{
		{ID: "x", Value: *schemas.NewSecretVar("sk-x"), Models: schemas.WhiteList{"*"}, Weight: 100},
	})
	client := newStreamTestClient(t, account)

	var wg sync.WaitGroup
	// nil-ctx passthrough with a fallback so the (guarded) machinery block would fire —
	// it substitutes bifrost.ctx and, WITHOUT the B5 guard, sets PassthroughSingleAttempt
	// on it. Retry-After: 1 holds that state live for ~1s per attempt.
	stop := make(chan struct{})
	wg.Add(1)
	go func() {
		defer wg.Done()
		for {
			select {
			case <-stop:
				return
			default:
			}
			stream, _ := client.PassthroughStream(nil, schemas.Anthropic, &schemas.BifrostPassthroughRequest{
				Method:    http.MethodPost,
				Path:      "/v1/messages",
				Body:      []byte(`{"model":"m"}`),
				Fallbacks: []schemas.Fallback{{Provider: schemas.OpenAI, Model: "gpt-4o-mini"}},
			})
			if stream != nil {
				drainPassthroughStream(stream)
			}
		}
	}()

	// Give the passthrough goroutine time to enter its Retry-After wait with the keys
	// set on bifrost.ctx, so the transforming request below reads the shared context
	// mid-leak.
	time.Sleep(50 * time.Millisecond)

	fail := func(format string, args ...any) {
		close(stop)
		wg.Wait()
		t.Fatalf(format, args...)
	}

	// Run the UNARY transforming request repeatedly WITH ctx == nil (shared bifrost.ctx)
	// while the passthrough holds its machinery keys live on that same context.
	for i := 0; i < 8; i++ {
		xHits.Store(0)
		resp, bifrostErr := client.ChatCompletionRequest(nil, &schemas.BifrostChatRequest{
			Provider: schemas.OpenAI,
			Model:    "gpt-4o-mini",
			Input: []schemas.ChatMessage{
				{Role: schemas.ChatMessageRoleUser, Content: &schemas.ChatMessageContent{ContentStr: schemas.Ptr("hi")}},
			},
		})
		if bifrostErr != nil {
			fail("iter %d: transforming request failed (hits=%d): %s — PassthroughSingleAttempt leaked onto the shared context and pinned the retry loop to one physical call", i, xHits.Load(), bifrostErr.Error.Message)
		}
		if resp == nil {
			fail("iter %d: transforming request returned nil resp (hits=%d)", i, xHits.Load())
		}
		if xHits.Load() != 3 {
			fail("iter %d: transforming physical calls = %d, want exactly 3 (MaxRetries=3, recovers on 3rd) — a leaked single-attempt pin would force 1", i, xHits.Load())
		}
		time.Sleep(10 * time.Millisecond)
	}
	close(stop)
	wg.Wait()
}
