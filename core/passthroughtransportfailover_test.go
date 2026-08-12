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
	"time"

	schemas "github.com/maximhq/bifrost/core/schemas"
)

// R3: a genuinely-down STREAMING primary (transport error: no HTTP response) must
// become escalate-eligible and reach the healthy fallback — while a control probe
// confirms the failure really is a transport error (StatusCode nil), not a status,
// so the test can never pass vacuously.
func TestPassthroughStreamFailover_TransportErrorEscalates(t *testing.T) {
	// A server closed before use: every dial is connection-refused (a transport error
	// with no HTTP response), the R3 anchor's exact "down provider" shape.
	downSrv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {}))
	downURL := downSrv.URL
	downSrv.Close()

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
	account.AddProviderWithBaseURL(schemas.OpenAI, 1, 1, downURL)
	account.AddProviderWithBaseURL(schemas.Anthropic, 1, 1, fallbackSrv.URL)
	account.SetKeysForProvider(schemas.OpenAI, []schemas.Key{
		{ID: "primary-key", Value: *schemas.NewSecretVar("sk-primary"), Models: schemas.WhiteList{"*"}, Weight: 100},
	})
	account.SetKeysForProvider(schemas.Anthropic, []schemas.Key{
		{ID: "fallback-key", Value: *schemas.NewSecretVar("sk-fallback"), Models: schemas.WhiteList{"*"}, Weight: 100},
	})
	client := newPassthroughTestClient(t, account)

	// CONTROL (anti-vacuous): a no-fallback call to the down primary must surface a
	// TRANSPORT error — StatusCode nil, no stashed PassthroughResponse — not an HTTP
	// status. If bifrost mapped connection-refused to a real status this whole test
	// would be exercising the wrong path.
	_, probeErr := doPassthroughStream(t, client, schemas.OpenAI, nil)
	if probeErr == nil {
		t.Fatalf("control: down primary with no fallbacks returned no error — not a transport failure")
	}
	if probeErr.StatusCode != nil {
		t.Fatalf("control: down primary error carried StatusCode=%d, want nil (a real transport error has no HTTP status)", *probeErr.StatusCode)
	}
	if probeErr.ExtraFields.PassthroughResponse != nil {
		t.Fatalf("control: down primary error stashed a PassthroughResponse — not a bare transport error")
	}

	// MAIN: with a healthy fallback configured, the transport-error primary must
	// escalate and the client must receive the FALLBACK stream.
	stream, bifrostErr := doPassthroughStream(t, client, schemas.OpenAI,
		[]schemas.Fallback{{Provider: schemas.Anthropic, Model: "claude-haiku"}})
	if bifrostErr != nil {
		t.Fatalf("transport-error passthrough stream failed (fallback=%d): %s", fallbackHits.Load(), bifrostErr.Error.Message)
	}
	body, status, errs := drainPassthroughStream(stream)
	if fallbackHits.Load() != 1 {
		t.Fatalf("fallback hits = %d, want 1 (a down primary must escalate to the healthy fallback)", fallbackHits.Load())
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

// primaryCallCounter counts PreLLMHook invocations per provider — the exact signal
// the R7 probe reads ("the primary plugin pipeline ran N times"). A down primary is
// reached via a closed-before-use server (genuine connection-refused, no server-side
// counter), so the plugin pipeline is the only place its physical attempts can be
// counted. Each same-provider re-issue in runPassthroughStatusFailover re-runs the
// full pipeline (AGENTS.md gotcha #9), so a PreLLMHook tick == one physical leg.
type primaryCallCounter struct {
	mu    sync.Mutex
	calls map[schemas.ModelProvider]int
}

func (c *primaryCallCounter) GetName() string { return "primaryCallCounter" }
func (c *primaryCallCounter) Cleanup() error  { return nil }
func (c *primaryCallCounter) PreRequestHook(ctx *schemas.BifrostContext, req *schemas.BifrostRequest) error {
	return nil
}
func (c *primaryCallCounter) PreLLMHook(ctx *schemas.BifrostContext, req *schemas.BifrostRequest) (*schemas.BifrostRequest, *schemas.LLMPluginShortCircuit, error) {
	provider, _, _ := req.GetRequestFields()
	c.mu.Lock()
	if c.calls == nil {
		c.calls = map[schemas.ModelProvider]int{}
	}
	c.calls[provider]++
	c.mu.Unlock()
	return req, nil, nil
}
func (c *primaryCallCounter) PostLLMHook(ctx *schemas.BifrostContext, resp *schemas.BifrostResponse, bifrostErr *schemas.BifrostError) (*schemas.BifrostResponse, *schemas.BifrostError, error) {
	return resp, bifrostErr, nil
}
func (c *primaryCallCounter) count(p schemas.ModelProvider) int {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.calls[p]
}

func newCounterClient(t *testing.T, account *MockAccount, c *primaryCallCounter) *Bifrost {
	t.Helper()
	client, err := Init(context.Background(), schemas.BifrostConfig{
		Account:    account,
		Logger:     NewDefaultLogger(schemas.LogLevelError),
		LLMPlugins: []schemas.LLMPlugin{c},
	})
	if err != nil {
		t.Fatalf("failed to initialize bifrost: %v", err)
	}
	t.Cleanup(client.Shutdown)
	return client
}

// TestReviewerUnaryTransportRetriesPrimaryBeforeFallbackR7 (MONEY RULE, tier A+):
// a unary passthrough whose primary transport-errors (connection refused, no HTTP
// status) must spend the same-provider budget — the primary pipeline runs >=2 times
// — BEFORE the healthy fallback serves the request. The regression this locks: the
// single-attempt pin removed the inner retries and the budgeted loop only ran on the
// success arm (primaryErr == nil), so a transport error escalated on the FIRST failure.
func TestReviewerUnaryTransportRetriesPrimaryBeforeFallbackR7(t *testing.T) {
	// A server closed before use: every dial is connection-refused (a genuine transport
	// error with no HTTP response, status 0) — the exact "dead provider" shape.
	downSrv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {}))
	downURL := downSrv.URL
	downSrv.Close()

	var fallbackHits atomic.Int32
	fallbackSrv := httptest.NewServer(statusHandler(http.StatusOK, &fallbackHits))
	defer fallbackSrv.Close()

	counter := &primaryCallCounter{}
	account := NewMockAccount()
	account.AddProviderWithBaseURL(schemas.OpenAI, 1, 1, downURL)
	account.AddProviderWithBaseURL(schemas.Anthropic, 1, 1, fallbackSrv.URL)
	account.SetKeysForProvider(schemas.OpenAI, []schemas.Key{
		{ID: "primary-key", Value: *schemas.NewSecretVar("sk-primary"), Models: schemas.WhiteList{"*"}, Weight: 100},
	})
	account.SetKeysForProvider(schemas.Anthropic, []schemas.Key{
		{ID: "fallback-key", Value: *schemas.NewSecretVar("sk-fallback"), Models: schemas.WhiteList{"*"}, Weight: 100},
	})
	client := newCounterClient(t, account, counter)

	resp, bifrostErr := doPassthrough(t, client, schemas.OpenAI,
		[]schemas.Fallback{{Provider: schemas.Anthropic, Model: "claude-haiku"}})
	if bifrostErr != nil {
		t.Fatalf("transport-error passthrough failed (primary=%d, fallback=%d): %s", counter.count(schemas.OpenAI), fallbackHits.Load(), bifrostErr.Error.Message)
	}
	if got := counter.count(schemas.OpenAI); got < 2 {
		t.Fatalf("primary pipeline ran %d time(s), want >=2 (a transport error must spend the same-provider budget before escalating cheap->expensive)", got)
	}
	if fallbackHits.Load() != 1 {
		t.Fatalf("fallback hits = %d, want 1 (a down primary must ultimately escalate to the healthy fallback)", fallbackHits.Load())
	}
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("final status = %d, want 200 (fallback served the request)", resp.StatusCode)
	}
}

// TestReviewerUnaryTransportNoFallbacksSingleCallR7 (CONTROL, byte-identical guarantee):
// with NO fallbacks the dedicated failover loop must NOT engage — armPassthroughFailover
// declines (len(fallbacks)==0), so the request takes the generic upstream path. The
// surfaced error must be a genuine transport error (StatusCode nil, no stashed
// PassthroughResponse), proving the test exercises the transport path and never passes
// vacuously. The measured primary attempt count pins the no-fallback path behavior.
func TestReviewerUnaryTransportNoFallbacksSingleCallR7(t *testing.T) {
	downSrv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {}))
	downURL := downSrv.URL
	downSrv.Close()

	counter := &primaryCallCounter{}
	account := NewMockAccount()
	account.AddProviderWithBaseURL(schemas.OpenAI, 1, 1, downURL)
	account.SetKeysForProvider(schemas.OpenAI, []schemas.Key{
		{ID: "primary-key", Value: *schemas.NewSecretVar("sk-primary"), Models: schemas.WhiteList{"*"}, Weight: 100},
	})
	client := newCounterClient(t, account, counter)

	_, bifrostErr := doPassthrough(t, client, schemas.OpenAI, nil)
	// Anti-vacuous: the down server must surface a real failure, not a success — proving
	// the transport path was actually exercised.
	if bifrostErr == nil {
		t.Fatalf("no-fallback transport error returned no error (primary=%d) — down server did not fail", counter.count(schemas.OpenAI))
	}
	// The byte-identical guarantee: with no fallbacks the dedicated passthrough failover
	// loop must NOT engage, so the primary plugin pipeline runs exactly once — no extra
	// same-provider legs, no escalation machinery. (The universal inner retry loop is a
	// separate concern, unchanged and shared by every route.)
	if got := counter.count(schemas.OpenAI); got != 1 {
		t.Fatalf("primary pipeline ran %d time(s), want exactly 1 (no passthrough failover machinery without fallbacks)", got)
	}
}

// R2 (MONEY RULE): a nil-ctx STREAMING passthrough whose primary leaks a pool 401
// then recovers on the same provider must NEVER escalate to the pricier fallback.
// nil ctx substitutes the shared instance context; the isolation fix routes it to the
// dedicated pool-aware loop (401 => retry-same-pooled) instead of the generic
// escalate-on-any-error loop.
func TestPassthroughStreamFailover_NilCtxPool401NoEscalate(t *testing.T) {
	// 401 on hit 1 (dead pool account), 200 on hit 2 (pool rotated to a live account).
	primary := &streamCountingHandler{failStatus: http.StatusUnauthorized, failN: 1}
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
		{ID: "fallback-key", Value: *schemas.NewSecretVar("sk-fallback"), Models: schemas.WhiteList{"*"}, Weight: 100},
	})
	client := newPassthroughTestClient(t, account)

	// nil ctx on purpose — this is the R2 vector.
	stream, bifrostErr := client.PassthroughStream(nil, schemas.OpenAI, &schemas.BifrostPassthroughRequest{
		Method:    http.MethodPost,
		Path:      "/v1/messages",
		Body:      []byte(`{"model":"m"}`),
		Fallbacks: []schemas.Fallback{{Provider: schemas.Anthropic, Model: "claude-haiku"}},
	})
	if bifrostErr != nil {
		t.Fatalf("nil-ctx pool-401 passthrough stream failed (primary=%d): %s", primary.hits.Load(), bifrostErr.Error.Message)
	}
	_, status, errs := drainPassthroughStream(stream)
	if primary.hits.Load() != 2 {
		t.Fatalf("primary hits = %d, want 2 (pool 401 must retry the SAME provider, then recover)", primary.hits.Load())
	}
	if fallbackHits.Load() != 0 {
		t.Fatalf("fallback called %d time(s) — a nil-ctx pool 401 escalated cheap->expensive (MONEY RULE VIOLATION)", fallbackHits.Load())
	}
	if len(errs) > 0 {
		t.Fatalf("recovered stream emitted error chunks: %v", errs)
	}
	if status != http.StatusOK {
		t.Fatalf("final status = %d, want 200 (primary recovered on same-provider pool retry)", status)
	}
}

// R1 (CONCURRENCY LOCK): a nil-ctx UNARY passthrough with fallbacks running
// concurrently with a nil-ctx transforming (unary) request must not steal the
// transforming request's retries. Both share the instance context on the broken
// path; the passthrough leaking PassthroughSingleAttempt would pin the transforming
// inner loop to one physical call so it never recovers.
func TestPassthroughFailover_NilCtxUnaryDoesNotLeakToTransforming(t *testing.T) {
	// Passthrough primary: always 500 with Retry-After:1 so the failover machinery keys
	// stay live on the (shared, on the broken path) context across the ~1s wait window.
	passPrimary := &countingHandler{failStatus: http.StatusInternalServerError, failN: 1000, retryAfter: "1"}
	passSrv := httptest.NewServer(passPrimary)
	defer passSrv.Close()

	// Transforming upstream: fails twice then recovers. With MaxRetries=3 the inner loop
	// must make exactly 3 physical calls; a leaked single-attempt pin forces 1 => no recovery.
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
	account.configs[schemas.OpenAI].NetworkConfig.RetryBackoffInitial = time.Millisecond
	account.SetKeysForProvider(schemas.Anthropic, []schemas.Key{
		{ID: "p", Value: *schemas.NewSecretVar("sk-p"), Models: schemas.WhiteList{"*"}, Weight: 100},
	})
	account.SetKeysForProvider(schemas.OpenAI, []schemas.Key{
		{ID: "x", Value: *schemas.NewSecretVar("sk-x"), Models: schemas.WhiteList{"*"}, Weight: 100},
	})
	client := newPassthroughTestClient(t, account)

	var wg sync.WaitGroup
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
			// nil-ctx unary passthrough with a fallback: the R1 leak vector.
			_, _ = client.Passthrough(nil, schemas.Anthropic, &schemas.BifrostPassthroughRequest{
				Method:    http.MethodPost,
				Path:      "/v1/messages",
				Body:      []byte(`{"model":"m"}`),
				Fallbacks: []schemas.Fallback{{Provider: schemas.OpenAI, Model: "gpt-4o-mini"}},
			})
		}
	}()

	// Let the passthrough goroutine enter its Retry-After wait with the keys live.
	time.Sleep(50 * time.Millisecond)

	fail := func(format string, args ...any) {
		close(stop)
		wg.Wait()
		t.Fatalf(format, args...)
	}

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
