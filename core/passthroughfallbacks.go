package bifrost

import (
	"context"
	"maps"
	"net/http"
	"strconv"
	"strings"
	"time"

	"github.com/google/uuid"
	schemas "github.com/maximhq/bifrost/core/schemas"
)

// passthroughFailoverAction is the classification verdict for a non-2xx passthrough
// status. It drives the status-aware failover branch: retry the same provider,
// escalate to the next (more expensive) fallback, fail fast, or treat as success.
type passthroughFailoverAction int

const (
	passthroughNotFailure      passthroughFailoverAction = iota // 2xx — no failover
	passthroughFailFast                                         // terminal request-bound error — return verbatim
	passthroughRetrySame                                        // retry same provider (429/408/5xx); never escalate for 429
	passthroughRetrySamePooled                                  // 401/403 pool-account symptom — retry same with a cooled key, never escalate
	passthroughEscalate                                         // escalate to the next fallback after same-provider retries are exhausted
)

// passthroughFailFastStatusCodes are request-bound errors: the request itself is
// wrong or forbidden for reasons no retry or provider swap can fix. The upstream
// body is returned verbatim. 401/403 are deliberately NOT here — reseller/gateway
// pools leak a single dead account as 401/403, and a same-provider retry lands on
// a different account, so they are classified as passthroughRetrySamePooled.
var passthroughFailFastStatusCodes = map[int]bool{
	400: true, // Bad Request
	404: true, // Not Found
	409: true, // Conflict
	422: true, // Unprocessable Entity
}

// passthroughPooledAccountStatusCodes are the symptoms a reseller/gateway pool
// leaks when one backing account is dead: retrying the same provider with a
// different key/account (the pool rotates) succeeds. They retry the SAME provider
// (never escalate to a pricier one) and cool the failed key so the retry avoids it.
var passthroughPooledAccountStatusCodes = map[int]bool{
	401: true, // Unauthorized — a pool account's key was rejected
	402: true, // Payment Required — billing issue on this key's account; another key may be funded
	403: true, // Forbidden — a pool account is blocked
}

// passthroughRetrySameStatusCodes are transient request-timing errors resolved by
// retrying the same provider — escalating to a different (pricier) provider gains
// nothing. 429 is handled separately in classifyPassthroughStatus so it can never
// escalate regardless of duration.
var passthroughRetrySameStatusCodes = map[int]bool{
	408: true, // Request Timeout
	425: true, // Too Early
}

// passthroughEscalateStatusCodes are real upstream server failures: retry the same
// provider first, then escalate to the next fallback once retries are exhausted.
// 529 (Anthropic overloaded) is included alongside the standard 5xx set.
var passthroughEscalateStatusCodes = map[int]bool{
	500: true, // Internal Server Error
	502: true, // Bad Gateway
	503: true, // Service Unavailable
	504: true, // Gateway Timeout
	529: true, // Overloaded (Anthropic)
}

// passthroughLocalFastFailMaxMs is the upper bound on request duration below which a
// 502/503 is treated as a local fasthttp fast-fail (the upstream was never reached:
// connection refused, pool exhausted) rather than a real upstream 5xx. Such a
// response retries the same provider immediately WITHOUT charging the escalation
// budget, but after the same-provider budget is exhausted it still escalates (a
// genuinely-down provider must reach the configured fallback). Only 502/503 are the
// fasthttp local-failure surface — 500/504 always reflect a reached upstream.
const passthroughLocalFastFailMaxMs int64 = 3

// Passthrough failover budgets and bounds, in one place. Claude Code has its own
// client timeout, so exhausting retries past it wastes the cheap provider's budget
// and still fails the client — every knob here is bounded.
const (
	// passthroughMaxRetrySame bounds retry-same attempts per provider for the
	// 429/408/5xx budget so a persistent transient error cannot hang.
	passthroughMaxRetrySame = 1

	// passthroughMaxRetryPooled bounds 401/403 pool-account retries per provider.
	// This is a SEPARATE budget from passthroughMaxRetrySame: a pool leaking 401s
	// on several accounts should get a couple of account rotations before giving up.
	passthroughMaxRetryPooled = 2

	// passthroughLocalFastFailMaxRetry bounds how many times a ≤3ms 502/503 retries
	// the same provider before escalating. Kept small — a provider that fast-fails
	// this many times in a row is genuinely down.
	passthroughLocalFastFailMaxRetry = 2

	// passthroughMaxTotalAttempts caps the total number of upstream attempts across
	// the primary and all fallbacks for one passthrough request, independent of the
	// per-class budgets, so no combination of statuses can loop unbounded.
	passthroughMaxTotalAttempts = 8

	// passthroughRetryAfterCap bounds how long a Retry-After header may pause a
	// same-provider retry. A provider returning "Retry-After: 3600" must not hang
	// the request for an hour past the client's own timeout.
	passthroughRetryAfterCap = 10 * time.Second
)

// boundedRetryAfter clamps a parsed Retry-After duration into [0, cap]. A negative
// or absent value yields 0 (retry immediately); a huge value is capped.
func boundedRetryAfter(d time.Duration) time.Duration {
	if d <= 0 {
		return 0
	}
	if d > passthroughRetryAfterCap {
		return passthroughRetryAfterCap
	}
	return d
}

// parsePassthroughRetryAfter reads the Retry-After header and returns the bounded wait,
// or 0 when absent or unparseable. Both RFC 9110 wire forms are honored: ignoring the
// HTTP-date form made a 503 retry fire immediately, exhausting the same-provider budget
// and escalating a transient error to a pricier provider.
func parsePassthroughRetryAfter(headers map[string]string) time.Duration {
	if len(headers) == 0 {
		return 0
	}
	for k, v := range headers {
		if !strings.EqualFold(k, "Retry-After") {
			continue
		}
		raw := strings.TrimSpace(v)
		if secs, err := strconv.ParseInt(raw, 10, 64); err == nil {
			// Clamp before the multiply: time.Duration is int64 nanoseconds, so a
			// large delay-seconds value overflows to negative and reads as 0.
			if secs > int64(passthroughRetryAfterCap/time.Second) {
				return passthroughRetryAfterCap
			}
			return boundedRetryAfter(time.Duration(secs) * time.Second)
		}
		if t, err := http.ParseTime(raw); err == nil {
			return boundedRetryAfter(time.Until(t))
		}
		return 0
	}
	return 0
}

// classifyPassthroughStatus maps a passthrough response status code and request
// duration to a failover action. This is money-critical: a wrong verdict can jump a
// cheap provider to an expensive one on a single error. 429 and 401/403 always
// retry the same provider and never escalate.
func classifyPassthroughStatus(statusCode int, durationMs int64) passthroughFailoverAction {
	if statusCode >= 200 && statusCode < 300 {
		return passthroughNotFailure
	}
	if statusCode == 0 {
		// A transport error (connection refused, no HTTP response) carries no status.
		// It is escalate-eligible — a genuinely-down provider must reach the fallback —
		// yet stays distinct from a pool 401/403 (retry-same-pooled), so the failover
		// loop retries the same provider a bounded number of times BEFORE escalating
		// and can never spin forever on a dead upstream.
		return passthroughEscalate
	}
	if statusCode == 429 {
		return passthroughRetrySame
	}
	if passthroughPooledAccountStatusCodes[statusCode] {
		return passthroughRetrySamePooled
	}
	if passthroughFailFastStatusCodes[statusCode] {
		return passthroughFailFast
	}
	if passthroughRetrySameStatusCodes[statusCode] {
		return passthroughRetrySame
	}
	if passthroughEscalateStatusCodes[statusCode] {
		if (statusCode == 502 || statusCode == 503) && durationMs <= passthroughLocalFastFailMaxMs {
			return passthroughRetrySame
		}
		return passthroughEscalate
	}
	return passthroughFailFast
}

// passthroughStreamFirstChunkError is the classifier hook handed to
// CheckFirstStreamChunkForError on the streaming-passthrough-with-fallbacks path.
// A passthrough stream reports a non-2xx upstream status as a data chunk
// (BifrostPassthroughResponse.StatusCode), not a BifrostError, so the generic peek
// can't see it. For every retry-eligible class (pool 401/403, retry-same 429/408/425,
// escalate 5xx) this synthesizes the *BifrostError that runPassthroughStreamFailover
// keys off, carrying the upstream StatusCode and stashing the full passthrough
// response (headers included) on ExtraFields.PassthroughResponse so the loop can
// re-classify and honor Retry-After without re-reading the drained stream. A 2xx or a
// fail-fast 4xx returns nil so the stream forwards to the client verbatim. The money
// rule lives in the failover loop, not here: this only reports "the leg failed, here
// is its status+headers" — whether that advances the ladder is the loop's decision.
func passthroughStreamFirstChunkError(chunk *schemas.BifrostStreamChunk) *schemas.BifrostError {
	pr := chunk.BifrostPassthroughResponse
	if pr == nil {
		return nil
	}
	switch classifyPassthroughStatus(pr.StatusCode, pr.ExtraFields.Latency) {
	case passthroughNotFailure, passthroughFailFast:
		return nil
	}
	status := pr.StatusCode
	return &schemas.BifrostError{
		IsBifrostError: false,
		StatusCode:     &status,
		Error: &schemas.ErrorField{
			Message: http.StatusText(status),
		},
		ExtraFields: schemas.BifrostErrorExtraFields{
			PassthroughResponse: pr,
		},
	}
}

// passthroughStatusResult carries the outcome of one passthrough attempt for the
// failover loop: the response to inspect and any transport error.
type passthroughStatusResult struct {
	resp *schemas.BifrostResponse
	err  *schemas.BifrostError
}

// passthroughStreamResult carries the outcome of one streaming passthrough attempt.
// A live stream is a success (stream != nil, err == nil). A failed first chunk is
// reported as err, whose ExtraFields.PassthroughResponse carries the upstream status
// and headers the loop classifies on (see passthroughStreamFirstChunkError).
type passthroughStreamResult struct {
	stream chan *schemas.BifrostStreamChunk
	err    *schemas.BifrostError
}

// streamFailoverStatus extracts the upstream status and headers a streaming attempt
// failed with, from the passthrough response the classifier stashed on the error.
// A transport error (no stashed response) reports (0, nil): the loop treats it like
// the generic streaming path — escalate-eligible, never a pool/retry-same class.
func streamFailoverStatus(err *schemas.BifrostError) (int, map[string]string, int64) {
	if err == nil || err.ExtraFields.PassthroughResponse == nil {
		if err != nil && err.StatusCode != nil {
			return *err.StatusCode, nil, 0
		}
		return 0, nil, 0
	}
	pr := err.ExtraFields.PassthroughResponse
	return pr.StatusCode, pr.Headers, pr.ExtraFields.Latency
}

// runPassthroughStreamFailover drives cost-aware, status-aware failover for a
// streaming passthrough request, mirroring runPassthroughStatusFailover exactly but
// over stream attempts. The primary already ran (base 0) and either produced a live
// stream (returned as-is) or a synthesized first-chunk error. Every invariant of the
// unary loop holds identically here — same budgets, same passthroughAttemptCounter,
// same Retry-After clamp — because both share the same constants and counter type:
//
//   - 429 and 401/403 retry the SAME provider and never escalate.
//   - an escalate-class 5xx exhausts the same-provider budget BEFORE escalating.
//   - a ≤3ms local fast-fail 502/503 retries same off-budget, then escalates.
//   - passthroughMaxTotalAttempts caps total physical stream attempts.
//
// attemptFn re-issues the primary against req; fallbackFn issues one fallback stream.
func (bifrost *Bifrost) runPassthroughStreamFailover(
	ctx *schemas.BifrostContext,
	req *schemas.BifrostRequest,
	primary passthroughStreamResult,
	attemptFn func() passthroughStreamResult,
	fallbackFn func(*schemas.BifrostRequest) passthroughStreamResult,
) passthroughStreamResult {
	_, _, fallbacks := req.GetRequestFields()
	counter := &passthroughAttemptCounter{ctx: ctx, base: 0} // primary already consumed base 0
	current := primary
	retrySameCount := 0
	retryPooledCount := 0
	localFastFailCount := 0

	retry := func(headers map[string]string) (passthroughStreamResult, bool) {
		if wait := parsePassthroughRetryAfter(headers); wait > 0 {
			select {
			case <-time.After(wait):
			case <-ctx.Done():
				return current, false
			}
		}
		if !counter.next() {
			return current, false
		}
		return attemptFn(), true
	}

	for {
		if current.err == nil {
			return current
		}
		// F4 (CWE-863): a plugin veto (AllowFallbacks=false) and a cancellation both carry
		// no HTTP status, so streamFailoverStatus reports status 0 — which classifies as
		// passthroughEscalate. Gate on the SAME predicate the generic loop uses (bifrost.go:4803)
		// BEFORE classifying, so a veto is never retried/escalated cheap→expensive; nil
		// AllowFallbacks stays allowed, so a genuine transport error (status 0) still escalates.
		if !bifrost.shouldTryFallbacks(req, current.err) {
			return current
		}
		status, headers, latency := streamFailoverStatus(current.err)
		action := classifyPassthroughStatus(status, latency)

		switch action {
		case passthroughNotFailure, passthroughFailFast:
			return current

		case passthroughRetrySamePooled:
			if retryPooledCount >= passthroughMaxRetryPooled {
				return current
			}
			retryPooledCount++
			recordPassthroughPooledKey(ctx)
			next, ok := retry(headers)
			if !ok {
				return current
			}
			current = next

		case passthroughRetrySame:
			isLocalFastFail := (status == 502 || status == 503) && latency <= passthroughLocalFastFailMaxMs
			if isLocalFastFail {
				if localFastFailCount >= passthroughLocalFastFailMaxRetry {
					if len(fallbacks) == 0 {
						return current
					}
					next, ok := bifrost.tryPassthroughStreamFallbacks(ctx, req, fallbacks, counter, fallbackFn)
					if !ok {
						return current
					}
					return next
				}
				localFastFailCount++
				next, ok := retry(headers)
				if !ok {
					return current
				}
				current = next
				continue
			}
			if retrySameCount >= passthroughMaxRetrySame {
				return current
			}
			retrySameCount++
			next, ok := retry(headers)
			if !ok {
				return current
			}
			current = next

		case passthroughEscalate:
			if retrySameCount < passthroughMaxRetrySame {
				retrySameCount++
				next, ok := retry(headers)
				if !ok {
					return current
				}
				current = next
				continue
			}
			if len(fallbacks) == 0 {
				return current
			}
			next, ok := bifrost.tryPassthroughStreamFallbacks(ctx, req, fallbacks, counter, fallbackFn)
			if !ok {
				return current
			}
			return next
		}
	}
}

// tryPassthroughStreamFallbacks walks the fallback ladder (cheap→expensive) for a
// streaming passthrough, mirroring tryPassthroughFallbacks: each fallback that fails
// with a retry-eligible status is retried on that same fallback (runPassthroughStreamFallbackAttempt)
// before the ladder advances, so a transient error on a cheap fallback never promotes
// traffic to a pricier one. Returns the first non-escalate outcome (a live stream, a
// fail-fast, or a retry-exhausted result), or (_, false) when every fallback was
// skipped/exhausted.
func (bifrost *Bifrost) tryPassthroughStreamFallbacks(
	ctx *schemas.BifrostContext,
	req *schemas.BifrostRequest,
	fallbacks []schemas.Fallback,
	counter *passthroughAttemptCounter,
	fallbackFn func(*schemas.BifrostRequest) passthroughStreamResult,
) (passthroughStreamResult, bool) {
	for i, fallback := range fallbacks {
		ctx.SetValue(schemas.BifrostContextKeyFallbackIndex, i+1)
		ctx.SetValue(schemas.BifrostContextKeyFallbackRequestID, uuid.New().String())
		clearCtxForFallback(ctx)

		fallbackReq := bifrost.prepareFallbackRequest(req, fallback)
		if fallbackReq == nil {
			continue
		}
		if !counter.next() {
			return passthroughStreamResult{}, false
		}
		result := bifrost.runPassthroughStreamFallbackAttempt(ctx, fallback, fallbackReq, counter, fallbackFn)
		if result.err == nil {
			return result, true
		}
		if !bifrost.shouldContinueWithFallbacks(fallback, result.err) {
			return result, true
		}
		status, _, latency := streamFailoverStatus(result.err)
		if classifyPassthroughStatus(status, latency) != passthroughEscalate {
			return result, true
		}
	}
	return passthroughStreamResult{}, false
}

// runPassthroughStreamFallbackAttempt issues one fallback stream and applies the
// same-provider retry policy (429/401/403/408 retry the SAME fallback) without
// escalating — escalation across fallbacks is the caller's ladder. Mirrors
// runPassthroughFallbackAttempt's budgets exactly.
func (bifrost *Bifrost) runPassthroughStreamFallbackAttempt(
	ctx *schemas.BifrostContext,
	fallback schemas.Fallback,
	fallbackReq *schemas.BifrostRequest,
	counter *passthroughAttemptCounter,
	fallbackFn func(*schemas.BifrostRequest) passthroughStreamResult,
) passthroughStreamResult {
	current := fallbackFn(fallbackReq)
	retrySameCount := 0
	retryPooledCount := 0

	for {
		if current.err == nil {
			return current
		}
		// C1 (CWE-863): a plugin veto (AllowFallbacks=false) on this fallback leg carries no
		// HTTP status, so streamFailoverStatus reports 0 → classifies as passthroughEscalate,
		// which would reissue the veto and advance the ladder cheap→expensive. Gate on the
		// same fallback-leg predicate the caller uses BEFORE classifying; nil AllowFallbacks
		// stays allowed, so a genuine transport error (status 0) still escalates.
		if !bifrost.shouldContinueWithFallbacks(fallback, current.err) {
			return current
		}
		status, headers, latency := streamFailoverStatus(current.err)
		action := classifyPassthroughStatus(status, latency)

		switch action {
		case passthroughRetrySamePooled:
			if retryPooledCount >= passthroughMaxRetryPooled {
				return current
			}
			retryPooledCount++
			recordPassthroughPooledKey(ctx)
		case passthroughRetrySame:
			if retrySameCount >= passthroughMaxRetrySame {
				return current
			}
			retrySameCount++
		case passthroughEscalate:
			if retrySameCount >= passthroughMaxRetrySame {
				return current
			}
			retrySameCount++
		default:
			return current
		}

		if wait := parsePassthroughRetryAfter(headers); wait > 0 {
			select {
			case <-time.After(wait):
			case <-ctx.Done():
				return current
			}
		}
		if !counter.next() {
			return current
		}
		current = fallbackFn(fallbackReq)
	}
}

// passthroughAttemptCounter enforces the global attempt bound and assigns each
// physical attempt a distinct base index via BifrostContextKeyAttemptBase. In
// passthrough failover the inner retry loop is pinned to one physical call
// (BifrostContextKeyPassthroughSingleAttempt), so it stamps base+0 == base onto
// BifrostContextKeyNumberOfRetries — the identity governance dedups billing on.
// A distinct base per physical attempt keeps the failed primary from colliding
// with the successful escalation on the same RequestID:AttemptNumber slot.
//
// base starts at 0: the primary already consumed base 0 before this counter runs.
type passthroughAttemptCounter struct {
	ctx  *schemas.BifrostContext
	base int
}

// next assigns the next physical attempt its base index and reports whether the
// global bound still allows another attempt. The primary is physical attempt 1
// (base 0), so up to passthroughMaxTotalAttempts-1 further attempts are permitted.
func (c *passthroughAttemptCounter) next() bool {
	if c.base >= passthroughMaxTotalAttempts-1 {
		return false
	}
	c.base++
	c.ctx.SetValue(schemas.BifrostContextKeyAttemptBase, c.base)
	return true
}

// armPassthroughFailover is the single gate both the unary (handleRequest) and
// streaming (handleStreamRequest) passthrough paths call, so the two can never
// drift apart. It reports whether the dedicated passthrough failover machinery
// should run and, when it should, pins the shared inner retry loop to one physical
// call per leg (PassthroughSingleAttempt) so passthroughMaxTotalAttempts is an honest
// upstream bound, giving each leg a distinct billing identity via AttemptBase.
//
// B5: it arms NOTHING on the shared instance context (bifrost.ctx) — a nil-ctx public
// call substitutes bifrost.ctx, and writing these keys there would make a concurrent
// transforming request's inner loop run a single physical call instead of its
// configured retries. Callers defer the returned cleanup unconditionally.
// isolatedPassthroughContext wraps the shared instance context in a request-scoped
// child for a nil-ctx passthrough call. The failover machinery stamps per-request keys
// (PassthroughSingleAttempt, AttemptBase, FallbackIndex); writing those on bifrost.ctx
// directly leaks into a concurrent request sharing it. The child isolates every write
// (reads still fall through to the instance context) AND, being != bifrost.ctx, lets the
// dedicated money-safe failover loop run so a pool 401 never escalates a nil-ctx passthrough
// cheap→expensive.
func isolatedPassthroughContext(parent *schemas.BifrostContext) *schemas.BifrostContext {
	// WithoutCancel strips the parent's Done()/deadline so NewBifrostContext spawns NO
	// watchCancellation goroutine. bifrost.ctx is cancellable but only closes at shutdown,
	// so a watcher here would block forever with nothing to Cancel() it — a leaked goroutine
	// per nil-ctx passthrough. Value() still delegates to the parent (reads fall through),
	// and the child is still a fresh *BifrostContext != bifrost.ctx with its own value map.
	// The RequestID is NOT redundant with handleRequest's "generate if absent": fall-through
	// would let the child READ an ID seeded by an earlier nil-ctx call, so that branch never
	// fires and two calls share one billing identity (dedup is RequestID:AttemptNumber).
	child := schemas.NewBifrostContextWithValue(
		context.WithoutCancel(parent), schemas.NoDeadline,
		schemas.BifrostContextKeyRequestID, uuid.New().String(),
	)
	// Capture parent.Done() BEFORE WithoutCancel above strips it, so a nil-ctx passthrough
	// stream still observes Shutdown() (which cancels bifrost.ctx == parent). SetupStreamCancellation's
	// already-bounded goroutine selects on this too — no new goroutine, no parked watcher.
	if done := parent.Done(); done != nil {
		child.SetValue(schemas.BifrostContextKeyShutdownDone, done)
	}
	return child
}

func (bifrost *Bifrost) armPassthroughFailover(ctx *schemas.BifrostContext, req *schemas.BifrostRequest, passthroughType schemas.RequestType) (bool, func()) {
	_, _, fallbacks := req.GetRequestFields()
	if req.RequestType != passthroughType || len(fallbacks) == 0 || ctx == bifrost.ctx {
		return false, func() {}
	}
	ctx.SetValue(schemas.BifrostContextKeyPassthroughSingleAttempt, true)
	ctx.ClearValue(schemas.BifrostContextKeyAttemptBase)
	// Scoped to ONE failover: a reused context must not inherit it, or a later request
	// would skip a credential that has since been rotated or repaired.
	ctx.ClearValue(schemas.BifrostContextKeyPassthroughRotatedKeys)
	return true, func() {
		ctx.ClearValue(schemas.BifrostContextKeyPassthroughSingleAttempt)
		ctx.ClearValue(schemas.BifrostContextKeyAttemptBase)
		ctx.ClearValue(schemas.BifrostContextKeyPassthroughRotatedKeys)
	}
}

// unaryFailoverStatus extracts the upstream status, headers, and latency a unary
// passthrough attempt failed with, from either arm — the twin of streamFailoverStatus.
// A non-2xx on the success arm (err == nil) stashes them in PassthroughResponse; a
// transport error (error arm) carries no HTTP response and reports (0, nil, 0), which
// classifies as passthroughEscalate — so a dead provider retries the same leg within
// budget BEFORE the ladder advances, exactly like the streaming path. A BifrostError
// that itself carries a StatusCode (synthesized status) is honored.
func unaryFailoverStatus(r passthroughStatusResult) (int, map[string]string, int64) {
	if r.err != nil {
		if r.err.ExtraFields.PassthroughResponse != nil {
			pr := r.err.ExtraFields.PassthroughResponse
			return pr.StatusCode, pr.Headers, pr.ExtraFields.Latency
		}
		if r.err.StatusCode != nil {
			return *r.err.StatusCode, nil, 0
		}
		return 0, nil, 0
	}
	if r.resp == nil || r.resp.PassthroughResponse == nil {
		return 0, nil, 0
	}
	pr := r.resp.PassthroughResponse
	return pr.StatusCode, pr.Headers, pr.ExtraFields.Latency
}

// runPassthroughStatusFailover drives cost-aware, status-aware failover for a
// passthrough request. The primary already ran (base 0). It handles BOTH arms: a
// non-2xx on the success arm (primaryErr == nil, status in PassthroughResponse) AND
// a transport error on the error arm (primaryErr != nil, status 0) — the latter is
// the connection-refused case that must spend the same-provider budget before
// escalating to a pricier fallback. It reuses prepareFallbackRequest for escalation.
// attemptFn issues one attempt against the given request.
//
// Money-critical invariants:
//   - 429 and 401/403 retry the SAME provider and never escalate to a pricier one.
//   - An escalate-class 5xx exhausts the same-provider retry budget BEFORE escalating.
//   - 401/403 (pool-account symptom) get their own retry budget and cool the failed
//     key so the retry lands on a different account.
//   - A ≤3ms local fast-fail 502/503 retries same off-budget, then escalates.
//
// Governance billing dedups on RequestID+AttemptNumber; this loop stamps a distinct
// AttemptNumber per physical attempt via passthroughAttemptCounter, which is the
// identity governance actually reads (it does NOT read FallbackRequestID/Index).
//
// This is the UNARY passthrough path. Streaming passthrough
// (PassthroughStreamRequest) reaches the same retry+fallback machinery from a
// different entry: passthroughStreamFirstChunkError synthesizes a *BifrostError
// from an escalate-class first chunk in CheckFirstStreamChunkForError, and
// handleStreamRequest's error-driven fallback loop takes it from there.
func (bifrost *Bifrost) runPassthroughStatusFailover(
	ctx *schemas.BifrostContext,
	req *schemas.BifrostRequest,
	primary passthroughStatusResult,
	attemptFn func(*schemas.BifrostRequest) passthroughStatusResult,
) passthroughStatusResult {
	_, _, fallbacks := req.GetRequestFields()
	counter := &passthroughAttemptCounter{ctx: ctx, base: 0} // primary already consumed base 0
	current := primary
	retrySameCount := 0
	retryPooledCount := 0
	localFastFailCount := 0

	retry := func(headers map[string]string) (passthroughStatusResult, bool) {
		if wait := parsePassthroughRetryAfter(headers); wait > 0 {
			select {
			case <-time.After(wait):
			case <-ctx.Done():
				return current, false
			}
		}
		if !counter.next() {
			return current, false
		}
		return attemptFn(req), true
	}

	for {
		// Success arm with no passthrough payload is terminal (nothing to classify).
		if current.err == nil && (current.resp == nil || current.resp.PassthroughResponse == nil) {
			return current
		}
		// Error arm only: a plugin veto (AllowFallbacks=false) or a cancellation carries
		// no status, so unaryFailoverStatus reports 0 → escalate; gate on the SAME
		// predicate the generic loop uses BEFORE classifying so a veto is never retried,
		// while a genuine transport error (nil AllowFallbacks, status 0) still escalates.
		if current.err != nil && !bifrost.shouldTryFallbacks(req, current.err) {
			return current
		}
		status, headers, latency := unaryFailoverStatus(current)
		action := classifyPassthroughStatus(status, latency)

		switch action {
		case passthroughNotFailure, passthroughFailFast:
			return current

		case passthroughRetrySamePooled:
			if retryPooledCount >= passthroughMaxRetryPooled {
				return current
			}
			retryPooledCount++
			recordPassthroughPooledKey(ctx)
			next, ok := retry(headers)
			if !ok {
				return current
			}
			current = next

		case passthroughRetrySame:
			// A ≤3ms local fast-fail 502/503 has its own off-budget retry count;
			// once exhausted it escalates rather than returning the 502 verbatim.
			isLocalFastFail := (status == 502 || status == 503) && latency <= passthroughLocalFastFailMaxMs
			if isLocalFastFail {
				if localFastFailCount >= passthroughLocalFastFailMaxRetry {
					if len(fallbacks) == 0 {
						return current
					}
					next, ok := bifrost.tryPassthroughFallbacks(ctx, req, fallbacks, counter, attemptFn)
					if !ok {
						return current
					}
					return next
				}
				localFastFailCount++
				next, ok := retry(headers)
				if !ok {
					return current
				}
				current = next
				continue
			}
			if retrySameCount >= passthroughMaxRetrySame {
				return current
			}
			retrySameCount++
			// 429 cools the key (a rotated pool account may have free quota);
			// 408/425 are request-timing, not per-account, so no cooling.
			next, ok := retry(headers)
			if !ok {
				return current
			}
			current = next

		case passthroughEscalate:
			// Exhaust the same-provider retry budget BEFORE escalating: one
			// transient 5xx (or a transport error, status 0) must not promote
			// traffic from a cheap provider to a pricier fallback on first failure.
			if retrySameCount < passthroughMaxRetrySame {
				retrySameCount++
				next, ok := retry(headers)
				if !ok {
					return current
				}
				current = next
				continue
			}
			if len(fallbacks) == 0 {
				return current
			}
			next, ok := bifrost.tryPassthroughFallbacks(ctx, req, fallbacks, counter, attemptFn)
			if !ok {
				return current
			}
			return next
		}
	}
}

// tryPassthroughFallbacks walks the caller-supplied fallback ladder (cheap→expensive),
// applying the SAME classification/retry policy to each fallback as to the primary:
// a fallback that returns 429/401/403/408 is retried on that same fallback before the
// ladder advances. It returns the first attempt that resolves to a non-escalate
// outcome (a success, a fail-fast, or a retry-exhausted result), or (_, false) when
// every fallback was skipped/exhausted.
func (bifrost *Bifrost) tryPassthroughFallbacks(
	ctx *schemas.BifrostContext,
	req *schemas.BifrostRequest,
	fallbacks []schemas.Fallback,
	counter *passthroughAttemptCounter,
	attemptFn func(*schemas.BifrostRequest) passthroughStatusResult,
) (passthroughStatusResult, bool) {
	for i, fallback := range fallbacks {
		ctx.SetValue(schemas.BifrostContextKeyFallbackIndex, i+1)
		ctx.SetValue(schemas.BifrostContextKeyFallbackRequestID, uuid.New().String())
		clearCtxForFallback(ctx)

		fallbackReq := bifrost.prepareFallbackRequest(req, fallback)
		if fallbackReq == nil {
			continue
		}
		if !counter.next() {
			return passthroughStatusResult{}, false
		}
		result := bifrost.runPassthroughFallbackAttempt(ctx, fallback, fallbackReq, counter, attemptFn)
		if result.err != nil {
			if !bifrost.shouldContinueWithFallbacks(fallback, result.err) {
				return result, true
			}
			continue
		}
		if result.resp == nil || result.resp.PassthroughResponse == nil {
			return result, true
		}
		if classifyPassthroughStatus(result.resp.PassthroughResponse.StatusCode, result.resp.PassthroughResponse.ExtraFields.Latency) != passthroughEscalate {
			return result, true
		}
	}
	return passthroughStatusResult{}, false
}

// runPassthroughFallbackAttempt issues one fallback attempt and applies the
// same-provider retry policy (429/401/403/408 retry the SAME fallback, cooling the
// failed key for the pool cases) without escalating further — escalation across
// fallbacks is the caller's ladder. An escalate-class status — including a status-0
// transport error (dead upstream) — also gets the primary's retry-same budget here;
// only once that is exhausted is the result returned so the caller advances to the
// next fallback. The exact twin of runPassthroughStreamFallbackAttempt: it handles
// BOTH arms via unaryFailoverStatus so a connection-refused cheap fallback spends its
// same-provider budget instead of instantly promoting traffic to a pricier one.
func (bifrost *Bifrost) runPassthroughFallbackAttempt(
	ctx *schemas.BifrostContext,
	fallback schemas.Fallback,
	fallbackReq *schemas.BifrostRequest,
	counter *passthroughAttemptCounter,
	attemptFn func(*schemas.BifrostRequest) passthroughStatusResult,
) passthroughStatusResult {
	current := attemptFn(fallbackReq)
	retrySameCount := 0
	retryPooledCount := 0

	for {
		// Success arm with no passthrough payload is terminal (nothing to classify).
		if current.err == nil && (current.resp == nil || current.resp.PassthroughResponse == nil) {
			return current
		}
		// C1 (CWE-863): a plugin veto (AllowFallbacks=false) on this fallback leg carries no
		// HTTP status, so unaryFailoverStatus reports 0 → classifies as passthroughEscalate,
		// which would reissue the veto and advance the ladder cheap→expensive. Gate on the
		// same fallback-leg predicate the caller uses BEFORE classifying; nil AllowFallbacks
		// stays allowed, so a genuine transport error (status 0) still escalates.
		if current.err != nil && !bifrost.shouldContinueWithFallbacks(fallback, current.err) {
			return current
		}
		status, headers, latency := unaryFailoverStatus(current)
		action := classifyPassthroughStatus(status, latency)

		switch action {
		case passthroughRetrySamePooled:
			if retryPooledCount >= passthroughMaxRetryPooled {
				return current
			}
			retryPooledCount++
			recordPassthroughPooledKey(ctx)
		case passthroughRetrySame:
			if retrySameCount >= passthroughMaxRetrySame {
				return current
			}
			retrySameCount++
		case passthroughEscalate:
			// Same budget the primary gets: a transient 5xx or a status-0 transport
			// error on a cheap fallback must not advance the ladder to a pricier one on
			// its first failure.
			if retrySameCount >= passthroughMaxRetrySame {
				return current
			}
			retrySameCount++
		default:
			return current
		}

		if wait := parsePassthroughRetryAfter(headers); wait > 0 {
			select {
			case <-time.After(wait):
			case <-ctx.Done():
				return current
			}
		}
		if !counter.next() {
			return current
		}
		current = attemptFn(fallbackReq)
	}
}

// loadPassthroughRotatedKeys returns the keys already rejected with a permanent per-key
// error on an EARLIER leg of this passthrough failover.
//
// The result seeds usedKeyIDs, never deadKeyIDs. That distinction is the whole fix: a
// deadKeyIDs entry is permanent and SHRINKS the eligible pool, so once every key is marked
// the provider reports errAllKeysDead and a pool 401 escalates cheap->expensive — a money-rule
// violation. usedKeyIDs is a soft skip the key provider RESETS when the pool is exhausted, so
// a single-key pool still retries its own key and never escalates.
//
// Returns a copy: the retry loop mutates its map (the provider clears it on reset), and only
// storePassthroughRotatedKey publishes back, so an abandoned leg cannot corrupt the shared set.
func loadPassthroughRotatedKeys(ctx *schemas.BifrostContext) map[string]bool {
	if ctx == nil {
		return nil
	}
	stored, _ := ctx.Value(schemas.BifrostContextKeyPassthroughRotatedKeys).(map[string]bool)
	if len(stored) == 0 {
		return nil
	}
	out := make(map[string]bool, len(stored))
	maps.Copy(out, stored)
	return out
}

// storePassthroughRotatedKey publishes a rejected key ID so the NEXT leg starts already
// preferring a different credential. Gated on the passthrough pin, so a transforming route
// never allocates or writes this key and its retry accounting is unchanged. The map is
// replaced rather than mutated in place so a concurrent reader cannot observe a torn write.
func storePassthroughRotatedKey(ctx *schemas.BifrostContext, keyID string) {
	if ctx == nil || keyID == "" {
		return
	}
	if single, _ := ctx.Value(schemas.BifrostContextKeyPassthroughSingleAttempt).(bool); !single {
		return
	}
	existing, _ := ctx.Value(schemas.BifrostContextKeyPassthroughRotatedKeys).(map[string]bool)
	next := make(map[string]bool, len(existing)+1)
	maps.Copy(next, existing)
	next[keyID] = true
	ctx.SetValue(schemas.BifrostContextKeyPassthroughRotatedKeys, next)
}

// recordPassthroughPooledKey remembers the credential that just drew a pool-account
// rejection (401/402/403) so the NEXT attempt prefers a different one.
//
// It must live here, not in executeRequestWithRetries' error arm: a passthrough 401 is
// carried on the SUCCESS arm inside PassthroughResponse, never as a *BifrostError, so that
// arm is never reached on this path (verified with a probe — it printed nothing). The key
// that served the attempt is still on the context: SelectedKeyID is cleared only when the
// call returns a BifrostError, which a status-carrying passthrough response does not.
func recordPassthroughPooledKey(ctx *schemas.BifrostContext) {
	if ctx == nil {
		return
	}
	keyID, _ := ctx.Value(schemas.BifrostContextKeySelectedKeyID).(string)
	storePassthroughRotatedKey(ctx, keyID)
}
