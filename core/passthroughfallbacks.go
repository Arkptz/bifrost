package bifrost

import (
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

// parsePassthroughRetryAfter reads the Retry-After header (delay-seconds form only;
// HTTP-date form is uncommon for rate limits and left unhonored) from a passthrough
// response and returns the bounded wait. Returns 0 when absent or unparseable.
func parsePassthroughRetryAfter(headers map[string]string) time.Duration {
	if len(headers) == 0 {
		return 0
	}
	for k, v := range headers {
		if !strings.EqualFold(k, "Retry-After") {
			continue
		}
		secs, err := strconv.Atoi(strings.TrimSpace(v))
		if err != nil {
			return 0
		}
		return boundedRetryAfter(time.Duration(secs) * time.Second)
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

// passthroughStatusResult carries the outcome of one passthrough attempt for the
// failover loop: the response to inspect and any transport error.
type passthroughStatusResult struct {
	resp *schemas.BifrostResponse
	err  *schemas.BifrostError
}

// passthroughAttemptCounter enforces the global attempt bound and stamps each
// physical attempt's index onto BifrostContextKeyNumberOfRetries. Governance dedups
// billing on RequestID+AttemptNumber (it reads AttemptNumber from that context key),
// so a distinct index per attempt keeps the failed primary from colliding with the
// successful escalation on the same billing slot.
type passthroughAttemptCounter struct {
	ctx *schemas.BifrostContext
	n   int
}

// next stamps the next attempt index and reports whether the global bound still
// allows another attempt.
func (c *passthroughAttemptCounter) next() bool {
	if c.n >= passthroughMaxTotalAttempts {
		return false
	}
	c.n++
	c.ctx.SetValue(schemas.BifrostContextKeyNumberOfRetries, c.n)
	return true
}

// coolLastSelectedKey appends the key the last attempt used to the per-provider
// cooled set so the next same-provider selection avoids that dead account. The key
// core selected is surfaced on BifrostContextKeySelectedKeyID after every attempt.
func coolLastSelectedKey(ctx *schemas.BifrostContext) {
	keyID, ok := ctx.Value(schemas.BifrostContextKeySelectedKeyID).(string)
	if !ok || keyID == "" {
		return
	}
	cooled, _ := ctx.Value(schemas.BifrostContextKeyPassthroughCooledKeyIDs).([]string)
	for _, id := range cooled {
		if id == keyID {
			return
		}
	}
	ctx.SetValue(schemas.BifrostContextKeyPassthroughCooledKeyIDs, append(append([]string{}, cooled...), keyID))
}

// runPassthroughStatusFailover drives cost-aware, status-aware failover for a
// passthrough request whose primary attempt returned a non-2xx status on the
// success arm (primaryErr == nil). It reuses prepareFallbackRequest for escalation.
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
// This is the UNARY passthrough path only. Streaming passthrough
// (PassthroughStreamRequest) is intentionally not covered this round — see the
// package-level note in handleStreamRequest's caller and the FIX-7 report.
func (bifrost *Bifrost) runPassthroughStatusFailover(
	ctx *schemas.BifrostContext,
	req *schemas.BifrostRequest,
	primary passthroughStatusResult,
	attemptFn func(*schemas.BifrostRequest) passthroughStatusResult,
) passthroughStatusResult {
	_, _, fallbacks := req.GetRequestFields()
	counter := &passthroughAttemptCounter{ctx: ctx, n: 1} // primary already consumed attempt 1
	current := primary
	retrySameCount := 0
	retryPooledCount := 0
	localFastFailCount := 0

	retry := func(status int, headers map[string]string, cool bool) (passthroughStatusResult, bool) {
		if cool {
			coolLastSelectedKey(ctx)
		}
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
		if current.err != nil || current.resp == nil || current.resp.PassthroughResponse == nil {
			return current
		}
		pr := current.resp.PassthroughResponse
		action := classifyPassthroughStatus(pr.StatusCode, pr.ExtraFields.Latency)

		switch action {
		case passthroughNotFailure, passthroughFailFast:
			return current

		case passthroughRetrySamePooled:
			if retryPooledCount >= passthroughMaxRetryPooled {
				return current
			}
			retryPooledCount++
			next, ok := retry(pr.StatusCode, pr.Headers, true)
			if !ok {
				return current
			}
			current = next

		case passthroughRetrySame:
			// A ≤3ms local fast-fail 502/503 has its own off-budget retry count;
			// once exhausted it escalates rather than returning the 502 verbatim.
			isLocalFastFail := (pr.StatusCode == 502 || pr.StatusCode == 503) && pr.ExtraFields.Latency <= passthroughLocalFastFailMaxMs
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
				next, ok := retry(pr.StatusCode, pr.Headers, false)
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
			next, ok := retry(pr.StatusCode, pr.Headers, pr.StatusCode == 429)
			if !ok {
				return current
			}
			current = next

		case passthroughEscalate:
			// Exhaust the same-provider retry budget BEFORE escalating: one
			// transient 5xx must not promote traffic from a cheap provider to a
			// pricier fallback.
			if retrySameCount < passthroughMaxRetrySame {
				retrySameCount++
				next, ok := retry(pr.StatusCode, pr.Headers, false)
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
		result := bifrost.runPassthroughFallbackAttempt(ctx, fallbackReq, counter, attemptFn)
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
// fallbacks is the caller's ladder. Escalate-class results are returned as-is so the
// caller advances to the next fallback.
func (bifrost *Bifrost) runPassthroughFallbackAttempt(
	ctx *schemas.BifrostContext,
	fallbackReq *schemas.BifrostRequest,
	counter *passthroughAttemptCounter,
	attemptFn func(*schemas.BifrostRequest) passthroughStatusResult,
) passthroughStatusResult {
	current := attemptFn(fallbackReq)
	retrySameCount := 0
	retryPooledCount := 0

	for {
		if current.err != nil || current.resp == nil || current.resp.PassthroughResponse == nil {
			return current
		}
		pr := current.resp.PassthroughResponse
		action := classifyPassthroughStatus(pr.StatusCode, pr.ExtraFields.Latency)

		var cool bool
		switch action {
		case passthroughRetrySamePooled:
			if retryPooledCount >= passthroughMaxRetryPooled {
				return current
			}
			retryPooledCount++
			cool = true
		case passthroughRetrySame:
			if retrySameCount >= passthroughMaxRetrySame {
				return current
			}
			retrySameCount++
			cool = pr.StatusCode == 429
		default:
			return current
		}

		if cool {
			coolLastSelectedKey(ctx)
		}
		if wait := parsePassthroughRetryAfter(pr.Headers); wait > 0 {
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
