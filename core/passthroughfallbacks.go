package bifrost

import (
	"github.com/google/uuid"
	schemas "github.com/maximhq/bifrost/core/schemas"
)

// passthroughFailoverAction is the classification verdict for a non-2xx passthrough
// status. It drives the status-aware failover branch: retry the same provider,
// escalate to the next (more expensive) fallback, fail fast, or treat as success.
type passthroughFailoverAction int

const (
	passthroughNotFailure passthroughFailoverAction = iota // 2xx — no failover
	passthroughFailFast                                    // terminal request-bound error — return verbatim
	passthroughRetrySame                                   // retry the same provider, do not escalate
	passthroughEscalate                                    // escalate to the next fallback after retries
)

// passthroughFailFastStatusCodes are request-bound errors: the request itself is
// wrong or forbidden, so neither retrying nor escalating to another provider can
// help. The upstream body is returned verbatim.
var passthroughFailFastStatusCodes = map[int]bool{
	400: true, // Bad Request
	401: true, // Unauthorized
	403: true, // Forbidden
	404: true, // Not Found
	409: true, // Conflict
	422: true, // Unprocessable Entity
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
// response retries the same provider immediately without charging the escalation
// budget. Only 502/503 are the fasthttp local-failure surface — 500/504 always
// reflect a reached upstream and still escalate.
const passthroughLocalFastFailMaxMs int64 = 3

// classifyPassthroughStatus maps a passthrough response status code and request
// duration to a failover action. This is money-critical: a wrong verdict can jump a
// cheap provider to an expensive one on a single error. 429 always retries the same
// provider and never escalates.
func classifyPassthroughStatus(statusCode int, durationMs int64) passthroughFailoverAction {
	if statusCode >= 200 && statusCode < 300 {
		return passthroughNotFailure
	}
	if statusCode == 429 {
		return passthroughRetrySame
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

// passthroughMaxRetrySame bounds retry-same attempts per provider so a persistent
// 429/5xx cannot hang. After the bound, a 429 is returned verbatim (never
// escalates) and an escalate-class status advances to the next fallback.
const passthroughMaxRetrySame = 1

// passthroughStatusResult carries the outcome of one passthrough attempt for the
// failover loop: the response to inspect and any transport error.
type passthroughStatusResult struct {
	resp *schemas.BifrostResponse
	err  *schemas.BifrostError
}

// runPassthroughStatusFailover drives cost-aware, status-aware failover for a
// passthrough request whose primary attempt returned a non-2xx status on the
// success arm (primaryErr == nil). It reuses prepareFallbackRequest for escalation
// and stamps BifrostContextKeyFallbackRequestID/FallbackIndex per attempt so
// governance billing dedups correctly. attemptFn issues one attempt against the
// given request. This is money-critical: 429 retries the same provider and never
// escalates to a pricier fallback.
//
// For streaming passthrough, escalation is resolved before any client-visible
// byte: the transport reads the first chunk (router.go:3246) — which carries the
// escalation-resolved status — before committing the body to the wire via
// SetBodyStream (router.go:3315). See TestPassthroughStreamCommitsBodyOnlyAfterFirstChunk.
func (bifrost *Bifrost) runPassthroughStatusFailover(
	ctx *schemas.BifrostContext,
	req *schemas.BifrostRequest,
	primary passthroughStatusResult,
	attemptFn func(*schemas.BifrostRequest) passthroughStatusResult,
) passthroughStatusResult {
	_, _, fallbacks := req.GetRequestFields()
	current := primary
	retrySameCount := 0

	for {
		if current.err != nil || current.resp == nil || current.resp.PassthroughResponse == nil {
			return current
		}
		pr := current.resp.PassthroughResponse
		action := classifyPassthroughStatus(pr.StatusCode, pr.ExtraFields.Latency)

		switch action {
		case passthroughNotFailure, passthroughFailFast:
			return current
		case passthroughRetrySame:
			if retrySameCount >= passthroughMaxRetrySame {
				return current
			}
			retrySameCount++
			current = attemptFn(req)
		case passthroughEscalate:
			if len(fallbacks) == 0 {
				return current
			}
			next, ok := bifrost.tryPassthroughFallbacks(ctx, req, fallbacks, attemptFn)
			if !ok {
				return current
			}
			return next
		}
	}
}

// tryPassthroughFallbacks walks the caller-supplied fallback ladder (cheap→expensive)
// once, issuing one attempt per configured fallback. It returns the first attempt
// that is not itself escalate-class (a success, a fail-fast, or a retry-exhausted
// result), or (_, false) when every fallback was skipped/exhausted.
func (bifrost *Bifrost) tryPassthroughFallbacks(
	ctx *schemas.BifrostContext,
	req *schemas.BifrostRequest,
	fallbacks []schemas.Fallback,
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
		result := attemptFn(fallbackReq)
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
