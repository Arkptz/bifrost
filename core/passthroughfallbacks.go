package bifrost

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
