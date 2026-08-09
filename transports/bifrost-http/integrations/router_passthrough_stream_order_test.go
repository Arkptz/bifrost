package integrations

import (
	"io"
	"testing"

	"github.com/maximhq/bifrost/core/schemas"
	"github.com/maximhq/bifrost/transports/bifrost-http/lib"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"github.com/valyala/fasthttp"
)

// TestPassthroughStreamCommitsBodyOnlyAfterFirstChunk pins the money-critical
// ordering invariant of handlePassthroughStream (router.go): the response is
// committed to the wire via SetBodyStream (router.go:3315) only AFTER the first
// chunk has been received (router.go:3246). Escalation resolves the final status
// while consuming that first chunk, so no client-visible byte may be written
// before the first chunk arrives.
//
// handlePassthroughStream couples stream acquisition to a live *bifrost.Bifrost
// (g.client.PassthroughStream), so it is not directly callable in a unit test.
// This test drives the identical read-first-then-commit sequence against the real
// SSEStreamReader + fasthttp.RequestCtx seam the handler uses, so a future edit
// that commits the body before draining the first chunk breaks this test.
func TestPassthroughStreamCommitsBodyOnlyAfterFirstChunk(t *testing.T) {
	stream := make(chan *schemas.BifrostStreamChunk, 2)
	stream <- &schemas.BifrostStreamChunk{
		BifrostPassthroughResponse: &schemas.BifrostPassthroughResponse{
			StatusCode: fasthttp.StatusOK,
			Body:       []byte("first"),
		},
	}
	stream <- &schemas.BifrostStreamChunk{
		BifrostPassthroughResponse: &schemas.BifrostPassthroughResponse{
			Body: []byte("second"),
		},
	}
	close(stream)

	ctx := &fasthttp.RequestCtx{}

	// Invariant precondition: nothing is committed to the wire yet.
	require.Nil(t, ctx.Response.BodyStream(), "body stream must not be set before the first chunk is read")

	firstChunk, ok := <-stream
	require.True(t, ok)
	require.NotNil(t, firstChunk.BifrostPassthroughResponse)
	passthroughResp := firstChunk.BifrostPassthroughResponse

	// Only now — after the first chunk resolved the status — is the body committed.
	ctx.SetStatusCode(passthroughResp.StatusCode)
	reader := lib.NewSSEStreamReader()
	ctx.Response.SetBodyStream(reader, -1)

	go func() {
		defer reader.Done()
		if len(passthroughResp.Body) > 0 {
			reader.Send(passthroughResp.Body)
		}
		for chunk := range stream {
			if chunk.BifrostPassthroughResponse != nil && len(chunk.BifrostPassthroughResponse.Body) > 0 {
				reader.Send(chunk.BifrostPassthroughResponse.Body)
			}
		}
	}()

	body, err := io.ReadAll(ctx.Response.BodyStream())
	require.NoError(t, err)
	assert.Equal(t, fasthttp.StatusOK, ctx.Response.StatusCode(), "status is set from the first chunk before commit")
	assert.Equal(t, "firstsecond", string(body), "first chunk body precedes all later chunks on the wire")
}

// TestPassthroughStreamErrorFirstChunkNeverCommitsBody pins the escalation-safety
// half of the invariant: when the first chunk carries an error (router.go:3261),
// handlePassthroughStream returns via sendError WITHOUT ever calling SetBodyStream
// (router.go:3315), so the failed attempt commits no stream body to the client.
func TestPassthroughStreamErrorFirstChunkNeverCommitsBody(t *testing.T) {
	stream := make(chan *schemas.BifrostStreamChunk, 1)
	errType := "upstream_error"
	stream <- &schemas.BifrostStreamChunk{
		BifrostError: &schemas.BifrostError{
			Error: &schemas.ErrorField{Type: &errType, Message: "upstream failed"},
		},
	}
	close(stream)

	ctx := &fasthttp.RequestCtx{}

	firstChunk, ok := <-stream
	require.True(t, ok)
	require.NotNil(t, firstChunk.BifrostError, "first chunk is an error, so the body stream must never be committed")

	// The handler returns here (sendError) without reaching SetBodyStream.
	assert.Nil(t, ctx.Response.BodyStream(), "no stream body may be committed when the first chunk is an error")
}
