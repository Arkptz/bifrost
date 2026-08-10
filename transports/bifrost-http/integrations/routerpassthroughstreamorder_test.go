package integrations

import (
	"context"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	bifrost "github.com/maximhq/bifrost/core"
	"github.com/maximhq/bifrost/core/schemas"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"github.com/valyala/fasthttp"
)

// streamOrderAccount is a minimal schemas.Account backing a real *bifrost.Bifrost
// with a single Anthropic provider pointed at an httptest server. It exists so the
// stream-order test can drive handlePassthroughStream against a live client rather
// than replaying the handler's steps by hand.
type streamOrderAccount struct {
	baseURL string
}

func (a *streamOrderAccount) GetConfiguredProviders() ([]schemas.ModelProvider, error) {
	return []schemas.ModelProvider{schemas.Anthropic}, nil
}

func (a *streamOrderAccount) GetKeysForProvider(_ context.Context, _ schemas.ModelProvider) ([]schemas.Key, error) {
	return []schemas.Key{
		{ID: "k1", Value: *schemas.NewSecretVar("sk-test"), Models: schemas.WhiteList{"*"}, Weight: 100},
	}, nil
}

func (a *streamOrderAccount) GetConfigForProvider(_ schemas.ModelProvider) (*schemas.ProviderConfig, error) {
	return &schemas.ProviderConfig{
		NetworkConfig: schemas.NetworkConfig{
			BaseURL:                        a.baseURL,
			DefaultRequestTimeoutInSeconds: 30,
			MaxRetries:                     0,
		},
		ConcurrencyAndBufferSize: schemas.ConcurrencyAndBufferSize{Concurrency: 1, BufferSize: 1},
	}, nil
}

// newStreamOrderRouter builds a real Bifrost client + GenericRouter wired to an
// Anthropic passthrough route whose upstream is `upstreamURL`.
func newStreamOrderRouter(t *testing.T, upstreamURL string) *GenericRouter {
	t.Helper()
	client, err := bifrost.Init(context.Background(), schemas.BifrostConfig{
		Account: &streamOrderAccount{baseURL: upstreamURL},
		Logger:  bifrost.NewDefaultLogger(schemas.LogLevelError),
	})
	require.NoError(t, err)
	t.Cleanup(client.Shutdown)

	return NewGenericRouter(client, &mockHandlerStore{}, nil, &PassthroughConfig{
		Provider: schemas.Anthropic,
	}, bifrost.NewNoOpLogger())
}

// newPassthroughStreamCtx builds a fasthttp.RequestCtx for a streaming passthrough
// request to `path`, plus the BifrostContext + cancel the handler expects.
func newPassthroughStreamCtx(path string) (*fasthttp.RequestCtx, *schemas.BifrostContext, context.CancelFunc) {
	ctx := &fasthttp.RequestCtx{}
	ctx.Request.Header.SetMethod(http.MethodPost)
	ctx.Request.SetRequestURI(path)
	ctx.Request.SetBody([]byte(`{"model":"claude-sonnet","stream":true}`))
	bifrostCtx := schemas.NewBifrostContext(context.Background(), time.Now().Add(30*time.Second))
	c, cancel := context.WithCancel(context.Background())
	_ = c
	return ctx, bifrostCtx, cancel
}

// sseServer serves an SSE passthrough stream: an optional leading status line,
// then N data events. Used to feed handlePassthroughStream a real upstream.
func sseServer(t *testing.T, status int, events ...string) *httptest.Server {
	t.Helper()
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(status)
		flusher, _ := w.(http.Flusher)
		for _, e := range events {
			_, _ = fmt.Fprint(w, e)
			if flusher != nil {
				flusher.Flush()
			}
		}
	}))
}

// TestHandlePassthroughStreamEmitsUpstreamBody genuinely exercises
// handlePassthroughStream end-to-end: a live Bifrost client streams from a real
// SSE upstream, and the handler must commit the upstream status and body to the
// fasthttp response. A panic or behavior change inside handlePassthroughStream
// (e.g. never reading the first chunk, or never calling SetBodyStream) fails this
// test — unlike the previous version, which replayed the handler's steps by hand
// and passed even when the handler body was replaced with panic().
func TestHandlePassthroughStreamEmitsUpstreamBody(t *testing.T) {
	upstream := sseServer(t, http.StatusOK,
		"event: message_start\ndata: {\"a\":1}\n\n",
		"event: message_stop\ndata: {\"b\":2}\n\n",
	)
	defer upstream.Close()

	router := newStreamOrderRouter(t, upstream.URL)
	ctx, bifrostCtx, cancel := newPassthroughStreamCtx("/v1/messages")

	router.handlePassthroughStream(ctx, bifrostCtx, cancel, schemas.Anthropic,
		&schemas.BifrostPassthroughRequest{
			Method:   http.MethodPost,
			Path:     "/v1/messages",
			Body:     []byte(`{"model":"claude-sonnet","stream":true}`),
			Provider: schemas.Anthropic,
			Model:    "claude-sonnet",
		})

	require.Equal(t, http.StatusOK, ctx.Response.StatusCode(),
		"handler must commit the upstream 200 status")
	bodyStream := ctx.Response.BodyStream()
	require.NotNil(t, bodyStream, "handler must commit a body stream after the first chunk")
	body, err := io.ReadAll(bodyStream)
	require.NoError(t, err)
	assert.Contains(t, string(body), "message_start",
		"handler must forward the first upstream event")
	assert.Contains(t, string(body), "message_stop",
		"handler must forward later upstream events")
}

// TestHandlePassthroughStreamForwardsUpstreamErrorStatus proves the handler
// commits a non-2xx upstream status verbatim on the streaming path (the first
// chunk carries the status). If the handler stopped reading the first chunk or
// hard-coded 200, this fails.
func TestHandlePassthroughStreamForwardsUpstreamErrorStatus(t *testing.T) {
	upstream := sseServer(t, http.StatusInternalServerError,
		"data: {\"error\":\"boom\"}\n\n",
	)
	defer upstream.Close()

	router := newStreamOrderRouter(t, upstream.URL)
	ctx, bifrostCtx, cancel := newPassthroughStreamCtx("/v1/messages")

	router.handlePassthroughStream(ctx, bifrostCtx, cancel, schemas.Anthropic,
		&schemas.BifrostPassthroughRequest{
			Method:   http.MethodPost,
			Path:     "/v1/messages",
			Body:     []byte(`{"model":"claude-sonnet","stream":true}`),
			Provider: schemas.Anthropic,
			Model:    "claude-sonnet",
		})

	assert.Equal(t, http.StatusInternalServerError, ctx.Response.StatusCode(),
		"handler must forward the upstream 500 status from the first chunk")
}
