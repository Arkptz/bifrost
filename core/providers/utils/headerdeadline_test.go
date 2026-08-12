package utils

import (
	"bufio"
	"fmt"
	"io"
	"net"
	"net/http"
	"net/http/httptest"
	"sync/atomic"
	"testing"
	"time"

	"github.com/valyala/fasthttp"
)

// rawHeaderServer accepts TCP conns, consumes each request's header block, then
// hands the raw conn to handle for a scripted response. Used by the trickle /
// split-terminator / slow-progress tests to control the exact byte timing of the
// response HEAD that headerDeadlineConn must bound.
func rawHeaderServer(t *testing.T, handle func(c net.Conn)) string {
	t.Helper()
	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatalf("listen: %v", err)
	}
	t.Cleanup(func() { _ = ln.Close() })
	go func() {
		for {
			conn, aerr := ln.Accept()
			if aerr != nil {
				return
			}
			go func(c net.Conn) {
				defer c.Close()
				if consumeHTTPHeaders(bufio.NewReader(c)) != nil {
					return
				}
				handle(c)
			}(conn)
		}
	}()
	return ln.Addr().String()
}

// TestHeaderDeadline_OneByteTrickle is THE regression: an upstream that sends a
// single byte ("H") then goes silent used to permanently disarm the header bound
// (disarm-on-first-successful-read), hanging Do() for the full server silence
// (~32s in the original probe). The head-boundary disarm must keep the bound armed
// until CRLFCRLF, so this is bounded near the 500ms bound.
func TestHeaderDeadline_OneByteTrickle(t *testing.T) {
	const bound = 500 * time.Millisecond
	addr := rawHeaderServer(t, func(c net.Conn) {
		_, _ = c.Write([]byte("H"))
		time.Sleep(30 * time.Second)
	})

	base := &fasthttp.Client{ReadTimeout: bound}
	ConfigureDialer(base, true)
	stream := BuildStreamingClient(base)

	req := fasthttp.AcquireRequest()
	resp := fasthttp.AcquireResponse()
	defer fasthttp.ReleaseRequest(req)
	defer fasthttp.ReleaseResponse(resp)
	req.SetRequestURI("http://" + addr + "/")
	req.Header.SetMethod(http.MethodGet)
	resp.StreamBody = true

	start := time.Now()
	doErr := stream.Do(req, resp)
	elapsed := time.Since(start)

	if doErr == nil {
		t.Fatal("Do returned nil on a one-byte-trickle upstream; expected a timeout")
	}
	if elapsed >= 6*time.Second {
		t.Fatalf("one-byte trickle hung %v (>= 6s): header bound disarmed by a single byte", elapsed)
	}
	t.Logf("one-byte trickle: err=%v elapsed=%v", doErr, elapsed)
}

// TestHeaderDeadline_PartialHeaderTrickle sends a complete status line but no
// terminator, then goes silent. A per-read scan that reset between reads, or a
// disarm on any header byte, would miss it; the incremental matcher must keep the
// bound armed because CRLFCRLF never arrives.
func TestHeaderDeadline_PartialHeaderTrickle(t *testing.T) {
	const bound = 500 * time.Millisecond
	addr := rawHeaderServer(t, func(c net.Conn) {
		_, _ = c.Write([]byte("HTTP/1.1 200 OK\r\n"))
		time.Sleep(30 * time.Second)
	})

	base := &fasthttp.Client{ReadTimeout: bound}
	ConfigureDialer(base, true)
	stream := BuildStreamingClient(base)

	req := fasthttp.AcquireRequest()
	resp := fasthttp.AcquireResponse()
	defer fasthttp.ReleaseRequest(req)
	defer fasthttp.ReleaseResponse(resp)
	req.SetRequestURI("http://" + addr + "/")
	req.Header.SetMethod(http.MethodGet)
	resp.StreamBody = true

	start := time.Now()
	doErr := stream.Do(req, resp)
	elapsed := time.Since(start)

	if doErr == nil {
		t.Fatal("Do returned nil on a partial-header upstream; expected a timeout")
	}
	if elapsed >= 6*time.Second {
		t.Fatalf("partial-header trickle hung %v (>= 6s): header bound not effective", elapsed)
	}
	t.Logf("partial-header trickle: err=%v elapsed=%v", doErr, elapsed)
}

// TestHeaderDeadline_SplitTerminator splits the CRLFCRLF head terminator across
// two writes ("...\r\n" then "\r\n"), proving the incremental matcher carries crlf
// state across Read boundaries. The head must be parsed and the body (chunks with
// gaps EXCEEDING the bound, to prove disarm) must survive to EOF.
func TestHeaderDeadline_SplitTerminator(t *testing.T) {
	const bound = 500 * time.Millisecond
	const totalChunks = 3
	addr := rawHeaderServer(t, func(c net.Conn) {
		_, _ = c.Write([]byte("HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nConnection: close\r\n"))
		time.Sleep(100 * time.Millisecond) // < bound: head still progressing
		_, _ = c.Write([]byte("\r\n"))     // final CRLF completes the terminator
		for i := 0; i < totalChunks; i++ {
			time.Sleep(700 * time.Millisecond) // > bound: proves the body is ungoverned
			fmt.Fprintf(c, "data: chunk-%d\n\n", i)
		}
	})

	base := &fasthttp.Client{ReadTimeout: bound}
	ConfigureDialer(base, true)
	stream := BuildStreamingClient(base)

	req := fasthttp.AcquireRequest()
	resp := fasthttp.AcquireResponse()
	defer fasthttp.ReleaseRequest(req)
	defer fasthttp.ReleaseResponse(resp)
	req.SetRequestURI("http://" + addr + "/")
	req.Header.SetMethod(http.MethodGet)
	resp.StreamBody = true

	if err := stream.Do(req, resp); err != nil {
		t.Fatalf("Do: %v (split terminator not parsed across reads)", err)
	}
	if resp.StatusCode() != http.StatusOK {
		t.Fatalf("status: %d", resp.StatusCode())
	}
	scanner := bufio.NewScanner(resp.BodyStream())
	got := 0
	for scanner.Scan() {
		if line := scanner.Text(); len(line) >= 5 && line[:5] == "data:" {
			got++
		}
	}
	if err := scanner.Err(); err != nil {
		t.Fatalf("scanner: %v (body killed after split terminator)", err)
	}
	if got != totalChunks {
		t.Errorf("chunks: got %d want %d", got, totalChunks)
	}
	t.Logf("split terminator: %d/%d chunks survived", got, totalChunks)
}

// TestHeaderDeadline_SlowProgressingHeaders drips header bytes with gaps SHORTER
// than the bound, then a body. Each progressing read must re-extend the deadline
// so a head that keeps arriving is never killed (only a stalled head fires).
func TestHeaderDeadline_SlowProgressingHeaders(t *testing.T) {
	const bound = 500 * time.Millisecond
	const totalChunks = 2
	head := []string{
		"HTTP/1.1 200 OK\r\n",
		"Content-Type: text/event-stream\r\n",
		"Connection: close\r\n",
		"\r\n",
	}
	addr := rawHeaderServer(t, func(c net.Conn) {
		for _, part := range head {
			time.Sleep(300 * time.Millisecond) // < bound: progressing, must not be killed
			if _, err := c.Write([]byte(part)); err != nil {
				return
			}
		}
		for i := 0; i < totalChunks; i++ {
			fmt.Fprintf(c, "data: chunk-%d\n\n", i)
			time.Sleep(50 * time.Millisecond)
		}
	})

	base := &fasthttp.Client{ReadTimeout: bound}
	ConfigureDialer(base, true)
	stream := BuildStreamingClient(base)

	req := fasthttp.AcquireRequest()
	resp := fasthttp.AcquireResponse()
	defer fasthttp.ReleaseRequest(req)
	defer fasthttp.ReleaseResponse(resp)
	req.SetRequestURI("http://" + addr + "/")
	req.Header.SetMethod(http.MethodGet)
	resp.StreamBody = true

	if err := stream.Do(req, resp); err != nil {
		t.Fatalf("Do: %v (slow-but-progressing head wrongly killed)", err)
	}
	if resp.StatusCode() != http.StatusOK {
		t.Fatalf("status: %d", resp.StatusCode())
	}
	scanner := bufio.NewScanner(resp.BodyStream())
	got := 0
	for scanner.Scan() {
		if line := scanner.Text(); len(line) >= 5 && line[:5] == "data:" {
			got++
		}
	}
	if got != totalChunks {
		t.Errorf("chunks: got %d want %d", got, totalChunks)
	}
	t.Logf("slow-progressing head: %d/%d chunks survived", got, totalChunks)
}

// TestHeaderDeadline_LongBodyUnaffected is the over-correction guard: a 12-chunk
// SSE stream with gaps FAR exceeding the bound must survive to clean EOF — once the
// head boundary disarms, the body must be entirely ungoverned.
func TestHeaderDeadline_LongBodyUnaffected(t *testing.T) {
	const bound = 300 * time.Millisecond
	const totalChunks = 12
	const chunkGap = 700 * time.Millisecond // >> bound

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)
		flusher, _ := w.(http.Flusher)
		for i := 0; i < totalChunks; i++ {
			fmt.Fprintf(w, "data: chunk-%d\n\n", i)
			if flusher != nil {
				flusher.Flush()
			}
			time.Sleep(chunkGap)
		}
	}))
	defer srv.Close()

	base := &fasthttp.Client{ReadTimeout: bound}
	ConfigureDialer(base, false)
	stream := BuildStreamingClient(base)

	req := fasthttp.AcquireRequest()
	resp := fasthttp.AcquireResponse()
	defer fasthttp.ReleaseRequest(req)
	defer fasthttp.ReleaseResponse(resp)
	req.SetRequestURI(srv.URL)
	req.Header.SetMethod(http.MethodGet)
	resp.StreamBody = true

	if err := stream.Do(req, resp); err != nil {
		t.Fatalf("Do: %v", err)
	}
	scanner := bufio.NewScanner(resp.BodyStream())
	got := 0
	for scanner.Scan() {
		if line := scanner.Text(); len(line) >= 5 && line[:5] == "data:" {
			got++
		}
	}
	if err := scanner.Err(); err != nil {
		t.Fatalf("scanner: %v (long body killed by leaked header deadline)", err)
	}
	if got != totalChunks {
		t.Errorf("chunks received: got %d, want %d (long body killed early)", got, totalChunks)
	}
	t.Logf("long body: %d/%d chunks survived", got, totalChunks)
}

// TestHeaderDeadline_RequestBodyWriteStallBounded covers B2 (client side): a
// provider that accepts TCP but stops reading blocks the client in Write on a
// large request body, before any header logic runs. The streaming client's
// WriteTimeout (= header bound) must bound that write. POST is non-idempotent, so
// fasthttp does not retry — a single ~bound attempt.
func TestHeaderDeadline_RequestBodyWriteStallBounded(t *testing.T) {
	const bound = 500 * time.Millisecond

	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatalf("listen: %v", err)
	}
	defer ln.Close()
	// Accept but NEVER read: the client's write blocks once kernel buffers fill.
	go func() {
		for {
			conn, aerr := ln.Accept()
			if aerr != nil {
				return
			}
			_ = conn
		}
	}()

	base := &fasthttp.Client{ReadTimeout: bound}
	ConfigureDialer(base, true)
	stream := BuildStreamingClient(base)

	req := fasthttp.AcquireRequest()
	resp := fasthttp.AcquireResponse()
	defer fasthttp.ReleaseRequest(req)
	defer fasthttp.ReleaseResponse(resp)
	req.SetRequestURI("http://" + ln.Addr().String() + "/")
	req.Header.SetMethod(http.MethodPost)
	req.SetBody(make([]byte, 32*1024*1024)) // 32MB: far exceeds kernel send + recv buffers
	resp.StreamBody = true

	start := time.Now()
	doErr := stream.Do(req, resp)
	elapsed := time.Since(start)

	if doErr == nil {
		t.Fatal("Do returned nil while writing to a non-reading peer; expected a write timeout")
	}
	if elapsed >= 6*time.Second {
		t.Fatalf("request-body write hung %v (>= 6s): WriteTimeout not bounding the write", elapsed)
	}
	t.Logf("request-body write stall: err=%v elapsed=%v", doErr, elapsed)
}

// TestHeaderDeadline_KeepAliveReuseStallBounded reproduces the pooled-connection
// gap: request 1 succeeds over a keep-alive conn (disarming the per-request read
// deadline), then request 2 on the SAME pooled conn stalls before headers. The
// bound must re-arm per request so the stall is bounded, not left to run until
// the server closes. Regression guard — this must never come back.
func TestHeaderDeadline_KeepAliveReuseStallBounded(t *testing.T) {
	const bound = 500 * time.Millisecond
	const serverSilence = 10 * time.Second // >> the per-request bound, so an unbounded stall is obvious

	var reqCount int32
	var dials int32

	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatalf("listen: %v", err)
	}
	defer ln.Close()

	go func() {
		for {
			conn, aerr := ln.Accept()
			if aerr != nil {
				return
			}
			go func(c net.Conn) {
				defer c.Close()
				br := bufio.NewReader(c)
				for {
					// Consume one request's header block (request line ... blank line).
					for {
						line, rerr := br.ReadString('\n')
						if rerr != nil {
							return
						}
						if line == "\r\n" {
							break
						}
					}
					if atomic.AddInt32(&reqCount, 1) == 1 {
						// Request 1: answer with keep-alive so fasthttp pools the conn.
						body := "ok"
						fmt.Fprintf(c, "HTTP/1.1 200 OK\r\nContent-Length: %d\r\nContent-Type: text/plain\r\n\r\n%s", len(body), body)
						continue
					}
					// Request 2+: go silent before sending any header byte.
					time.Sleep(serverSilence)
					return
				}
			}(conn)
		}
	}()

	base := &fasthttp.Client{ReadTimeout: bound}
	base.Dial = func(addr string) (net.Conn, error) {
		atomic.AddInt32(&dials, 1)
		return net.Dial("tcp", addr)
	}
	ConfigureDialer(base, true)
	stream := BuildStreamingClient(base)

	addr := ln.Addr().String()

	// Request 1 — succeeds, fully drained so the conn returns to the pool.
	req1 := fasthttp.AcquireRequest()
	resp1 := fasthttp.AcquireResponse()
	req1.SetRequestURI("http://" + addr + "/")
	req1.Header.SetMethod(http.MethodGet)
	resp1.StreamBody = true
	if err := stream.Do(req1, resp1); err != nil {
		fasthttp.ReleaseRequest(req1)
		fasthttp.ReleaseResponse(resp1)
		t.Fatalf("request 1 Do: %v", err)
	}
	if bs := resp1.BodyStream(); bs != nil {
		_, _ = io.Copy(io.Discard, bs)
	}
	fasthttp.ReleaseRequest(req1)
	fasthttp.ReleaseResponse(resp1)

	// Request 2 — stalls before headers on the reused conn.
	req2 := fasthttp.AcquireRequest()
	resp2 := fasthttp.AcquireResponse()
	defer fasthttp.ReleaseRequest(req2)
	defer fasthttp.ReleaseResponse(resp2)
	req2.SetRequestURI("http://" + addr + "/")
	req2.Header.SetMethod(http.MethodGet)
	resp2.StreamBody = true

	start := time.Now()
	doErr := stream.Do(req2, resp2)
	elapsed := time.Since(start)

	if doErr == nil {
		t.Fatal("request 2 returned nil error on a silent reused conn; expected a timeout")
	}
	// 5 idempotent retries × bound = 2.5s ceiling; well under the server's 10s silence.
	const upperBound = 5 * time.Second
	if elapsed >= upperBound {
		t.Fatalf("request 2 took %v (>= %v): header bound not re-armed per request on pooled reuse (dials=%d)",
			elapsed, upperBound, atomic.LoadInt32(&dials))
	}
	t.Logf("keep-alive reuse stall: err=%v elapsed=%v dials=%d", doErr, elapsed, atomic.LoadInt32(&dials))
}

// TestHeaderDeadline_StalledHeadersTimeOut verifies that an upstream which
// accepts the TCP connection but never sends response headers causes Do() to
// return a timeout error within a bounded time — not hang forever. The bound is
// derived from base.ReadTimeout; fasthttp may retry the idempotent GET up to 5x,
// so we assert a bounded TOTAL, never an exact single-attempt duration.
func TestHeaderDeadline_StalledHeadersTimeOut(t *testing.T) {
	const bound = 500 * time.Millisecond

	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatalf("listen: %v", err)
	}
	defer ln.Close()

	// Accept connections and hold them silent — never write any response bytes.
	go func() {
		for {
			conn, aerr := ln.Accept()
			if aerr != nil {
				return
			}
			_ = conn // keep the conn open, send nothing
		}
	}()

	base := &fasthttp.Client{ReadTimeout: bound}
	ConfigureDialer(base, true)
	stream := BuildStreamingClient(base)

	req := fasthttp.AcquireRequest()
	resp := fasthttp.AcquireResponse()
	defer fasthttp.ReleaseRequest(req)
	defer fasthttp.ReleaseResponse(resp)

	req.SetRequestURI("http://" + ln.Addr().String() + "/")
	req.Header.SetMethod(http.MethodGet)
	resp.StreamBody = true

	start := time.Now()
	doErr := stream.Do(req, resp)
	elapsed := time.Since(start)

	if doErr == nil {
		t.Fatal("Do returned nil error on a silent upstream; expected a timeout")
	}
	// 5 idempotent retries × bound = 2.5s ceiling; allow generous slack for CI.
	const upperBound = 6 * time.Second
	if elapsed >= upperBound {
		t.Fatalf("Do took %v (>= %v): header deadline not bounding the stall", elapsed, upperBound)
	}
	t.Logf("stalled headers: err=%v elapsed=%v", doErr, elapsed)
}

// TestHeaderDeadline_LiveSlowBodySurvives verifies that a live SSE upstream that
// sends headers promptly but drips body chunks with 400ms gaps (past the header
// bound) is NOT killed — the deadline must disarm on the first read so the body
// streams to EOF.
func TestHeaderDeadline_LiveSlowBodySurvives(t *testing.T) {
	const bound = 500 * time.Millisecond
	const totalChunks = 4

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)
		flusher, _ := w.(http.Flusher)
		for i := 0; i < totalChunks; i++ {
			fmt.Fprintf(w, "data: chunk-%d\n\n", i)
			if flusher != nil {
				flusher.Flush()
			}
			time.Sleep(400 * time.Millisecond)
		}
	}))
	defer srv.Close()

	base := &fasthttp.Client{ReadTimeout: bound}
	ConfigureDialer(base, false)
	stream := BuildStreamingClient(base)

	req := fasthttp.AcquireRequest()
	resp := fasthttp.AcquireResponse()
	defer fasthttp.ReleaseRequest(req)
	defer fasthttp.ReleaseResponse(resp)

	req.SetRequestURI(srv.URL)
	req.Header.SetMethod(http.MethodGet)
	resp.StreamBody = true

	if err := stream.Do(req, resp); err != nil {
		t.Fatalf("Do: %v", err)
	}
	if resp.StatusCode() != http.StatusOK {
		t.Fatalf("status: %d", resp.StatusCode())
	}

	scanner := bufio.NewScanner(resp.BodyStream())
	got := 0
	for scanner.Scan() {
		if line := scanner.Text(); len(line) >= 5 && line[:5] == "data:" {
			got++
		}
	}
	if err := scanner.Err(); err != nil {
		t.Fatalf("scanner: %v (body killed by leaked header deadline)", err)
	}
	if got != totalChunks {
		t.Errorf("chunks received: got %d, want %d (slow body killed early)", got, totalChunks)
	}
}
