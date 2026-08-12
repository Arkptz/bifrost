package utils

import (
	"net"
	"sync"
	"sync/atomic"
	"time"

	"github.com/valyala/fasthttp"
)

// streamConnRegistry bridges the dial site to the streaming caller. fasthttp
// dials through a client-level Dial closure (no per-request context) and, after
// Do returns, exposes ONLY resp.LocalAddr()/RemoteAddr() — never the underlying
// net.Conn. So a passthrough stream recovers the *headerDeadlineConn serving it
// (to move its read deadline into the past instead of firing fasthttp's
// pool-releasing close from a third goroutine) by looking the conn up here by its
// address 4-tuple. Entries are added on dial, removed on Close, so the map is
// bounded by live connections.
var streamConnRegistry sync.Map // connRegistryKey(laddr,raddr) -> *headerDeadlineConn

// connRegistryKey identifies a live conn by its (local addr, remote addr) pair,
// which is unique while the connection is open.
func connRegistryKey(laddr, raddr net.Addr) string {
	if laddr == nil || raddr == nil {
		return ""
	}
	return laddr.String() + "|" + raddr.String()
}

// lookupStreamConn returns the headerDeadlineConn for the given address pair, or
// nil when none is registered (a provider on a plain net.Conn, or a dial that
// never completed ParseNetConn).
func lookupStreamConn(laddr, raddr net.Addr) *headerDeadlineConn {
	key := connRegistryKey(laddr, raddr)
	if key == "" {
		return nil
	}
	if v, ok := streamConnRegistry.Load(key); ok {
		return v.(*headerDeadlineConn)
	}
	return nil
}

// defaultHeaderReadBound is the fallback pre-header read deadline used when the
// base client has no ReadTimeout to derive from. Streaming clients zero their
// ReadTimeout so a stalled upstream (TCP connected, silent before headers) would
// otherwise pin a worker goroutine forever. See headerDeadlineConn.
const defaultHeaderReadBound = 300 * time.Second

// headerMode records, per connection, whether the response head can be scanned
// as plaintext HTTP or must be treated as an opaque byte stream (TLS ciphertext,
// which this wrapper sees BELOW the tls.Client layer, or any non-HTTP prologue).
type headerMode uint8

const (
	modeUnknown   headerMode = iota
	modePlaintext            // first response byte was 'H' → scan for the CRLFCRLF head boundary
	modeOpaque               // TLS/ciphertext or non-HTTP → cannot scan; use an idle bound
)

// headerDeadlineConn bounds the wait for a complete response HEAD, then gets out
// of the way so a live-but-slow streaming body is not killed.
//
// Why this exists: streaming clients zero ReadTimeout/MaxConnDuration so fasthttp
// won't pre-empt a healthy stream, but that also removes any bound on the pre-header
// window — an upstream that connects then trickles/stalls before the head completes
// hangs Do() forever. Setting ReadTimeout back on, or using DoDeadline, also kills a
// live slow body (the deadline leaks into BodyStream reads).
//
// Disarm rule — the header BOUNDARY, not the first byte. fasthttp parses response
// headers through a bufio.Reader that issues MULTIPLE conn.Read calls; an upstream
// that sends one byte then goes silent satisfies "n>0 && err==nil" and, under the
// old disarm-on-first-byte rule, permanently removed the bound before any header
// arrived. This wrapper is a net.Conn and sees only bytes, so it detects the end of
// the HTTP response head — the "\r\n\r\n" terminator — with an incremental matcher
// that SURVIVES the terminator being split across reads (crlf state persists between
// Read calls). 1xx interim heads (e.g. "100 Continue") have their own terminator;
// the matcher classifies the status line and keeps the bound armed until a FINAL
// (non-1xx) head completes. Until then the deadline is extended on every progressing
// read, so a slow-but-progressing head survives while a stalled one still fires.
//
// Per-transport guarantee:
//   - Plaintext HTTP (first byte 'H'): precise — disarm exactly on the final CRLFCRLF,
//     after which the body is fully ungoverned (any chunk gap survives).
//   - TLS / opaque (ciphertext, cannot see plaintext CRLFCRLF): best achievable is an
//     IDLE bound — the deadline is re-armed on every progressing read and never
//     precisely disarmed, so a silent/trickled stall fires within `bound` while a
//     stream that keeps producing bytes within `bound` survives. In production `bound`
//     is the base client's ReadTimeout (DefaultRequestTimeoutInSeconds, default 300s),
//     so this idle bound never fires on a real SSE stream; the tighter app-layer
//     NewIdleTimeoutReader remains the real per-chunk governor.
//
// The boundary is PER-REQUEST: fasthttp reuses keep-alive conns from its pool, so a
// new request always begins by WRITING request bytes — Write re-arms the bound and
// resets the scan state. HTTP/1.1 sends the whole request before reading any response,
// so Write and Read never interleave on one conn for one request, and a pooled conn is
// never used by two requests concurrently; the scan fields therefore need no locking.
//
// fasthttp (client.go) unconditionally calls conn.SetReadDeadline(readDeadline) after
// dialing; with a zeroed ReadTimeout and no Do-deadline that value is the zero
// time.Time = "clear deadline", which would wipe an armed bound. SetReadDeadline below
// ignores that zero-value wipe while still armed.
type headerDeadlineConn struct {
	net.Conn
	bound time.Duration
	armed atomic.Bool

	// Scan state — touched only from Read, reset from arm(). Write (arm) and Read
	// are serialized per request by fasthttp's HTTP/1.1 client, so no locking.
	mode    headerMode
	crlf    int      // consecutive bytes of "\r\n\r\n" matched so far (0..4)
	lineBuf [16]byte // start of the current header block, for 1xx interim detection
	lineLen int
}

// arm sets the read+write deadlines for the current request's header phase and
// resets the per-request scan state (mode persists — a conn is plaintext or TLS
// for life).
//
// The WRITE deadline bounds request-body writes only (an upstream that accepts TCP
// then stops reading blocks the client in Write, before any header logic runs).
// HTTP/1.1 writes the whole request before reading, so re-arming on every Write is
// self-maintaining and cannot bound the response read; it is cleared with the read
// bound at the head boundary so it never touches a slow SSE body. No zero-wipe
// defence is needed here: fasthttp's conn.SetWriteDeadline(zero) runs BEFORE
// req.Write, i.e. before this hook re-arms (the read wipe runs AFTER, so
// SetReadDeadline must ignore it).
func (c *headerDeadlineConn) arm() {
	c.armed.Store(true)
	c.crlf = 0
	c.lineLen = 0
	deadline := time.Now().Add(c.bound)
	_ = c.Conn.SetReadDeadline(deadline)
	_ = c.Conn.SetWriteDeadline(deadline)
}

// Write re-arms the header bound at the start of every request (each request
// begins by writing its bytes). Re-arming on multiple writes of the same request
// only pushes the deadline out and is harmless — the head boundary disarms it.
func (c *headerDeadlineConn) Write(p []byte) (int, error) {
	c.arm()
	return c.Conn.Write(p)
}

// SetReadDeadline ignores fasthttp's zero-value "clear deadline" call while the
// header bound is still armed; any non-zero deadline (or any call after disarm)
// delegates to the underlying conn.
//
// A streaming cancellation/idle owner calls this with a past time to unblock a
// parked body read; that owner runs AFTER the header phase disarmed (armed==false
// once the head boundary was seen), so the zero-wipe guard never suppresses it.
func (c *headerDeadlineConn) SetReadDeadline(t time.Time) error {
	if c.armed.Load() && t.IsZero() {
		return nil
	}
	return c.Conn.SetReadDeadline(t)
}

// Close deregisters the conn from streamConnRegistry before closing the
// underlying connection, so the map never outlives the connection. fasthttp
// calls this via CloseConn when it retires the conn (whether pooled-then-idle or
// hard-closed on ConnectionClose), which is the single point every teardown path
// funnels through.
func (c *headerDeadlineConn) Close() error {
	if key := connRegistryKey(c.Conn.LocalAddr(), c.Conn.RemoteAddr()); key != "" {
		streamConnRegistry.Delete(key)
	}
	return c.Conn.Close()
}

// Read governs the pre-header window. On the first read it classifies the conn as
// plaintext HTTP or opaque (see headerMode). Plaintext: scan for the final CRLFCRLF
// and disarm exactly there so the body is ungoverned. Opaque or still-in-flight
// head: extend the bound so a progressing peer survives and a stalled one fires.
func (c *headerDeadlineConn) Read(p []byte) (int, error) {
	n, err := c.Conn.Read(p)
	if !c.armed.Load() || n <= 0 {
		return n, err
	}
	if c.mode == modeUnknown {
		// TLS record type bytes are 0x14-0x17, never 'H' (0x48); a leading 'H'
		// reliably means a plaintext "HTTP/..." status line.
		if p[0] == 'H' {
			c.mode = modePlaintext
		} else {
			c.mode = modeOpaque
		}
	}
	if c.mode == modePlaintext && c.scanHeadComplete(p[:n]) {
		c.armed.Store(false)
		_ = c.Conn.SetReadDeadline(time.Time{})
		_ = c.Conn.SetWriteDeadline(time.Time{})
		return n, err
	}
	_ = c.Conn.SetReadDeadline(time.Now().Add(c.bound))
	return n, err
}

// scanHeadComplete feeds freshly read plaintext bytes through an incremental
// "\r\n\r\n" matcher that survives the terminator being split across reads (crlf
// persists on the conn between calls). It returns true only when a FINAL (non-1xx)
// response head completes; a 1xx interim head resets the scan so the real head is
// still bounded.
func (c *headerDeadlineConn) scanHeadComplete(p []byte) bool {
	for _, b := range p {
		if c.lineLen < len(c.lineBuf) {
			c.lineBuf[c.lineLen] = b
			c.lineLen++
		}
		switch {
		case (c.crlf == 0 || c.crlf == 2) && b == '\r':
			c.crlf++
		case (c.crlf == 1 || c.crlf == 3) && b == '\n':
			c.crlf++
		case b == '\r':
			c.crlf = 1
		default:
			c.crlf = 0
		}
		if c.crlf != 4 {
			continue
		}
		// A header block terminated. Classify it: in "HTTP/1.x SSS ..." the first
		// status digit is at index 9; a leading '1' marks a 1xx interim response.
		interim := c.lineLen > 9 && c.lineBuf[9] == '1'
		c.crlf = 0
		c.lineLen = 0
		if !interim {
			return true
		}
		// interim: keep scanning for the real head in the remaining bytes.
	}
	return false
}

// withHeaderDeadlineDial wraps client.Dial so every dialed connection arms a
// per-request read deadline of `bound` for the header phase. It COMPOSES with any
// existing Dial (e.g. ConfigureDialer's SSRF-protecting dialer); it does not
// replace it. A non-positive bound is a no-op.
func withHeaderDeadlineDial(client *fasthttp.Client, bound time.Duration) {
	if bound <= 0 {
		return
	}
	inner := client.Dial
	if inner == nil {
		inner = func(addr string) (net.Conn, error) { return fasthttp.Dial(addr) }
	}
	client.Dial = func(addr string) (net.Conn, error) {
		conn, err := inner(addr)
		if err != nil {
			return nil, err
		}
		hc := &headerDeadlineConn{Conn: conn, bound: bound}
		hc.arm()
		// Register under the live 4-tuple so a streaming caller can recover this
		// conn from resp.LocalAddr()+RemoteAddr() after Do; Close deregisters it.
		if key := connRegistryKey(conn.LocalAddr(), conn.RemoteAddr()); key != "" {
			streamConnRegistry.Store(key, hc)
		}
		return hc, nil
	}
}
