package utils

import (
	"bufio"
	"crypto/ecdsa"
	"crypto/elliptic"
	"crypto/rand"
	"crypto/tls"
	"crypto/x509"
	"encoding/pem"
	"fmt"
	"io"
	"math/big"
	"net"
	"net/http"
	"sync/atomic"
	"testing"
	"time"

	"github.com/valyala/fasthttp"
)

// testTLSConfig generates a throwaway self-signed ECDSA P-256 cert for
// 127.0.0.1. No files, no deps beyond crypto stdlib.
func testTLSConfig(t *testing.T) (server, client *tls.Config) {
	t.Helper()
	key, err := ecdsa.GenerateKey(elliptic.P256(), rand.Reader)
	if err != nil {
		t.Fatalf("keygen: %v", err)
	}
	tmpl := &x509.Certificate{
		SerialNumber: big.NewInt(1),
		NotBefore:    time.Now().Add(-time.Minute),
		NotAfter:     time.Now().Add(time.Hour),
		IPAddresses:  []net.IP{net.ParseIP("127.0.0.1")},
	}
	der, err := x509.CreateCertificate(rand.Reader, tmpl, tmpl, &key.PublicKey, key)
	if err != nil {
		t.Fatalf("cert: %v", err)
	}
	certPEM := pem.EncodeToMemory(&pem.Block{Type: "CERTIFICATE", Bytes: der})
	keyDER, err := x509.MarshalECPrivateKey(key)
	if err != nil {
		t.Fatalf("marshal key: %v", err)
	}
	keyPEM := pem.EncodeToMemory(&pem.Block{Type: "EC PRIVATE KEY", Bytes: keyDER})
	cert, err := tls.X509KeyPair(certPEM, keyPEM)
	if err != nil {
		t.Fatalf("keypair: %v", err)
	}
	return &tls.Config{Certificates: []tls.Certificate{cert}},
		&tls.Config{InsecureSkipVerify: true} //nolint:gosec // test-only self-signed
}

// consumeHTTPHeaders reads one HTTP request header block (through the blank \r\n).
func consumeHTTPHeaders(br *bufio.Reader) error {
	for {
		line, err := br.ReadString('\n')
		if err != nil {
			return err
		}
		if line == "\r\n" {
			return nil
		}
	}
}

// buildTLSStreamClient wires header-deadline + SSRF dialer + TLS for tests.
func buildTLSStreamClient(bound time.Duration, cTLS *tls.Config) *fasthttp.Client {
	base := &fasthttp.Client{ReadTimeout: bound, TLSConfig: cTLS}
	ConfigureDialer(base, true)
	return BuildStreamingClient(base)
}

// TestHeaderDeadlineTLS_FreshConnStallBounded is the gap investigation.
//
// Hypothesis: TLS handshake reads through headerDeadlineConn.Read disarm the
// bound before any HTTP response byte arrives. If fasthttp's subsequent HTTP
// request Write re-arms (as expected), the stall is bounded. If not, it hangs.
func TestHeaderDeadlineTLS_FreshConnStallBounded(t *testing.T) {
	const bound = 500 * time.Millisecond

	sCfg, cCfg := testTLSConfig(t)
	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatalf("listen: %v", err)
	}
	defer ln.Close()
	tlsLn := tls.NewListener(ln, sCfg)
	defer tlsLn.Close()

	go func() {
		for {
			conn, aerr := tlsLn.Accept()
			if aerr != nil {
				return
			}
			go func(c net.Conn) {
				defer c.Close()
				if consumeHTTPHeaders(bufio.NewReader(c)) != nil {
					return
				}
				// Gap scenario: handshake done, request consumed, now silent.
				time.Sleep(10 * time.Second)
			}(conn)
		}
	}()

	stream := buildTLSStreamClient(bound, cCfg)

	req := fasthttp.AcquireRequest()
	resp := fasthttp.AcquireResponse()
	defer fasthttp.ReleaseRequest(req)
	defer fasthttp.ReleaseResponse(resp)
	req.SetRequestURI("https://" + ln.Addr().String() + "/")
	req.Header.SetMethod(http.MethodGet)
	resp.StreamBody = true

	start := time.Now()
	doErr := stream.Do(req, resp)
	elapsed := time.Since(start)

	if doErr == nil {
		t.Fatal("expected timeout on silent TLS server; got nil")
	}
	// 5 idempotent retries × bound ≈ 2.5s; allow 6s for CI + TLS overhead.
	if elapsed >= 6*time.Second {
		t.Fatalf("elapsed %v >= 6s: header bound not effective over TLS (gap is real)", elapsed)
	}
	t.Logf("VERDICT: GAP NOT REAL — fresh TLS stall bounded in %v err=%v", elapsed, doErr)
}

// TestHeaderDeadlineTLS_SlowBodySurvives verifies a live slow SSE stream over
// TLS is NOT killed — the disarm-on-first-read clears the deadline so chunk
// gaps exceeding the bound are tolerated. Guards against an over-correction.
func TestHeaderDeadlineTLS_SlowBodySurvives(t *testing.T) {
	const bound = 500 * time.Millisecond
	const totalChunks = 4

	sCfg, cCfg := testTLSConfig(t)
	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatalf("listen: %v", err)
	}
	defer ln.Close()
	tlsLn := tls.NewListener(ln, sCfg)
	defer tlsLn.Close()

	go func() {
		for {
			conn, aerr := tlsLn.Accept()
			if aerr != nil {
				return
			}
			go func(c net.Conn) {
				defer c.Close()
				if consumeHTTPHeaders(bufio.NewReader(c)) != nil {
					return
				}
				fmt.Fprint(c, "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nConnection: close\r\n\r\n")
				for i := 0; i < totalChunks; i++ {
					fmt.Fprintf(c, "data: chunk-%d\n\n", i)
					time.Sleep(400 * time.Millisecond)
				}
			}(conn)
		}
	}()

	stream := buildTLSStreamClient(bound, cCfg)

	req := fasthttp.AcquireRequest()
	resp := fasthttp.AcquireResponse()
	defer fasthttp.ReleaseRequest(req)
	defer fasthttp.ReleaseResponse(resp)
	req.SetRequestURI("https://" + ln.Addr().String() + "/")
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
		t.Fatalf("scanner: %v (TLS body killed by leaked header deadline)", err)
	}
	if got != totalChunks {
		t.Errorf("chunks: got %d want %d (slow TLS body killed early)", got, totalChunks)
	}
	t.Logf("TLS slow body: %d/%d chunks survived", got, totalChunks)
}

// TestHeaderDeadlineTLS_KeepAliveReuseBounded verifies that on a pooled TLS
// conn, request 2 stalling before headers is bounded. Write on request 2
// re-arms the deadline even though the handshake and request 1's reads
// disarmed it.
func TestHeaderDeadlineTLS_KeepAliveReuseBounded(t *testing.T) {
	const bound = 500 * time.Millisecond

	sCfg, cCfg := testTLSConfig(t)
	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatalf("listen: %v", err)
	}
	defer ln.Close()
	tlsLn := tls.NewListener(ln, sCfg)
	defer tlsLn.Close()

	var reqCount int32

	go func() {
		for {
			conn, aerr := tlsLn.Accept()
			if aerr != nil {
				return
			}
			go func(c net.Conn) {
				defer c.Close()
				br := bufio.NewReader(c)
				for {
					if consumeHTTPHeaders(br) != nil {
						return
					}
					if atomic.AddInt32(&reqCount, 1) == 1 {
						body := "ok"
						fmt.Fprintf(c, "HTTP/1.1 200 OK\r\nContent-Length: %d\r\nContent-Type: text/plain\r\n\r\n%s", len(body), body)
						continue
					}
					// Request 2+: silent.
					time.Sleep(10 * time.Second)
					return
				}
			}(conn)
		}
	}()

	stream := buildTLSStreamClient(bound, cCfg)
	addr := ln.Addr().String()

	// Request 1: succeed, drain body so conn returns to pool.
	req1 := fasthttp.AcquireRequest()
	resp1 := fasthttp.AcquireResponse()
	req1.SetRequestURI("https://" + addr + "/")
	req1.Header.SetMethod(http.MethodGet)
	resp1.StreamBody = true
	if err := stream.Do(req1, resp1); err != nil {
		fasthttp.ReleaseRequest(req1)
		fasthttp.ReleaseResponse(resp1)
		t.Fatalf("req1: %v", err)
	}
	if bs := resp1.BodyStream(); bs != nil {
		_, _ = io.Copy(io.Discard, bs)
	}
	fasthttp.ReleaseRequest(req1)
	fasthttp.ReleaseResponse(resp1)

	// Request 2: stalls on reused TLS conn.
	req2 := fasthttp.AcquireRequest()
	resp2 := fasthttp.AcquireResponse()
	defer fasthttp.ReleaseRequest(req2)
	defer fasthttp.ReleaseResponse(resp2)
	req2.SetRequestURI("https://" + addr + "/")
	req2.Header.SetMethod(http.MethodGet)
	resp2.StreamBody = true

	start := time.Now()
	doErr := stream.Do(req2, resp2)
	elapsed := time.Since(start)

	if doErr == nil {
		t.Fatal("req2 nil error on silent reused TLS conn; expected timeout")
	}
	if elapsed >= 6*time.Second {
		t.Fatalf("req2 took %v >= 6s: header bound not re-armed on TLS keep-alive reuse", elapsed)
	}
	t.Logf("TLS keep-alive reuse: err=%v elapsed=%v", doErr, elapsed)
}
