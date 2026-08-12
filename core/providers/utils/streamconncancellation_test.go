package utils

import (
	"context"
	"io"
	"runtime"
	"strconv"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/maximhq/bifrost/core/schemas"
)

// goroutineID parses the current goroutine's numeric ID from its stack header
// ("goroutine 123 [running]:"). Used to assert CloseWithError runs on the reader
// goroutine — a revert to third-goroutine close fails this even without -race.
func goroutineID() int64 {
	var buf [64]byte
	n := runtime.Stack(buf[:], false)
	fields := strings.Fields(string(buf[:n]))
	id, _ := strconv.ParseInt(fields[1], 10, 64)
	return id
}

// deadlineParkConn is the fake conn handle stashed under
// BifrostContextKeyStreamConnHandle. SetReadDeadline with a non-zero time
// unblocks the parked body read by closing the shared channel — mirroring how a
// past read deadline wakes headerDeadlineConn's blocked conn.Read WITHOUT firing
// fasthttp's pool-releasing close.
type deadlineParkConn struct {
	parked    chan struct{}
	closeOnce *sync.Once
}

func (c *deadlineParkConn) SetReadDeadline(t time.Time) error {
	if !t.IsZero() {
		c.closeOnce.Do(func() { close(c.parked) })
	}
	return nil
}

// readAfterReleaseBody mirrors fasthttp's requestStream (streaming.go). Read
// reads chunkLeft at line 45 BEFORE parking at line 46, then mutates it at line
// 48 after unblock — the pre-park read is the access that races a third
// goroutine's zeroing (a channel close→receive edge would order a post-park-only
// access, hiding the bug, so the faithful line-45 read is load-bearing).
// CloseWithError zeroes chunkLeft (releaseRequestStream) then unblocks
// (write-then-unblock, matching client.go's closeFunc order) and records the
// goroutine that ran it.
type readAfterReleaseBody struct {
	parked     chan struct{}
	closeOnce  *sync.Once
	chunkLeft  int
	totalRead  int
	closeGID   int64
	closeGuard sync.Once
}

func (b *readAfterReleaseBody) Read(p []byte) (int, error) {
	bytesToRead := b.chunkLeft // streaming.go:45 — read chunkLeft before the blocking read
	if bytesToRead > len(p) {
		bytesToRead = len(p)
	}
	<-b.parked                 // streaming.go:46 — park inside conn.Read
	b.totalRead += bytesToRead // streaming.go:47
	b.chunkLeft -= bytesToRead // streaming.go:48
	return bytesToRead, io.EOF
}

func (b *readAfterReleaseBody) CloseWithError(error) error {
	b.closeGuard.Do(func() {
		b.closeGID = goroutineID()
		b.chunkLeft = 0                            // releaseRequestStream zeroing
		b.closeOnce.Do(func() { close(b.parked) }) // unblock (no-op if deadline already closed it)
	})
	return nil
}

// TestSetupStreamCancellation_PassthroughDeadlineReleasesOnReader is the
// revert-proof regression for the read-after-release race. With a conn handle
// present, the shutdown arm must unblock the parked read via SetReadDeadline (not
// close), so the SINGLE release (CloseWithError) runs on the reader goroutine.
//
// FIXED code: only the reader touches chunkLeft — clean under -race, and
// closeGID == reader GID. Revert the shutdown arm to closeBodyStream and the
// cancel goroutine's chunkLeft=0 races the reader's line-45 read (fails -race)
// AND closeGID != reader GID (fails the assertion even without -race).
func TestSetupStreamCancellation_PassthroughDeadlineReleasesOnReader(t *testing.T) {
	parked := make(chan struct{})
	closeOnce := &sync.Once{}
	conn := &deadlineParkConn{parked: parked, closeOnce: closeOnce}
	body := &readAfterReleaseBody{parked: parked, closeOnce: closeOnce, chunkLeft: 4}

	shutdownDone := make(chan struct{})
	ctx := schemas.NewBifrostContext(context.Background(), time.Time{})
	ctx.SetValue(schemas.BifrostContextKeyShutdownDone, (<-chan struct{})(shutdownDone))
	ctx.SetValue(schemas.BifrostContextKeyStreamConnHandle, conn)

	stop := SetupStreamCancellation(ctx, body, getLogger())

	readerGIDCh := make(chan int64, 1)
	readerDone := make(chan struct{})
	go func() {
		readerGIDCh <- goroutineID()
		buf := make([]byte, 2)
		_, _ = body.Read(buf) // parks until the shutdown arm sets a past deadline
		// Simulate the reader's defer chain (ReleaseStreamingResponse): the SINGLE
		// release runs here, on the reader goroutine.
		_ = body.CloseWithError(nil)
		close(readerDone)
	}()

	readerGID := <-readerGIDCh
	close(shutdownDone) // fire shutdown → cancel goroutine unblocks via deadline

	select {
	case <-readerDone:
	case <-time.After(2 * time.Second):
		t.Fatal("reader did not unblock after shutdown set a read deadline")
	}
	stop()

	if body.closeGID != readerGID {
		t.Fatalf("CloseWithError ran on goroutine %d, want reader goroutine %d — a revert to third-goroutine close", body.closeGID, readerGID)
	}
	if body.chunkLeft != 0 {
		t.Fatalf("chunkLeft = %d, want 0 — release did not run on the reader", body.chunkLeft)
	}
}

// SHARED-CONTEXT SAFETY. BifrostContext values are mutable and the public API lets one
// context drive two concurrent passthrough streams: the second stream's StreamPassthrough
// overwrites BifrostContextKeyStreamConnHandle (verified end-to-end — the key held a
// different pointer after the second call). The idle timer used to re-read that key when it
// fired, so stream A's timeout moved the read deadline on stream B's LIVE connection,
// aborting an unrelated request while A stayed parked. SetupStreamCancellation never had the
// bug because it captures the handle once at setup; this pins the timer to the same rule.
//
// Discriminator (revert-proof): make connHandle() re-read the context key and this FAILS —
// the timer unblocks the second stream's conn instead of its own.
func TestIdleTimeoutReader_ConnHandlePinnedAtSetup(t *testing.T) {
	ctx := schemas.NewBifrostContext(context.Background(), schemas.NoDeadline)

	mine := &deadlineParkConn{parked: make(chan struct{}), closeOnce: &sync.Once{}}
	ctx.SetValue(schemas.BifrostContextKeyStreamConnHandle, streamReadDeadliner(mine))

	body := &blockingBody{parked: make(chan struct{})}
	reader, stop := NewIdleTimeoutReader(body, body, 20*time.Millisecond, ctx)
	defer stop()

	// A SECOND stream on the SAME context replaces the handle after this reader was built.
	other := &deadlineParkConn{parked: make(chan struct{}), closeOnce: &sync.Once{}}
	ctx.SetValue(schemas.BifrostContextKeyStreamConnHandle, streamReadDeadliner(other))

	done := make(chan struct{})
	go func() {
		defer close(done)
		buf := make([]byte, 8)
		_, _ = reader.Read(buf)
	}()

	select {
	case <-mine.parked:
	case <-time.After(2 * time.Second):
		t.Fatalf("idle timer did not unblock THIS stream's conn — it targeted the handle the other stream stashed, so an unrelated live request was aborted while this one stayed parked")
	}

	select {
	case <-other.parked:
		t.Fatalf("idle timer set a read deadline on the OTHER stream's connection — a closer must never touch a conn it does not own")
	default:
	}

	close(body.parked)
	<-done
}

// blockingBody parks until released, so the idle timer is guaranteed to fire mid-read.
type blockingBody struct{ parked chan struct{} }

func (b *blockingBody) Read(p []byte) (int, error) {
	<-b.parked
	return 0, io.EOF
}
