package llm

import (
	"context"
	"errors"
	"io"
	"net/http"
	"net/url"
	"strings"
	"testing"
)

// erroringReader yields data once, then fails every subsequent Read with err
// (never io.EOF) — simulating a mid-stream TCP drop / ctx cancellation.
type erroringReader struct {
	data []byte
	off  int
	err  error
}

func (r *erroringReader) Read(p []byte) (int, error) {
	if r.off < len(r.data) {
		n := copy(p, r.data[r.off:])
		r.off += n
		return n, nil
	}
	return 0, r.err
}

func (r *erroringReader) Close() error { return nil }

func sseErrResp(data string, err error) *http.Response {
	r := &http.Response{
		StatusCode: 200,
		Header:     http.Header{},
		Body:       &erroringReader{data: []byte(data), err: err},
	}
	r.Header.Set("Content-Type", "text/event-stream")
	return r
}

// TestStreamScannerErrorSurfaced: a read error AFTER a valid delta must surface as
// a *ConnectorError, not be swallowed into a clean io.EOF (truncation-as-success).
func TestStreamScannerErrorSurfaced(t *testing.T) {
	fake := newFake(sseErrResp("data: {\"choices\":[{\"delta\":{\"content\":\"Hel\"}}]}\n\n", errors.New("connection reset by peer")))
	chat, _ := NewChat(OpenAI, OpenAICompatConfig{APIKey: "k"}, WithHTTPClient(fake))
	stream, err := chat.Stream(context.Background(), ChatInput{Model: "m", Messages: []ChatMessage{{Role: "user", Content: "hi"}}})
	if err != nil {
		t.Fatal(err)
	}
	defer stream.Close()

	d, err := stream.Recv()
	if err != nil || d.ContentDelta != "Hel" {
		t.Fatalf("first delta should arrive cleanly: d=%+v err=%v", d, err)
	}

	_, err = stream.Recv()
	if err == io.EOF {
		t.Fatal("truncated stream surfaced as a clean io.EOF (swallowed scanner error)")
	}
	var ce *ConnectorError
	if !errors.As(err, &ce) {
		t.Fatalf("expected *ConnectorError, got %T: %v", err, err)
	}
	if ce.ProviderCode != CodeProviderUnavailable {
		t.Fatalf("expected provider_unavailable, got %s", ce.ProviderCode)
	}
}

// TestStreamScannerErrorPreservesUnwrap: the underlying cause (e.g. a cancelled
// context) stays reachable via errors.Is after being wrapped.
func TestStreamScannerErrorPreservesUnwrap(t *testing.T) {
	fake := newFake(sseErrResp("data: {\"choices\":[{\"delta\":{\"content\":\"x\"}}]}\n\n", context.Canceled))
	chat, _ := NewChat(OpenAI, OpenAICompatConfig{APIKey: "k"}, WithHTTPClient(fake))
	stream, err := chat.Stream(context.Background(), ChatInput{Model: "m", Messages: []ChatMessage{{Role: "user", Content: "hi"}}})
	if err != nil {
		t.Fatal(err)
	}
	defer stream.Close()

	if _, err := stream.Recv(); err != nil {
		t.Fatalf("first recv: %v", err)
	}
	_, err = stream.Recv()
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("ctx cancellation not preserved through ConnectorError.Unwrap: %v", err)
	}
}

// TestTransportErrorSanitizesCause: a *url.Error carrying a credential in its
// exported URL field must be redacted before it lands on Cause, while keeping the
// errors.Is / errors.As chain intact.
func TestTransportErrorSanitizesCause(t *testing.T) {
	inner := errors.New("dial tcp: connection refused")
	ue := &url.Error{Op: "Post", URL: "https://api.example.com/v1/chat?api_key=SECRETVALUE", Err: inner}
	ce := transportError(ue)

	if strings.Contains(ce.Message, "SECRETVALUE") {
		t.Fatalf("message leaks credential: %q", ce.Message)
	}
	var got *url.Error
	if !errors.As(ce, &got) {
		t.Fatalf("cause is not reachable as *url.Error via errors.As")
	}
	if strings.Contains(got.URL, "SECRETVALUE") {
		t.Fatalf("cause URL leaks credential: %q", got.URL)
	}
	if !strings.Contains(got.URL, "[REDACTED]") {
		t.Fatalf("cause URL was not redacted: %q", got.URL)
	}
	if !errors.Is(ce, inner) {
		t.Fatal("errors.Is chain to the inner error broken after sanitizing")
	}
}

// filler reads endless bytes without EOF, to exercise the readBody size cap.
type filler struct{}

func (filler) Read(p []byte) (int, error) {
	for i := range p {
		p[i] = 'a'
	}
	return len(p), nil
}

// TestReadBodyCapsResponse: an unbounded upstream body is truncated at the cap.
func TestReadBodyCapsResponse(t *testing.T) {
	r := &http.Response{StatusCode: 200, Header: http.Header{}, Body: io.NopCloser(filler{})}
	data := readBody(r)
	if len(data) != maxResponseBytes {
		t.Fatalf("readBody must cap at %d bytes, got %d", maxResponseBytes, len(data))
	}
}
