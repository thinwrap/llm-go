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
	data, _ := readBody(r)
	if len(data) != maxResponseBytes {
		t.Fatalf("readBody must cap at %d bytes, got %d", maxResponseBytes, len(data))
	}
}

// TestBodyReadErrorOn2xxIsTransportError: a 2xx whose body read fails midway
// (deadline, reset) is a transport failure carrying the cause, not an "empty
// body" with status 200.
func TestBodyReadErrorOn2xxIsTransportError(t *testing.T) {
	cfgs := []struct {
		id  ProviderID
		cfg ChatConfig
	}{
		{Anthropic, AnthropicConfig{APIKey: "k"}},
		{Bedrock, BedrockConfig{Region: "us-east-1", AccessKeyID: "a", SecretAccessKey: "s"}},
		{Gemini, GeminiConfig{APIKey: "k"}},
		{OpenAI, OpenAICompatConfig{APIKey: "k"}},
	}
	for _, c := range cfgs {
		r := &http.Response{StatusCode: 200, Header: http.Header{}, Body: &erroringReader{data: []byte(`{"cont`), err: context.DeadlineExceeded}}
		chat, _ := NewChat(c.id, c.cfg, WithHTTPClient(newFake(r)))
		_, err := chat.Complete(context.Background(), ChatInput{Model: "m", Messages: []ChatMessage{{Role: "user", Content: "x"}}})
		var ce *ConnectorError
		if !errors.As(err, &ce) || ce.StatusCode != nil || !errors.Is(err, context.DeadlineExceeded) {
			t.Fatalf("%s: expected transport error wrapping the read failure, got %#v", c.id, err)
		}
	}
	r := &http.Response{StatusCode: 200, Header: http.Header{}, Body: &erroringReader{data: []byte(`{"da`), err: context.DeadlineExceeded}}
	emb, _ := NewEmbeddings(OpenAI, OpenAICompatConfig{APIKey: "k"}, WithHTTPClient(newFake(r)))
	_, err := emb.Create(context.Background(), EmbeddingsInput{Model: "m", Input: []string{"a"}})
	if !errors.Is(err, context.DeadlineExceeded) {
		t.Fatalf("embeddings: expected transport error, got %v", err)
	}
}

// TestUnsupportedContentPartType: a zero-value or misspelled Type would go out
// as an empty image; it is refused before any request is sent.
func TestUnsupportedContentPartType(t *testing.T) {
	fake := newFake(resp(200, `{"choices":[{"message":{"content":"ok"}}]}`))
	chat, _ := NewChat(OpenAI, OpenAICompatConfig{APIKey: "k"}, WithHTTPClient(fake))
	in := ChatInput{Model: "m", Messages: []ChatMessage{{Role: "user", ContentParts: []ContentPart{{Text: "no type"}}}}}
	_, err := chat.Complete(context.Background(), in)
	var ce *ConnectorError
	if !errors.As(err, &ce) || ce.ProviderCode != CodeInvalidRequest || !strings.Contains(ce.Message, "messages[0]") {
		t.Fatalf("Complete: expected invalid_request, got %v", err)
	}
	if _, err := chat.Stream(context.Background(), in); !errors.As(err, &ce) || ce.ProviderCode != CodeInvalidRequest {
		t.Fatalf("Stream: expected invalid_request, got %v", err)
	}
	if len(fake.calls) != 0 {
		t.Fatalf("no request may be sent, got %d", len(fake.calls))
	}
}

// TestRetryAfterIgnoresEmptyAndNegative: an empty or negative Retry-After is
// not a delay of 0; the raw value is still surfaced when non-empty.
func TestRetryAfterIgnoresEmptyAndNegative(t *testing.T) {
	for _, v := range []string{"", "   ", "-5", "1.5", "soon"} {
		ce := vendorError(429, CodeRateLimited, "slow", nil, http.Header{"Retry-After": []string{v}})
		cause := ce.Cause.(map[string]any)
		if _, ok := cause["retryAfterSeconds"]; ok {
			t.Fatalf("Retry-After %q must not yield retryAfterSeconds: %v", v, cause)
		}
	}
	ce := vendorError(429, CodeRateLimited, "slow", nil, http.Header{"Retry-After": []string{"0"}})
	if ce.Cause.(map[string]any)["retryAfterSeconds"] != 0 {
		t.Fatalf("an explicit 0 is a real value: %v", ce.Cause)
	}
}

func streamCfgs() []struct {
	id  ProviderID
	cfg ChatConfig
} {
	return []struct {
		id  ProviderID
		cfg ChatConfig
	}{
		{Anthropic, AnthropicConfig{APIKey: "k"}},
		{Gemini, GeminiConfig{APIKey: "k"}},
		{OpenAI, OpenAICompatConfig{APIKey: "k"}},
	}
}

// TestStreamNonSSEBodyRejected: a 2xx labelled as anything but
// text/event-stream holds no events; it must fail, not end as an empty stream.
func TestStreamNonSSEBodyRejected(t *testing.T) {
	for _, c := range streamCfgs() {
		r := respH(200, `{"choices":[{"message":{"content":"full answer"}}]}`, "Content-Type", "application/json; charset=utf-8")
		chat, _ := NewChat(c.id, c.cfg, WithHTTPClient(newFake(r)))
		_, err := chat.Stream(context.Background(), ChatInput{Model: "m", Messages: []ChatMessage{{Role: "user", Content: "x"}}})
		var ce *ConnectorError
		if !errors.As(err, &ce) || ce.ProviderCode != CodeProviderUnavailable || ce.StatusCode == nil || *ce.StatusCode != 200 ||
			!strings.Contains(ce.Message, "(application/json)") {
			t.Fatalf("%s: expected provider_unavailable for a non-SSE body, got %v", c.id, err)
		}
	}
	// A missing Content-Type is let through (the end-of-stream rule still applies).
	r := resp(200, "data: {\"choices\":[{\"delta\":{\"content\":\"ok\"},\"finish_reason\":\"stop\"}]}\n\n")
	chat, _ := NewChat(OpenAI, OpenAICompatConfig{APIKey: "k"}, WithHTTPClient(newFake(r)))
	stream, err := chat.Stream(context.Background(), ChatInput{Model: "m", Messages: []ChatMessage{{Role: "user", Content: "x"}}})
	if err != nil {
		t.Fatal(err)
	}
	drain(t, stream)
}

// TestStreamEOFBeforeTerminalEvent: a stream cut off before its terminal event
// is a truncated answer, not a short one.
func TestStreamEOFBeforeTerminalEvent(t *testing.T) {
	truncated := map[ProviderID]string{
		Anthropic: `data: {"type":"message_start","message":{"usage":{"input_tokens":3}}}` + "\n\n" +
			`data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hel"}}` + "\n\n",
		Gemini: `data: {"candidates":[{"content":{"parts":[{"text":"Hel"}]}}]}` + "\n\n",
		OpenAI: `data: {"choices":[{"index":0,"delta":{"content":"Hel"}}]}` + "\n\n",
	}
	for _, c := range streamCfgs() {
		chat, _ := NewChat(c.id, c.cfg, WithHTTPClient(newFake(sseResp(truncated[c.id]))))
		stream, err := chat.Stream(context.Background(), ChatInput{Model: "m", Messages: []ChatMessage{{Role: "user", Content: "x"}}})
		if err != nil {
			t.Fatal(err)
		}
		var last error
		for {
			_, err := stream.Recv()
			if err != nil {
				last = err
				break
			}
		}
		stream.Close()
		var ce *ConnectorError
		if last == io.EOF || !errors.As(last, &ce) || ce.ProviderCode != CodeProviderUnavailable || ce.StatusCode != nil {
			t.Fatalf("%s: expected provider_unavailable on early EOF, got %v", c.id, last)
		}
	}
	// An empty 2xx stream body has no terminal event either.
	chat, _ := NewChat(OpenAI, OpenAICompatConfig{APIKey: "k"}, WithHTTPClient(newFake(sseResp(""))))
	stream, _ := chat.Stream(context.Background(), ChatInput{Model: "m", Messages: []ChatMessage{{Role: "user", Content: "x"}}})
	if _, err := stream.Recv(); err == io.EOF || err == nil {
		t.Fatalf("empty stream must not end cleanly, got %v", err)
	}
}

// TestStreamTerminalEvents: each connector's own terminal event ends cleanly.
func TestStreamTerminalEvents(t *testing.T) {
	cases := []struct {
		id  ProviderID
		cfg ChatConfig
		sse string
	}{
		// compat: [DONE] alone (no finish_reason) is a complete stream.
		{OpenAI, OpenAICompatConfig{APIKey: "k"}, `data: {"choices":[{"delta":{"content":"a"}}]}` + "\n\ndata: [DONE]\n\n"},
		// compat: a finish_reason without [DONE].
		{OpenAI, OpenAICompatConfig{APIKey: "k"}, `data: {"choices":[{"delta":{"content":"a"},"finish_reason":"stop"}]}` + "\n\n"},
		// Anthropic: message_stop alone.
		{Anthropic, AnthropicConfig{APIKey: "k"}, `data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"a"}}` + "\n\n" + `data: {"type":"message_stop"}` + "\n\n"},
		// Gemini: a prompt-level block (no candidate).
		{Gemini, GeminiConfig{APIKey: "k"}, `data: {"promptFeedback":{"blockReason":"SAFETY"}}` + "\n\n"},
	}
	for i, c := range cases {
		chat, _ := NewChat(c.id, c.cfg, WithHTTPClient(newFake(sseResp(c.sse))))
		stream, err := chat.Stream(context.Background(), ChatInput{Model: "m", Messages: []ChatMessage{{Role: "user", Content: "x"}}})
		if err != nil {
			t.Fatalf("case %d: %v", i, err)
		}
		drain(t, stream)
	}
}

// TestRetryAfterStrictForms: delay-seconds is digits only (RFC 9110), and only
// the three HTTP-date formats count; a past date means 0.
func TestRetryAfterStrictForms(t *testing.T) {
	for _, v := range []string{"+5", "1.5", "5s", "tomorrow", "2026-10-18T00:00:00Z"} {
		if _, ok := parseRetryAfter(v); ok {
			t.Fatalf("%q must be ignored", v)
		}
	}
	if n, ok := parseRetryAfter("007"); !ok || n != 7 {
		t.Fatalf("digits: %d %v", n, ok)
	}
	for _, past := range []string{"Sun, 06 Nov 1994 08:49:37 GMT", "Sunday, 06-Nov-94 08:49:37 GMT", "Sun Nov  6 08:49:37 1994"} {
		if n, ok := parseRetryAfter(past); !ok || n != 0 {
			t.Fatalf("past date %q: %d %v", past, n, ok)
		}
	}
}
