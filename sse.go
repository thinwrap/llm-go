package llm

import (
	"bufio"
	"errors"
	"io"
	"net/http"
	"strings"
)

// maxSSEEventBytes bounds a single un-terminated SSE event to avoid unbounded
// buffer growth from a malicious/broken upstream.
const maxSSEEventBytes = 16 * 1024 * 1024

// errSSEEventTooLarge is surfaced (via scanner.err → transportError) when the
// accumulated data: payload of one un-terminated event exceeds maxSSEEventBytes.
var errSSEEventTooLarge = errors.New("SSE event exceeded size limit")

// sseScanner reads a Server-Sent-Events body and yields the JSON payload of each
// event's data: line(s). Skips comments/blank lines, stops at [DONE], tolerates
// both \n\n and \r\n\r\n boundaries.
type sseScanner struct {
	sc   *bufio.Scanner
	body io.ReadCloser
	done bool
	// sawDone records the OpenAI-style `[DONE]` sentinel, one of the ways a
	// stream proves it was not cut off.
	sawDone bool
	// err holds a non-EOF read failure (mid-stream TCP drop, ctx cancellation, or
	// bufio.ErrTooLong) so recv can surface it instead of a clean EOF.
	err error
}

func newSSEScanner(body io.ReadCloser) *sseScanner {
	if body == nil {
		// A contract-violating BYO client may hand back a 200 with a nil Body;
		// treat it as an already-complete (empty) stream rather than panicking in
		// bufio.NewScanner / the first Read. Mirrors readBody's nil-Body guard.
		return &sseScanner{done: true}
	}
	sc := bufio.NewScanner(body)
	sc.Buffer(make([]byte, 0, 64*1024), maxSSEEventBytes)
	return &sseScanner{sc: sc, body: body}
}

// next returns the next event's data payload; ok is false at end of stream.
func (s *sseScanner) next() (payload string, ok bool) {
	if s.done {
		return "", false
	}
	var dataParts []string
	// eventBytes tracks the running total of data: payload accumulated within the
	// CURRENT event; it resets each call (i.e. at every event boundary). The
	// bufio.Scanner caps a single line; this caps the whole multi-line event so a
	// never-terminated event cannot grow dataParts without bound.
	var eventBytes int
	for s.sc.Scan() {
		line := strings.TrimSuffix(s.sc.Text(), "\r")
		if line == "" {
			if len(dataParts) == 0 {
				continue // blank line between events
			}
			p := strings.Join(dataParts, "\n")
			if p == "[DONE]" {
				s.done = true
				s.sawDone = true
				return "", false
			}
			return p, true
		}
		if strings.HasPrefix(line, "data:") {
			part := strings.TrimPrefix(strings.TrimPrefix(line, "data:"), " ")
			eventBytes += len(part)
			if eventBytes > maxSSEEventBytes {
				// Un-terminated event over the cap: stop accumulating and surface a
				// transport error rather than OOM (mirrors the non-stream 32 MiB cap).
				s.done = true
				s.err = errSSEEventTooLarge
				return "", false
			}
			dataParts = append(dataParts, part)
		}
		// comments (":" prefix) and other fields are ignored
	}
	s.done = true
	// bufio.Scanner reports false on BOTH a clean EOF and a read error; only the
	// latter sets Err(). Capture it so a truncated stream is not mistaken for a
	// normal completion.
	if err := s.sc.Err(); err != nil {
		s.err = err
	}
	if s.err == nil && len(dataParts) > 0 {
		if p := strings.Join(dataParts, "\n"); p != "[DONE]" {
			return p, true
		}
		s.sawDone = true
	}
	return "", false
}

func (s *sseScanner) close() error {
	if s.body != nil {
		return s.body.Close()
	}
	return nil
}

// oneDelta adapts a single-delta mapper (Anthropic) to the multi-delta
// mapChunk shape newChatStream expects: it forwards an error, drops a nil delta,
// and otherwise wraps the delta in a one-element slice.
func oneDelta(d *ChatStreamDelta, err error) ([]*ChatStreamDelta, error) {
	if err != nil {
		return nil, err
	}
	if d == nil {
		return nil, nil
	}
	return []*ChatStreamDelta{d}, nil
}

// requireEventStream guards a 2xx streaming response before it is read as SSE.
// A body labelled as anything but text/event-stream (a JSON completion from a
// proxy that dropped `stream: true`, an HTML gateway page) holds no events, so
// reading it would end the stream with nothing and no error. A missing
// Content-Type is let through; the end-of-stream check still catches an empty
// body.
func requireEventStream(resp *http.Response) error {
	ct := strings.TrimSpace(resp.Header.Get("Content-Type"))
	if ct == "" || strings.Contains(strings.ToLower(ct), "text/event-stream") {
		return nil
	}
	if resp.Body != nil {
		resp.Body.Close()
	}
	mediaType := strings.TrimSpace(strings.SplitN(ct, ";", 2)[0])
	msg := "Provider returned a successful status with a non-SSE body (" + mediaType + ") for a streaming request"
	s := resp.StatusCode
	return &ConnectorError{StatusCode: &s, ProviderCode: CodeProviderUnavailable, Message: msg, ProviderMessage: msg}
}

// streamEndedEarlyError is the error for a stream whose body ended before the
// provider's terminal event. Every supported provider closes a complete stream
// with one, so an EOF before it is a truncated answer (dropped connection,
// proxy timeout), not a short one.
func streamEndedEarlyError() *ConnectorError {
	msg := "Stream ended before the provider sent its terminal event"
	return &ConnectorError{ProviderCode: CodeProviderUnavailable, Message: msg, ProviderMessage: msg}
}

// newChatStream wires a *ChatStream over an SSE response. mapChunk maps a decoded
// vendor chunk to zero or more normalized deltas (a single chunk may carry, e.g.,
// several tool-call fragments); return an empty slice to skip a chunk. A mapChunk
// error (e.g. a mid-stream vendor error frame) is surfaced to the consumer.
//
// A clean EOF counts as the end of the answer only after a terminal event: the
// `[DONE]` sentinel, a delta carrying a FinishReason, or — when terminal is
// non-nil — whatever else the connector reports through it. Otherwise the
// stream ends with streamEndedEarlyError.
func newChatStream(resp *http.Response, mapChunk func(chunk any) ([]*ChatStreamDelta, error), terminal func() bool) *ChatStream {
	scanner := newSSEScanner(resp.Body)
	sawFinish := false
	var pending []*ChatStreamDelta
	recv := func() (*ChatStreamDelta, error) {
		for {
			if len(pending) > 0 {
				d := pending[0]
				pending = pending[1:]
				return d, nil
			}
			payload, ok := scanner.next()
			if !ok {
				// A captured read error is a truncated stream, not a clean end.
				if scanner.err != nil {
					return nil, transportError(scanner.err)
				}
				if !scanner.sawDone && !sawFinish && (terminal == nil || !terminal()) {
					return nil, streamEndedEarlyError()
				}
				return nil, io.EOF
			}
			chunk := decodeAny([]byte(payload))
			if chunk == nil {
				continue
			}
			ds, err := mapChunk(chunk)
			if err != nil {
				return nil, err
			}
			for _, d := range ds {
				if d != nil {
					if d.FinishReason != "" {
						sawFinish = true
					}
					pending = append(pending, d)
				}
			}
		}
	}
	return &ChatStream{recv: recv, closer: scanner.close}
}
