package llm

import (
	"io"
	"net/http"
	"net/url"
	"strings"
)

type recordedRequest struct {
	Method string
	URL    *url.URL
	Header http.Header
	Body   string
}

// fakeClient returns queued responses in order (repeating the last) and records
// every request it saw.
type fakeClient struct {
	queue []*http.Response
	calls []recordedRequest
}

func newFake(responses ...*http.Response) *fakeClient {
	return &fakeClient{queue: responses}
}

func (f *fakeClient) Do(req *http.Request) (*http.Response, error) {
	var body string
	if req.Body != nil {
		b, _ := io.ReadAll(req.Body)
		req.Body.Close()
		body = string(b)
	}
	f.calls = append(f.calls, recordedRequest{Method: req.Method, URL: req.URL, Header: req.Header.Clone(), Body: body})
	idx := len(f.calls) - 1
	switch {
	case idx < len(f.queue):
		return f.queue[idx], nil
	case len(f.queue) > 0:
		return f.queue[len(f.queue)-1], nil
	default:
		return &http.Response{StatusCode: 200, Header: http.Header{}, Body: io.NopCloser(strings.NewReader(""))}, nil
	}
}

func (f *fakeClient) last() recordedRequest  { return f.calls[len(f.calls)-1] }
func (f *fakeClient) first() recordedRequest { return f.calls[0] }

func resp(status int, body string) *http.Response {
	return &http.Response{StatusCode: status, Header: http.Header{}, Body: io.NopCloser(strings.NewReader(body))}
}

func respH(status int, body string, kv ...string) *http.Response {
	r := resp(status, body)
	for i := 0; i+1 < len(kv); i += 2 {
		r.Header.Set(kv[i], kv[i+1])
	}
	return r
}

// sseResp builds a text/event-stream response from raw SSE text.
func sseResp(body string) *http.Response {
	r := resp(200, body)
	r.Header.Set("Content-Type", "text/event-stream")
	return r
}
