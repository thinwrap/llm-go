package llm

import (
	"bytes"
	"context"
	"io"
	"net/http"
	"regexp"
)

// baseConnector centralizes the HTTP seam: the injectable client, request
// building, and transport-error normalization. Vendor specifics stay in the
// per-provider connectors that embed it.
type baseConnector struct {
	httpClient HTTPClient
}

func (b baseConnector) client() HTTPClient {
	if b.httpClient != nil {
		return b.httpClient
	}
	return defaultHTTPClient
}

// sendPostJSON marshals body and POSTs it as application/json, returning the raw
// response (the caller reads/streams the body).
func (b baseConnector) sendPostJSON(ctx context.Context, url string, body any, headers map[string]string) (*http.Response, error) {
	data, err := marshalJSON(body)
	if err != nil {
		return nil, &ConnectorError{ProviderCode: CodeInvalidRequest, Message: "Failed to encode request body", ProviderMessage: "Failed to encode request body", Cause: err}
	}
	return b.post(ctx, url, "application/json", data, headers)
}

// post sends a POST with caller-serialized bytes and an explicit content type.
func (b baseConnector) post(ctx context.Context, url, contentType string, body []byte, headers map[string]string) (*http.Response, error) {
	if ctx == nil {
		ctx = context.Background()
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(body))
	if err != nil {
		return nil, transportError(err)
	}
	if contentType != "" {
		req.Header.Set("Content-Type", contentType)
	}
	for k, v := range headers {
		req.Header.Set(k, v)
	}
	resp, err := b.client().Do(req)
	if err != nil {
		return nil, transportError(err)
	}
	return resp, nil
}

// maxResponseBytes caps a non-streaming response body to guard against a
// malicious or broken upstream sending an unbounded body. It is deliberately
// generous (large tool-call / JSON payloads are well under it); the SSE streaming
// path is separately bounded per-event by maxSSEEventBytes.
const maxResponseBytes = 32 << 20 // 32 MiB

func readBody(resp *http.Response) []byte {
	if resp.Body == nil {
		return nil
	}
	defer resp.Body.Close()
	data, err := io.ReadAll(io.LimitReader(resp.Body, maxResponseBytes))
	if err != nil {
		return nil
	}
	return data
}

var (
	// credentialRe catches URL query-string forms: ...key=secret&...
	credentialRe = regexp.MustCompile(`(?i)((?:access_token|apikey|api_key|key|token|sig|signature|auth)=)[^&\s"']+`)
	// bearerRe catches an HTTP "Bearer <token>" value (run first so the token after
	// "Bearer" is redacted before authHeaderRe collapses the header name).
	bearerRe = regexp.MustCompile(`(?i)(bearer\s+)\S+`)
	// authHeaderRe catches header-shaped auth pairs a BYO client might embed in an
	// error string: "Authorization: <v>", "x-api-key: <v>", "api-key=<v>".
	authHeaderRe = regexp.MustCompile(`(?i)(authorization|x-api-key|api-key)(\s*[:=]\s*)\S+`)
)

func redactCredentials(s string) string {
	s = bearerRe.ReplaceAllString(s, `${1}[REDACTED]`)
	s = authHeaderRe.ReplaceAllString(s, `${1}${2}[REDACTED]`)
	s = credentialRe.ReplaceAllString(s, `${1}[REDACTED]`)
	return s
}
