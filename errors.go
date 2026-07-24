package llm

import (
	"errors"
	"net/http"
	"net/url"
	"strconv"
	"strings"
	"time"
)

// ProviderCode is the normalized, cross-provider error classification surfaced
// on every *ConnectorError. The values are byte-identical across the thinwrap
// llm siblings (TypeScript, PHP, Go).
type ProviderCode string

const (
	CodeRateLimited           ProviderCode = "rate_limited"
	CodeAuthFailed            ProviderCode = "auth_failed"
	CodeProviderUnavailable   ProviderCode = "provider_unavailable"
	CodeInvalidRequest        ProviderCode = "invalid_request"
	CodeContextLengthExceeded ProviderCode = "context_length_exceeded"
	CodeContentFiltered       ProviderCode = "content_filtered"
	CodeUnknown               ProviderCode = "unknown"
)

// ConnectorError is the single typed error every operation can return. Retrieve
// it with errors.As:
//
//	var ce *llm.ConnectorError
//	if errors.As(err, &ce) { ... }
//
// There is deliberately no top-level RetryAfterSeconds field: for a vendor HTTP
// error the raw Retry-After header rides in Cause["retryAfter"] and its parsed
// seconds in Cause["retryAfterSeconds"].
type ConnectorError struct {
	StatusCode      *int
	ProviderCode    ProviderCode
	ProviderMessage string
	Message         string
	Cause           any
}

func (e *ConnectorError) Error() string {
	if e.Message != "" {
		return e.Message
	}
	if e.ProviderMessage != "" {
		return e.ProviderMessage
	}
	return "Connector error"
}

// Unwrap exposes an underlying error cause (e.g. a transport error).
func (e *ConnectorError) Unwrap() error {
	if err, ok := e.Cause.(error); ok {
		return err
	}
	return nil
}

func invalidReq(msg string) *ConnectorError {
	return &ConnectorError{ProviderCode: CodeInvalidRequest, Message: msg, ProviderMessage: msg}
}

func transportError(err error) *ConnectorError {
	msg := redactCredentials(err.Error())
	return &ConnectorError{ProviderCode: CodeProviderUnavailable, Message: msg, ProviderMessage: msg, Cause: sanitizeCause(err)}
}

// emptyBodyError is returned when a 2xx response carries no decodable JSON body
// (a proxy/captive-portal HTML page, or a truncated/aborted read that readBody
// surfaced as nil). Without this guard the connector would parse nil into an
// empty "successful" result — silent data loss indistinguishable from a real
// empty completion.
func emptyBodyError(status int) *ConnectorError {
	s := status
	msg := "provider returned a successful status with an empty or non-JSON body"
	return &ConnectorError{StatusCode: &s, ProviderCode: CodeProviderUnavailable, Message: msg, ProviderMessage: msg}
}

// sanitizeCause redacts credentials embedded in a transport error before it is
// stored (and JSON-serialized) on ConnectorError.Cause. The stdlib's *url.Error
// carries the full request URL in an exported field — a credential passed via
// Passthrough.Query would otherwise leak. The rebuilt *url.Error preserves the
// errors.Is / errors.As / Unwrap chain.
func sanitizeCause(err error) error {
	var ue *url.Error
	if errors.As(err, &ue) {
		return &url.Error{Op: ue.Op, URL: redactCredentials(ue.URL), Err: ue.Err}
	}
	return err
}

// streamFrameError builds a *ConnectorError from a decoded mid-stream vendor
// error frame. SSE error frames carry no HTTP status, so StatusCode stays nil and
// the decoded frame rides in Cause["raw"] — matching the non-stream error shape.
func streamFrameError(code ProviderCode, message string, raw any) *ConnectorError {
	if message == "" {
		message = "provider returned an error mid-stream"
	}
	if code == "" {
		code = CodeProviderUnavailable
	}
	return &ConnectorError{ProviderCode: code, ProviderMessage: message, Message: message, Cause: map[string]any{"raw": raw}}
}

// vendorError builds a *ConnectorError for a non-2xx HTTP response. The
// classification and message are computed per-connector; the shared Retry-After
// surfacing lives here (raw header → Cause["retryAfter"], parsed seconds →
// Cause["retryAfterSeconds"]).
func vendorError(status int, code ProviderCode, providerMessage string, rawBody any, headers http.Header) *ConnectorError {
	cause := map[string]any{"raw": rawBody}
	if headers != nil {
		if ra := headers.Get("Retry-After"); ra != "" {
			cause["retryAfter"] = ra
			if secs, ok := parseRetryAfter(ra); ok {
				cause["retryAfterSeconds"] = secs
			}
		}
	}
	sc := status
	msg := providerMessage
	if msg == "" {
		msg = "HTTP " + strconv.Itoa(status)
	}
	return &ConnectorError{StatusCode: &sc, ProviderCode: code, ProviderMessage: msg, Message: msg, Cause: cause}
}

func parseRetryAfter(v string) (int, bool) {
	s := strings.TrimSpace(v)
	if s == "" {
		return 0, false
	}
	if secs, err := strconv.Atoi(s); err == nil {
		if secs < 0 {
			return 0, false
		}
		return secs, true
	}
	if t, err := http.ParseTime(s); err == nil {
		delta := int(time.Until(t).Round(time.Second) / time.Second)
		if delta < 0 {
			delta = 0
		}
		return delta, true
	}
	return 0, false
}
