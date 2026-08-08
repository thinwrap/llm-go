package llm

import "net/http"

// HTTPClient is the bring-your-own HTTP seam, satisfied by *http.Client. Inject
// any implementation for tracing, retries, mocking, or proxying.
//
// Contract: a non-2xx must be RETURNED as a *http.Response, not reported as an
// error — the same rule *http.Client follows. Every connector's status mapping
// (429 -> rate_limited, 401 -> auth_failed, ...) reads the response, so an
// implementation that converts a non-2xx into an error collapses every provider
// error into provider_unavailable with no status. Return errors only for genuine
// transport failures: DNS, connection, TLS, timeout.
type HTTPClient interface {
	Do(req *http.Request) (*http.Response, error)
}

// defaultHTTPClient is used when the consumer injects none. It never follows
// redirects (a 3xx surfaces as an error rather than re-sending auth headers).
var defaultHTTPClient HTTPClient = &http.Client{
	CheckRedirect: func(req *http.Request, via []*http.Request) error {
		return http.ErrUseLastResponse
	},
}
