package llm

import "net/http"

// HTTPClient is the bring-your-own HTTP seam, satisfied by *http.Client. Inject
// any implementation for tracing, retries, mocking, or proxying.
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
