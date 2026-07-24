package llm

// Option configures a facade constructor.
type Option func(*facadeOptions)

type facadeOptions struct {
	httpClient HTTPClient
}

// WithHTTPClient injects a custom HTTP client (see HTTPClient). Without it, a
// non-redirect-following default is used.
func WithHTTPClient(c HTTPClient) Option {
	return func(o *facadeOptions) { o.httpClient = c }
}

func applyOptions(opts []Option) facadeOptions {
	var o facadeOptions
	for _, fn := range opts {
		fn(&o)
	}
	return o
}
