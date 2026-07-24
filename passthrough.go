package llm

// Passthrough is the per-request escape hatch for sub-baseline / provider-
// specific fields. Body is shallow-merged onto (and overrides) the connector-
// built request body; Headers are merged onto request headers; Query is appended
// to the URL. Never emulated — forwarded verbatim.
type Passthrough struct {
	Body    map[string]any    `json:"body,omitempty"`
	Headers map[string]string `json:"headers,omitempty"`
	Query   map[string]string `json:"query,omitempty"`
}

// applyBodyPassthrough shallow-merges pt.Body onto body (consumer wins).
func applyBodyPassthrough(body map[string]any, pt *Passthrough) {
	if pt == nil {
		return
	}
	for k, v := range pt.Body {
		body[k] = v
	}
}

// mergeHeaders returns base with pt.Headers overlaid.
func mergeHeaders(base map[string]string, pt *Passthrough) map[string]string {
	out := map[string]string{}
	for k, v := range base {
		out[k] = v
	}
	if pt != nil {
		for k, v := range pt.Headers {
			out[k] = v
		}
	}
	return out
}
