package llm

import (
	"bytes"
	"encoding/json"
)

// marshalJSON encodes v without HTML escaping and without the trailing newline
// the encoder appends (matching JSON.stringify).
func marshalJSON(v any) ([]byte, error) {
	var buf bytes.Buffer
	enc := json.NewEncoder(&buf)
	enc.SetEscapeHTML(false)
	if err := enc.Encode(v); err != nil {
		return nil, err
	}
	b := buf.Bytes()
	if n := len(b); n > 0 && b[n-1] == '\n' {
		b = b[:n-1]
	}
	return b, nil
}

// decodeAny decodes arbitrary JSON into an any tree, returning nil on empty or
// malformed input.
func decodeAny(data []byte) any {
	if len(bytes.TrimSpace(data)) == 0 {
		return nil
	}
	var v any
	if err := json.Unmarshal(data, &v); err != nil {
		return nil
	}
	return v
}

func okStatus(status int) bool { return status >= 200 && status < 300 }
