package llm

// Nil/absence-tolerant helpers for navigating a decoded JSON tree on the error
// and raw paths.

func jsonGet(v any, key string) any {
	if m, ok := v.(map[string]any); ok {
		return m[key]
	}
	return nil
}

func jsonStr(v any) string {
	s, _ := v.(string)
	return s
}

func jsonNum(v any) float64 {
	f, _ := v.(float64)
	return f
}

func jsonSliceVal(v any) ([]any, bool) {
	s, ok := v.([]any)
	return s, ok
}

// extractErrorMessage pulls a human message out of a decoded vendor error body
// across the common shapes (string, {error:string}, {error:{message}},
// {message}, {errors:[{message}]}).
func extractErrorMessage(body any) string {
	switch b := body.(type) {
	case string:
		return b
	case map[string]any:
		if s, ok := b["error"].(string); ok {
			return s
		}
		if em, ok := b["error"].(map[string]any); ok {
			if s := jsonStr(em["message"]); s != "" {
				return s
			}
		}
		if s := jsonStr(b["message"]); s != "" {
			return s
		}
		if errs, ok := jsonSliceVal(b["errors"]); ok && len(errs) > 0 {
			if s := jsonStr(jsonGet(errs[0], "message")); s != "" {
				return s
			}
		}
	}
	return ""
}
