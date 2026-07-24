package llm

import (
	"context"
	"net/http"
	"strings"
)

// openAICompatConnector is the single connector shared by every first-class
// OpenAI-compatible provider. Provider differences are expressed by the spec,
// not by subclassing.
type openAICompatConnector struct {
	baseConnector
	spec    openAICompatSpec
	cfg     OpenAICompatConfig
	baseURL string
}

func newOpenAICompatChat(spec openAICompatSpec, cfg OpenAICompatConfig, o facadeOptions) (*openAICompatConnector, error) {
	base := cfg.BaseURL
	if base == "" {
		base = spec.defaultBaseURL
	}
	if base == "" {
		return nil, invalidReq("Provider '" + string(spec.id) + "' requires an explicit baseUrl in config")
	}
	return &openAICompatConnector{
		baseConnector: baseConnector{o.httpClient},
		spec:          spec,
		cfg:           cfg,
		baseURL:       strings.TrimRight(base, "/"),
	}, nil
}

func (c *openAICompatConnector) complete(ctx context.Context, in ChatInput) (*ChatResult, error) {
	url, body, headers := c.buildRequest(in, false)
	resp, err := c.sendPostJSON(ctx, url, body, headers)
	if err != nil {
		return nil, err
	}
	data := readBody(resp)
	raw := decodeAny(data)
	if !okStatus(resp.StatusCode) {
		return nil, compatVendorError(resp, raw)
	}
	if raw == nil {
		return nil, emptyBodyError(resp.StatusCode)
	}
	return c.parseResult(raw, in.Model), nil
}

func (c *openAICompatConnector) stream(ctx context.Context, in ChatInput) (*ChatStream, error) {
	url, body, headers := c.buildRequest(in, true)
	headers["Accept"] = "text/event-stream"
	resp, err := c.sendPostJSON(ctx, url, body, headers)
	if err != nil {
		return nil, err
	}
	if !okStatus(resp.StatusCode) {
		raw := decodeAny(readBody(resp))
		return nil, compatVendorError(resp, raw)
	}
	return newChatStream(resp, func(chunk any) ([]*ChatStreamDelta, error) {
		return c.mapStreamChunk(chunk)
	}), nil
}

func (c *openAICompatConnector) buildRequest(in ChatInput, stream bool) (string, map[string]any, map[string]string) {
	msgs := make([]any, 0, len(in.Messages))
	for _, m := range in.Messages {
		msgs = append(msgs, compatMapMessage(m))
	}
	body := map[string]any{"model": in.Model, "messages": msgs}

	toolsAllowed := c.spec.supportsTools == nil || *c.spec.supportsTools
	if toolsAllowed && len(in.Tools) > 0 {
		tools := make([]any, 0, len(in.Tools))
		for _, t := range in.Tools {
			fn := map[string]any{"name": t.Name}
			if t.Description != "" {
				fn["description"] = t.Description
			}
			if t.Parameters != nil {
				fn["parameters"] = t.Parameters
			}
			tools = append(tools, map[string]any{"type": "function", "function": fn})
		}
		body["tools"] = tools
	}
	if toolsAllowed && in.ToolChoice != nil {
		body["tool_choice"] = compatToolChoice(in.ToolChoice)
	}
	if in.ResponseFormat != nil {
		body["response_format"] = compatResponseFormat(in.ResponseFormat)
	}
	if in.Temperature != nil {
		body["temperature"] = *in.Temperature
	}
	if in.TopP != nil {
		body["top_p"] = *in.TopP
	}
	if in.MaxOutputTokens != nil {
		field := c.spec.maxTokensField
		if field == "" {
			field = "max_tokens"
		}
		body[field] = *in.MaxOutputTokens
	}
	if len(in.Stop) > 0 {
		if len(in.Stop) == 1 {
			body["stop"] = in.Stop[0]
		} else {
			body["stop"] = in.Stop
		}
	}
	if in.Reasoning != nil {
		body["reasoning_effort"] = in.Reasoning.Effort
	}
	if stream {
		body["stream"] = true
		if !c.spec.strictParams {
			body["stream_options"] = map[string]any{"include_usage": true}
		}
	}
	applyBodyPassthrough(body, in.Passthrough)

	headers := map[string]string{}
	for k, v := range c.spec.buildAuthHeaders(c.cfg) {
		headers[k] = v
	}
	for k, v := range c.cfg.Headers {
		headers[k] = v
	}
	headers = mergeHeaders(headers, in.Passthrough)

	url := c.baseURL + "/chat/completions"
	if in.Passthrough != nil && len(in.Passthrough.Query) > 0 {
		url += "?" + buildCanonicalQuery(in.Passthrough.Query)
	}
	return url, body, headers
}

func (c *openAICompatConnector) parseResult(raw any, requestedModel string) *ChatResult {
	choices, _ := jsonSliceVal(jsonGet(raw, "choices"))
	var choice any
	if len(choices) > 0 {
		choice = choices[0]
	}
	msg := jsonGet(choice, "message")
	res := &ChatResult{
		Message:      AssistantMessage{Role: "assistant", Content: jsonStr(jsonGet(msg, "content"))},
		FinishReason: compatFinishReason(jsonStr(jsonGet(choice, "finish_reason"))),
		Usage:        compatUsage(jsonGet(raw, "usage")),
		Model:        firstStr(jsonStr(jsonGet(raw, "model")), requestedModel),
		Raw:          raw,
	}
	if tcs, ok := jsonSliceVal(jsonGet(msg, "tool_calls")); ok {
		for _, tc := range tcs {
			res.Message.ToolCalls = append(res.Message.ToolCalls, compatToolCall(tc))
		}
	}
	return res
}

func (c *openAICompatConnector) mapStreamChunk(chunk any) ([]*ChatStreamDelta, error) {
	// A mid-stream {"error":{...}} frame must abort with a *ConnectorError rather
	// than be silently dropped (which would truncate the result cleanly).
	if jsonGet(chunk, "error") != nil {
		return nil, streamFrameError(CodeProviderUnavailable, extractErrorMessage(chunk), chunk)
	}

	choices, _ := jsonSliceVal(jsonGet(chunk, "choices"))
	var choice any
	if len(choices) > 0 {
		choice = choices[0]
	}
	delta := jsonGet(choice, "delta")

	var out []*ChatStreamDelta
	if s := jsonStr(jsonGet(delta, "content")); s != "" {
		out = append(out, &ChatStreamDelta{ContentDelta: s, Raw: chunk})
	}
	// Emit one delta per tool-call entry: a single chunk may carry several.
	if tcs, ok := jsonSliceVal(jsonGet(delta, "tool_calls")); ok {
		for _, tc := range tcs {
			fn := jsonGet(tc, "function")
			out = append(out, &ChatStreamDelta{
				ToolCallDelta: &ToolCallDelta{
					Index:          int(jsonNum(jsonGet(tc, "index"))),
					ID:             jsonStr(jsonGet(tc, "id")),
					FunctionName:   jsonStr(jsonGet(fn, "name")),
					ArgumentsDelta: jsonStr(jsonGet(fn, "arguments")),
				},
				Raw: chunk,
			})
		}
	}

	// finish_reason + usage ride on a trailing delta.
	var tail *ChatStreamDelta
	if fr := jsonStr(jsonGet(choice, "finish_reason")); fr != "" {
		tail = &ChatStreamDelta{Raw: chunk, FinishReason: compatFinishReason(fr)}
	}
	usageRaw := jsonGet(chunk, "usage")
	if usageRaw == nil && c.spec.streamUsagePath == "x_groq" {
		usageRaw = jsonGet(jsonGet(chunk, "x_groq"), "usage")
	}
	if usageRaw != nil {
		if tail == nil {
			tail = &ChatStreamDelta{Raw: chunk}
		}
		u := compatUsage(usageRaw)
		tail.Usage = &u
	}
	if tail != nil {
		out = append(out, tail)
	}

	return out, nil
}

func compatVendorError(resp *http.Response, raw any) *ConnectorError {
	msg := extractErrorMessage(raw)
	return vendorError(resp.StatusCode, compatMapCode(resp.StatusCode, raw), msg, raw, resp.Header)
}

// ---- module-private mappers ----

func compatMapMessage(m ChatMessage) map[string]any {
	out := map[string]any{"role": m.Role, "content": compatContent(m)}
	if m.Name != "" {
		out["name"] = m.Name
	}
	if len(m.ToolCalls) > 0 {
		tcs := make([]any, 0, len(m.ToolCalls))
		for _, tc := range m.ToolCalls {
			tcs = append(tcs, map[string]any{
				"id": tc.ID, "type": "function",
				"function": map[string]any{"name": tc.Name, "arguments": tc.Arguments},
			})
		}
		out["tool_calls"] = tcs
	}
	if m.ToolCallID != "" {
		out["tool_call_id"] = m.ToolCallID
	}
	return out
}

func compatContent(m ChatMessage) any {
	if len(m.ContentParts) > 0 {
		parts := make([]any, 0, len(m.ContentParts))
		for _, p := range m.ContentParts {
			if p.Type == "text" {
				parts = append(parts, map[string]any{"type": "text", "text": p.Text})
			} else {
				parts = append(parts, map[string]any{
					"type":      "image_url",
					"image_url": map[string]any{"url": "data:" + p.MediaType + ";base64," + p.Base64},
				})
			}
		}
		return parts
	}
	if m.Content != "" {
		return m.Content
	}
	if m.Role == "assistant" && len(m.ToolCalls) > 0 {
		return nil
	}
	return ""
}

func compatToolCall(tc any) ToolCall {
	fn := jsonGet(tc, "function")
	return ToolCall{ID: jsonStr(jsonGet(tc, "id")), Name: jsonStr(jsonGet(fn, "name")), Arguments: jsonStr(jsonGet(fn, "arguments"))}
}

func compatToolChoice(tc *ToolChoice) any {
	switch tc.Mode {
	case "function":
		return map[string]any{"type": "function", "function": map[string]any{"name": tc.FunctionName}}
	default:
		return tc.Mode
	}
}

func compatResponseFormat(rf *ResponseFormat) map[string]any {
	if rf.Type == "json_schema" {
		return map[string]any{
			"type":        "json_schema",
			"json_schema": map[string]any{"name": rf.JSONSchemaName, "schema": rf.JSONSchema},
		}
	}
	return map[string]any{"type": rf.Type}
}

func compatFinishReason(fr string) FinishReason {
	switch fr {
	case "stop", "eos": // Together AI reports natural end-of-sequence as "eos"
		return "stop"
	case "length":
		return "length"
	case "tool_calls", "function_call":
		return "tool_calls"
	case "content_filter":
		return "content_filter"
	default:
		return "unknown"
	}
}

func compatUsage(u any) Usage {
	in := int(jsonNum(jsonGet(u, "prompt_tokens")))
	out := int(jsonNum(jsonGet(u, "completion_tokens")))
	total := int(jsonNum(jsonGet(u, "total_tokens")))
	if total == 0 {
		total = in + out
	}
	return Usage{InputTokens: in, OutputTokens: out, TotalTokens: total}
}

func compatMapCode(status int, body any) ProviderCode {
	switch {
	case status == 401 || status == 403:
		return CodeAuthFailed
	case status == 429:
		return CodeRateLimited
	case status >= 500:
		return CodeProviderUnavailable
	}
	if status == 400 || status == 404 || status == 413 || status == 422 {
		// 413 (Payload Too Large) is a client-side request problem — a too-big
		// prompt/body — not an opaque failure; it must not fall through to Unknown.
		msg := strings.ToLower(extractErrorMessage(body))
		if strings.Contains(msg, "context length") || strings.Contains(msg, "maximum context") || strings.Contains(msg, "context_length") {
			return CodeContextLengthExceeded
		}
		if strings.Contains(msg, "content") && strings.Contains(msg, "filter") {
			return CodeContentFiltered
		}
		return CodeInvalidRequest
	}
	return CodeUnknown
}

func firstStr(a, b string) string {
	if a != "" {
		return a
	}
	return b
}
