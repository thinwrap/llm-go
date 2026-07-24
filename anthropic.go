package llm

import (
	"context"
	"net/http"
	"strconv"
	"strings"
)

// Native adapter for the Anthropic Messages API (POST /v1/messages). Anthropic is
// structurally different from OpenAI Chat Completions — different auth, a
// top-level system, typed content blocks, tool results carried inside a user
// turn, stop_sequences, and Anthropic-shaped SSE events — so this is a full
// translation layer, not an OpenAI passthrough. It emits the identical
// normalized ChatResult / ChatStreamDelta / ConnectorError as every other
// connector, so it stays drop-in swappable via the Chat facade.
//
// Not normalized here (surfaced via Raw or Passthrough, never emulated):
// ResponseFormat (Anthropic has no response_format — use tool-based structured
// output via Passthrough); reasoning CoT output (thinking blocks land in Raw);
// prompt caching (cache_control via Passthrough, cached-token counts in Raw).

const (
	anthropicDefaultBaseURL   = "https://api.anthropic.com/v1"
	anthropicDefaultVersion   = "2023-06-01"
	anthropicDefaultMaxTokens = 4096
)

func init() { registerNativeChat(Anthropic, newAnthropicChat) }

func newAnthropicChat(cfg ChatConfig, o facadeOptions) (chatConnector, error) {
	c, ok := cfg.(AnthropicConfig)
	if !ok {
		return nil, invalidReq("anthropic requires an AnthropicConfig")
	}
	base := c.BaseURL
	if base == "" {
		base = anthropicDefaultBaseURL
	}
	return &anthropicConnector{
		baseConnector: baseConnector{o.httpClient},
		cfg:           c,
		baseURL:       strings.TrimRight(base, "/"),
	}, nil
}

type anthropicConnector struct {
	baseConnector
	cfg     AnthropicConfig
	baseURL string
}

func (c *anthropicConnector) complete(ctx context.Context, in ChatInput) (*ChatResult, error) {
	url, body, headers, err := c.buildRequest(in, false)
	if err != nil {
		return nil, err
	}
	resp, err := c.sendPostJSON(ctx, url, body, headers)
	if err != nil {
		return nil, err
	}
	raw := decodeAny(readBody(resp))
	if !okStatus(resp.StatusCode) {
		return nil, anthropicVendorError(resp, raw)
	}
	if raw == nil {
		return nil, emptyBodyError(resp.StatusCode)
	}
	return c.parseResult(raw, in.Model), nil
}

func (c *anthropicConnector) stream(ctx context.Context, in ChatInput) (*ChatStream, error) {
	url, body, headers, err := c.buildRequest(in, true)
	if err != nil {
		return nil, err
	}
	headers["Accept"] = "text/event-stream"
	resp, err := c.sendPostJSON(ctx, url, body, headers)
	if err != nil {
		return nil, err
	}
	if !okStatus(resp.StatusCode) {
		raw := decodeAny(readBody(resp))
		return nil, anthropicVendorError(resp, raw)
	}
	// inputTokens is captured across events: message_start reports it, message_delta
	// reads it back when it reports output_tokens (Anthropic's SSE is stateful).
	inputTokens := 0
	nextToolIndex := 0
	toolIndexByBlock := map[int]int{}
	return newChatStream(resp, func(chunk any) ([]*ChatStreamDelta, error) {
		return oneDelta(anthropicMapStreamEvent(chunk, &inputTokens, &nextToolIndex, toolIndexByBlock))
	}), nil
}

func (c *anthropicConnector) authHeaders(in ChatInput) map[string]string {
	version := c.cfg.AnthropicVersion
	if version == "" {
		version = anthropicDefaultVersion
	}
	headers := map[string]string{
		"x-api-key":         c.cfg.APIKey,
		"anthropic-version": version,
	}
	for k, v := range c.cfg.Headers {
		headers[k] = v
	}
	return mergeHeaders(headers, in.Passthrough)
}

func (c *anthropicConnector) buildRequest(in ChatInput, stream bool) (string, map[string]any, map[string]string, error) {
	system, messages := anthropicSplitMessages(in.Messages)

	var thinking map[string]any
	budget := 0
	if in.Reasoning != nil {
		budget = anthropicEffortBudget(in.Reasoning.Effort)
		thinking = map[string]any{"type": "enabled", "budget_tokens": budget}
	}

	maxTokens := anthropicDefaultMaxTokens
	if c.cfg.DefaultMaxTokens > 0 {
		maxTokens = c.cfg.DefaultMaxTokens
	}
	if in.MaxOutputTokens != nil {
		maxTokens = *in.MaxOutputTokens
	}
	if thinking != nil && maxTokens <= budget {
		// Anthropic requires max_tokens > thinking.budget_tokens (max_tokens is the
		// TOTAL, thinking included). If the CALLER set an explicit MaxOutputTokens
		// below the budget, silently inflating it would spend past their cap — return
		// an error so they lower reasoning.Effort or raise the cap. Only a
		// non-explicit default is inflated to keep the request valid.
		if in.MaxOutputTokens != nil {
			return "", nil, nil, invalidReq("max_output_tokens (" + strconv.Itoa(maxTokens) +
				") must be greater than the reasoning budget (" + strconv.Itoa(budget) +
				") for extended thinking — raise max_output_tokens or lower reasoning.effort")
		}
		maxTokens = budget + anthropicDefaultMaxTokens
	}

	body := map[string]any{
		"model":      in.Model,
		"max_tokens": maxTokens,
		"messages":   messages,
	}
	if system != "" {
		body["system"] = system
	}

	if len(in.Tools) > 0 {
		tools := make([]any, 0, len(in.Tools))
		for _, t := range in.Tools {
			tool := map[string]any{"name": t.Name}
			if t.Description != "" {
				tool["description"] = t.Description
			}
			if t.Parameters != nil {
				tool["input_schema"] = t.Parameters
			} else {
				tool["input_schema"] = map[string]any{"type": "object", "properties": map[string]any{}}
			}
			tools = append(tools, tool)
		}
		body["tools"] = tools
	}
	if in.ToolChoice != nil {
		if tc := anthropicMapToolChoice(in.ToolChoice); tc != nil {
			body["tool_choice"] = tc
		}
	}

	if thinking != nil {
		// Extended thinking requires temperature=1 and disallows top_p/top_k — omit sampling.
		body["thinking"] = thinking
	} else {
		if in.Temperature != nil {
			body["temperature"] = *in.Temperature
		}
		if in.TopP != nil {
			body["top_p"] = *in.TopP
		}
	}
	if len(in.Stop) > 0 {
		body["stop_sequences"] = in.Stop
	}
	// NOTE: ResponseFormat is intentionally NOT mapped — Anthropic has no
	// response_format; structured output is via tools / Passthrough.

	if stream {
		body["stream"] = true
	}
	applyBodyPassthrough(body, in.Passthrough)

	url := c.baseURL + "/messages"
	if in.Passthrough != nil && len(in.Passthrough.Query) > 0 {
		url += "?" + buildCanonicalQuery(in.Passthrough.Query)
	}
	return url, body, c.authHeaders(in), nil
}

func (c *anthropicConnector) parseResult(raw any, requestedModel string) *ChatResult {
	var text strings.Builder
	var toolCalls []ToolCall
	if blocks, ok := jsonSliceVal(jsonGet(raw, "content")); ok {
		for _, b := range blocks {
			switch jsonStr(jsonGet(b, "type")) {
			case "text":
				text.WriteString(jsonStr(jsonGet(b, "text")))
			case "tool_use":
				input := jsonGet(b, "input")
				if input == nil {
					input = map[string]any{}
				}
				args, _ := marshalJSON(input)
				toolCalls = append(toolCalls, ToolCall{
					ID:        jsonStr(jsonGet(b, "id")),
					Name:      jsonStr(jsonGet(b, "name")),
					Arguments: string(args),
				})
			}
			// thinking blocks → Raw only
		}
	}
	usage := jsonGet(raw, "usage")
	inTok := int(jsonNum(jsonGet(usage, "input_tokens")))
	outTok := int(jsonNum(jsonGet(usage, "output_tokens")))
	return &ChatResult{
		Message:      AssistantMessage{Role: "assistant", Content: text.String(), ToolCalls: toolCalls},
		FinishReason: anthropicMapStopReason(jsonStr(jsonGet(raw, "stop_reason"))),
		Usage:        Usage{InputTokens: inTok, OutputTokens: outTok, TotalTokens: inTok + outTok},
		Model:        firstStr(jsonStr(jsonGet(raw, "model")), requestedModel),
		Raw:          raw,
	}
}

func anthropicVendorError(resp *http.Response, raw any) *ConnectorError {
	msg := extractErrorMessage(raw)
	if msg == "" {
		msg = "HTTP " + strconv.Itoa(resp.StatusCode)
	}
	return vendorError(resp.StatusCode, anthropicMapCode(resp.StatusCode, msg), msg, raw, resp.Header)
}

// ---- module-private mappers ----

// anthropicMapStreamEvent maps one decoded Anthropic SSE event to a normalized
// delta. inputTokens is shared state: message_start writes it, message_delta
// reads it back. Returns (nil, nil) for events that produce no delta, and a
// non-nil error for a mid-stream `error` event. Anthropic indexes
// CONTENT BLOCKS (text + tool_use interleaved), so the first tool call can arrive
// at block index 1; nextToolIndex/toolIndexByBlock remap that to a 0-based
// tool-call-relative index (matching the OpenAI-compat surface and the other
// languages), and the argument continuation deltas reuse the mapping.
func anthropicMapStreamEvent(evt any, inputTokens *int, nextToolIndex *int, toolIndexByBlock map[int]int) (*ChatStreamDelta, error) {
	switch jsonStr(jsonGet(evt, "type")) {
	case "message_start":
		usage := jsonGet(jsonGet(evt, "message"), "usage")
		*inputTokens = int(jsonNum(jsonGet(usage, "input_tokens")))
		return nil, nil
	case "content_block_start":
		cb := jsonGet(evt, "content_block")
		if jsonStr(jsonGet(cb, "type")) == "tool_use" {
			block := int(jsonNum(jsonGet(evt, "index")))
			idx := *nextToolIndex
			toolIndexByBlock[block] = idx
			*nextToolIndex++
			return &ChatStreamDelta{
				ToolCallDelta: &ToolCallDelta{
					Index:        idx,
					ID:           jsonStr(jsonGet(cb, "id")),
					FunctionName: jsonStr(jsonGet(cb, "name")),
				},
				Raw: evt,
			}, nil
		}
		return nil, nil
	case "content_block_delta":
		d := jsonGet(evt, "delta")
		switch jsonStr(jsonGet(d, "type")) {
		case "text_delta":
			if s := jsonStr(jsonGet(d, "text")); s != "" {
				return &ChatStreamDelta{ContentDelta: s, Raw: evt}, nil
			}
		case "input_json_delta":
			if s, ok := jsonGet(d, "partial_json").(string); ok {
				block := int(jsonNum(jsonGet(evt, "index")))
				return &ChatStreamDelta{
					ToolCallDelta: &ToolCallDelta{Index: toolIndexByBlock[block], ArgumentsDelta: s},
					Raw:           evt,
				}, nil
			}
		}
		// thinking_delta → CoT output not normalized; available only via Raw.
		return nil, nil
	case "message_delta":
		out := &ChatStreamDelta{Raw: evt}
		if sr := jsonStr(jsonGet(jsonGet(evt, "delta"), "stop_reason")); sr != "" {
			out.FinishReason = anthropicMapStopReason(sr)
		}
		outTok := int(jsonNum(jsonGet(jsonGet(evt, "usage"), "output_tokens")))
		out.Usage = &Usage{InputTokens: *inputTokens, OutputTokens: outTok, TotalTokens: *inputTokens + outTok}
		return out, nil
	case "error":
		// Anthropic sends `event: error` with {"type":"error","error":{...}} on a
		// mid-stream failure (e.g. overloaded_error) — surface it, don't drop it.
		return nil, streamFrameError(CodeProviderUnavailable, extractErrorMessage(evt), evt)
	}
	// message_stop / content_block_stop / ping → ignored
	return nil, nil
}

func anthropicSplitMessages(messages []ChatMessage) (string, []any) {
	system := ""
	out := make([]any, 0, len(messages))
	for _, m := range messages {
		switch m.Role {
		case "system":
			text := anthropicContentToText(m)
			if system != "" {
				system = system + "\n\n" + text
			} else {
				system = text
			}
		case "tool":
			block := map[string]any{
				"type":        "tool_result",
				"tool_use_id": m.ToolCallID,
				"content":     anthropicContentToText(m),
			}
			if n := len(out); n > 0 {
				if last, ok := out[n-1].(map[string]any); ok && last["role"] == "user" && anthropicIsToolResultContent(last["content"]) {
					last["content"] = append(last["content"].([]any), block)
					continue
				}
			}
			out = append(out, map[string]any{"role": "user", "content": []any{block}})
		default:
			out = append(out, map[string]any{"role": m.Role, "content": anthropicMapContentBlocks(m)})
		}
	}
	return system, out
}

func anthropicContentToText(m ChatMessage) string {
	if len(m.ContentParts) > 0 {
		var sb strings.Builder
		for _, p := range m.ContentParts {
			if p.Type == "text" {
				sb.WriteString(p.Text)
			}
		}
		return sb.String()
	}
	return m.Content
}

func anthropicMapContentBlocks(m ChatMessage) []any {
	blocks := make([]any, 0, 1)
	if len(m.ContentParts) > 0 {
		for _, p := range m.ContentParts {
			if p.Type == "text" {
				blocks = append(blocks, map[string]any{"type": "text", "text": p.Text})
			} else {
				blocks = append(blocks, map[string]any{
					"type":   "image",
					"source": map[string]any{"type": "base64", "media_type": p.MediaType, "data": p.Base64},
				})
			}
		}
	} else if m.Content != "" {
		blocks = append(blocks, map[string]any{"type": "text", "text": m.Content})
	}
	if m.Role == "assistant" && len(m.ToolCalls) > 0 {
		for _, tc := range m.ToolCalls {
			blocks = append(blocks, map[string]any{
				"type":  "tool_use",
				"id":    tc.ID,
				"name":  tc.Name,
				"input": anthropicSafeJSONParse(tc.Arguments),
			})
		}
	}
	if len(blocks) == 0 {
		blocks = append(blocks, map[string]any{"type": "text", "text": ""})
	}
	return blocks
}

func anthropicIsToolResultContent(content any) bool {
	arr, ok := content.([]any)
	if !ok || len(arr) == 0 {
		return false
	}
	first, ok := arr[0].(map[string]any)
	return ok && first["type"] == "tool_result"
}

func anthropicSafeJSONParse(s string) any {
	if v := decodeAny([]byte(s)); v != nil {
		return v
	}
	return map[string]any{}
}

func anthropicMapToolChoice(tc *ToolChoice) map[string]any {
	switch tc.Mode {
	case "auto":
		return map[string]any{"type": "auto"}
	case "required":
		return map[string]any{"type": "any"}
	case "none":
		return map[string]any{"type": "none"}
	case "function":
		return map[string]any{"type": "tool", "name": tc.FunctionName}
	}
	return nil
}

func anthropicMapStopReason(sr string) FinishReason {
	switch sr {
	case "end_turn", "stop_sequence":
		return "stop"
	case "max_tokens":
		return "length"
	case "tool_use":
		return "tool_calls"
	case "refusal":
		return "content_filter"
	default:
		return "unknown"
	}
}

func anthropicEffortBudget(effort string) int {
	switch effort {
	case "low":
		return 1024
	case "high":
		return 12288
	default: // "medium" and any unset/unknown value
		return 4096
	}
}

func anthropicMapCode(status int, message string) ProviderCode {
	switch {
	case status == 401 || status == 403:
		return CodeAuthFailed
	case status == 429:
		return CodeRateLimited
	// Any 5xx is a provider-side failure. A fixed {500,502,503,529} whitelist
	// dropped 504/520/522/524 (gateway / Cloudflare-origin) to Unknown, so a
	// consumer's retry policy wouldn't fire on them.
	case status >= 500:
		return CodeProviderUnavailable
	case status == 400 || status == 404 || status == 413 || status == 422:
		msg := strings.ToLower(message)
		if strings.Contains(msg, "too long") || strings.Contains(msg, "context") || strings.Contains(msg, "max_tokens") {
			return CodeContextLengthExceeded
		}
		return CodeInvalidRequest
	}
	return CodeUnknown
}
