package llm

import (
	"context"
	"net/http"
	"reflect"
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
// output via Passthrough); reasoning CoT output (thinking blocks land in Raw and,
// for replay, in ProviderMetadata["anthropic"]); prompt caching (cache_control
// via Passthrough; cached tokens are folded into Usage.InputTokens).

const (
	anthropicDefaultBaseURL = "https://api.anthropic.com/v1"
	anthropicDefaultVersion = "2023-06-01"
	// anthropicDefaultMaxTokens is a ceiling, not a charge. Current models think
	// by default (Opus 5.5 always does, at medium effort), and thinking counts
	// toward max_tokens, so a small default often ends at "length" with no text.
	// 16000 is Anthropic's recommended non-streaming default.
	anthropicDefaultMaxTokens = 16000
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
	url, body, headers := c.buildRequest(in, false)
	resp, err := c.sendPostJSON(ctx, url, body, headers)
	if err != nil {
		return nil, err
	}
	raw, readErr := readJSON(resp)
	if !okStatus(resp.StatusCode) {
		return nil, anthropicVendorError(resp, raw)
	}
	if readErr != nil {
		return nil, transportError(readErr)
	}
	// A Messages response always carries a content array (possibly empty); a
	// body without one is not a completion.
	if _, ok := jsonSliceVal(jsonGet(raw, "content")); !ok {
		return nil, emptyBodyError(resp.StatusCode)
	}
	return c.parseResult(raw, in.Model), nil
}

func (c *anthropicConnector) stream(ctx context.Context, in ChatInput) (*ChatStream, error) {
	url, body, headers := c.buildRequest(in, true)
	headers["Accept"] = "text/event-stream"
	resp, err := c.sendPostJSON(ctx, url, body, headers)
	if err != nil {
		return nil, err
	}
	if !okStatus(resp.StatusCode) {
		raw, _ := readJSON(resp)
		return nil, anthropicVendorError(resp, raw)
	}
	if err := requireEventStream(resp); err != nil {
		return nil, err
	}
	st := newAnthropicStreamState()
	st.status = resp.StatusCode
	// A complete stream ends with a message_delta stop_reason (a finish delta)
	// followed by message_stop; either proves the answer was not cut off.
	return newChatStream(resp, func(chunk any) ([]*ChatStreamDelta, error) {
		return oneDelta(st.mapEvent(chunk))
	}, func() bool { return st.sawStop }), nil
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

func (c *anthropicConnector) buildRequest(in ChatInput, stream bool) (string, map[string]any, map[string]string) {
	system, messages := anthropicSplitMessages(in.Messages)

	maxTokens := anthropicDefaultMaxTokens
	if c.cfg.DefaultMaxTokens > 0 {
		maxTokens = c.cfg.DefaultMaxTokens
	}
	if in.MaxOutputTokens != nil {
		maxTokens = *in.MaxOutputTokens
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

	if in.Reasoning != nil {
		// Adaptive thinking is the only mode Claude 4.7+ accepts (budget_tokens is
		// a 400 there), and it lets the model size its own thinking, so there is
		// no budget to fit under max_tokens. Models before Claude 4.6 reject it; they
		// take thinking via Passthrough.Body instead. Thinking also forbids
		// changing temperature/top_p, so sampling is omitted.
		body["thinking"] = map[string]any{"type": "adaptive"}
		body["output_config"] = map[string]any{"effort": anthropicEffort(in.Reasoning.Effort)}
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
	return url, body, c.authHeaders(in)
}

func (c *anthropicConnector) parseResult(raw any, requestedModel string) *ChatResult {
	var text strings.Builder
	var toolCalls []ToolCall
	var metadata map[string]any
	if blocks, ok := jsonSliceVal(jsonGet(raw, "content")); ok {
		metadata = map[string]any{"anthropic": map[string]any{"content": blocks}}
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
			// thinking / redacted_thinking blocks travel only in ProviderMetadata and Raw.
		}
	}
	fields := map[string]float64{}
	anthropicMergeUsage(fields, jsonGet(raw, "usage"))
	return &ChatResult{
		Message:      AssistantMessage{Role: "assistant", Content: text.String(), ToolCalls: toolCalls, ProviderMetadata: metadata},
		FinishReason: anthropicMapStopReason(jsonStr(jsonGet(raw, "stop_reason"))),
		Usage:        anthropicUsage(fields),
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

// anthropicStreamState carries what Anthropic's stateful SSE spreads across
// events: usage is cumulative and merged field by field across message_start
// and message_delta, tool calls are numbered by their ordinal among tool calls (the
// event index counts every content block, so the first tool call can sit at
// block 1), and the response's content array is rebuilt so the terminal delta
// can hand it back as ProviderMetadata, exactly as a non-stream call would.
type anthropicStreamState struct {
	status           int
	usage            map[string]float64
	nextToolIndex    int
	toolIndexByBlock map[int]int
	blocks           map[int]map[string]any
	blockOrder       []int
	partialJSON      map[int]*strings.Builder
	sawStop          bool
}

func newAnthropicStreamState() *anthropicStreamState {
	return &anthropicStreamState{
		usage:            map[string]float64{},
		toolIndexByBlock: map[int]int{},
		blocks:           map[int]map[string]any{},
		partialJSON:      map[int]*strings.Builder{},
	}
}

// mapEvent maps one decoded Anthropic SSE event to a normalized delta. It
// returns (nil, nil) for events that produce no delta, and a non-nil error for
// a mid-stream `error` event.
func (s *anthropicStreamState) mapEvent(evt any) (*ChatStreamDelta, error) {
	switch jsonStr(jsonGet(evt, "type")) {
	case "message_start":
		s.mergeUsage(jsonGet(jsonGet(evt, "message"), "usage"))
		return nil, nil
	case "content_block_start":
		block := int(jsonNum(jsonGet(evt, "index")))
		cb := jsonGet(evt, "content_block")
		s.startBlock(block, cb)
		if jsonStr(jsonGet(cb, "type")) == "tool_use" {
			idx := s.nextToolIndex
			s.toolIndexByBlock[block] = idx
			s.nextToolIndex++
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
		block := int(jsonNum(jsonGet(evt, "index")))
		d := jsonGet(evt, "delta")
		s.applyDelta(block, d)
		switch jsonStr(jsonGet(d, "type")) {
		case "text_delta":
			if t := jsonStr(jsonGet(d, "text")); t != "" {
				return &ChatStreamDelta{ContentDelta: t, Raw: evt}, nil
			}
		case "input_json_delta":
			if pj, ok := jsonGet(d, "partial_json").(string); ok {
				if idx, isTool := s.toolIndexByBlock[block]; isTool {
					return &ChatStreamDelta{
						ToolCallDelta: &ToolCallDelta{Index: idx, ArgumentsDelta: pj},
						Raw:           evt,
					}, nil
				}
			}
		}
		// thinking_delta / signature_delta → CoT is not normalized; it is kept
		// only in the rebuilt content array.
		return nil, nil
	case "message_delta":
		out := &ChatStreamDelta{Raw: evt}
		if sr := jsonStr(jsonGet(jsonGet(evt, "delta"), "stop_reason")); sr != "" {
			out.FinishReason = anthropicMapStopReason(sr)
			// The content array rides only on the delta that ends the turn, so
			// a consumer never mistakes a partial array for the final one.
			out.ProviderMetadata = map[string]any{"anthropic": map[string]any{"content": s.content()}}
		}
		s.mergeUsage(jsonGet(evt, "usage"))
		u := anthropicUsage(s.usage)
		out.Usage = &u
		return out, nil
	case "error":
		// Anthropic sends `event: error` with {"type":"error","error":{...}} on a
		// mid-stream failure (e.g. overloaded_error) — surface it, don't drop it.
		msg := extractErrorMessage(evt)
		return nil, streamFrameError(s.status, anthropicMapErrorType(jsonStr(jsonGet(jsonGet(evt, "error"), "type")), msg), msg, evt)
	case "message_stop":
		s.sawStop = true
	}
	// content_block_stop / ping → ignored
	return nil, nil
}

func (s *anthropicStreamState) startBlock(index int, cb any) {
	m, ok := cb.(map[string]any)
	if !ok {
		return
	}
	block := make(map[string]any, len(m))
	for k, v := range m {
		block[k] = v
	}
	if _, seen := s.blocks[index]; !seen {
		s.blockOrder = append(s.blockOrder, index)
	}
	s.blocks[index] = block
}

func (s *anthropicStreamState) applyDelta(index int, d any) {
	block := s.blocks[index]
	if block == nil {
		return
	}
	switch jsonStr(jsonGet(d, "type")) {
	case "text_delta":
		block["text"] = jsonStr(block["text"]) + jsonStr(jsonGet(d, "text"))
	case "thinking_delta":
		block["thinking"] = jsonStr(block["thinking"]) + jsonStr(jsonGet(d, "thinking"))
	case "signature_delta":
		block["signature"] = jsonStr(block["signature"]) + jsonStr(jsonGet(d, "signature"))
	case "citations_delta":
		citations, _ := jsonSliceVal(block["citations"])
		block["citations"] = append(citations, jsonGet(d, "citation"))
	case "input_json_delta":
		b := s.partialJSON[index]
		if b == nil {
			b = &strings.Builder{}
			s.partialJSON[index] = b
		}
		b.WriteString(jsonStr(jsonGet(d, "partial_json")))
	}
}

// content returns the rebuilt content array in block order, with each tool
// block's input parsed from its accumulated partial_json (Anthropic starts
// tool_use blocks with an empty input and streams the real one as fragments).
func (s *anthropicStreamState) content() []any {
	out := make([]any, 0, len(s.blockOrder))
	for _, idx := range s.blockOrder {
		block := s.blocks[idx]
		if b := s.partialJSON[idx]; b != nil && b.Len() > 0 {
			block["input"] = anthropicSafeJSONParse(b.String())
		}
		out = append(out, block)
	}
	return out
}

func anthropicSplitMessages(messages []ChatMessage) (string, []any) {
	system := ""
	rebuilt := false
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
			var content []any
			if m.Role == "assistant" && !rebuilt {
				content, _ = anthropicReplayContent(m)
			}
			if content == nil {
				content = anthropicMapContentBlocks(m)
				// Anthropic binds each thinking block to the history before it:
				// once an assistant turn is edited or stripped, every later
				// thinking block is invalid (a 400 on accounts that enforce
				// preserved thinking), so later turns are rebuilt too.
				if m.Role == "assistant" {
					rebuilt = true
				}
			}
			out = append(out, map[string]any{"role": m.Role, "content": content})
		}
	}
	return system, out
}

// anthropicReplayContent returns the assistant turn's original content blocks
// from ProviderMetadata["anthropic"]["content"], so its thinking blocks go back
// to the model. They are used only while they still describe the message: the
// same tool_use ids in the same order with the same inputs, and the same text.
// A caller that edited the turn gets it rebuilt from the normalized fields
// instead, which drops the thinking blocks and so loses the earlier reasoning;
// the edit is never silently replaced by the stored blocks.
func anthropicReplayContent(m ChatMessage) ([]any, bool) {
	if m.Role != "assistant" {
		return nil, false
	}
	blocks, ok := jsonSliceVal(jsonGet(m.ProviderMetadata["anthropic"], "content"))
	if !ok || len(blocks) == 0 {
		return nil, false
	}
	var text strings.Builder
	toolIdx := 0
	for _, b := range blocks {
		switch jsonStr(jsonGet(b, "type")) {
		case "text":
			text.WriteString(jsonStr(jsonGet(b, "text")))
		case "tool_use":
			if toolIdx >= len(m.ToolCalls) || m.ToolCalls[toolIdx].ID != jsonStr(jsonGet(b, "id")) ||
				!anthropicSameJSON(jsonGet(b, "input"), anthropicSafeJSONParse(m.ToolCalls[toolIdx].Arguments)) {
				return nil, false
			}
			toolIdx++
		}
	}
	if toolIdx != len(m.ToolCalls) || text.String() != anthropicContentToText(m) {
		return nil, false
	}
	return blocks, true
}

// anthropicSameJSON compares two decoded JSON values after a round trip through
// encoding/json, so numbers compare by value whatever their Go type (a history
// decoded with UseNumber holds json.Number, a fresh one float64).
func anthropicSameJSON(a, b any) bool {
	ea, errA := marshalJSON(a)
	eb, errB := marshalJSON(b)
	if errA != nil || errB != nil {
		return false
	}
	return reflect.DeepEqual(decodeAny(ea), decodeAny(eb))
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
				// Anthropic rejects an empty text block, so an empty part is dropped.
				if p.Text != "" {
					blocks = append(blocks, map[string]any{"type": "text", "text": p.Text})
				}
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
	case "max_tokens", "model_context_window_exceeded":
		// Both cut the answer short at a token limit (the output cap, or the
		// model's context window).
		return "length"
	case "tool_use":
		return "tool_calls"
	case "refusal":
		return "content_filter"
	default:
		// pause_turn (a server-tool turn the caller must send back to resume)
		// has no normalized equivalent; it is neither finished nor truncated.
		return "unknown"
	}
}

// anthropicEffort maps the normalized effort onto output_config.effort. Unset
// means "medium"; any other value is sent as given, so Anthropic validates it
// (and a caller can reach its own levels, such as "max").
func anthropicEffort(effort string) string {
	if effort == "" {
		return "medium"
	}
	return effort
}

// anthropicUsageFields are the counters a usage object can carry.
var anthropicUsageFields = []string{"input_tokens", "cache_read_input_tokens", "cache_creation_input_tokens", "output_tokens"}

// anthropicMergeUsage copies each counter that is present and a number into
// fields. Stream usage is cumulative and a message_delta may restate only some
// counters (or send null), so an absent or null one keeps its earlier value,
// as the official SDK's MessageStream does.
func anthropicMergeUsage(fields map[string]float64, usage any) {
	for _, k := range anthropicUsageFields {
		if n, ok := jsonGet(usage, k).(float64); ok {
			fields[k] = n
		}
	}
}

func (s *anthropicStreamState) mergeUsage(usage any) { anthropicMergeUsage(s.usage, usage) }

// anthropicUsage normalizes merged counters. Anthropic reports cache reads and
// cache writes apart from input_tokens, but all three are billed input.
func anthropicUsage(fields map[string]float64) Usage {
	in := int(fields["input_tokens"] + fields["cache_read_input_tokens"] + fields["cache_creation_input_tokens"])
	out := int(fields["output_tokens"])
	return Usage{InputTokens: in, OutputTokens: out, TotalTokens: in + out}
}

// anthropicIsContextOverflow matches Anthropic's context-window messages
// ("prompt is too long: N tokens > M maximum", "input length and `max_tokens`
// exceed context limit"). A looser match ("max_tokens", "context") would also
// catch ordinary validation errors, and a consumer that trims history on
// context_length_exceeded would then loop forever.
func anthropicIsContextOverflow(message string) bool {
	msg := strings.ToLower(message)
	return strings.Contains(msg, "prompt is too long") || strings.Contains(msg, "exceed context limit")
}

// anthropicMapErrorType classifies a mid-stream error event by its documented
// error.type, since the stream's own HTTP status (200) says nothing about it.
// A type Anthropic has not documented is unknown rather than guessed.
func anthropicMapErrorType(errType, message string) ProviderCode {
	switch errType {
	case "authentication_error", "permission_error":
		return CodeAuthFailed
	case "rate_limit_error", "billing_error":
		return CodeRateLimited
	case "invalid_request_error", "not_found_error", "request_too_large", "conflict_error":
		if errType == "invalid_request_error" && anthropicIsContextOverflow(message) {
			return CodeContextLengthExceeded
		}
		return CodeInvalidRequest
	case "api_error", "overloaded_error", "timeout_error":
		return CodeProviderUnavailable
	}
	return CodeUnknown
}

func anthropicMapCode(status int, message string) ProviderCode {
	switch {
	case status == 401 || status == 403:
		return CodeAuthFailed
	// 402 billing_error: like a spend cap (which Anthropic itself answers with
	// 429), the call cannot succeed until the account is topped up.
	case status == 429 || status == 402:
		return CodeRateLimited
	// Any 5xx is a provider-side failure. A fixed {500,502,503,529} whitelist
	// dropped 504/520/522/524 (gateway / Cloudflare-origin) to Unknown, so a
	// consumer's retry policy wouldn't fire on them. 408 is a transient timeout.
	case status >= 500 || status == 408:
		return CodeProviderUnavailable
	// 413 request_too_large is the 32 MB byte limit, not the context window.
	case status == 413:
		return CodeInvalidRequest
	case status == 400 || status == 404 || status == 422:
		if anthropicIsContextOverflow(message) {
			return CodeContextLengthExceeded
		}
		return CodeInvalidRequest
	// Any other 4xx (409 conflict_error, …): the request cannot succeed as sent.
	case status >= 400:
		return CodeInvalidRequest
	}
	return CodeUnknown
}
