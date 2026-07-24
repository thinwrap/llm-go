package llm

import (
	"context"
	"encoding/json"
	"math"
	"net/http"
	"net/url"
	"regexp"
	"strconv"
	"strings"
)

// Native adapter for the Google Gemini (AI Studio) generateContent API. Gemini
// uses contents[]/parts[] (roles user/model), a top-level systemInstruction,
// functionDeclarations, and keys tool results by function NAME (not id). This
// connector translates the normalized surface to that shape and back, emitting
// the identical ChatResult/ChatStreamDelta/ConnectorError. Unlike Bedrock,
// Gemini streaming IS SSE (:streamGenerateContent?alt=sse), so streaming is
// truly incremental.

const geminiDefaultBaseURL = "https://generativelanguage.googleapis.com/v1beta"

func init() { registerNativeChat(Gemini, newGeminiChat) }

func newGeminiChat(cfg ChatConfig, o facadeOptions) (chatConnector, error) {
	c, ok := cfg.(GeminiConfig)
	if !ok {
		return nil, invalidReq("gemini requires a GeminiConfig")
	}
	base := c.BaseURL
	if base == "" {
		base = geminiDefaultBaseURL
	}
	return &geminiConnector{
		baseConnector: baseConnector{o.httpClient},
		cfg:           c,
		baseURL:       strings.TrimRight(base, "/"),
	}, nil
}

type geminiConnector struct {
	baseConnector
	cfg     GeminiConfig
	baseURL string
}

func (c *geminiConnector) complete(ctx context.Context, in ChatInput) (*ChatResult, error) {
	reqURL := c.baseURL + "/models/" + url.PathEscape(in.Model) + ":generateContent"
	if in.Passthrough != nil && len(in.Passthrough.Query) > 0 {
		reqURL += "?" + buildCanonicalQuery(in.Passthrough.Query)
	}
	resp, err := c.sendPostJSON(ctx, reqURL, c.buildRequest(in), c.authHeaders(in))
	if err != nil {
		return nil, err
	}
	raw := decodeAny(readBody(resp))
	if !okStatus(resp.StatusCode) {
		return nil, geminiVendorError(resp, raw)
	}
	if raw == nil {
		return nil, emptyBodyError(resp.StatusCode)
	}
	return c.parseResult(raw, in.Model), nil
}

func (c *geminiConnector) stream(ctx context.Context, in ChatInput) (*ChatStream, error) {
	reqURL := c.baseURL + "/models/" + url.PathEscape(in.Model) + ":streamGenerateContent?alt=sse"
	if in.Passthrough != nil && len(in.Passthrough.Query) > 0 {
		reqURL += "&" + buildCanonicalQuery(in.Passthrough.Query)
	}
	headers := c.authHeaders(in)
	headers["Accept"] = "text/event-stream"
	resp, err := c.sendPostJSON(ctx, reqURL, c.buildRequest(in), headers)
	if err != nil {
		return nil, err
	}
	if !okStatus(resp.StatusCode) {
		raw := decodeAny(readBody(resp))
		return nil, geminiVendorError(resp, raw)
	}
	sawFunctionCall := false
	toolIndex := 0
	return newChatStream(resp, func(chunk any) ([]*ChatStreamDelta, error) {
		return c.mapStreamChunk(chunk, &sawFunctionCall, &toolIndex)
	}), nil
}

func (c *geminiConnector) authHeaders(in ChatInput) map[string]string {
	headers := map[string]string{"x-goog-api-key": c.cfg.APIKey}
	for k, v := range c.cfg.Headers {
		headers[k] = v
	}
	if in.Passthrough != nil {
		for k, v := range in.Passthrough.Headers {
			headers[k] = v
		}
	}
	return headers
}

func (c *geminiConnector) buildRequest(in ChatInput) map[string]any {
	systemInstruction, contents := geminiSplitMessages(in.Messages)

	body := map[string]any{"contents": contents}
	if systemInstruction != nil {
		body["systemInstruction"] = systemInstruction
	}

	if len(in.Tools) > 0 {
		decls := make([]any, 0, len(in.Tools))
		for _, t := range in.Tools {
			fn := map[string]any{"name": t.Name}
			if t.Description != "" {
				fn["description"] = t.Description
			}
			if t.Parameters != nil {
				fn["parameters"] = t.Parameters
			}
			decls = append(decls, fn)
		}
		body["tools"] = []any{map[string]any{"functionDeclarations": decls}}
		if fc := geminiFunctionCallingConfig(in.ToolChoice); fc != nil {
			body["toolConfig"] = map[string]any{"functionCallingConfig": fc}
		}
	}

	gc := map[string]any{}
	if in.MaxOutputTokens != nil {
		gc["maxOutputTokens"] = *in.MaxOutputTokens
	} else if c.cfg.DefaultMaxTokens != 0 {
		gc["maxOutputTokens"] = c.cfg.DefaultMaxTokens
	}
	if in.Temperature != nil {
		gc["temperature"] = *in.Temperature
	}
	if in.TopP != nil {
		gc["topP"] = *in.TopP
	}
	if len(in.Stop) > 0 {
		gc["stopSequences"] = in.Stop
	}
	geminiApplyResponseFormat(gc, in.ResponseFormat)
	if in.Reasoning != nil {
		gc["thinkingConfig"] = map[string]any{"thinkingBudget": geminiEffortBudget(in.Reasoning.Effort)}
	}
	if len(gc) > 0 {
		body["generationConfig"] = gc
	}

	applyBodyPassthrough(body, in.Passthrough)
	return body
}

func (c *geminiConnector) parseResult(raw any, requestedModel string) *ChatResult {
	cand := geminiFirstCandidate(raw)
	parts, _ := jsonSliceVal(jsonGet(jsonGet(cand, "content"), "parts"))

	var text strings.Builder
	var toolCalls []ToolCall
	for _, p := range parts {
		if b, ok := jsonGet(p, "thought").(bool); ok && b {
			// Gemini-3 reasoning parts (thought:true) are not answer content; skip
			// them (still available to the consumer via Raw).
			continue
		} else if t, ok := jsonGet(p, "text").(string); ok {
			text.WriteString(t)
		} else if fc := jsonGet(p, "functionCall"); fc != nil {
			id := jsonStr(jsonGet(fc, "id"))
			if id == "" {
				// Same synthetic-ID scheme as the streaming path (name-based, not
				// positional) so a call's id is stable across stream vs non-stream.
				id = "call_" + jsonStr(jsonGet(fc, "name"))
			}
			toolCalls = append(toolCalls, ToolCall{
				ID:        id,
				Name:      jsonStr(jsonGet(fc, "name")),
				Arguments: geminiStringifyArgs(jsonGet(fc, "args")),
			})
		}
	}

	res := &ChatResult{
		Message: AssistantMessage{Role: "assistant", Content: text.String()},
		Usage:   geminiMapUsage(jsonGet(raw, "usageMetadata")),
		Model:   firstStr(jsonStr(jsonGet(raw, "modelVersion")), requestedModel),
		Raw:     raw,
	}
	if len(toolCalls) > 0 {
		res.Message.ToolCalls = toolCalls
		res.FinishReason = "tool_calls"
	} else if fr := jsonStr(jsonGet(cand, "finishReason")); fr == "" && jsonStr(jsonGet(jsonGet(raw, "promptFeedback"), "blockReason")) != "" {
		// No candidate came back: a prompt-level safety block reports only
		// promptFeedback.blockReason. Surface it as content_filter rather than a
		// silent empty completion with an unknown finish.
		res.FinishReason = "content_filter"
	} else {
		res.FinishReason = geminiMapFinishReason(fr)
	}
	return res
}

// mapStreamChunk normalizes one Gemini SSE chunk into zero or more deltas. A
// chunk may carry text parts, one OR MORE functionCall parts, a finishReason,
// and/or usageMetadata. Each functionCall part becomes its own ToolCallDelta
// with a monotonically increasing Index (toolIndex, stateful across chunks) so
// parallel/multiple calls are never merged; text is concatenated into a single
// content delta and any finish/usage ride on a trailing delta. sawFunctionCall
// is stateful so a terminal finishReason maps to "tool_calls".
func (c *geminiConnector) mapStreamChunk(chunk any, sawFunctionCall *bool, toolIndex *int) ([]*ChatStreamDelta, error) {
	// A mid-stream {"error":{...}} frame must abort with a *ConnectorError; the
	// frame carries an HTTP-style code, so classify with it.
	if e := jsonGet(chunk, "error"); e != nil {
		msg := extractErrorMessage(chunk)
		return nil, streamFrameError(geminiMapCode(int(jsonNum(jsonGet(e, "code"))), msg), msg, chunk)
	}

	var deltas []*ChatStreamDelta
	cand := geminiFirstCandidate(chunk)
	// A prompt-level safety block arrives as a chunk with no candidate and only
	// promptFeedback.blockReason — emit a content_filter finish so the consumer
	// sees the block rather than a silent, finish-less end.
	if cand == nil && jsonStr(jsonGet(jsonGet(chunk, "promptFeedback"), "blockReason")) != "" {
		return []*ChatStreamDelta{{Raw: chunk, FinishReason: "content_filter"}}, nil
	}
	parts, _ := jsonSliceVal(jsonGet(jsonGet(cand, "content"), "parts"))
	var text strings.Builder
	for _, p := range parts {
		if b, ok := jsonGet(p, "thought").(bool); ok && b {
			// Gemini-3 reasoning parts (thought:true) are not answer content; skip.
			continue
		} else if t, ok := jsonGet(p, "text").(string); ok && t != "" {
			text.WriteString(t)
		} else if fc := jsonGet(p, "functionCall"); fc != nil {
			*sawFunctionCall = true
			id := jsonStr(jsonGet(fc, "id"))
			if id == "" {
				id = "call_" + jsonStr(jsonGet(fc, "name"))
			}
			deltas = append(deltas, &ChatStreamDelta{
				Raw: chunk,
				ToolCallDelta: &ToolCallDelta{
					Index:          *toolIndex,
					ID:             id,
					FunctionName:   jsonStr(jsonGet(fc, "name")),
					ArgumentsDelta: geminiStringifyArgs(jsonGet(fc, "args")),
				},
			})
			*toolIndex++
		}
	}
	// Text precedes the tool-call deltas (Gemini emits any prose first).
	if text.Len() > 0 {
		deltas = append([]*ChatStreamDelta{{Raw: chunk, ContentDelta: text.String()}}, deltas...)
	}

	fr := jsonStr(jsonGet(cand, "finishReason"))
	usageRaw := jsonGet(chunk, "usageMetadata")
	if fr != "" || usageRaw != nil {
		tail := &ChatStreamDelta{Raw: chunk}
		if fr != "" {
			if *sawFunctionCall {
				tail.FinishReason = "tool_calls"
			} else {
				tail.FinishReason = geminiMapFinishReason(fr)
			}
		}
		if usageRaw != nil {
			u := geminiMapUsage(usageRaw)
			tail.Usage = &u
		}
		deltas = append(deltas, tail)
	}

	return deltas, nil
}

func geminiVendorError(resp *http.Response, raw any) *ConnectorError {
	msg := extractErrorMessage(raw)
	ce := vendorError(resp.StatusCode, geminiMapCode(resp.StatusCode, msg), msg, raw, resp.Header)
	// Gemini has no rate-limit headers — retry timing lives in the 429 body's
	// RetryInfo detail. Fill it in only when the Retry-After header didn't.
	if cause, ok := ce.Cause.(map[string]any); ok {
		if _, present := cause["retryAfterSeconds"]; !present {
			if secs, ok := geminiRetryDelaySeconds(raw); ok {
				cause["retryAfterSeconds"] = secs
			}
		}
	}
	return ce
}

// ---- module-private mappers ----

func geminiFirstCandidate(raw any) any {
	candidates, _ := jsonSliceVal(jsonGet(raw, "candidates"))
	if len(candidates) > 0 {
		return candidates[0]
	}
	return nil
}

// geminiSplitMessages folds system messages into a single systemInstruction and
// maps the rest into Gemini's contents[]. Tool-result messages become
// functionResponse parts keyed by function name and are batched onto a trailing
// user turn (matching how Gemini expects parallel tool results).
func geminiSplitMessages(messages []ChatMessage) (systemInstruction map[string]any, contents []any) {
	nameByID := map[string]string{}
	for _, m := range messages {
		if m.Role == "assistant" {
			for _, tc := range m.ToolCalls {
				nameByID[tc.ID] = tc.Name
			}
		}
	}

	var systemText string
	contents = []any{}
	for _, m := range messages {
		switch m.Role {
		case "system":
			t := geminiContentToText(m)
			if systemText != "" {
				systemText = systemText + "\n\n" + t
			} else {
				systemText = t
			}
		case "tool":
			name := m.Name
			if name == "" && m.ToolCallID != "" {
				name = nameByID[m.ToolCallID]
			}
			if name == "" {
				name = m.ToolCallID
			}
			if name == "" {
				name = "tool"
			}
			part := map[string]any{
				"functionResponse": map[string]any{
					"name":     name,
					"response": map[string]any{"result": geminiParseJSONOrString(geminiContentToText(m))},
				},
			}
			if n := len(contents); n > 0 {
				if last, ok := contents[n-1].(map[string]any); ok &&
					last["role"] == "user" && geminiIsFunctionResponseParts(last["parts"]) {
					last["parts"] = append(last["parts"].([]any), part)
					continue
				}
			}
			contents = append(contents, map[string]any{"role": "user", "parts": []any{part}})
		default:
			role := "user"
			if m.Role == "assistant" {
				role = "model"
			}
			contents = append(contents, map[string]any{"role": role, "parts": geminiMapParts(m)})
		}
	}

	if systemText != "" {
		systemInstruction = map[string]any{"parts": []any{map[string]any{"text": systemText}}}
	}
	return systemInstruction, contents
}

func geminiContentToText(m ChatMessage) string {
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

func geminiMapParts(m ChatMessage) []any {
	parts := []any{}
	if len(m.ContentParts) > 0 {
		for _, p := range m.ContentParts {
			if p.Type == "text" {
				parts = append(parts, map[string]any{"text": p.Text})
			} else {
				parts = append(parts, map[string]any{"inlineData": map[string]any{"mimeType": p.MediaType, "data": p.Base64}})
			}
		}
	} else if m.Content != "" {
		parts = append(parts, map[string]any{"text": m.Content})
	}
	if m.Role == "assistant" && len(m.ToolCalls) > 0 {
		for _, tc := range m.ToolCalls {
			parts = append(parts, map[string]any{"functionCall": map[string]any{"name": tc.Name, "args": geminiSafeJSONParse(tc.Arguments)}})
		}
	}
	if len(parts) == 0 {
		parts = append(parts, map[string]any{"text": ""})
	}
	return parts
}

func geminiIsFunctionResponseParts(parts any) bool {
	ps, ok := parts.([]any)
	if !ok || len(ps) == 0 {
		return false
	}
	m, ok := ps[0].(map[string]any)
	if !ok {
		return false
	}
	_, has := m["functionResponse"]
	return has
}

func geminiFunctionCallingConfig(tc *ToolChoice) map[string]any {
	if tc == nil {
		return nil
	}
	switch tc.Mode {
	case "auto":
		return map[string]any{"mode": "AUTO"}
	case "required":
		return map[string]any{"mode": "ANY"}
	case "none":
		return map[string]any{"mode": "NONE"}
	case "function":
		return map[string]any{"mode": "ANY", "allowedFunctionNames": []any{tc.FunctionName}}
	}
	return nil
}

func geminiApplyResponseFormat(gc map[string]any, rf *ResponseFormat) {
	if rf == nil {
		return
	}
	switch rf.Type {
	case "json_object":
		gc["responseMimeType"] = "application/json"
	case "json_schema":
		gc["responseMimeType"] = "application/json"
		if rf.JSONSchema != nil {
			gc["responseSchema"] = rf.JSONSchema
		}
	}
}

func geminiMapFinishReason(fr string) FinishReason {
	switch fr {
	case "STOP":
		return "stop"
	case "MAX_TOKENS":
		return "length"
	case "SAFETY", "RECITATION", "BLOCKLIST", "PROHIBITED_CONTENT", "SPII":
		return "content_filter"
	default:
		return "unknown"
	}
}

func geminiMapUsage(u any) Usage {
	inp := int(jsonNum(jsonGet(u, "promptTokenCount")))
	out := int(jsonNum(jsonGet(u, "candidatesTokenCount")))
	total := int(jsonNum(jsonGet(u, "totalTokenCount")))
	if total == 0 {
		total = inp + out
	}
	return Usage{InputTokens: inp, OutputTokens: out, TotalTokens: total}
}

func geminiEffortBudget(effort string) int {
	switch effort {
	case "low":
		return 1024
	case "high":
		return 24576
	default: // "medium" and any unset/unknown value
		// A budget of 0 would DISABLE thinking; since Reasoning was explicitly
		// requested, an unrecognized effort must fall back to a sensible non-zero
		// budget (medium) — matching Anthropic's "default → medium" normalization.
		return 8192
	}
}

func geminiStringifyArgs(args any) string {
	if args == nil {
		args = map[string]any{}
	}
	b, err := marshalJSON(args)
	if err != nil {
		return "{}"
	}
	return string(b)
}

func geminiSafeJSONParse(s string) any {
	var v any
	if json.Unmarshal([]byte(s), &v) != nil {
		return map[string]any{}
	}
	return v
}

func geminiParseJSONOrString(s string) any {
	var v any
	if json.Unmarshal([]byte(s), &v) != nil {
		return s
	}
	return v
}

func geminiMapCode(status int, message string) ProviderCode {
	switch {
	case status == 401 || status == 403:
		return CodeAuthFailed
	case status == 429:
		return CodeRateLimited
	case status >= 500:
		return CodeProviderUnavailable
	}
	if status == 400 || status == 404 || status == 422 {
		msg := strings.ToLower(message)
		if strings.Contains(msg, "too long") || strings.Contains(msg, "context") || strings.Contains(msg, "token count") {
			return CodeContextLengthExceeded
		}
		return CodeInvalidRequest
	}
	return CodeUnknown
}

var geminiRetryDelayRe = regexp.MustCompile(`^(\d+(?:\.\d+)?)s$`)

func geminiRetryDelaySeconds(body any) (int, bool) {
	details, ok := jsonSliceVal(jsonGet(jsonGet(body, "error"), "details"))
	if !ok {
		return 0, false
	}
	for _, d := range details {
		rd := jsonStr(jsonGet(d, "retryDelay"))
		if rd == "" {
			continue
		}
		if m := geminiRetryDelayRe.FindStringSubmatch(rd); m != nil {
			if f, err := strconv.ParseFloat(m[1], 64); err == nil {
				return int(math.Round(f)), true
			}
		}
	}
	return 0, false
}
