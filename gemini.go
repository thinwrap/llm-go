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
// functionDeclarations, and keys tool results by function NAME (an id is
// optional and often absent). This connector translates the normalized surface
// to that shape and back, emitting the identical
// ChatResult/ChatStreamDelta/ConnectorError. Unlike Bedrock,
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
	raw, readErr := readJSON(resp)
	if !okStatus(resp.StatusCode) {
		return nil, geminiVendorError(resp, raw)
	}
	if readErr != nil {
		return nil, transportError(readErr)
	}
	// A prompt-level safety block legitimately omits candidates (parseResult
	// surfaces it as content_filter); any other body without a candidates array
	// is not a completion.
	if _, ok := jsonSliceVal(jsonGet(raw, "candidates")); !ok && jsonStr(jsonGet(jsonGet(raw, "promptFeedback"), "blockReason")) == "" {
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
		raw, _ := readJSON(resp)
		return nil, geminiVendorError(resp, raw)
	}
	if err := requireEventStream(resp); err != nil {
		return nil, err
	}
	st := &geminiStreamState{status: resp.StatusCode}
	// The terminal event is a chunk with a finishReason (or a promptFeedback
	// block); both surface as a finish delta.
	return newChatStream(resp, func(chunk any) ([]*ChatStreamDelta, error) {
		return c.mapStreamChunk(chunk, st)
	}, nil), nil
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
				// parametersJsonSchema takes standard JSON Schema; `parameters`
				// is an OpenAPI subset that rejects additionalProperties, $ref, ….
				fn["parametersJsonSchema"] = t.Parameters
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
			toolCalls = append(toolCalls, geminiToolCall(p, fc, len(toolCalls)))
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

// geminiStreamState is what a Gemini stream carries across chunks: the next
// tool-call Index (so parallel calls are never merged and synthetic ids match
// the non-stream ones), whether any functionCall was seen (so the terminal
// finishReason maps to "tool_calls"), and whether the finish was already sent.
type geminiStreamState struct {
	status          int
	toolIndex       int
	sawFunctionCall bool
	finished        bool
	lastUsage       any
}

// mapStreamChunk normalizes one Gemini SSE chunk into zero or more deltas. A
// chunk may carry text parts, one OR MORE functionCall parts, a finishReason,
// and/or usageMetadata. Each functionCall part becomes its own ToolCallDelta;
// text is concatenated into a single content delta and the finish rides on a
// trailing delta. usageMetadata is cumulative on every chunk, so it is reported
// exactly once, on the finish delta; a chunk after the finish adds nothing.
func (c *geminiConnector) mapStreamChunk(chunk any, st *geminiStreamState) ([]*ChatStreamDelta, error) {
	// A mid-stream {"error":{...}} frame must abort with a *ConnectorError,
	// classified by the frame's HTTP-style code or, without one, its google.rpc
	// status name.
	if e := jsonGet(chunk, "error"); e != nil {
		msg := extractErrorMessage(chunk)
		return nil, streamFrameError(st.status, geminiMapCode(geminiFrameStatus(e), msg, chunk), msg, chunk)
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
			st.sawFunctionCall = true
			tc := geminiToolCall(p, fc, st.toolIndex)
			deltas = append(deltas, &ChatStreamDelta{
				Raw: chunk,
				ToolCallDelta: &ToolCallDelta{
					Index:            st.toolIndex,
					ID:               tc.ID,
					FunctionName:     tc.Name,
					ArgumentsDelta:   tc.Arguments,
					ProviderMetadata: tc.ProviderMetadata,
				},
			})
			st.toolIndex++
		}
	}
	// Text precedes the tool-call deltas (Gemini emits any prose first).
	if text.Len() > 0 {
		deltas = append([]*ChatStreamDelta{{Raw: chunk, ContentDelta: text.String()}}, deltas...)
	}

	fr := jsonStr(jsonGet(cand, "finishReason"))
	if u := jsonGet(chunk, "usageMetadata"); u != nil {
		st.lastUsage = u
	}
	if fr != "" && !st.finished {
		st.finished = true
		tail := &ChatStreamDelta{Raw: chunk}
		if st.sawFunctionCall {
			tail.FinishReason = "tool_calls"
		} else {
			tail.FinishReason = geminiMapFinishReason(fr)
		}
		if st.lastUsage != nil {
			u := geminiMapUsage(st.lastUsage)
			tail.Usage = &u
		}
		deltas = append(deltas, tail)
	}

	return deltas, nil
}

func geminiVendorError(resp *http.Response, raw any) *ConnectorError {
	msg := extractErrorMessage(raw)
	ce := vendorError(resp.StatusCode, geminiMapCode(resp.StatusCode, msg, raw), msg, raw, resp.Header)
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

// geminiSkipSignature is Google's documented placeholder thoughtSignature. It
// tells the validator to accept a functionCall whose real signature is unknown
// (a hand-built history, or one from another provider); without any signature
// Gemini 3 rejects the turn with a 400.
const geminiSkipSignature = "skip_thought_signature_validator"

// geminiSyntheticIDRe matches the ids this connector invents when Gemini sends
// a functionCall without one. They never go back on the wire: Gemini did not
// issue them, and it pairs a response with its call by name when ids are absent.
var geminiSyntheticIDRe = regexp.MustCompile(`^call_[0-9]+$`)

// geminiToolCall normalizes one functionCall part. Gemini's own id is kept;
// when it is absent the id is synthesized from the call's ordinal, so parallel
// calls to the same function stay distinct and stream and non-stream agree.
// The part's thoughtSignature is kept in ProviderMetadata["gemini"] so the
// next request can hand it back.
func geminiToolCall(part, fc any, ordinal int) ToolCall {
	id := jsonStr(jsonGet(fc, "id"))
	if id == "" {
		id = "call_" + strconv.Itoa(ordinal)
	}
	tc := ToolCall{
		ID:        id,
		Name:      jsonStr(jsonGet(fc, "name")),
		Arguments: geminiStringifyArgs(jsonGet(fc, "args")),
	}
	if sig := jsonStr(jsonGet(part, "thoughtSignature")); sig != "" {
		tc.ProviderMetadata = map[string]any{"gemini": map[string]any{"thoughtSignature": sig}}
	}
	return tc
}

// geminiVendorID reports whether a ToolCall id may go back to Gemini as a
// functionCall / functionResponse id.
func geminiVendorID(id string) bool {
	return id != "" && !geminiSyntheticIDRe.MatchString(id)
}

// geminiSplitMessages folds system messages into a single systemInstruction and
// maps the rest into Gemini's contents[]. Tool-result messages become
// functionResponse parts keyed by function name and are batched onto a trailing
// user turn (matching how Gemini expects parallel tool results).
func geminiSplitMessages(messages []ChatMessage) (systemInstruction map[string]any, contents []any) {
	// nameByID is filled as the history is walked, so a tool result resolves
	// against the assistant turn before it; synthetic ids repeat from turn to
	// turn ("call_0"), so a whole-history map would resolve to the wrong name.
	nameByID := map[string]string{}

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
			fr := map[string]any{
				"name":     name,
				"response": map[string]any{"result": geminiParseJSONOrString(geminiContentToText(m))},
			}
			if geminiVendorID(m.ToolCallID) {
				fr["id"] = m.ToolCallID
			}
			part := map[string]any{"functionResponse": fr}
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
				for _, tc := range m.ToolCalls {
					nameByID[tc.ID] = tc.Name
				}
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
		signed := false
		first := -1
		for _, tc := range m.ToolCalls {
			fc := map[string]any{"name": tc.Name, "args": geminiSafeJSONParse(tc.Arguments)}
			if geminiVendorID(tc.ID) {
				fc["id"] = tc.ID
			}
			part := map[string]any{"functionCall": fc}
			if sig := jsonStr(jsonGet(tc.ProviderMetadata["gemini"], "thoughtSignature")); sig != "" {
				part["thoughtSignature"] = sig
				signed = true
			}
			if first < 0 {
				first = len(parts)
			}
			parts = append(parts, part)
		}
		// Gemini validates the signature on the turn's first functionCall; a
		// turn that carries none of its own gets the documented placeholder.
		if !signed {
			parts[first].(map[string]any)["thoughtSignature"] = geminiSkipSignature
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
			// responseJsonSchema takes standard JSON Schema; responseSchema is the
			// OpenAPI subset Google now steers away from.
			gc["responseJsonSchema"] = rf.JSONSchema
		}
	}
}

func geminiMapFinishReason(fr string) FinishReason {
	switch fr {
	case "STOP":
		return "stop"
	case "MAX_TOKENS", "CONTINUATION":
		// CONTINUATION: the per-request token limit cut generation short.
		return "length"
	case "SAFETY", "RECITATION", "BLOCKLIST", "PROHIBITED_CONTENT", "SPII",
		"IMAGE_SAFETY", "IMAGE_PROHIBITED_CONTENT", "IMAGE_RECITATION":
		return "content_filter"
	default:
		// LANGUAGE, MALFORMED_FUNCTION_CALL, UNEXPECTED_TOOL_CALL,
		// TOO_MANY_TOOL_CALLS, NO_IMAGE, IMAGE_OTHER, OTHER, unspecified, and any
		// value added later have no normalized equivalent.
		return "unknown"
	}
}

// geminiMapUsage counts every billed token: tool-use prompt tokens are input
// and thinking tokens are output, though Gemini reports both apart from
// promptTokenCount / candidatesTokenCount. The total is their sum, which is
// what totalTokenCount documents.
func geminiMapUsage(u any) Usage {
	inp := int(jsonNum(jsonGet(u, "promptTokenCount")) + jsonNum(jsonGet(u, "toolUsePromptTokenCount")))
	out := int(jsonNum(jsonGet(u, "candidatesTokenCount")) + jsonNum(jsonGet(u, "thoughtsTokenCount")))
	return Usage{InputTokens: inp, OutputTokens: out, TotalTokens: inp + out}
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

func geminiMapCode(status int, message string, body any) ProviderCode {
	switch {
	case status == 401 || status == 403:
		return CodeAuthFailed
	// 402 is RESOURCE_EXHAUSTED (prepay credit depleted), the same status as
	// a 429 quota hit.
	case status == 429 || status == 402:
		return CodeRateLimited
	// 408 is a transient timeout; Gemini's troubleshooting guide says to retry it.
	case status >= 500 || status == 408:
		return CodeProviderUnavailable
	}
	if status == 400 || status == 404 || status == 413 || status == 422 {
		// A bad key is a 400 INVALID_ARGUMENT whose ErrorInfo reason is
		// API_KEY_INVALID; it is a credential failure, not a malformed request.
		if geminiErrorReason(body) == "API_KEY_INVALID" {
			return CodeAuthFailed
		}
		// "The input token count (N) exceeds the maximum number of tokens
		// allowed (M)." A looser match ("context", "maximum number of tokens"
		// alone) also catches maxOutputTokens validation errors.
		if strings.Contains(strings.ToLower(message), "exceeds the maximum number of tokens") {
			return CodeContextLengthExceeded
		}
		return CodeInvalidRequest
	}
	// Any other 4xx: the request cannot succeed as sent.
	if status >= 400 && status < 500 {
		return CodeInvalidRequest
	}
	return CodeUnknown
}

// geminiRPCStatus is the HTTP status for each google.rpc status name an error
// frame can carry instead of a numeric code.
var geminiRPCStatus = map[string]int{
	"INVALID_ARGUMENT": 400, "FAILED_PRECONDITION": 400, "OUT_OF_RANGE": 400,
	"UNAUTHENTICATED": 401, "PERMISSION_DENIED": 403, "NOT_FOUND": 404,
	"ABORTED": 409, "ALREADY_EXISTS": 409, "RESOURCE_EXHAUSTED": 429, "CANCELLED": 499,
	"INTERNAL": 500, "UNKNOWN": 500, "DATA_LOSS": 500, "UNIMPLEMENTED": 501,
	"UNAVAILABLE": 503, "DEADLINE_EXCEEDED": 504,
}

// geminiFrameStatus reads the HTTP status a mid-stream error frame stands for:
// error.code, else the google.rpc error.status name, else 0 (which classifies
// as unknown).
func geminiFrameStatus(e any) int {
	if n, ok := jsonGet(e, "code").(float64); ok && n >= 400 && n <= 599 && n == math.Trunc(n) {
		return int(n)
	}
	return geminiRPCStatus[jsonStr(jsonGet(e, "status"))]
}

// geminiErrorReason returns the reason of the first google.rpc.ErrorInfo detail.
func geminiErrorReason(body any) string {
	details, _ := jsonSliceVal(jsonGet(jsonGet(body, "error"), "details"))
	for _, d := range details {
		if r := jsonStr(jsonGet(d, "reason")); r != "" {
			return r
		}
	}
	return ""
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
