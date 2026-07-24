package llm

import (
	"context"
	"io"
	"net/http"
	"net/url"
	"strconv"
	"strings"
	"time"
)

// bedrockDefaultMaxTokens is used when neither ChatInput.MaxOutputTokens nor
// BedrockConfig.DefaultMaxTokens is set (Converse requires inferenceConfig.maxTokens).
const bedrockDefaultMaxTokens = 4096

// bedrockService is the AWS service name folded into the SigV4 credential scope.
const bedrockService = "bedrock"

func init() { registerNativeChat(Bedrock, newBedrockChat) }

// newBedrockChat builds the native adapter for AWS Bedrock's unified Converse API
// (POST /model/{modelId}/converse). Requests are signed with hand-rolled AWS SigV4
// (signAWSv4) — zero third-party deps, same approach as the notifications SES/SNS
// connectors. Converse is structurally different from OpenAI (top-level system[],
// typed content blocks, inferenceConfig, toolConfig, tool results in a user turn),
// so this is a full translation layer emitting the identical normalized ChatResult
// / ChatStreamDelta / ConnectorError.
func newBedrockChat(cfg ChatConfig, o facadeOptions) (chatConnector, error) {
	c, ok := cfg.(BedrockConfig)
	if !ok {
		return nil, invalidReq("bedrock requires a BedrockConfig")
	}
	origin := c.BaseURL
	if origin == "" {
		origin = "https://bedrock-runtime." + c.Region + ".amazonaws.com"
	}
	origin = strings.TrimRight(origin, "/")
	u, err := url.Parse(origin)
	if err != nil || u.Host == "" {
		return nil, invalidReq("Bedrock: invalid `baseUrl`/`region` — could not derive a host from '" + origin + "'")
	}
	return &bedrockConnector{
		baseConnector: baseConnector{o.httpClient},
		cfg:           c,
		origin:        origin,
		host:          u.Host,
	}, nil
}

type bedrockConnector struct {
	baseConnector
	cfg    BedrockConfig
	origin string // baseURL, trailing slashes trimmed
	host   string // host derived from origin (participates in the SigV4 signature)
}

func (c *bedrockConnector) complete(ctx context.Context, in ChatInput) (*ChatResult, error) {
	url, bodyBytes, headers, err := c.buildSignedRequest(in)
	if err != nil {
		return nil, err
	}
	// Send the EXACT signed bytes so the request the transport writes matches the
	// SigV4 payload hash + canonical path.
	resp, err := c.post(ctx, url, "application/json", bodyBytes, headers)
	if err != nil {
		return nil, err
	}
	raw := decodeAny(readBody(resp))
	if !okStatus(resp.StatusCode) {
		return nil, bedrockVendorError(resp, raw)
	}
	if raw == nil {
		return nil, emptyBodyError(resp.StatusCode)
	}
	return c.parseResult(raw, in.Model), nil
}

// stream is a NON-INCREMENTAL fallback, faithfully mirroring the TS connector.
// Bedrock's converse-stream uses AWS's binary event-stream framing
// (application/vnd.amazon.eventstream), not SSE; this port does NOT parse that
// binary protocol. Instead it issues a single Converse call and yields the full
// result as deltas: an optional content delta, one delta per tool call, then a
// terminal finish/usage/raw delta. True incremental streaming (a binary
// event-stream parser) is a documented follow-up.
func (c *bedrockConnector) stream(ctx context.Context, in ChatInput) (*ChatStream, error) {
	result, err := c.complete(ctx, in)
	if err != nil {
		return nil, err
	}
	var deltas []*ChatStreamDelta
	if result.Message.Content != "" {
		deltas = append(deltas, &ChatStreamDelta{ContentDelta: result.Message.Content})
	}
	for i, tc := range result.Message.ToolCalls {
		deltas = append(deltas, &ChatStreamDelta{ToolCallDelta: &ToolCallDelta{
			Index:          i,
			ID:             tc.ID,
			FunctionName:   tc.Name,
			ArgumentsDelta: tc.Arguments,
		}})
	}
	usage := result.Usage
	deltas = append(deltas, &ChatStreamDelta{
		FinishReason: result.FinishReason,
		Usage:        &usage,
		Raw:          result.Raw,
	})

	idx := 0
	recv := func() (*ChatStreamDelta, error) {
		if idx >= len(deltas) {
			return nil, io.EOF
		}
		d := deltas[idx]
		idx++
		return d, nil
	}
	return &ChatStream{recv: recv, closer: func() error { return nil }}, nil
}

// setHeaderCI sets m[k]=v, first removing any existing entry whose name matches
// k case-insensitively — so an HTTP header appears at most once regardless of the
// casing each source used (later writer wins).
func setHeaderCI(m map[string]string, k, v string) {
	for existing := range m {
		if strings.EqualFold(existing, k) {
			delete(m, existing)
		}
	}
	m[k] = v
}

// buildSignedRequest serializes the Converse body ONCE, then SigV4-signs those
// exact bytes. The WIRE path is single-encoded; the SigV4 canonical path must be
// DOUBLE-encoded because every non-S3 AWS service URI-encodes the received path a
// second time before recomputing the signature (see bedrockEscapeModelID).
func (c *bedrockConnector) buildSignedRequest(in ChatInput) (string, []byte, map[string]string, error) {
	bodyBytes, err := marshalJSON(c.buildRequest(in))
	if err != nil {
		return "", nil, nil, invalidReq("bedrock: failed to encode request body")
	}
	// wirePath goes on the wire (single-encoded); canonicalPath goes into the
	// signature (double-encoded). They differ for any model id containing a
	// reserved char (e.g. the ':' in "...-v2:0", cross-region "us." profiles, ARNs).
	wirePath := "/model/" + bedrockEscapeModelID(in.Model) + "/converse"
	canonicalPath := "/model/" + bedrockEscapeModelID(bedrockEscapeModelID(in.Model)) + "/converse"
	// A per-request Passthrough.Query must be BOTH signed (SigV4 CanonicalQuery)
	// and put on the wire, using the same canonical (sorted, RFC3986) encoding, so
	// the signature matches what the transport sends (mirrors the openai-compat
	// connector). buildCanonicalQuery is safe on a nil/empty map.
	var canonicalQuery string
	if in.Passthrough != nil {
		canonicalQuery = buildCanonicalQuery(in.Passthrough.Query)
	}
	reqURL := c.origin + wirePath
	if canonicalQuery != "" {
		reqURL += "?" + canonicalQuery
	}

	// Headers folded into the signature: Content-Type + config.Headers +
	// per-request passthrough headers (mirrors the TS `extra` set). Merge
	// case-insensitively: HTTP header names are case-insensitive, but a Go map
	// keys on the exact string, so a config/passthrough "content-type" next to the
	// default "Content-Type" would leave TWO entries. The SigV4 signer lowercases
	// keys when canonicalizing, so the duplicate collapses nondeterministically
	// (map order) to a value that may differ from what the wire sends → sporadic
	// SignatureDoesNotMatch. setHeaderCI keeps one entry per name (last wins:
	// Content-Type default < config < passthrough).
	extra := map[string]string{"Content-Type": "application/json"}
	for k, v := range c.cfg.Headers {
		setHeaderCI(extra, k, v)
	}
	if in.Passthrough != nil {
		for k, v := range in.Passthrough.Headers {
			setHeaderCI(extra, k, v)
		}
	}

	signed := signAWSv4(sigV4Params{
		Method:          "POST",
		Service:         bedrockService,
		Region:          c.cfg.Region,
		Host:            c.host,
		Path:            canonicalPath,
		CanonicalQuery:  canonicalQuery,
		Payload:         bodyBytes,
		AccessKeyID:     c.cfg.AccessKeyID,
		SecretAccessKey: c.cfg.SecretAccessKey,
		SessionToken:    c.cfg.SessionToken,
		Signed:          extra,
		Now:             time.Now(),
	})

	headers := map[string]string{}
	for k, v := range extra {
		headers[k] = v
	}
	for k, v := range signed {
		// Host is derived by the transport from the URL; attaching it as a request
		// header is a no-op (net/http uses req.Host), so drop it.
		if strings.EqualFold(k, "Host") {
			continue
		}
		headers[k] = v
	}
	return reqURL, bodyBytes, headers, nil
}

func (c *bedrockConnector) buildRequest(in ChatInput) map[string]any {
	system, messages := bedrockSplitMessages(in.Messages)

	body := map[string]any{"messages": messages}
	if len(system) > 0 {
		body["system"] = system
	}

	maxTokens := bedrockDefaultMaxTokens
	if c.cfg.DefaultMaxTokens != 0 {
		maxTokens = c.cfg.DefaultMaxTokens
	}
	if in.MaxOutputTokens != nil {
		maxTokens = *in.MaxOutputTokens
	}
	inferenceConfig := map[string]any{"maxTokens": maxTokens}
	if in.Temperature != nil {
		inferenceConfig["temperature"] = *in.Temperature
	}
	if in.TopP != nil {
		inferenceConfig["topP"] = *in.TopP
	}
	if len(in.Stop) > 0 {
		inferenceConfig["stopSequences"] = in.Stop
	}
	body["inferenceConfig"] = inferenceConfig

	if len(in.Tools) > 0 {
		tools := make([]any, 0, len(in.Tools))
		for _, t := range in.Tools {
			spec := map[string]any{"name": t.Name}
			if t.Description != "" {
				spec["description"] = t.Description
			}
			var schema any = t.Parameters
			if t.Parameters == nil {
				schema = map[string]any{"type": "object", "properties": map[string]any{}}
			}
			spec["inputSchema"] = map[string]any{"json": schema}
			tools = append(tools, map[string]any{"toolSpec": spec})
		}
		toolConfig := map[string]any{"tools": tools}
		if tc := bedrockMapToolChoice(in.ToolChoice); tc != nil {
			toolConfig["toolChoice"] = tc
		}
		body["toolConfig"] = toolConfig
	}
	// reasoning control is model-dependent on Bedrock → not auto-mapped;
	// pass additionalModelRequestFields via Passthrough.Body.

	applyBodyPassthrough(body, in.Passthrough)
	return body
}

func (c *bedrockConnector) parseResult(raw any, requestedModel string) *ChatResult {
	blocks, _ := jsonSliceVal(jsonGet(jsonGet(jsonGet(raw, "output"), "message"), "content"))
	var text strings.Builder
	var toolCalls []ToolCall
	for _, b := range blocks {
		if t, ok := jsonGet(b, "text").(string); ok {
			text.WriteString(t)
			continue
		}
		if tu := jsonGet(b, "toolUse"); tu != nil {
			args := "{}"
			if input := jsonGet(tu, "input"); input != nil {
				if enc, err := marshalJSON(input); err == nil {
					args = string(enc)
				}
			}
			toolCalls = append(toolCalls, ToolCall{
				ID:        jsonStr(jsonGet(tu, "toolUseId")),
				Name:      jsonStr(jsonGet(tu, "name")),
				Arguments: args,
			})
		}
	}

	usage := jsonGet(raw, "usage")
	inTok := int(jsonNum(jsonGet(usage, "inputTokens")))
	outTok := int(jsonNum(jsonGet(usage, "outputTokens")))
	total := int(jsonNum(jsonGet(usage, "totalTokens")))
	if total == 0 {
		total = inTok + outTok
	}

	return &ChatResult{
		Message:      AssistantMessage{Role: "assistant", Content: text.String(), ToolCalls: toolCalls},
		FinishReason: bedrockMapStopReason(jsonStr(jsonGet(raw, "stopReason"))),
		Usage:        Usage{InputTokens: inTok, OutputTokens: outTok, TotalTokens: total},
		Model:        requestedModel,
		Raw:          raw,
	}
}

func bedrockVendorError(resp *http.Response, raw any) *ConnectorError {
	msg := bedrockExtractError(raw)
	if msg == "" {
		msg = "HTTP " + strconv.Itoa(resp.StatusCode)
	}
	return vendorError(resp.StatusCode, bedrockMapCode(resp.StatusCode, msg), msg, raw, resp.Header)
}

// ---- module-private mappers ----

// bedrockSplitMessages hoists system turns into a top-level system[] and maps the
// rest into Converse {role, content[]} turns; a tool result folds into the
// preceding user turn (Converse carries tool results in a user message).
func bedrockSplitMessages(messages []ChatMessage) (system []any, out []any) {
	for _, m := range messages {
		switch m.Role {
		case "system":
			if t := bedrockContentToText(m); t != "" {
				system = append(system, map[string]any{"text": t})
			}
		case "tool":
			block := map[string]any{"toolResult": map[string]any{
				"toolUseId": m.ToolCallID,
				"content":   []any{map[string]any{"text": bedrockContentToText(m)}},
			}}
			if n := len(out); n > 0 {
				if last, ok := out[n-1].(map[string]any); ok && last["role"] == "user" {
					if lc, ok := last["content"].([]any); ok && bedrockIsToolResultContent(lc) {
						last["content"] = append(lc, block)
						continue
					}
				}
			}
			out = append(out, map[string]any{"role": "user", "content": []any{block}})
		default:
			out = append(out, map[string]any{"role": m.Role, "content": bedrockMapContentBlocks(m)})
		}
	}
	return system, out
}

func bedrockContentToText(m ChatMessage) string {
	if len(m.ContentParts) > 0 {
		var b strings.Builder
		for _, p := range m.ContentParts {
			if p.Type == "text" {
				b.WriteString(p.Text)
			}
		}
		return b.String()
	}
	return m.Content
}

func bedrockMapContentBlocks(m ChatMessage) []any {
	blocks := []any{}
	if len(m.ContentParts) > 0 {
		for _, p := range m.ContentParts {
			if p.Type == "text" {
				blocks = append(blocks, map[string]any{"text": p.Text})
			} else {
				blocks = append(blocks, map[string]any{"image": map[string]any{
					"format": bedrockImageFormat(p.MediaType),
					"source": map[string]any{"bytes": p.Base64},
				}})
			}
		}
	} else if m.Content != "" {
		blocks = append(blocks, map[string]any{"text": m.Content})
	}
	if m.Role == "assistant" && len(m.ToolCalls) > 0 {
		for _, tc := range m.ToolCalls {
			blocks = append(blocks, map[string]any{"toolUse": map[string]any{
				"toolUseId": tc.ID,
				"name":      tc.Name,
				"input":     bedrockSafeJSONParse(tc.Arguments),
			}})
		}
	}
	if len(blocks) == 0 {
		blocks = append(blocks, map[string]any{"text": ""})
	}
	return blocks
}

func bedrockIsToolResultContent(content []any) bool {
	if len(content) == 0 {
		return false
	}
	m, ok := content[0].(map[string]any)
	if !ok {
		return false
	}
	_, has := m["toolResult"]
	return has
}

func bedrockMapToolChoice(tc *ToolChoice) map[string]any {
	if tc == nil {
		return nil
	}
	switch tc.Mode {
	case "auto":
		return map[string]any{"auto": map[string]any{}}
	case "required":
		return map[string]any{"any": map[string]any{}}
	case "function":
		return map[string]any{"tool": map[string]any{"name": tc.FunctionName}}
	default:
		// Bedrock has no "none"; omit toolChoice.
		return nil
	}
}

func bedrockMapStopReason(sr string) FinishReason {
	switch sr {
	case "end_turn", "stop_sequence":
		return "stop"
	case "max_tokens":
		return "length"
	case "tool_use":
		return "tool_calls"
	case "content_filtered", "guardrail_intervened":
		return "content_filter"
	default:
		return "unknown"
	}
}

func bedrockImageFormat(mediaType string) string {
	m := strings.ToLower(mediaType)
	switch {
	case strings.Contains(m, "png"):
		return "png"
	case strings.Contains(m, "jpeg"), strings.Contains(m, "jpg"):
		return "jpeg"
	case strings.Contains(m, "gif"):
		return "gif"
	case strings.Contains(m, "webp"):
		return "webp"
	default:
		return "png"
	}
}

func bedrockSafeJSONParse(s string) any {
	if v := decodeAny([]byte(s)); v != nil {
		return v
	}
	return map[string]any{}
}

func bedrockExtractError(body any) string {
	switch b := body.(type) {
	case string:
		return b
	case map[string]any:
		if s := jsonStr(b["message"]); s != "" {
			return s
		}
		if s := jsonStr(b["Message"]); s != "" {
			return s
		}
		if s := jsonStr(b["__type"]); s != "" {
			return s
		}
	}
	return ""
}

func bedrockMapCode(status int, message string) ProviderCode {
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
		if strings.Contains(msg, "throttl") {
			return CodeRateLimited
		}
		if strings.Contains(msg, "too long") || strings.Contains(msg, "context") ||
			strings.Contains(msg, "token") || strings.Contains(msg, "max") {
			return CodeContextLengthExceeded
		}
		return CodeInvalidRequest
	}
	return CodeUnknown
}

// bedrockEscapeModelID percent-encodes a model id for a path segment, leaving
// only the RFC 3986 unreserved set (A-Za-z0-9-._~) untouched so that sub-delims
// (! * ' ( )) are always percent-encoded — matching AWS SigV4 canonicalization
// (a versioned id like "...-v2:0" becomes "...-v2%3A0"). The wire path applies
// this ONCE; the SigV4 canonical path applies it TWICE (so "%3A" → "%253A"),
// because non-S3 AWS services re-encode the received path before recomputing the
// signature — the double-encoded canonical URI is what the server reconstructs
// and verifies.
func bedrockEscapeModelID(s string) string {
	const upperhex = "0123456789ABCDEF"
	var b strings.Builder
	for i := 0; i < len(s); i++ {
		c := s[i]
		if (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') || (c >= '0' && c <= '9') ||
			c == '-' || c == '_' || c == '.' || c == '~' {
			b.WriteByte(c)
			continue
		}
		b.WriteByte('%')
		b.WriteByte(upperhex[c>>4])
		b.WriteByte(upperhex[c&0xf])
	}
	return b.String()
}
