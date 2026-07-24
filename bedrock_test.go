package llm

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"io"
	"regexp"
	"strings"
	"testing"
)

const bedrockModel = "anthropic.claude-3-5-sonnet-20241022-v2:0"

// authRe matches a well-formed SigV4 Authorization header for the bedrock service.
var bedrockAuthRe = regexp.MustCompile(
	`^AWS4-HMAC-SHA256 Credential=AKIATEST/\d{8}/us-east-1/bedrock/aws4_request, ` +
		`SignedHeaders=([a-z0-9;.-]+), Signature=([0-9a-f]{64})$`)

func bedrockCfg() BedrockConfig {
	return BedrockConfig{Region: "us-east-1", AccessKeyID: "AKIATEST", SecretAccessKey: "secret/key+x"}
}

func TestBedrockComplete(t *testing.T) {
	body := `{
		"output":{"message":{"content":[{"text":"Hello from Claude"}]}},
		"stopReason":"end_turn",
		"usage":{"inputTokens":11,"outputTokens":7,"totalTokens":18}
	}`
	fake := newFake(resp(200, body))
	chat, err := NewChat(Bedrock, bedrockCfg(), WithHTTPClient(fake))
	if err != nil {
		t.Fatal(err)
	}
	temp := 0.5
	maxTok := 256
	res, err := chat.Complete(context.Background(), ChatInput{
		Model:           bedrockModel,
		Messages:        []ChatMessage{{Role: "system", Content: "Be brief"}, {Role: "user", Content: "Hi"}},
		Temperature:     &temp,
		MaxOutputTokens: &maxTok,
		Stop:            []string{"STOP"},
	})
	if err != nil {
		t.Fatal(err)
	}

	// normalized result
	if res.Message.Content != "Hello from Claude" {
		t.Fatalf("bad content: %q", res.Message.Content)
	}
	if res.FinishReason != "stop" {
		t.Fatalf("bad finishReason: %q", res.FinishReason)
	}
	if res.Usage.InputTokens != 11 || res.Usage.OutputTokens != 7 || res.Usage.TotalTokens != 18 {
		t.Fatalf("bad usage: %+v", res.Usage)
	}
	if res.Model != bedrockModel {
		t.Fatalf("bad model: %q", res.Model)
	}

	req := fake.last()

	// URL + path: colon in the model id must be percent-encoded and preserved on the wire.
	wantURL := "https://bedrock-runtime.us-east-1.amazonaws.com/model/" +
		"anthropic.claude-3-5-sonnet-20241022-v2%3A0/converse"
	if req.URL.String() != wantURL {
		t.Fatalf("bad url: %s", req.URL.String())
	}
	if req.URL.RequestURI() != "/model/anthropic.claude-3-5-sonnet-20241022-v2%3A0/converse" {
		t.Fatalf("bad wire path: %s", req.URL.RequestURI())
	}

	// SigV4 Authorization header present + well-formed.
	auth := req.Header.Get("Authorization")
	m := bedrockAuthRe.FindStringSubmatch(auth)
	if m == nil {
		t.Fatalf("malformed Authorization: %q", auth)
	}
	if m[1] != "content-type;host;x-amz-content-sha256;x-amz-date" {
		t.Fatalf("unexpected SignedHeaders: %q", m[1])
	}
	if req.Header.Get("X-Amz-Date") == "" {
		t.Fatalf("missing X-Amz-Date")
	}
	// no session token configured → no security-token header
	if req.Header.Get("X-Amz-Security-Token") != "" {
		t.Fatalf("unexpected X-Amz-Security-Token")
	}
	// X-Amz-Content-Sha256 must equal the sha256 of the exact bytes sent.
	wantHash := sha256.Sum256([]byte(req.Body))
	if got := req.Header.Get("X-Amz-Content-Sha256"); got != hex.EncodeToString(wantHash[:]) {
		t.Fatalf("payload hash mismatch: header=%s body-hash=%s", got, hex.EncodeToString(wantHash[:]))
	}
	if req.Header.Get("Content-Type") != "application/json" {
		t.Fatalf("bad content-type: %s", req.Header.Get("Content-Type"))
	}

	// request body shape (Converse).
	var sent map[string]any
	if err := json.Unmarshal([]byte(req.Body), &sent); err != nil {
		t.Fatalf("body not json: %v", err)
	}
	sys, _ := sent["system"].([]any)
	if len(sys) != 1 || sys[0].(map[string]any)["text"] != "Be brief" {
		t.Fatalf("bad system: %v", sent["system"])
	}
	msgs, _ := sent["messages"].([]any)
	if len(msgs) != 1 {
		t.Fatalf("bad messages: %v", sent["messages"])
	}
	first := msgs[0].(map[string]any)
	if first["role"] != "user" {
		t.Fatalf("bad first role: %v", first["role"])
	}
	inf, _ := sent["inferenceConfig"].(map[string]any)
	if inf["maxTokens"].(float64) != 256 {
		t.Fatalf("bad maxTokens: %v", inf["maxTokens"])
	}
	if inf["temperature"].(float64) != 0.5 {
		t.Fatalf("bad temperature: %v", inf["temperature"])
	}
	stops, _ := inf["stopSequences"].([]any)
	if len(stops) != 1 || stops[0] != "STOP" {
		t.Fatalf("bad stopSequences: %v", inf["stopSequences"])
	}
}

func TestBedrockToolUseResult(t *testing.T) {
	body := `{
		"output":{"message":{"content":[
			{"toolUse":{"toolUseId":"tool-1","name":"get_weather","input":{"city":"Paris"}}}
		]}},
		"stopReason":"tool_use",
		"usage":{"inputTokens":20,"outputTokens":5}
	}`
	fake := newFake(resp(200, body))
	chat, _ := NewChat(Bedrock, bedrockCfg(), WithHTTPClient(fake))
	res, err := chat.Complete(context.Background(), ChatInput{
		Model:    bedrockModel,
		Messages: []ChatMessage{{Role: "user", Content: "weather?"}},
		Tools: []ToolDefinition{{
			Name:        "get_weather",
			Description: "Get weather",
			Parameters:  map[string]any{"type": "object", "properties": map[string]any{"city": map[string]any{"type": "string"}}},
		}},
		ToolChoice: &ToolChoice{Mode: "required"},
	})
	if err != nil {
		t.Fatal(err)
	}
	if res.FinishReason != "tool_calls" {
		t.Fatalf("bad finishReason: %q", res.FinishReason)
	}
	if len(res.Message.ToolCalls) != 1 {
		t.Fatalf("expected 1 tool call, got %d", len(res.Message.ToolCalls))
	}
	tc := res.Message.ToolCalls[0]
	if tc.ID != "tool-1" || tc.Name != "get_weather" {
		t.Fatalf("bad tool call: %+v", tc)
	}
	if tc.Arguments != `{"city":"Paris"}` {
		t.Fatalf("bad tool args: %q", tc.Arguments)
	}
	// usage totalTokens defaulted to input+output when absent.
	if res.Usage.TotalTokens != 25 {
		t.Fatalf("bad total: %d", res.Usage.TotalTokens)
	}

	// toolConfig shape in the request body.
	var sent map[string]any
	_ = json.Unmarshal([]byte(fake.last().Body), &sent)
	toolCfg, _ := sent["toolConfig"].(map[string]any)
	if toolCfg == nil {
		t.Fatalf("missing toolConfig")
	}
	tools, _ := toolCfg["tools"].([]any)
	if len(tools) != 1 {
		t.Fatalf("bad tools: %v", toolCfg["tools"])
	}
	spec := tools[0].(map[string]any)["toolSpec"].(map[string]any)
	if spec["name"] != "get_weather" {
		t.Fatalf("bad toolSpec name: %v", spec["name"])
	}
	if _, ok := spec["inputSchema"].(map[string]any)["json"]; !ok {
		t.Fatalf("missing inputSchema.json: %v", spec["inputSchema"])
	}
	if _, ok := toolCfg["toolChoice"].(map[string]any)["any"]; !ok {
		t.Fatalf("expected toolChoice.any (required): %v", toolCfg["toolChoice"])
	}
}

func TestBedrockSessionTokenSigning(t *testing.T) {
	fake := newFake(resp(200, `{"output":{"message":{"content":[{"text":"ok"}]}},"stopReason":"end_turn","usage":{"inputTokens":1,"outputTokens":1}}`))
	cfg := bedrockCfg()
	cfg.SessionToken = "FQoGZXIvSESSIONtoken=="
	chat, _ := NewChat(Bedrock, cfg, WithHTTPClient(fake))
	if _, err := chat.Complete(context.Background(), ChatInput{
		Model:    bedrockModel,
		Messages: []ChatMessage{{Role: "user", Content: "hi"}},
	}); err != nil {
		t.Fatal(err)
	}
	req := fake.last()
	if got := req.Header.Get("X-Amz-Security-Token"); got != "FQoGZXIvSESSIONtoken==" {
		t.Fatalf("bad security token header: %q", got)
	}
	// the token must be folded into SignedHeaders.
	m := bedrockAuthRe.FindStringSubmatch(req.Header.Get("Authorization"))
	if m == nil {
		t.Fatalf("malformed Authorization: %q", req.Header.Get("Authorization"))
	}
	if m[1] != "content-type;host;x-amz-content-sha256;x-amz-date;x-amz-security-token" {
		t.Fatalf("session token not in SignedHeaders: %q", m[1])
	}
}

func TestBedrockBaseURLOverride(t *testing.T) {
	fake := newFake(resp(200, `{"output":{"message":{"content":[{"text":"ok"}]}},"stopReason":"end_turn"}`))
	cfg := bedrockCfg()
	cfg.BaseURL = "https://bedrock.example.com/"
	chat, err := NewChat(Bedrock, cfg, WithHTTPClient(fake))
	if err != nil {
		t.Fatal(err)
	}
	if _, err := chat.Complete(context.Background(), ChatInput{
		Model:    "amazon.titan-text-express-v1",
		Messages: []ChatMessage{{Role: "user", Content: "hi"}},
	}); err != nil {
		t.Fatal(err)
	}
	req := fake.last()
	if req.URL.String() != "https://bedrock.example.com/model/amazon.titan-text-express-v1/converse" {
		t.Fatalf("bad override url: %s", req.URL.String())
	}
	// host in the signature must be the override host.
	if !strings.Contains(req.Header.Get("Authorization"), "/us-east-1/bedrock/aws4_request") {
		t.Fatalf("bad credential scope: %s", req.Header.Get("Authorization"))
	}
}

func TestBedrockStream(t *testing.T) {
	body := `{
		"output":{"message":{"content":[
			{"text":"streamed text"},
			{"toolUse":{"toolUseId":"t1","name":"fn","input":{"a":1}}}
		]}},
		"stopReason":"tool_use",
		"usage":{"inputTokens":3,"outputTokens":4,"totalTokens":7}
	}`
	fake := newFake(resp(200, body))
	chat, _ := NewChat(Bedrock, bedrockCfg(), WithHTTPClient(fake))
	stream, err := chat.Stream(context.Background(), ChatInput{
		Model:    bedrockModel,
		Messages: []ChatMessage{{Role: "user", Content: "go"}},
	})
	if err != nil {
		t.Fatal(err)
	}
	defer stream.Close()

	var text string
	var toolDeltas []*ToolCallDelta
	var finish FinishReason
	var usage *Usage
	for {
		d, err := stream.Recv()
		if err == io.EOF {
			break
		}
		if err != nil {
			t.Fatal(err)
		}
		text += d.ContentDelta
		if d.ToolCallDelta != nil {
			toolDeltas = append(toolDeltas, d.ToolCallDelta)
		}
		if d.FinishReason != "" {
			finish = d.FinishReason
		}
		if d.Usage != nil {
			usage = d.Usage
		}
	}
	if text != "streamed text" {
		t.Fatalf("bad streamed text: %q", text)
	}
	if len(toolDeltas) != 1 || toolDeltas[0].Index != 0 || toolDeltas[0].ID != "t1" ||
		toolDeltas[0].FunctionName != "fn" || toolDeltas[0].ArgumentsDelta != `{"a":1}` {
		t.Fatalf("bad tool delta: %+v", toolDeltas)
	}
	if finish != "tool_calls" {
		t.Fatalf("bad finish: %q", finish)
	}
	if usage == nil || usage.TotalTokens != 7 {
		t.Fatalf("bad stream usage: %+v", usage)
	}
	// stream() hits the non-streaming /converse endpoint (documented fallback).
	if !strings.HasSuffix(fake.last().URL.RequestURI(), "/converse") {
		t.Fatalf("stream should call /converse, got %s", fake.last().URL.RequestURI())
	}
}

func TestBedrockErrorMapping(t *testing.T) {
	tests := []struct {
		name   string
		status int
		body   string
		want   ProviderCode
	}{
		{"auth", 403, `{"message":"The security token included in the request is invalid."}`, CodeAuthFailed},
		{"throttle", 429, `{"message":"Too many requests"}`, CodeRateLimited},
		{"server", 500, `{"message":"internal"}`, CodeProviderUnavailable},
		{"throttle-400", 400, `{"message":"ThrottlingException: rate exceeded"}`, CodeRateLimited},
		{"context", 400, `{"message":"input is too long for requested model"}`, CodeContextLengthExceeded},
		{"validation", 400, `{"message":"malformed input"}`, CodeInvalidRequest},
		{"typed", 400, `{"__type":"ValidationException"}`, CodeInvalidRequest}, // no context/token keyword
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			fake := newFake(resp(tt.status, tt.body))
			chat, _ := NewChat(Bedrock, bedrockCfg(), WithHTTPClient(fake))
			_, err := chat.Complete(context.Background(), ChatInput{
				Model:    bedrockModel,
				Messages: []ChatMessage{{Role: "user", Content: "x"}},
			})
			var ce *ConnectorError
			if !errors.As(err, &ce) {
				t.Fatalf("expected *ConnectorError, got %v", err)
			}
			if ce.ProviderCode != tt.want {
				t.Fatalf("status %d: want %s, got %s (msg=%q)", tt.status, tt.want, ce.ProviderCode, ce.ProviderMessage)
			}
			if ce.StatusCode == nil || *ce.StatusCode != tt.status {
				t.Fatalf("bad status code on error: %+v", ce.StatusCode)
			}
		})
	}
}

func TestBedrockRetryAfterCause(t *testing.T) {
	fake := newFake(respH(429, `{"message":"slow down"}`, "Retry-After", "30"))
	chat, _ := NewChat(Bedrock, bedrockCfg(), WithHTTPClient(fake))
	_, err := chat.Complete(context.Background(), ChatInput{
		Model:    bedrockModel,
		Messages: []ChatMessage{{Role: "user", Content: "x"}},
	})
	var ce *ConnectorError
	if !errors.As(err, &ce) {
		t.Fatalf("expected *ConnectorError, got %v", err)
	}
	cause, _ := ce.Cause.(map[string]any)
	if cause == nil {
		t.Fatalf("expected cause map, got %T", ce.Cause)
	}
	if cause["retryAfter"] != "30" {
		t.Fatalf("bad retryAfter: %v", cause["retryAfter"])
	}
	if cause["retryAfterSeconds"] != 30 {
		t.Fatalf("bad retryAfterSeconds: %v", cause["retryAfterSeconds"])
	}
}

// recomputeBedrockSig independently re-derives the SigV4 signature over a given
// canonical path, using the exact amzDate/payload-hash the connector emitted (read
// back from the captured request, so it is time-independent). The SignedHeaders
// for these fixtures are content-type;host;x-amz-content-sha256;x-amz-date, already
// in sorted order.
func recomputeBedrockSig(req recordedRequest, secret, canonicalPath string) string {
	amzDate := req.Header.Get("X-Amz-Date")
	dateStamp := amzDate[:8]
	hashedPayload := req.Header.Get("X-Amz-Content-Sha256")
	canonicalHeaders := "content-type:" + collapseWS(req.Header.Get("Content-Type")) + "\n" +
		"host:" + collapseWS(req.URL.Host) + "\n" +
		"x-amz-content-sha256:" + hashedPayload + "\n" +
		"x-amz-date:" + amzDate + "\n"
	signedHeaders := "content-type;host;x-amz-content-sha256;x-amz-date"
	canonicalRequest := "POST\n" + canonicalPath + "\n" + "" + "\n" +
		canonicalHeaders + "\n" + signedHeaders + "\n" + hashedPayload
	credentialScope := dateStamp + "/us-east-1/bedrock/aws4_request"
	stringToSign := "AWS4-HMAC-SHA256\n" + amzDate + "\n" + credentialScope + "\n" +
		hexSHA256([]byte(canonicalRequest))
	kDate := hmacSHA256([]byte("AWS4"+secret), dateStamp)
	kRegion := hmacSHA256(kDate, "us-east-1")
	kService := hmacSHA256(kRegion, "bedrock")
	kSigning := hmacSHA256(kService, "aws4_request")
	return hex.EncodeToString(hmacSHA256(kSigning, stringToSign))
}

// TestBedrockCanonicalPathDoubleEncoded proves the WIRE path is single-encoded
// while the SigV4 canonical path is DOUBLE-encoded — the fix for the 403
// SignatureDoesNotMatch on any model id containing a reserved char (e.g. ':').
func TestBedrockCanonicalPathDoubleEncoded(t *testing.T) {
	fake := newFake(resp(200, `{"output":{"message":{"content":[{"text":"ok"}]}},"stopReason":"end_turn","usage":{"inputTokens":1,"outputTokens":1}}`))
	chat, _ := NewChat(Bedrock, bedrockCfg(), WithHTTPClient(fake))
	if _, err := chat.Complete(context.Background(), ChatInput{
		Model:    bedrockModel, // "...-v2:0" — contains a ':'
		Messages: []ChatMessage{{Role: "user", Content: "hi"}},
	}); err != nil {
		t.Fatal(err)
	}
	req := fake.last()

	singlePath := "/model/" + bedrockEscapeModelID(bedrockModel) + "/converse"
	doublePath := "/model/" + bedrockEscapeModelID(bedrockEscapeModelID(bedrockModel)) + "/converse"

	// The wire path is single-encoded (%3A present, %253A absent).
	if got := req.URL.RequestURI(); got != singlePath {
		t.Fatalf("wire path must be single-encoded\n got=%s\nwant=%s", got, singlePath)
	}
	if !strings.Contains(singlePath, "-v2%3A0/") || strings.Contains(singlePath, "%253A") {
		t.Fatalf("singlePath should be single-encoded: %s", singlePath)
	}
	if !strings.Contains(doublePath, "-v2%253A0/") {
		t.Fatalf("doublePath should be double-encoded (%%253A): %s", doublePath)
	}
	if singlePath == doublePath {
		t.Fatal("single and double encodings must differ for a colon-bearing model id")
	}

	m := bedrockAuthRe.FindStringSubmatch(req.Header.Get("Authorization"))
	if m == nil {
		t.Fatalf("malformed Authorization: %q", req.Header.Get("Authorization"))
	}
	gotSig := m[2]

	secret := bedrockCfg().SecretAccessKey
	sigDouble := recomputeBedrockSig(req, secret, doublePath)
	sigSingle := recomputeBedrockSig(req, secret, singlePath)

	// The emitted signature must be computed over the DOUBLE-encoded canonical path.
	if gotSig != sigDouble {
		t.Fatalf("signature is not over the double-encoded canonical path\n got=%s\n dbl=%s\n sgl=%s", gotSig, sigDouble, sigSingle)
	}
	// ...and must NOT match the (previously buggy) single-encoded canonical path.
	if gotSig == sigSingle {
		t.Fatal("regression: signature matches the single-encoded canonical path")
	}
}

// TestBedrockHeaderCasingDeterministicSignature proves a passthrough header that
// differs only in CASE from the default Content-Type collapses to a single
// canonical header, so the SigV4-signed value matches the wire value. Before the
// case-insensitive merge, "Content-Type" and "content-type" coexisted in the map
// and the signer's lowercasing collapsed them nondeterministically → sporadic
// SignatureDoesNotMatch.
func TestBedrockHeaderCasingDeterministicSignature(t *testing.T) {
	fake := newFake(resp(200, `{"output":{"message":{"content":[{"text":"ok"}]}},"stopReason":"end_turn","usage":{"inputTokens":1,"outputTokens":1}}`))
	chat, _ := NewChat(Bedrock, bedrockCfg(), WithHTTPClient(fake))
	model := "anthropic.claude-3-haiku" // no reserved chars → wire path == canonical path
	if _, err := chat.Complete(context.Background(), ChatInput{
		Model:       model,
		Messages:    []ChatMessage{{Role: "user", Content: "hi"}},
		Passthrough: &Passthrough{Headers: map[string]string{"content-type": "application/json; charset=utf-8"}},
	}); err != nil {
		t.Fatal(err)
	}
	req := fake.last()

	if got := req.Header.Get("Content-Type"); got != "application/json; charset=utf-8" {
		t.Fatalf("content-type override not applied: %q", got)
	}
	if vals := req.Header.Values("Content-Type"); len(vals) != 1 {
		t.Fatalf("expected exactly one Content-Type on the wire, got %v", vals)
	}

	m := bedrockAuthRe.FindStringSubmatch(req.Header.Get("Authorization"))
	if m == nil {
		t.Fatalf("malformed Authorization: %q", req.Header.Get("Authorization"))
	}
	path := "/model/" + bedrockEscapeModelID(model) + "/converse"
	if want := recomputeBedrockSig(req, bedrockCfg().SecretAccessKey, path); m[2] != want {
		t.Fatalf("signature not over the single canonical content-type\n got=%s\nwant=%s", m[2], want)
	}
}

func TestBedrockWrongConfig(t *testing.T) {
	_, err := NewChat(Bedrock, OpenAICompatConfig{APIKey: "k"})
	var ce *ConnectorError
	if !errors.As(err, &ce) || ce.ProviderCode != CodeInvalidRequest {
		t.Fatalf("expected invalid_request for wrong config type, got %v", err)
	}
}
