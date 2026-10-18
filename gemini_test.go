package llm

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"reflect"
	"strconv"
	"strings"
	"testing"
)

func TestGeminiComplete(t *testing.T) {
	body := `{"modelVersion":"gemini-3.8-flash-001","candidates":[{"content":{"role":"model","parts":[{"text":"Hi there"}]},"finishReason":"STOP"}],"usageMetadata":{"promptTokenCount":9,"candidatesTokenCount":12,"totalTokenCount":21}}`
	fake := newFake(resp(200, body))
	temp := 0.7
	maxTok := 256
	chat, err := NewChat(Gemini, GeminiConfig{APIKey: "gk-x"}, WithHTTPClient(fake))
	if err != nil {
		t.Fatal(err)
	}
	res, err := chat.Complete(context.Background(), ChatInput{
		Model: "gemini-3.8-flash",
		Messages: []ChatMessage{
			{Role: "system", Content: "You are helpful"},
			{Role: "user", Content: "Hello"},
		},
		Temperature:     &temp,
		MaxOutputTokens: &maxTok,
	})
	if err != nil {
		t.Fatal(err)
	}
	if res.Message.Content != "Hi there" || res.FinishReason != "stop" {
		t.Fatalf("bad result: %+v", res)
	}
	if res.Usage.InputTokens != 9 || res.Usage.OutputTokens != 12 || res.Usage.TotalTokens != 21 {
		t.Fatalf("bad usage: %+v", res.Usage)
	}
	if res.Model != "gemini-3.8-flash-001" {
		t.Fatalf("bad model: %s", res.Model)
	}

	req := fake.last()
	if req.URL.String() != "https://generativelanguage.googleapis.com/v1beta/models/gemini-3.8-flash:generateContent" {
		t.Fatalf("bad url: %s", req.URL)
	}
	if req.Header.Get("x-goog-api-key") != "gk-x" {
		t.Fatalf("bad auth header: %q", req.Header.Get("x-goog-api-key"))
	}
	if req.Header.Get("Authorization") != "" {
		t.Fatalf("gemini must not send Authorization header")
	}

	var sent map[string]any
	if err := json.Unmarshal([]byte(req.Body), &sent); err != nil {
		t.Fatal(err)
	}
	si, _ := sent["systemInstruction"].(map[string]any)
	siParts, _ := si["parts"].([]any)
	if len(siParts) != 1 || siParts[0].(map[string]any)["text"] != "You are helpful" {
		t.Fatalf("bad systemInstruction: %v", sent["systemInstruction"])
	}
	contents, _ := sent["contents"].([]any)
	if len(contents) != 1 {
		t.Fatalf("system message must not leak into contents: %v", contents)
	}
	c0 := contents[0].(map[string]any)
	if c0["role"] != "user" {
		t.Fatalf("bad content role: %v", c0["role"])
	}
	p0 := c0["parts"].([]any)[0].(map[string]any)
	if p0["text"] != "Hello" {
		t.Fatalf("bad content part: %v", p0)
	}
	gc, _ := sent["generationConfig"].(map[string]any)
	if gc["maxOutputTokens"].(float64) != 256 {
		t.Fatalf("bad maxOutputTokens: %v", gc["maxOutputTokens"])
	}
	if gc["temperature"].(float64) != 0.7 {
		t.Fatalf("bad temperature: %v", gc["temperature"])
	}
}

func TestGeminiCompleteToolCall(t *testing.T) {
	body := `{"candidates":[{"content":{"role":"model","parts":[{"functionCall":{"name":"get_weather","args":{"city":"Paris"}}}]},"finishReason":"STOP"}],"usageMetadata":{"promptTokenCount":5,"candidatesTokenCount":7,"totalTokenCount":12}}`
	fake := newFake(resp(200, body))
	chat, _ := NewChat(Gemini, GeminiConfig{APIKey: "k"}, WithHTTPClient(fake))
	res, err := chat.Complete(context.Background(), ChatInput{
		Model:    "gemini-3.8-flash",
		Messages: []ChatMessage{{Role: "user", Content: "weather in Paris?"}},
		Tools: []ToolDefinition{{
			Name:        "get_weather",
			Description: "Get weather",
			Parameters:  map[string]any{"type": "object"},
		}},
		ToolChoice: &ToolChoice{Mode: "required"},
	})
	if err != nil {
		t.Fatal(err)
	}
	if res.FinishReason != "tool_calls" {
		t.Fatalf("bad finish: %s", res.FinishReason)
	}
	if len(res.Message.ToolCalls) != 1 {
		t.Fatalf("bad tool calls: %+v", res.Message.ToolCalls)
	}
	tc := res.Message.ToolCalls[0]
	// Gemini sent no id, so it is synthesized from the call's ordinal.
	if tc.Name != "get_weather" || tc.ID != "call_0" {
		t.Fatalf("bad tool call: %+v", tc)
	}
	var args map[string]any
	if err := json.Unmarshal([]byte(tc.Arguments), &args); err != nil {
		t.Fatalf("arguments not valid JSON %q: %v", tc.Arguments, err)
	}
	if args["city"] != "Paris" {
		t.Fatalf("bad args: %s", tc.Arguments)
	}

	var sent map[string]any
	_ = json.Unmarshal([]byte(fake.last().Body), &sent)
	tools, _ := sent["tools"].([]any)
	if len(tools) != 1 {
		t.Fatalf("bad tools: %v", tools)
	}
	decls := tools[0].(map[string]any)["functionDeclarations"].([]any)
	if decls[0].(map[string]any)["name"] != "get_weather" {
		t.Fatalf("bad functionDeclarations: %v", decls)
	}
	fcc := sent["toolConfig"].(map[string]any)["functionCallingConfig"].(map[string]any)
	if fcc["mode"] != "ANY" {
		t.Fatalf("bad functionCallingConfig: %v", fcc)
	}
}

func TestGeminiStream(t *testing.T) {
	sse := strings.Join([]string{
		`data: {"candidates":[{"content":{"role":"model","parts":[{"text":"Hel"}]}}]}`,
		"",
		`data: {"candidates":[{"content":{"role":"model","parts":[{"text":"lo"}]}}]}`,
		"",
		`data: {"candidates":[{"content":{"role":"model","parts":[{"text":"!"}]},"finishReason":"STOP"}],"usageMetadata":{"promptTokenCount":3,"candidatesTokenCount":2,"totalTokenCount":5}}`,
		"",
	}, "\n")
	fake := newFake(sseResp(sse))
	chat, _ := NewChat(Gemini, GeminiConfig{APIKey: "k"}, WithHTTPClient(fake))
	stream, err := chat.Stream(context.Background(), ChatInput{
		Model:    "gemini-3.8-flash",
		Messages: []ChatMessage{{Role: "user", Content: "hi"}},
	})
	if err != nil {
		t.Fatal(err)
	}
	defer stream.Close()

	var text string
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
		if d.FinishReason != "" {
			finish = d.FinishReason
		}
		if d.Usage != nil {
			usage = d.Usage
		}
	}
	if text != "Hello!" || finish != "stop" {
		t.Fatalf("stream text=%q finish=%q", text, finish)
	}
	if usage == nil || usage.TotalTokens != 5 {
		t.Fatalf("bad stream usage: %+v", usage)
	}
	if fake.last().URL.String() != "https://generativelanguage.googleapis.com/v1beta/models/gemini-3.8-flash:streamGenerateContent?alt=sse" {
		t.Fatalf("bad stream url: %s", fake.last().URL)
	}
	if fake.last().Header.Get("Accept") != "text/event-stream" {
		t.Fatalf("stream must send Accept: text/event-stream")
	}
}

func TestGeminiStreamToolCall(t *testing.T) {
	sse := strings.Join([]string{
		`data: {"candidates":[{"content":{"role":"model","parts":[{"functionCall":{"name":"lookup","args":{"q":"x"}}}]},"finishReason":"STOP"}],"usageMetadata":{"promptTokenCount":4,"candidatesTokenCount":6,"totalTokenCount":10}}`,
		"",
	}, "\n")
	fake := newFake(sseResp(sse))
	chat, _ := NewChat(Gemini, GeminiConfig{APIKey: "k"}, WithHTTPClient(fake))
	stream, err := chat.Stream(context.Background(), ChatInput{
		Model:    "gemini-3.8-flash",
		Messages: []ChatMessage{{Role: "user", Content: "hi"}},
	})
	if err != nil {
		t.Fatal(err)
	}
	defer stream.Close()

	var toolName, finish string
	for {
		d, err := stream.Recv()
		if err == io.EOF {
			break
		}
		if err != nil {
			t.Fatal(err)
		}
		if d.ToolCallDelta != nil {
			toolName = d.ToolCallDelta.FunctionName
			if d.ToolCallDelta.ID != "call_0" {
				t.Fatalf("bad stream tool-call id: %q", d.ToolCallDelta.ID)
			}
		}
		if d.FinishReason != "" {
			finish = d.FinishReason
		}
	}
	if toolName != "lookup" {
		t.Fatalf("bad stream tool name: %q", toolName)
	}
	if finish != "tool_calls" {
		t.Fatalf("expected tool_calls finish after a functionCall, got %q", finish)
	}
}

func TestGeminiStreamParallelToolCalls(t *testing.T) {
	// Two functionCall parts in ONE chunk must yield TWO tool-call deltas with
	// distinct indices (0, 1) — not collapse to the last part at index 0.
	sse := strings.Join([]string{
		`data: {"candidates":[{"content":{"role":"model","parts":[{"functionCall":{"name":"get_weather","args":{"city":"SF"}}},{"functionCall":{"name":"get_time","args":{"tz":"PST"}}}]},"finishReason":"STOP"}]}`,
		"",
	}, "\n")
	fake := newFake(sseResp(sse))
	chat, _ := NewChat(Gemini, GeminiConfig{APIKey: "k"}, WithHTTPClient(fake))
	stream, err := chat.Stream(context.Background(), ChatInput{
		Model:    "gemini-3.8-flash",
		Messages: []ChatMessage{{Role: "user", Content: "hi"}},
	})
	if err != nil {
		t.Fatal(err)
	}
	defer stream.Close()

	var names []string
	var indices []int
	for {
		d, err := stream.Recv()
		if err == io.EOF {
			break
		}
		if err != nil {
			t.Fatal(err)
		}
		if d.ToolCallDelta != nil {
			names = append(names, d.ToolCallDelta.FunctionName)
			indices = append(indices, d.ToolCallDelta.Index)
		}
	}
	if len(names) != 2 || names[0] != "get_weather" || names[1] != "get_time" {
		t.Fatalf("expected both parallel calls, got %v", names)
	}
	if len(indices) != 2 || indices[0] != 0 || indices[1] != 1 {
		t.Fatalf("expected distinct indices [0 1], got %v", indices)
	}
}

func TestGeminiStreamMidStreamError(t *testing.T) {
	sse := strings.Join([]string{
		`data: {"candidates":[{"content":{"role":"model","parts":[{"text":"Hi"}]}}]}`,
		"",
		`data: {"error":{"code":429,"message":"Resource exhausted","status":"RESOURCE_EXHAUSTED"}}`,
		"",
	}, "\n")
	fake := newFake(sseResp(sse))
	chat, _ := NewChat(Gemini, GeminiConfig{APIKey: "k"}, WithHTTPClient(fake))
	stream, err := chat.Stream(context.Background(), ChatInput{Model: "m", Messages: []ChatMessage{{Role: "user", Content: "hi"}}})
	if err != nil {
		t.Fatal(err)
	}
	defer stream.Close()

	d, err := stream.Recv()
	if err != nil || d.ContentDelta != "Hi" {
		t.Fatalf("first delta should arrive: d=%+v err=%v", d, err)
	}
	_, err = stream.Recv()
	if err == io.EOF {
		t.Fatal("gemini mid-stream error frame was swallowed as io.EOF")
	}
	var ce *ConnectorError
	if !errors.As(err, &ce) {
		t.Fatalf("expected *ConnectorError, got %v", err)
	}
	// The frame carries an HTTP code (429) → classified via geminiMapCode.
	if ce.ProviderCode != CodeRateLimited {
		t.Fatalf("expected rate_limited from code 429, got %s", ce.ProviderCode)
	}
	if ce.ProviderMessage != "Resource exhausted" {
		t.Fatalf("bad message: %s", ce.ProviderMessage)
	}
}

func TestGeminiErrorMapping(t *testing.T) {
	fake := newFake(resp(429, `{"error":{"code":429,"message":"Resource exhausted","status":"RESOURCE_EXHAUSTED","details":[{"@type":"type.googleapis.com/google.rpc.RetryInfo","retryDelay":"17s"}]}}`))
	chat, _ := NewChat(Gemini, GeminiConfig{APIKey: "k"}, WithHTTPClient(fake))
	_, err := chat.Complete(context.Background(), ChatInput{
		Model:    "gemini-3.8-flash",
		Messages: []ChatMessage{{Role: "user", Content: "x"}},
	})
	var ce *ConnectorError
	if !errors.As(err, &ce) || ce.ProviderCode != CodeRateLimited {
		t.Fatalf("expected rate_limited, got %v", err)
	}
	if ce.ProviderMessage != "Resource exhausted" {
		t.Fatalf("bad message: %s", ce.ProviderMessage)
	}
	cause, _ := ce.Cause.(map[string]any)
	if cause["retryAfterSeconds"] != 17 {
		t.Fatalf("expected retryAfterSeconds from RetryInfo body, got %v (%T)", cause["retryAfterSeconds"], cause["retryAfterSeconds"])
	}
}

func TestGeminiPromptBlockContentFilter(t *testing.T) {
	// No candidates, only promptFeedback.blockReason → content_filter, not unknown.
	fake := newFake(resp(200, `{"promptFeedback":{"blockReason":"SAFETY"},"usageMetadata":{"promptTokenCount":9,"totalTokenCount":9}}`))
	chat, _ := NewChat(Gemini, GeminiConfig{APIKey: "k"}, WithHTTPClient(fake))
	res, err := chat.Complete(context.Background(), ChatInput{Model: "m", Messages: []ChatMessage{{Role: "user", Content: "x"}}})
	if err != nil {
		t.Fatal(err)
	}
	if res.FinishReason != "content_filter" {
		t.Fatalf("expected content_filter, got %q", res.FinishReason)
	}
	if res.Message.Content != "" {
		t.Fatalf("expected empty content, got %q", res.Message.Content)
	}
}

func TestGeminiStreamPromptBlockContentFilter(t *testing.T) {
	sse := strings.Join([]string{`data: {"promptFeedback":{"blockReason":"SAFETY"}}`, "", ""}, "\n")
	fake := newFake(sseResp(sse))
	chat, _ := NewChat(Gemini, GeminiConfig{APIKey: "k"}, WithHTTPClient(fake))
	stream, err := chat.Stream(context.Background(), ChatInput{Model: "m", Messages: []ChatMessage{{Role: "user", Content: "x"}}})
	if err != nil {
		t.Fatal(err)
	}
	defer stream.Close()
	var finish FinishReason
	for {
		d, err := stream.Recv()
		if err == io.EOF {
			break
		}
		if err != nil {
			t.Fatal(err)
		}
		if d.FinishReason != "" {
			finish = d.FinishReason
		}
	}
	if finish != "content_filter" {
		t.Fatalf("expected content_filter finish, got %q", finish)
	}
}

func TestGeminiThoughtPartsExcludedFromContent(t *testing.T) {
	body := `{"candidates":[{"content":{"role":"model","parts":[{"text":"secret cot","thought":true},{"text":"the answer"}]},"finishReason":"STOP"}],"usageMetadata":{}}`
	fake := newFake(resp(200, body))
	chat, _ := NewChat(Gemini, GeminiConfig{APIKey: "k"}, WithHTTPClient(fake))
	res, err := chat.Complete(context.Background(), ChatInput{Model: "gemini-3.8-flash", Messages: []ChatMessage{{Role: "user", Content: "hi"}}})
	if err != nil {
		t.Fatal(err)
	}
	if res.Message.Content != "the answer" {
		t.Fatalf("thought part leaked into content: %q", res.Message.Content)
	}
}

func TestGeminiPassthroughQuery(t *testing.T) {
	fake := newFake(resp(200, `{"candidates":[{"content":{"parts":[{"text":"ok"}]},"finishReason":"STOP"}],"usageMetadata":{}}`))
	chat, _ := NewChat(Gemini, GeminiConfig{APIKey: "k"}, WithHTTPClient(fake))
	_, err := chat.Complete(context.Background(), ChatInput{
		Model:       "gemini-3.8-flash",
		Messages:    []ChatMessage{{Role: "user", Content: "hi"}},
		Passthrough: &Passthrough{Query: map[string]string{"extra": "v"}},
	})
	if err != nil {
		t.Fatal(err)
	}
	if got := fake.last().URL.String(); got != "https://generativelanguage.googleapis.com/v1beta/models/gemini-3.8-flash:generateContent?extra=v" {
		t.Fatalf("query not appended: %s", got)
	}
}

func TestGeminiContextLengthError(t *testing.T) {
	fake := newFake(resp(400, `{"error":{"message":"The input token count (12000) exceeds the maximum number of tokens allowed (8192)."}}`))
	chat, _ := NewChat(Gemini, GeminiConfig{APIKey: "k"}, WithHTTPClient(fake))
	_, err := chat.Complete(context.Background(), ChatInput{
		Model:    "gemini-3.8-flash",
		Messages: []ChatMessage{{Role: "user", Content: "x"}},
	})
	var ce *ConnectorError
	if !errors.As(err, &ce) || ce.ProviderCode != CodeContextLengthExceeded {
		t.Fatalf("expected context_length_exceeded, got %v", err)
	}
}

// TestGeminiSyntheticIDConsistency proves the fallback tool-call id (when the
// vendor omits one) uses the SAME scheme on both the non-stream and stream paths.
func TestGeminiSyntheticIDConsistency(t *testing.T) {
	const fn = "lookup"

	// non-stream
	nsBody := `{"candidates":[{"content":{"role":"model","parts":[{"functionCall":{"name":"` + fn + `","args":{}}}]},"finishReason":"STOP"}]}`
	nsFake := newFake(resp(200, nsBody))
	nsChat, _ := NewChat(Gemini, GeminiConfig{APIKey: "k"}, WithHTTPClient(nsFake))
	nsRes, err := nsChat.Complete(context.Background(), ChatInput{Model: "m", Messages: []ChatMessage{{Role: "user", Content: "x"}}})
	if err != nil {
		t.Fatal(err)
	}
	if len(nsRes.Message.ToolCalls) != 1 {
		t.Fatalf("expected 1 tool call, got %d", len(nsRes.Message.ToolCalls))
	}
	nonStreamID := nsRes.Message.ToolCalls[0].ID

	// stream
	sse := strings.Join([]string{
		`data: {"candidates":[{"content":{"role":"model","parts":[{"functionCall":{"name":"` + fn + `","args":{}}}]},"finishReason":"STOP"}]}`,
		"",
	}, "\n")
	sChat, _ := NewChat(Gemini, GeminiConfig{APIKey: "k"}, WithHTTPClient(newFake(sseResp(sse))))
	stream, err := sChat.Stream(context.Background(), ChatInput{Model: "m", Messages: []ChatMessage{{Role: "user", Content: "x"}}})
	if err != nil {
		t.Fatal(err)
	}
	defer stream.Close()
	var streamID string
	for {
		d, err := stream.Recv()
		if err == io.EOF {
			break
		}
		if err != nil {
			t.Fatal(err)
		}
		if d.ToolCallDelta != nil {
			streamID = d.ToolCallDelta.ID
		}
	}

	if nonStreamID != streamID {
		t.Fatalf("synthetic tool-call id differs across paths: non-stream=%q stream=%q", nonStreamID, streamID)
	}
	if nonStreamID != "call_0" {
		t.Fatalf("unexpected synthetic id scheme: %q", nonStreamID)
	}
}

func TestGeminiWrongConfig(t *testing.T) {
	_, err := NewChat(Gemini, OpenAICompatConfig{APIKey: "k"})
	var ce *ConnectorError
	if !errors.As(err, &ce) || ce.ProviderCode != CodeInvalidRequest {
		t.Fatalf("expected invalid_request for wrong config type, got %v", err)
	}
}

// Gemini 3 puts the thoughtSignature on the functionCall part (the first one of
// a parallel set); Gemini's own functionCall.id, when present, is kept.
const geminiSignedToolBody = `{"candidates":[{"content":{"role":"model","parts":[` +
	`{"functionCall":{"id":"fc-a1","name":"get_weather","args":{"city":"Paris"}},"thoughtSignature":"c2lnLUE="},` +
	`{"functionCall":{"name":"get_weather","args":{"city":"Rome"}}}]},"finishReason":"STOP"}],` +
	`"usageMetadata":{"promptTokenCount":10,"candidatesTokenCount":8,"thoughtsTokenCount":30,"totalTokenCount":48},"modelVersion":"gemini-3.8-flash"}`

func TestGeminiThoughtSignatureCaptured(t *testing.T) {
	fake := newFake(resp(200, geminiSignedToolBody))
	chat, _ := NewChat(Gemini, GeminiConfig{APIKey: "k"}, WithHTTPClient(fake))
	res, err := chat.Complete(context.Background(), ChatInput{Model: "gemini-3.8-flash", Messages: []ChatMessage{{Role: "user", Content: "weather?"}}})
	if err != nil {
		t.Fatal(err)
	}
	tcs := res.Message.ToolCalls
	if len(tcs) != 2 {
		t.Fatalf("tool calls: %+v", tcs)
	}
	if tcs[0].ID != "fc-a1" || tcs[1].ID != "call_1" {
		t.Fatalf("ids: vendor id must be kept, missing one synthesized by ordinal: %q %q", tcs[0].ID, tcs[1].ID)
	}
	want := map[string]any{"gemini": map[string]any{"thoughtSignature": "c2lnLUE="}}
	if !reflect.DeepEqual(tcs[0].ProviderMetadata, want) {
		t.Fatalf("signature not captured: %v", tcs[0].ProviderMetadata)
	}
	if tcs[1].ProviderMetadata != nil {
		t.Fatalf("unsigned call must carry no metadata: %v", tcs[1].ProviderMetadata)
	}
	if res.Message.ProviderMetadata != nil {
		t.Fatalf("gemini has no message-level metadata: %v", res.Message.ProviderMetadata)
	}
	// Thinking tokens are billed output.
	if res.Usage != (Usage{InputTokens: 10, OutputTokens: 38, TotalTokens: 48}) {
		t.Fatalf("usage: %+v", res.Usage)
	}
}

func TestGeminiStreamThoughtSignatureCaptured(t *testing.T) {
	sse := "data: " + geminiSignedToolBody + "\n\n"
	fake := newFake(sseResp(sse))
	chat, _ := NewChat(Gemini, GeminiConfig{APIKey: "k"}, WithHTTPClient(fake))
	stream, err := chat.Stream(context.Background(), ChatInput{Model: "gemini-3.8-flash", Messages: []ChatMessage{{Role: "user", Content: "weather?"}}})
	if err != nil {
		t.Fatal(err)
	}
	var tcds []*ToolCallDelta
	for _, d := range drain(t, stream) {
		if d.ToolCallDelta != nil {
			tcds = append(tcds, d.ToolCallDelta)
		}
	}
	if len(tcds) != 2 || tcds[0].ID != "fc-a1" || tcds[1].ID != "call_1" {
		t.Fatalf("stream tool deltas: %+v", tcds)
	}
	if jsonStr(jsonGet(tcds[0].ProviderMetadata["gemini"], "thoughtSignature")) != "c2lnLUE=" || tcds[1].ProviderMetadata != nil {
		t.Fatalf("stream signature: %v / %v", tcds[0].ProviderMetadata, tcds[1].ProviderMetadata)
	}
}

// geminiModelTurnParts returns the parts of the first `model` turn sent.
func geminiModelTurnParts(t *testing.T, sent map[string]any) []any {
	t.Helper()
	for _, c := range sent["contents"].([]any) {
		if c.(map[string]any)["role"] == "model" {
			return c.(map[string]any)["parts"].([]any)
		}
	}
	t.Fatalf("no model turn in %v", sent["contents"])
	return nil
}

func TestGeminiThoughtSignatureReplayed(t *testing.T) {
	// Round trip: the result's tool calls go back verbatim as the next turn.
	fake := newFake(resp(200, geminiSignedToolBody), resp(200, `{"candidates":[{"content":{"parts":[{"text":"Sunny in both."}]},"finishReason":"STOP"}]}`))
	chat, _ := NewChat(Gemini, GeminiConfig{APIKey: "k"}, WithHTTPClient(fake))
	first, err := chat.Complete(context.Background(), ChatInput{Model: "gemini-3.8-flash", Messages: []ChatMessage{{Role: "user", Content: "weather?"}}})
	if err != nil {
		t.Fatal(err)
	}
	history := []ChatMessage{
		{Role: "user", Content: "weather?"},
		{Role: "assistant", Content: first.Message.Content, ToolCalls: first.Message.ToolCalls, ProviderMetadata: first.Message.ProviderMetadata},
		{Role: "tool", ToolCallID: first.Message.ToolCalls[0].ID, Content: "sunny"},
		{Role: "tool", ToolCallID: first.Message.ToolCalls[1].ID, Content: "sunny"},
	}
	if _, err := chat.Complete(context.Background(), ChatInput{Model: "gemini-3.8-flash", Messages: history}); err != nil {
		t.Fatal(err)
	}
	sent := sentBody(t, fake)
	parts := geminiModelTurnParts(t, sent)
	p0, p1 := parts[0].(map[string]any), parts[1].(map[string]any)
	if p0["thoughtSignature"] != "c2lnLUE=" {
		t.Fatalf("signature not replayed: %v", p0)
	}
	if _, ok := p1["thoughtSignature"]; ok {
		t.Fatalf("a real signature in the turn means no placeholder anywhere: %v", p1)
	}
	if p0["functionCall"].(map[string]any)["id"] != "fc-a1" {
		t.Fatalf("vendor id must be echoed on the functionCall: %v", p0)
	}
	if _, ok := p1["functionCall"].(map[string]any)["id"]; ok {
		t.Fatalf("synthetic id must not go on the wire: %v", p1)
	}
	contents := sent["contents"].([]any)
	resps := contents[len(contents)-1].(map[string]any)["parts"].([]any)
	fr0 := resps[0].(map[string]any)["functionResponse"].(map[string]any)
	fr1 := resps[1].(map[string]any)["functionResponse"].(map[string]any)
	if fr0["id"] != "fc-a1" || fr0["name"] != "get_weather" {
		t.Fatalf("functionResponse must echo the vendor id: %v", fr0)
	}
	if _, ok := fr1["id"]; ok || fr1["name"] != "get_weather" {
		t.Fatalf("synthetic-id response is keyed by name only: %v", fr1)
	}
}

func TestGeminiSignaturePlaceholderOnlyOnFirstCall(t *testing.T) {
	// A hand-built history (or one from another provider) has no signature:
	// the first functionCall gets Google's documented placeholder, the rest none.
	// Foreign provider keys are ignored and never reach the wire.
	fake := newFake(resp(200, `{"candidates":[{"content":{"parts":[{"text":"ok"}]},"finishReason":"STOP"}]}`))
	chat, _ := NewChat(Gemini, GeminiConfig{APIKey: "k"}, WithHTTPClient(fake))
	foreign := map[string]any{"anthropic": map[string]any{"content": []any{map[string]any{"type": "thinking", "thinking": "x", "signature": "anthropic-sig"}}}}
	_, err := chat.Complete(context.Background(), ChatInput{Model: "gemini-3.8-flash", Messages: []ChatMessage{
		{Role: "user", Content: "weather?"},
		{Role: "assistant", ProviderMetadata: foreign, ToolCalls: []ToolCall{
			{ID: "toolu_01", Name: "get_weather", Arguments: `{"city":"Paris"}`, ProviderMetadata: foreign},
			{ID: "toolu_02", Name: "get_time", Arguments: `{}`},
		}},
		{Role: "tool", ToolCallID: "toolu_01", Content: "sunny"},
		{Role: "tool", ToolCallID: "toolu_02", Content: "noon"},
	}})
	if err != nil {
		t.Fatal(err)
	}
	parts := geminiModelTurnParts(t, sentBody(t, fake))
	if parts[0].(map[string]any)["thoughtSignature"] != "skip_thought_signature_validator" {
		t.Fatalf("first call must carry the placeholder: %v", parts[0])
	}
	if _, ok := parts[1].(map[string]any)["thoughtSignature"]; ok {
		t.Fatalf("placeholder must go on the first call only: %v", parts[1])
	}
	if body := fake.last().Body; strings.Contains(body, "anthropic") || strings.Contains(body, "anthropic-sig") {
		t.Fatalf("foreign provider metadata leaked to the wire: %s", body)
	}
	// A turn with text only (no calls) gets no placeholder at all.
	fake2 := newFake(resp(200, `{"candidates":[{"content":{"parts":[{"text":"ok"}]},"finishReason":"STOP"}]}`))
	chat2, _ := NewChat(Gemini, GeminiConfig{APIKey: "k"}, WithHTTPClient(fake2))
	_, _ = chat2.Complete(context.Background(), ChatInput{Model: "m", Messages: []ChatMessage{{Role: "user", Content: "a"}, {Role: "assistant", Content: "b"}, {Role: "user", Content: "c"}}})
	if strings.Contains(fake2.last().Body, "thoughtSignature") {
		t.Fatalf("placeholder on a turn without calls: %s", fake2.last().Body)
	}
}

func TestGeminiToolNameResolvedPerTurn(t *testing.T) {
	// Synthetic ids restart at call_0 every turn; each tool result must resolve
	// against the assistant turn right before it.
	fake := newFake(resp(200, `{"candidates":[{"content":{"parts":[{"text":"ok"}]},"finishReason":"STOP"}]}`))
	chat, _ := NewChat(Gemini, GeminiConfig{APIKey: "k"}, WithHTTPClient(fake))
	_, err := chat.Complete(context.Background(), ChatInput{Model: "m", Messages: []ChatMessage{
		{Role: "user", Content: "q"},
		{Role: "assistant", ToolCalls: []ToolCall{{ID: "call_0", Name: "first_tool", Arguments: `{}`}}},
		{Role: "tool", ToolCallID: "call_0", Content: "1"},
		{Role: "assistant", ToolCalls: []ToolCall{{ID: "call_0", Name: "second_tool", Arguments: `{}`}}},
		{Role: "tool", ToolCallID: "call_0", Content: "2"},
	}})
	if err != nil {
		t.Fatal(err)
	}
	contents := sentBody(t, fake)["contents"].([]any)
	name := func(i int) any {
		return contents[i].(map[string]any)["parts"].([]any)[0].(map[string]any)["functionResponse"].(map[string]any)["name"]
	}
	if name(2) != "first_tool" || name(4) != "second_tool" {
		t.Fatalf("tool names resolved across turns: %v / %v", name(2), name(4))
	}
}

func TestGeminiJSONSchemaFields(t *testing.T) {
	// Standard JSON Schema (additionalProperties, $ref) goes in the *JsonSchema
	// fields; the OpenAPI-subset `parameters` / `responseSchema` reject it.
	schema := map[string]any{"type": "object", "properties": map[string]any{"city": map[string]any{"type": "string"}}, "additionalProperties": false}
	fake := newFake(resp(200, `{"candidates":[{"content":{"parts":[{"text":"{}"}]},"finishReason":"STOP"}]}`))
	chat, _ := NewChat(Gemini, GeminiConfig{APIKey: "k"}, WithHTTPClient(fake))
	_, err := chat.Complete(context.Background(), ChatInput{
		Model:          "gemini-3.8-flash",
		Messages:       []ChatMessage{{Role: "user", Content: "x"}},
		Tools:          []ToolDefinition{{Name: "get_weather", Parameters: schema}},
		ResponseFormat: &ResponseFormat{Type: "json_schema", JSONSchemaName: "out", JSONSchema: schema},
	})
	if err != nil {
		t.Fatal(err)
	}
	sent := sentBody(t, fake)
	decl := sent["tools"].([]any)[0].(map[string]any)["functionDeclarations"].([]any)[0].(map[string]any)
	if !reflect.DeepEqual(decl["parametersJsonSchema"], schema) {
		t.Fatalf("parametersJsonSchema: %v", decl)
	}
	if _, ok := decl["parameters"]; ok {
		t.Fatalf("parameters must not be sent: %v", decl)
	}
	gc := sent["generationConfig"].(map[string]any)
	if !reflect.DeepEqual(gc["responseJsonSchema"], schema) || gc["responseMimeType"] != "application/json" {
		t.Fatalf("responseJsonSchema: %v", gc)
	}
	if _, ok := gc["responseSchema"]; ok {
		t.Fatalf("deprecated responseSchema must not be sent: %v", gc)
	}
}

func TestGeminiStreamUsageNotDoubleCounted(t *testing.T) {
	// usageMetadata is cumulative on every chunk: it must surface once, on the
	// finish delta, with the final totals (thoughts counted as output).
	sse := strings.Join([]string{
		`data: {"candidates":[{"content":{"role":"model","parts":[{"text":"Hel"}]}}],"usageMetadata":{"promptTokenCount":5,"candidatesTokenCount":1,"thoughtsTokenCount":20,"totalTokenCount":26}}`,
		"",
		`data: {"candidates":[{"content":{"role":"model","parts":[{"text":"lo"}]},"finishReason":"STOP"}],"usageMetadata":{"promptTokenCount":5,"candidatesTokenCount":2,"thoughtsTokenCount":20,"totalTokenCount":27}}`,
		"",
	}, "\n")
	chat, _ := NewChat(Gemini, GeminiConfig{APIKey: "k"}, WithHTTPClient(newFake(sseResp(sse))))
	stream, err := chat.Stream(context.Background(), ChatInput{Model: "m", Messages: []ChatMessage{{Role: "user", Content: "hi"}}})
	if err != nil {
		t.Fatal(err)
	}
	var usages []Usage
	for _, d := range drain(t, stream) {
		if d.Usage != nil {
			usages = append(usages, *d.Usage)
		}
	}
	if len(usages) != 1 || usages[0] != (Usage{InputTokens: 5, OutputTokens: 22, TotalTokens: 27}) {
		t.Fatalf("usage deltas: %+v", usages)
	}

	// A finish chunk without usageMetadata reports the latest cumulative usage.
	sse2 := strings.Join([]string{
		`data: {"candidates":[{"content":{"parts":[{"text":"a"}]}}],"usageMetadata":{"promptTokenCount":3,"candidatesTokenCount":1,"totalTokenCount":4}}`,
		"",
		`data: {"candidates":[{"content":{"parts":[{"text":"b"}]},"finishReason":"MAX_TOKENS"}]}`,
		"",
	}, "\n")
	chat2, _ := NewChat(Gemini, GeminiConfig{APIKey: "k"}, WithHTTPClient(newFake(sseResp(sse2))))
	stream2, _ := chat2.Stream(context.Background(), ChatInput{Model: "m", Messages: []ChatMessage{{Role: "user", Content: "hi"}}})
	ds := drain(t, stream2)
	last := ds[len(ds)-1]
	if last.FinishReason != "length" || last.Usage == nil || last.Usage.TotalTokens != 4 {
		t.Fatalf("finish delta: %+v", last)
	}
}

func TestGeminiAPIKeyInvalidIsAuthFailed(t *testing.T) {
	body := `{"error":{"code":400,"message":"API key not valid. Please pass a valid API key.","status":"INVALID_ARGUMENT",` +
		`"details":[{"@type":"type.googleapis.com/google.rpc.ErrorInfo","reason":"API_KEY_INVALID","domain":"googleapis.com","metadata":{"service":"generativelanguage.googleapis.com"}}]}}`
	chat, _ := NewChat(Gemini, GeminiConfig{APIKey: "bad"}, WithHTTPClient(newFake(resp(400, body))))
	_, err := chat.Complete(context.Background(), ChatInput{Model: "m", Messages: []ChatMessage{{Role: "user", Content: "x"}}})
	var ce *ConnectorError
	if !errors.As(err, &ce) || ce.ProviderCode != CodeAuthFailed || *ce.StatusCode != 400 {
		t.Fatalf("expected auth_failed, got %v", err)
	}
}

func TestGeminiErrorClassification(t *testing.T) {
	cases := []struct {
		status int
		msg    string
		want   ProviderCode
	}{
		// An ordinary validation error that merely mentions "context" is not
		// a context overflow.
		{400, "Invalid JSON payload received. Unknown name \"context\" at 'contents[0]': Cannot find field.", CodeInvalidRequest},
		{400, "Function call is missing a thought_signature in functionCall parts.", CodeInvalidRequest},
		{400, "The input token count (1200000) exceeds the maximum number of tokens allowed (1048576).", CodeContextLengthExceeded},
		{402, "Your prepay credit balance is depleted.", CodeRateLimited},
		{413, "Request payload size exceeds the limit.", CodeInvalidRequest},
		{404, "models/gemini-2.5-flash is no longer available to new users.", CodeInvalidRequest},
	}
	for _, c := range cases {
		body := `{"error":{"code":` + strconv.Itoa(c.status) + `,"message":` + strconv.Quote(c.msg) + `}}`
		chat, _ := NewChat(Gemini, GeminiConfig{APIKey: "k"}, WithHTTPClient(newFake(resp(c.status, body))))
		_, err := chat.Complete(context.Background(), ChatInput{Model: "m", Messages: []ChatMessage{{Role: "user", Content: "x"}}})
		var ce *ConnectorError
		if !errors.As(err, &ce) || ce.ProviderCode != c.want {
			t.Fatalf("%d %q: got %v, want %s", c.status, c.msg, err, c.want)
		}
	}
}

func TestGeminiFinishReasonMapping(t *testing.T) {
	for fr, want := range map[string]FinishReason{
		"STOP": "stop", "MAX_TOKENS": "length", "CONTINUATION": "length", "SAFETY": "content_filter",
		"IMAGE_SAFETY": "content_filter", "LANGUAGE": "unknown", "IMAGE_PROHIBITED_CONTENT": "content_filter", "NO_IMAGE": "unknown", "MALFORMED_FUNCTION_CALL": "unknown",
		"UNEXPECTED_TOOL_CALL": "unknown", "TOO_MANY_TOOL_CALLS": "unknown", "MISSING_THOUGHT_SIGNATURE": "unknown",
		"OTHER": "unknown",
	} {
		if got := geminiMapFinishReason(fr); got != want {
			t.Fatalf("%s → %s, want %s", fr, got, want)
		}
	}
}

func TestGeminiUsageCountsToolUsePrompt(t *testing.T) {
	body := `{"candidates":[{"content":{"parts":[{"text":"ok"}]},"finishReason":"STOP"}],` +
		`"usageMetadata":{"promptTokenCount":10,"toolUsePromptTokenCount":5,"candidatesTokenCount":4,"thoughtsTokenCount":6,"totalTokenCount":25}}`
	chat, _ := NewChat(Gemini, GeminiConfig{APIKey: "k"}, WithHTTPClient(newFake(resp(200, body))))
	res, err := chat.Complete(context.Background(), ChatInput{Model: "m", Messages: []ChatMessage{{Role: "user", Content: "x"}}})
	if err != nil {
		t.Fatal(err)
	}
	if res.Usage != (Usage{InputTokens: 15, OutputTokens: 10, TotalTokens: 25}) {
		t.Fatalf("usage: %+v", res.Usage)
	}
}

func TestGeminiStreamUsageNotRepeatedAfterFinish(t *testing.T) {
	sse := strings.Join([]string{
		`data: {"candidates":[{"content":{"parts":[{"text":"a"}]},"finishReason":"STOP"}],"usageMetadata":{"promptTokenCount":3,"candidatesTokenCount":1}}`,
		"",
		`data: {"candidates":[{"content":{"parts":[]}}],"usageMetadata":{"promptTokenCount":3,"candidatesTokenCount":1}}`,
		"",
	}, "\n")
	chat, _ := NewChat(Gemini, GeminiConfig{APIKey: "k"}, WithHTTPClient(newFake(sseResp(sse))))
	stream, _ := chat.Stream(context.Background(), ChatInput{Model: "m", Messages: []ChatMessage{{Role: "user", Content: "x"}}})
	n := 0
	for _, d := range drain(t, stream) {
		if d.Usage != nil {
			n++
			if d.FinishReason != "stop" {
				t.Fatalf("usage must ride on the finish delta: %+v", d)
			}
		}
	}
	if n != 1 {
		t.Fatalf("usage reported %d times, want exactly once", n)
	}
}

func TestGeminiReview5ErrorRules(t *testing.T) {
	cases := []struct {
		status int
		msg    string
		want   ProviderCode
	}{
		// A bare "maximum number of tokens" is a maxOutputTokens validation error.
		{400, "maxOutputTokens is above the maximum number of tokens allowed for this model", CodeInvalidRequest},
		{400, "The input token count (1200000) exceeds the maximum number of tokens allowed (1048576).", CodeContextLengthExceeded},
		{408, "Request timed out", CodeProviderUnavailable},
		{409, "Aborted", CodeInvalidRequest},
	}
	for _, c := range cases {
		body := `{"error":{"code":` + strconv.Itoa(c.status) + `,"message":` + strconv.Quote(c.msg) + `}}`
		chat, _ := NewChat(Gemini, GeminiConfig{APIKey: "k"}, WithHTTPClient(newFake(resp(c.status, body))))
		_, err := chat.Complete(context.Background(), ChatInput{Model: "m", Messages: []ChatMessage{{Role: "user", Content: "x"}}})
		var ce *ConnectorError
		if !errors.As(err, &ce) || ce.ProviderCode != c.want {
			t.Fatalf("%d %q: got %v, want %s", c.status, c.msg, err, c.want)
		}
	}
	// Mid-stream frames: the stream's status, classified by code, else by the
	// google.rpc status name, else unknown.
	for frame, want := range map[string]ProviderCode{
		`{"error":{"code":503,"message":"overloaded","status":"UNAVAILABLE"}}`: CodeProviderUnavailable,
		`{"error":{"message":"slow down","status":"RESOURCE_EXHAUSTED"}}`:      CodeRateLimited,
		`{"error":{"message":"mystery"}}`:                                      CodeUnknown,
	} {
		chat, _ := NewChat(Gemini, GeminiConfig{APIKey: "k"}, WithHTTPClient(newFake(sseResp("data: "+frame+"\n\n"))))
		stream, _ := chat.Stream(context.Background(), ChatInput{Model: "m", Messages: []ChatMessage{{Role: "user", Content: "x"}}})
		_, err := stream.Recv()
		var ce *ConnectorError
		if !errors.As(err, &ce) || ce.ProviderCode != want || ce.StatusCode == nil || *ce.StatusCode != 200 {
			t.Fatalf("%s: got %v, want %s with status 200", frame, err, want)
		}
	}
}
