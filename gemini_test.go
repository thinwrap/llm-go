package llm

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"strings"
	"testing"
)

func TestGeminiComplete(t *testing.T) {
	body := `{"modelVersion":"gemini-1.5-flash-002","candidates":[{"content":{"role":"model","parts":[{"text":"Hi there"}]},"finishReason":"STOP"}],"usageMetadata":{"promptTokenCount":9,"candidatesTokenCount":12,"totalTokenCount":21}}`
	fake := newFake(resp(200, body))
	temp := 0.7
	maxTok := 256
	chat, err := NewChat(Gemini, GeminiConfig{APIKey: "gk-x"}, WithHTTPClient(fake))
	if err != nil {
		t.Fatal(err)
	}
	res, err := chat.Complete(context.Background(), ChatInput{
		Model: "gemini-1.5-flash",
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
	if res.Model != "gemini-1.5-flash-002" {
		t.Fatalf("bad model: %s", res.Model)
	}

	req := fake.last()
	if req.URL.String() != "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent" {
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
		Model:    "gemini-1.5-pro",
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
	// Synthetic id is name-based and consistent with the streaming path (fix #7).
	if tc.Name != "get_weather" || tc.ID != "call_get_weather" {
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
		Model:    "gemini-1.5-flash",
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
	if fake.last().URL.String() != "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:streamGenerateContent?alt=sse" {
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
		Model:    "gemini-1.5-flash",
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
			if d.ToolCallDelta.ID != "call_lookup" {
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
		Model:    "gemini-1.5-flash",
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
		Model:    "gemini-1.5-flash",
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
	res, err := chat.Complete(context.Background(), ChatInput{Model: "gemini-3-pro", Messages: []ChatMessage{{Role: "user", Content: "hi"}}})
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
		Model:       "gemini-1.5-flash",
		Messages:    []ChatMessage{{Role: "user", Content: "hi"}},
		Passthrough: &Passthrough{Query: map[string]string{"extra": "v"}},
	})
	if err != nil {
		t.Fatal(err)
	}
	if got := fake.last().URL.String(); got != "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?extra=v" {
		t.Fatalf("query not appended: %s", got)
	}
}

func TestGeminiContextLengthError(t *testing.T) {
	fake := newFake(resp(400, `{"error":{"message":"The input token count (12000) exceeds the maximum number of tokens allowed (8192)."}}`))
	chat, _ := NewChat(Gemini, GeminiConfig{APIKey: "k"}, WithHTTPClient(fake))
	_, err := chat.Complete(context.Background(), ChatInput{
		Model:    "gemini-1.5-flash",
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
	if nonStreamID != "call_"+fn {
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
