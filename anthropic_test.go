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

func TestAnthropicComplete(t *testing.T) {
	body := `{"model":"claude-opus-5-5","stop_reason":"end_turn","usage":{"input_tokens":10,"output_tokens":7},"content":[{"type":"text","text":"Hello "},{"type":"text","text":"world"}]}`
	fake := newFake(resp(200, body))
	chat, err := NewChat(Anthropic, AnthropicConfig{APIKey: "sk-ant"}, WithHTTPClient(fake))
	if err != nil {
		t.Fatal(err)
	}
	res, err := chat.Complete(context.Background(), ChatInput{
		Model: "claude-opus-5-5",
		Messages: []ChatMessage{
			{Role: "system", Content: "Be terse"},
			{Role: "user", Content: "Hi"},
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	if res.Message.Content != "Hello world" {
		t.Fatalf("bad content: %q", res.Message.Content)
	}
	if res.FinishReason != "stop" {
		t.Fatalf("bad finish: %q", res.FinishReason)
	}
	if res.Usage.InputTokens != 10 || res.Usage.OutputTokens != 7 || res.Usage.TotalTokens != 17 {
		t.Fatalf("bad usage: %+v", res.Usage)
	}
	if res.Model != "claude-opus-5-5" {
		t.Fatalf("bad model: %q", res.Model)
	}

	req := fake.last()
	if req.URL.String() != "https://api.anthropic.com/v1/messages" {
		t.Fatalf("bad url: %s", req.URL)
	}
	if req.Header.Get("x-api-key") != "sk-ant" {
		t.Fatalf("bad api key header: %q", req.Header.Get("x-api-key"))
	}
	if req.Header.Get("anthropic-version") != "2023-06-01" {
		t.Fatalf("bad version header: %q", req.Header.Get("anthropic-version"))
	}

	var sent map[string]any
	if err := json.Unmarshal([]byte(req.Body), &sent); err != nil {
		t.Fatalf("body not JSON: %v", err)
	}
	if sent["model"] != "claude-opus-5-5" {
		t.Fatalf("bad model in body: %v", sent["model"])
	}
	if sent["system"] != "Be terse" {
		t.Fatalf("system not split to top-level: %v", sent["system"])
	}
	if sent["max_tokens"] != float64(16000) {
		t.Fatalf("bad max_tokens: %v", sent["max_tokens"])
	}
	msgs, ok := sent["messages"].([]any)
	if !ok || len(msgs) != 1 {
		t.Fatalf("expected 1 non-system message, got: %v", sent["messages"])
	}
	if msgs[0].(map[string]any)["role"] != "user" {
		t.Fatalf("bad first message role: %v", msgs[0])
	}
}

func TestAnthropicPassthroughQuery(t *testing.T) {
	fake := newFake(resp(200, `{"model":"m","stop_reason":"end_turn","usage":{"input_tokens":1,"output_tokens":1},"content":[{"type":"text","text":"ok"}]}`))
	chat, _ := NewChat(Anthropic, AnthropicConfig{APIKey: "sk-ant"}, WithHTTPClient(fake))
	_, err := chat.Complete(context.Background(), ChatInput{
		Model:       "m",
		Messages:    []ChatMessage{{Role: "user", Content: "hi"}},
		Passthrough: &Passthrough{Query: map[string]string{"beta": "1"}},
	})
	if err != nil {
		t.Fatal(err)
	}
	if got := fake.last().URL.String(); got != "https://api.anthropic.com/v1/messages?beta=1" {
		t.Fatalf("passthrough query not appended: %s", got)
	}
}

func TestAnthropicReasoningKeepsCallerMaxTokens(t *testing.T) {
	// Adaptive thinking has no budget, so a small explicit cap goes out as-is
	// (no error, no inflation) and the default stays the connector default.
	cases := []struct {
		mx   *int
		want float64
	}{{intPtr(100), 100}, {nil, 16000}}
	for _, c := range cases {
		fake := newFake(resp(200, `{"model":"m","stop_reason":"end_turn","usage":{"input_tokens":1,"output_tokens":1},"content":[{"type":"text","text":"ok"}]}`))
		chat, _ := NewChat(Anthropic, AnthropicConfig{APIKey: "sk-ant"}, WithHTTPClient(fake))
		if _, err := chat.Complete(context.Background(), ChatInput{
			Model:           "claude-opus-5-5",
			Messages:        []ChatMessage{{Role: "user", Content: "hi"}},
			MaxOutputTokens: c.mx,
			Reasoning:       &ReasoningOptions{Effort: "low"},
		}); err != nil {
			t.Fatal(err)
		}
		var sent map[string]any
		_ = json.Unmarshal([]byte(fake.last().Body), &sent)
		if sent["max_tokens"] != c.want {
			t.Fatalf("max_tokens = %v, want %v", sent["max_tokens"], c.want)
		}
	}
}

func intPtr(n int) *int { return &n }

func TestAnthropicToolUse(t *testing.T) {
	body := `{"model":"claude-opus-5-5","stop_reason":"tool_use","usage":{"input_tokens":20,"output_tokens":15},"content":[{"type":"text","text":"Let me check"},{"type":"tool_use","id":"toolu_1","name":"get_weather","input":{"city":"Paris"}}]}`
	fake := newFake(resp(200, body))
	chat, _ := NewChat(Anthropic, AnthropicConfig{APIKey: "k"}, WithHTTPClient(fake))
	res, err := chat.Complete(context.Background(), ChatInput{
		Model:    "claude-opus-5-5",
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
		t.Fatalf("bad finish: %q", res.FinishReason)
	}
	if len(res.Message.ToolCalls) != 1 {
		t.Fatalf("expected 1 tool call, got %d", len(res.Message.ToolCalls))
	}
	tc := res.Message.ToolCalls[0]
	if tc.ID != "toolu_1" || tc.Name != "get_weather" {
		t.Fatalf("bad tool call: %+v", tc)
	}
	var args map[string]any
	if err := json.Unmarshal([]byte(tc.Arguments), &args); err != nil {
		t.Fatalf("tool args not valid JSON: %q", tc.Arguments)
	}
	if args["city"] != "Paris" {
		t.Fatalf("bad tool args: %v", args)
	}
	if res.Message.Content != "Let me check" {
		t.Fatalf("bad content: %q", res.Message.Content)
	}

	var sent map[string]any
	_ = json.Unmarshal([]byte(fake.last().Body), &sent)
	tools := sent["tools"].([]any)
	if len(tools) != 1 {
		t.Fatalf("expected 1 tool, got %d", len(tools))
	}
	tool0 := tools[0].(map[string]any)
	if tool0["name"] != "get_weather" {
		t.Fatalf("bad tool name: %v", tool0["name"])
	}
	if _, ok := tool0["input_schema"]; !ok {
		t.Fatalf("tool missing input_schema: %v", tool0)
	}
	if _, ok := tool0["parameters"]; ok {
		t.Fatalf("tool must not carry OpenAI-shaped 'parameters'")
	}
	if tc := sent["tool_choice"].(map[string]any); tc["type"] != "any" {
		t.Fatalf("required must map to tool_choice type 'any', got %v", tc["type"])
	}
}

func TestAnthropicStream(t *testing.T) {
	sse := strings.Join([]string{
		`event: message_start`,
		`data: {"type":"message_start","message":{"model":"claude","usage":{"input_tokens":11}}}`,
		``,
		`event: content_block_start`,
		`data: {"type":"content_block_start","index":0,"content_block":{"type":"text"}}`,
		``,
		`event: content_block_delta`,
		`data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hel"}}`,
		``,
		`event: content_block_delta`,
		`data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"lo"}}`,
		``,
		`event: message_delta`,
		`data: {"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":4}}`,
		``,
		`event: message_stop`,
		`data: {"type":"message_stop"}`,
		``,
	}, "\n")
	fake := newFake(sseResp(sse))
	chat, _ := NewChat(Anthropic, AnthropicConfig{APIKey: "k"}, WithHTTPClient(fake))
	stream, err := chat.Stream(context.Background(), ChatInput{Model: "claude", Messages: []ChatMessage{{Role: "user", Content: "hi"}}})
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
	if text != "Hello" {
		t.Fatalf("bad stream text: %q", text)
	}
	if finish != "stop" {
		t.Fatalf("bad finish: %q", finish)
	}
	if usage == nil || usage.InputTokens != 11 || usage.OutputTokens != 4 || usage.TotalTokens != 15 {
		t.Fatalf("bad stream usage (input tokens must carry from message_start): %+v", usage)
	}

	var sent map[string]any
	_ = json.Unmarshal([]byte(fake.last().Body), &sent)
	if sent["stream"] != true {
		t.Fatalf("expected stream:true in body")
	}
}

func TestAnthropicStreamToolUse(t *testing.T) {
	sse := strings.Join([]string{
		`data: {"type":"message_start","message":{"usage":{"input_tokens":5}}}`,
		``,
		`data: {"type":"content_block_start","index":0,"content_block":{"type":"tool_use","id":"toolu_9","name":"lookup"}}`,
		``,
		`data: {"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"{\"q\":"}}`,
		``,
		`data: {"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"\"hi\"}"}}`,
		``,
		`data: {"type":"message_delta","delta":{"stop_reason":"tool_use"},"usage":{"output_tokens":3}}`,
		``,
	}, "\n")
	fake := newFake(sseResp(sse))
	chat, _ := NewChat(Anthropic, AnthropicConfig{APIKey: "k"}, WithHTTPClient(fake))
	stream, err := chat.Stream(context.Background(), ChatInput{Model: "claude", Messages: []ChatMessage{{Role: "user", Content: "hi"}}})
	if err != nil {
		t.Fatal(err)
	}
	defer stream.Close()

	var id, name, args string
	var finish FinishReason
	for {
		d, err := stream.Recv()
		if err == io.EOF {
			break
		}
		if err != nil {
			t.Fatal(err)
		}
		if d.ToolCallDelta != nil {
			if d.ToolCallDelta.ID != "" {
				id = d.ToolCallDelta.ID
			}
			if d.ToolCallDelta.FunctionName != "" {
				name = d.ToolCallDelta.FunctionName
			}
			args += d.ToolCallDelta.ArgumentsDelta
		}
		if d.FinishReason != "" {
			finish = d.FinishReason
		}
	}
	if id != "toolu_9" || name != "lookup" {
		t.Fatalf("bad tool id/name: %q / %q", id, name)
	}
	if args != `{"q":"hi"}` {
		t.Fatalf("bad accumulated args: %q", args)
	}
	if finish != "tool_calls" {
		t.Fatalf("bad finish: %q", finish)
	}
}

func TestAnthropicStreamMidStreamError(t *testing.T) {
	sse := strings.Join([]string{
		`event: message_start`,
		`data: {"type":"message_start","message":{"usage":{"input_tokens":3}}}`,
		``,
		`event: content_block_delta`,
		`data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hi"}}`,
		``,
		`event: error`,
		`data: {"type":"error","error":{"type":"overloaded_error","message":"Overloaded"}}`,
		``,
	}, "\n")
	fake := newFake(sseResp(sse))
	chat, _ := NewChat(Anthropic, AnthropicConfig{APIKey: "k"}, WithHTTPClient(fake))
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
		t.Fatal("anthropic mid-stream `error` event was swallowed as io.EOF")
	}
	var ce *ConnectorError
	if !errors.As(err, &ce) {
		t.Fatalf("expected *ConnectorError, got %v", err)
	}
	if ce.ProviderMessage != "Overloaded" {
		t.Fatalf("bad message: %s", ce.ProviderMessage)
	}
}

func TestAnthropicReasoningThinking(t *testing.T) {
	fake := newFake(resp(200, `{"content":[{"type":"text","text":"ok"}],"stop_reason":"end_turn","usage":{"input_tokens":1,"output_tokens":1}}`))
	chat, _ := NewChat(Anthropic, AnthropicConfig{APIKey: "k"}, WithHTTPClient(fake))
	temp := 0.7
	_, err := chat.Complete(context.Background(), ChatInput{
		Model:       "claude",
		Messages:    []ChatMessage{{Role: "user", Content: "hi"}},
		Reasoning:   &ReasoningOptions{Effort: "high"},
		Temperature: &temp,
	})
	if err != nil {
		t.Fatal(err)
	}
	// D1: adaptive thinking + output_config.effort for every model, never a
	// budget, no model sniffing.
	var sent map[string]any
	_ = json.Unmarshal([]byte(fake.last().Body), &sent)
	if !reflect.DeepEqual(sent["thinking"], map[string]any{"type": "adaptive"}) {
		t.Fatalf("bad thinking: %v", sent["thinking"])
	}
	if !reflect.DeepEqual(sent["output_config"], map[string]any{"effort": "high"}) {
		t.Fatalf("bad output_config: %v", sent["output_config"])
	}
	if strings.Contains(fake.last().Body, "budget_tokens") {
		t.Fatalf("budget_tokens must never be sent: %s", fake.last().Body)
	}
	if sent["max_tokens"] != float64(16000) {
		t.Fatalf("max_tokens must stay the default: %v", sent["max_tokens"])
	}
	if _, ok := sent["temperature"]; ok {
		t.Fatalf("temperature must be omitted when thinking is enabled")
	}
	if _, ok := sent["top_p"]; ok {
		t.Fatalf("top_p must be omitted when thinking is enabled")
	}
}

func TestAnthropicReasoningEffortValues(t *testing.T) {
	for effort, want := range map[string]string{"low": "low", "medium": "medium", "high": "high", "": "medium", "max": "max"} {
		fake := newFake(resp(200, `{"content":[],"stop_reason":"end_turn"}`))
		chat, _ := NewChat(Anthropic, AnthropicConfig{APIKey: "k"}, WithHTTPClient(fake))
		if _, err := chat.Complete(context.Background(), ChatInput{Model: "m", Messages: []ChatMessage{{Role: "user", Content: "hi"}}, Reasoning: &ReasoningOptions{Effort: effort}}); err != nil {
			t.Fatal(err)
		}
		var sent map[string]any
		_ = json.Unmarshal([]byte(fake.last().Body), &sent)
		if got := sent["output_config"].(map[string]any)["effort"]; got != want {
			t.Fatalf("effort %q → %v, want %s", effort, got, want)
		}
	}
	// Without Reasoning neither field is sent, and sampling goes through.
	fake := newFake(resp(200, `{"content":[],"stop_reason":"end_turn"}`))
	chat, _ := NewChat(Anthropic, AnthropicConfig{APIKey: "k"}, WithHTTPClient(fake))
	temp := 0.2
	if _, err := chat.Complete(context.Background(), ChatInput{Model: "m", Messages: []ChatMessage{{Role: "user", Content: "hi"}}, Temperature: &temp}); err != nil {
		t.Fatal(err)
	}
	var sent map[string]any
	_ = json.Unmarshal([]byte(fake.last().Body), &sent)
	if _, ok := sent["thinking"]; ok {
		t.Fatalf("thinking sent without Reasoning")
	}
	if _, ok := sent["output_config"]; ok {
		t.Fatalf("output_config sent without Reasoning")
	}
	if sent["temperature"] != 0.2 {
		t.Fatalf("temperature dropped without Reasoning: %v", sent["temperature"])
	}
}

func TestAnthropicToolResultInUserTurn(t *testing.T) {
	fake := newFake(resp(200, `{"content":[{"type":"text","text":"ok"}],"stop_reason":"end_turn","usage":{"input_tokens":1,"output_tokens":1}}`))
	chat, _ := NewChat(Anthropic, AnthropicConfig{APIKey: "k"}, WithHTTPClient(fake))
	_, err := chat.Complete(context.Background(), ChatInput{
		Model: "claude",
		Messages: []ChatMessage{
			{Role: "user", Content: "weather?"},
			{Role: "assistant", ToolCalls: []ToolCall{{ID: "t1", Name: "get_weather", Arguments: `{"city":"Paris"}`}}},
			{Role: "tool", ToolCallID: "t1", Content: "sunny"},
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	var sent map[string]any
	_ = json.Unmarshal([]byte(fake.last().Body), &sent)
	msgs := sent["messages"].([]any)
	if len(msgs) != 3 {
		t.Fatalf("expected 3 messages (user, assistant tool_use, user tool_result), got %d: %v", len(msgs), msgs)
	}

	asst := msgs[1].(map[string]any)
	if asst["role"] != "assistant" {
		t.Fatalf("bad assistant role: %v", asst["role"])
	}
	ablock := asst["content"].([]any)[0].(map[string]any)
	if ablock["type"] != "tool_use" || ablock["name"] != "get_weather" || ablock["id"] != "t1" {
		t.Fatalf("bad assistant tool_use block: %v", ablock)
	}

	last := msgs[2].(map[string]any)
	if last["role"] != "user" {
		t.Fatalf("tool result must be carried in a user turn, got role %v", last["role"])
	}
	block := last["content"].([]any)[0].(map[string]any)
	if block["type"] != "tool_result" || block["tool_use_id"] != "t1" || block["content"] != "sunny" {
		t.Fatalf("bad tool_result block: %v", block)
	}
}

func TestAnthropicErrorMapping(t *testing.T) {
	fake := newFake(respH(429, `{"type":"error","error":{"type":"rate_limit_error","message":"slow down"}}`, "Retry-After", "12"))
	chat, _ := NewChat(Anthropic, AnthropicConfig{APIKey: "k"}, WithHTTPClient(fake))
	_, err := chat.Complete(context.Background(), ChatInput{Model: "m", Messages: []ChatMessage{{Role: "user", Content: "x"}}})
	var ce *ConnectorError
	if !errors.As(err, &ce) || ce.ProviderCode != CodeRateLimited {
		t.Fatalf("expected rate_limited, got %v", err)
	}
	if ce.ProviderMessage != "slow down" {
		t.Fatalf("bad message: %q", ce.ProviderMessage)
	}
	cause, _ := ce.Cause.(map[string]any)
	if cause["retryAfter"] != "12" {
		t.Fatalf("expected retryAfter=12 in cause, got %v", ce.Cause)
	}
	if cause["retryAfterSeconds"] != 12 {
		t.Fatalf("expected retryAfterSeconds=12, got %v", cause["retryAfterSeconds"])
	}
}

func TestAnthropicContextLengthError(t *testing.T) {
	fake := newFake(resp(400, `{"type":"error","error":{"type":"invalid_request_error","message":"prompt is too long: 250000 tokens > 200000 maximum"}}`))
	chat, _ := NewChat(Anthropic, AnthropicConfig{APIKey: "k"}, WithHTTPClient(fake))
	_, err := chat.Complete(context.Background(), ChatInput{Model: "m", Messages: []ChatMessage{{Role: "user", Content: "x"}}})
	var ce *ConnectorError
	if !errors.As(err, &ce) || ce.ProviderCode != CodeContextLengthExceeded {
		t.Fatalf("expected context_length_exceeded, got %v", err)
	}
}

func TestAnthropicConfigTypeMismatch(t *testing.T) {
	_, err := NewChat(Anthropic, OpenAICompatConfig{APIKey: "k"})
	var ce *ConnectorError
	if !errors.As(err, &ce) || ce.ProviderCode != CodeInvalidRequest {
		t.Fatalf("expected invalid_request for wrong config type, got %v", err)
	}
}

// anthropicThinkingToolBody is a Messages response with every block kind the
// replay slot must carry: thinking (with signature), redacted_thinking, text
// and tool_use.
const anthropicThinkingToolBody = `{"id":"msg_01","type":"message","role":"assistant","model":"claude-opus-5-5",` +
	`"content":[{"type":"thinking","thinking":"The user wants the weather.","signature":"EqQBCgIYAhIM"},` +
	`{"type":"redacted_thinking","data":"EmwKAhgBEgy3va3pzix"},` +
	`{"type":"text","text":"Let me check."},` +
	`{"type":"tool_use","id":"toolu_01A","name":"get_weather","input":{"city":"Paris"}}],` +
	`"stop_reason":"tool_use","usage":{"input_tokens":50,"cache_creation_input_tokens":20,"cache_read_input_tokens":100,"output_tokens":30}}`

func TestAnthropicProviderMetadataCaptured(t *testing.T) {
	fake := newFake(resp(200, anthropicThinkingToolBody))
	chat, _ := NewChat(Anthropic, AnthropicConfig{APIKey: "k"}, WithHTTPClient(fake))
	res, err := chat.Complete(context.Background(), ChatInput{Model: "claude-opus-5-5", Messages: []ChatMessage{{Role: "user", Content: "weather?"}}})
	if err != nil {
		t.Fatal(err)
	}
	var raw map[string]any
	_ = json.Unmarshal([]byte(anthropicThinkingToolBody), &raw)
	want := map[string]any{"anthropic": map[string]any{"content": raw["content"]}}
	if !reflect.DeepEqual(res.Message.ProviderMetadata, want) {
		t.Fatalf("metadata must be the verbatim content array:\n got %v\nwant %v", res.Message.ProviderMetadata, want)
	}
	if res.Message.Content != "Let me check." || res.Message.ToolCalls[0].ProviderMetadata != nil {
		t.Fatalf("normalized fields: %+v", res.Message)
	}
	// Cache reads and writes are billed input.
	if res.Usage != (Usage{InputTokens: 170, OutputTokens: 30, TotalTokens: 200}) {
		t.Fatalf("usage: %+v", res.Usage)
	}
}

// anthropicAssistantContent returns the content of the first assistant turn sent.
func anthropicAssistantContent(t *testing.T, f *fakeClient) []any {
	t.Helper()
	for _, m := range sentBody(t, f)["messages"].([]any) {
		if m.(map[string]any)["role"] == "assistant" {
			return m.(map[string]any)["content"].([]any)
		}
	}
	t.Fatal("no assistant turn sent")
	return nil
}

func TestAnthropicVerbatimReplayWhenMatching(t *testing.T) {
	fake := newFake(resp(200, anthropicThinkingToolBody), resp(200, `{"content":[{"type":"text","text":"Sunny."}],"stop_reason":"end_turn"}`))
	chat, _ := NewChat(Anthropic, AnthropicConfig{APIKey: "k"}, WithHTTPClient(fake))
	first, err := chat.Complete(context.Background(), ChatInput{Model: "claude-opus-5-5", Messages: []ChatMessage{{Role: "user", Content: "weather?"}}})
	if err != nil {
		t.Fatal(err)
	}
	_, err = chat.Complete(context.Background(), ChatInput{Model: "claude-opus-5-5", Messages: []ChatMessage{
		{Role: "user", Content: "weather?"},
		{Role: "assistant", Content: first.Message.Content, ToolCalls: first.Message.ToolCalls, ProviderMetadata: first.Message.ProviderMetadata},
		{Role: "tool", ToolCallID: "toolu_01A", Content: "sunny"},
	}})
	if err != nil {
		t.Fatal(err)
	}
	got := anthropicAssistantContent(t, fake)
	var raw map[string]any
	_ = json.Unmarshal([]byte(anthropicThinkingToolBody), &raw)
	if !reflect.DeepEqual(got, raw["content"]) {
		t.Fatalf("assistant turn must replay the original blocks verbatim:\n got %v\nwant %v", got, raw["content"])
	}
}

func TestAnthropicRebuildWhenMetadataDoesNotMatch(t *testing.T) {
	var raw map[string]any
	_ = json.Unmarshal([]byte(anthropicThinkingToolBody), &raw)
	meta := map[string]any{"anthropic": map[string]any{"content": raw["content"]}}
	call := ToolCall{ID: "toolu_01A", Name: "get_weather", Arguments: `{"city":"Paris"}`}
	cases := map[string]ChatMessage{
		"edited text":     {Role: "assistant", Content: "Checking now.", ToolCalls: []ToolCall{call}, ProviderMetadata: meta},
		"other tool id":   {Role: "assistant", Content: "Let me check.", ToolCalls: []ToolCall{{ID: "toolu_99", Name: "get_weather", Arguments: `{}`}}, ProviderMetadata: meta},
		"extra tool call": {Role: "assistant", Content: "Let me check.", ToolCalls: []ToolCall{call, {ID: "toolu_02", Name: "x", Arguments: `{}`}}, ProviderMetadata: meta},
		"not a list":      {Role: "assistant", Content: "Let me check.", ToolCalls: []ToolCall{call}, ProviderMetadata: map[string]any{"anthropic": map[string]any{"content": "nope"}}},
		"foreign key":     {Role: "assistant", Content: "Let me check.", ToolCalls: []ToolCall{call}, ProviderMetadata: map[string]any{"gemini": map[string]any{"thoughtSignature": "g-sig"}}},
	}
	for name, asst := range cases {
		fake := newFake(resp(200, `{"content":[{"type":"text","text":"ok"}],"stop_reason":"end_turn"}`))
		chat, _ := NewChat(Anthropic, AnthropicConfig{APIKey: "k"}, WithHTTPClient(fake))
		_, err := chat.Complete(context.Background(), ChatInput{Model: "m", Messages: []ChatMessage{
			{Role: "user", Content: "weather?"}, asst, {Role: "tool", ToolCallID: asst.ToolCalls[0].ID, Content: "sunny"},
		}})
		if err != nil {
			t.Fatal(err)
		}
		got := anthropicAssistantContent(t, fake)
		for _, b := range got {
			if ty := b.(map[string]any)["type"]; ty == "thinking" || ty == "redacted_thinking" {
				t.Fatalf("%s: thinking blocks must be dropped on a rebuilt turn: %v", name, got)
			}
		}
		if got[0].(map[string]any)["text"] != asst.Content {
			t.Fatalf("%s: rebuilt turn must use the message's own fields: %v", name, got)
		}
		if body := fake.last().Body; strings.Contains(body, "g-sig") || strings.Contains(body, "gemini") {
			t.Fatalf("%s: foreign metadata leaked: %s", name, body)
		}
	}
}

func TestAnthropicStreamRebuildsContentArray(t *testing.T) {
	// The same response as anthropicThinkingToolBody, streamed per the
	// streaming docs: thinking via thinking_delta + signature_delta,
	// redacted_thinking whole in content_block_start, tool input as partial_json.
	sse := strings.Join([]string{
		`event: message_start`,
		`data: {"type":"message_start","message":{"id":"msg_01","type":"message","role":"assistant","model":"claude-opus-5-5","content":[],"stop_reason":null,"usage":{"input_tokens":50,"cache_creation_input_tokens":20,"cache_read_input_tokens":100,"output_tokens":1}}}`,
		``,
		`event: content_block_start`,
		`data: {"type":"content_block_start","index":0,"content_block":{"type":"thinking","thinking":"","signature":""}}`,
		``,
		`event: content_block_delta`,
		`data: {"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":"The user wants "}}`,
		``,
		`event: content_block_delta`,
		`data: {"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":"the weather."}}`,
		``,
		`event: content_block_delta`,
		`data: {"type":"content_block_delta","index":0,"delta":{"type":"signature_delta","signature":"EqQBCgIYAhIM"}}`,
		``,
		`event: content_block_stop`,
		`data: {"type":"content_block_stop","index":0}`,
		``,
		`event: content_block_start`,
		`data: {"type":"content_block_start","index":1,"content_block":{"type":"redacted_thinking","data":"EmwKAhgBEgy3va3pzix"}}`,
		``,
		`event: content_block_stop`,
		`data: {"type":"content_block_stop","index":1}`,
		``,
		`event: content_block_start`,
		`data: {"type":"content_block_start","index":2,"content_block":{"type":"text","text":""}}`,
		``,
		`event: content_block_delta`,
		`data: {"type":"content_block_delta","index":2,"delta":{"type":"text_delta","text":"Let me "}}`,
		``,
		`event: content_block_delta`,
		`data: {"type":"content_block_delta","index":2,"delta":{"type":"text_delta","text":"check."}}`,
		``,
		`event: content_block_stop`,
		`data: {"type":"content_block_stop","index":2}`,
		``,
		`event: content_block_start`,
		`data: {"type":"content_block_start","index":3,"content_block":{"type":"tool_use","id":"toolu_01A","name":"get_weather","input":{}}}`,
		``,
		`event: content_block_delta`,
		`data: {"type":"content_block_delta","index":3,"delta":{"type":"input_json_delta","partial_json":"{\"city\": "}}`,
		``,
		`event: content_block_delta`,
		`data: {"type":"content_block_delta","index":3,"delta":{"type":"input_json_delta","partial_json":"\"Paris\"}"}}`,
		``,
		`event: content_block_stop`,
		`data: {"type":"content_block_stop","index":3}`,
		``,
		`event: message_delta`,
		`data: {"type":"message_delta","delta":{"stop_reason":"tool_use","stop_sequence":null},"usage":{"output_tokens":30}}`,
		``,
		`event: message_stop`,
		`data: {"type":"message_stop"}`,
		``,
	}, "\n")
	chat, _ := NewChat(Anthropic, AnthropicConfig{APIKey: "k"}, WithHTTPClient(newFake(sseResp(sse))))
	stream, err := chat.Stream(context.Background(), ChatInput{Model: "claude-opus-5-5", Messages: []ChatMessage{{Role: "user", Content: "weather?"}}})
	if err != nil {
		t.Fatal(err)
	}
	ds := drain(t, stream)
	var terminal *ChatStreamDelta
	for _, d := range ds {
		if d.ProviderMetadata != nil {
			if terminal != nil {
				t.Fatal("ProviderMetadata must ride only on the terminal delta")
			}
			terminal = d
		}
		if d.ToolCallDelta != nil && d.ToolCallDelta.Index != 0 {
			// The tool sits at content block 3 but is the first tool call.
			t.Fatalf("tool-call index must be the call ordinal, got %d", d.ToolCallDelta.Index)
		}
	}
	if terminal == nil || terminal.FinishReason != "tool_calls" {
		t.Fatalf("terminal delta: %+v", terminal)
	}
	var raw map[string]any
	_ = json.Unmarshal([]byte(anthropicThinkingToolBody), &raw)
	want := map[string]any{"anthropic": map[string]any{"content": raw["content"]}}
	if !reflect.DeepEqual(terminal.ProviderMetadata, want) {
		t.Fatalf("stream must rebuild the non-stream content array:\n got %v\nwant %v", terminal.ProviderMetadata, want)
	}
	if *terminal.Usage != (Usage{InputTokens: 170, OutputTokens: 30, TotalTokens: 200}) {
		t.Fatalf("stream usage: %+v", *terminal.Usage)
	}
}

func TestAnthropicStreamUsageFromMessageDelta(t *testing.T) {
	// message_delta usage is cumulative and may restate the input side; it
	// then wins over message_start.
	sse := strings.Join([]string{
		`data: {"type":"message_start","message":{"usage":{"input_tokens":5,"output_tokens":1}}}`,
		``,
		`data: {"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"input_tokens":10682,"cache_creation_input_tokens":0,"cache_read_input_tokens":8,"output_tokens":510}}`,
		``,
	}, "\n")
	chat, _ := NewChat(Anthropic, AnthropicConfig{APIKey: "k"}, WithHTTPClient(newFake(sseResp(sse))))
	stream, _ := chat.Stream(context.Background(), ChatInput{Model: "m", Messages: []ChatMessage{{Role: "user", Content: "hi"}}})
	ds := drain(t, stream)
	if u := ds[len(ds)-1].Usage; u == nil || *u != (Usage{InputTokens: 10690, OutputTokens: 510, TotalTokens: 11200}) {
		t.Fatalf("usage: %+v", u)
	}
}

func TestAnthropicStreamToolIndexAfterTextBlock(t *testing.T) {
	sse := strings.Join([]string{
		`data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}`,
		``,
		`data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Checking"}}`,
		``,
		`data: {"type":"content_block_start","index":1,"content_block":{"type":"tool_use","id":"toolu_1","name":"a","input":{}}}`,
		``,
		`data: {"type":"content_block_delta","index":1,"delta":{"type":"input_json_delta","partial_json":"{}"}}`,
		``,
		`data: {"type":"content_block_start","index":2,"content_block":{"type":"tool_use","id":"toolu_2","name":"b","input":{}}}`,
		``,
		`data: {"type":"content_block_delta","index":2,"delta":{"type":"input_json_delta","partial_json":"{}"}}`,
		``,
		`data: {"type":"message_delta","delta":{"stop_reason":"tool_use"},"usage":{"output_tokens":3}}`,
		``,
	}, "\n")
	chat, _ := NewChat(Anthropic, AnthropicConfig{APIKey: "k"}, WithHTTPClient(newFake(sseResp(sse))))
	stream, _ := chat.Stream(context.Background(), ChatInput{Model: "m", Messages: []ChatMessage{{Role: "user", Content: "hi"}}})
	var idx []int
	for _, d := range drain(t, stream) {
		if d.ToolCallDelta != nil {
			idx = append(idx, d.ToolCallDelta.Index)
		}
	}
	if !reflect.DeepEqual(idx, []int{0, 0, 1, 1}) {
		t.Fatalf("tool-call indices must be call ordinals: %v", idx)
	}
}

func TestAnthropicStreamErrorTypes(t *testing.T) {
	for errType, want := range map[string]ProviderCode{
		"overloaded_error": CodeProviderUnavailable, "api_error": CodeProviderUnavailable,
		"rate_limit_error": CodeRateLimited, "authentication_error": CodeAuthFailed,
		"invalid_request_error": CodeInvalidRequest, "request_too_large": CodeInvalidRequest,
	} {
		sse := `data: {"type":"error","error":{"type":"` + errType + `","message":"boom"}}` + "\n\n"
		chat, _ := NewChat(Anthropic, AnthropicConfig{APIKey: "k"}, WithHTTPClient(newFake(sseResp(sse))))
		stream, _ := chat.Stream(context.Background(), ChatInput{Model: "m", Messages: []ChatMessage{{Role: "user", Content: "hi"}}})
		_, err := stream.Recv()
		var ce *ConnectorError
		if !errors.As(err, &ce) || ce.ProviderCode != want {
			t.Fatalf("%s: got %v, want %s", errType, err, want)
		}
	}
}

func TestAnthropicErrorClassification(t *testing.T) {
	cases := []struct {
		status int
		msg    string
		want   ProviderCode
	}{
		{400, "prompt is too long: 250000 tokens > 200000 maximum", CodeContextLengthExceeded},
		{400, "input length and `max_tokens` exceed context limit: 195000 + 8192 > 200000, decrease input length or `max_tokens` and try again", CodeContextLengthExceeded},
		// Ordinary validation errors that merely mention max_tokens / context.
		{400, "max_tokens: 200000 > 128000, which is the maximum allowed number of output tokens for claude-opus-5-5", CodeInvalidRequest},
		{400, "tool_choice: type \"tool\" and \"any\" are not supported for this model.", CodeInvalidRequest},
		{400, "adaptive thinking is not supported on this model", CodeInvalidRequest},
		{402, "Your credit balance is too low to access the Anthropic API.", CodeRateLimited},
		{413, "Request exceeds the maximum allowed number of bytes.", CodeInvalidRequest},
		{529, "Overloaded", CodeProviderUnavailable},
	}
	for _, c := range cases {
		body := `{"type":"error","error":{"type":"x","message":` + strconv.Quote(c.msg) + `}}`
		chat, _ := NewChat(Anthropic, AnthropicConfig{APIKey: "k"}, WithHTTPClient(newFake(resp(c.status, body))))
		_, err := chat.Complete(context.Background(), ChatInput{Model: "m", Messages: []ChatMessage{{Role: "user", Content: "x"}}})
		var ce *ConnectorError
		if !errors.As(err, &ce) || ce.ProviderCode != c.want {
			t.Fatalf("%d %q: got %v, want %s", c.status, c.msg, err, c.want)
		}
	}
}

func TestAnthropicStopReasonMapping(t *testing.T) {
	for sr, want := range map[string]FinishReason{
		"end_turn": "stop", "stop_sequence": "stop", "max_tokens": "length", "model_context_window_exceeded": "length",
		"tool_use": "tool_calls", "refusal": "content_filter", "pause_turn": "unknown",
	} {
		if got := anthropicMapStopReason(sr); got != want {
			t.Fatalf("%s → %s, want %s", sr, got, want)
		}
	}
}

func TestAnthropicEmptyTextPartDropped(t *testing.T) {
	fake := newFake(resp(200, `{"content":[],"stop_reason":"end_turn"}`))
	chat, _ := NewChat(Anthropic, AnthropicConfig{APIKey: "k"}, WithHTTPClient(fake))
	_, err := chat.Complete(context.Background(), ChatInput{Model: "m", Messages: []ChatMessage{
		{Role: "user", ContentParts: []ContentPart{TextPart(""), ImagePart("aGk=", "image/png")}},
	}})
	if err != nil {
		t.Fatal(err)
	}
	blocks := sentBody(t, fake)["messages"].([]any)[0].(map[string]any)["content"].([]any)
	if len(blocks) != 1 || blocks[0].(map[string]any)["type"] != "image" {
		t.Fatalf("empty text block must be dropped: %v", blocks)
	}
}

func TestAnthropicRebuildCascadesToLaterTurns(t *testing.T) {
	// Once an assistant turn is rebuilt (edited here), a later turn that still
	// matches must be rebuilt too: Anthropic invalidates every thinking block
	// after an edited or stripped turn.
	turn := func(id, sig string) (ChatMessage, ChatMessage) {
		blocks := []any{
			map[string]any{"type": "thinking", "thinking": "plan", "signature": sig},
			map[string]any{"type": "text", "text": "Checking."},
			map[string]any{"type": "tool_use", "id": id, "name": "lookup", "input": map[string]any{"q": "x"}},
		}
		return ChatMessage{Role: "assistant", Content: "Checking.", ToolCalls: []ToolCall{{ID: id, Name: "lookup", Arguments: `{"q":"x"}`}},
				ProviderMetadata: map[string]any{"anthropic": map[string]any{"content": blocks}}},
			ChatMessage{Role: "tool", ToolCallID: id, Content: "ok"}
	}
	a, aRes := turn("toolu_A", "SIG_A")
	b, bRes := turn("toolu_B", "SIG_B")
	send := func(msgs []ChatMessage) string {
		fake := newFake(resp(200, `{"content":[{"type":"text","text":"done"}],"stop_reason":"end_turn"}`))
		chat, _ := NewChat(Anthropic, AnthropicConfig{APIKey: "k"}, WithHTTPClient(fake))
		if _, err := chat.Complete(context.Background(), ChatInput{Model: "m", Messages: msgs}); err != nil {
			t.Fatal(err)
		}
		return fake.last().Body
	}
	// Both unedited: both replay verbatim.
	if body := send([]ChatMessage{{Role: "user", Content: "q"}, a, aRes, b, bRes}); !strings.Contains(body, "SIG_A") || !strings.Contains(body, "SIG_B") {
		t.Fatalf("unedited turns must replay verbatim: %s", body)
	}
	// Turn A edited: A and the still-matching B are both rebuilt.
	edited := a
	edited.Content = "Let me look."
	if body := send([]ChatMessage{{Role: "user", Content: "q"}, edited, aRes, b, bRes}); strings.Contains(body, "SIG_A") || strings.Contains(body, "SIG_B") {
		t.Fatalf("thinking after a rebuilt turn must be dropped: %s", body)
	}
	// A turn with no metadata at all (hand-built, or from another provider)
	// counts as rebuilt as well.
	plain := ChatMessage{Role: "assistant", Content: "Hello."}
	if body := send([]ChatMessage{{Role: "user", Content: "hi"}, plain, {Role: "user", Content: "q"}, b, bRes}); strings.Contains(body, "SIG_B") {
		t.Fatalf("thinking after a rebuilt turn must be dropped: %s", body)
	}
}

func TestAnthropicReplayRequiresSameToolInput(t *testing.T) {
	blocks := []any{
		map[string]any{"type": "thinking", "thinking": "plan", "signature": "SIG"},
		map[string]any{"type": "tool_use", "id": "toolu_1", "name": "lookup", "input": map[string]any{"q": "paris", "n": 2.0}},
	}
	meta := map[string]any{"anthropic": map[string]any{"content": blocks}}
	for args, wantVerbatim := range map[string]bool{
		`{"n":2,"q":"paris"}`:    true, // same value, other key order / number form
		`{"q": "paris", "n": 2}`: true,
		`{"q":"rome","n":2}`:     false, // edited arguments must win over the stored input
		`{"q":"paris"}`:          false,
	} {
		fake := newFake(resp(200, `{"content":[{"type":"text","text":"ok"}],"stop_reason":"end_turn"}`))
		chat, _ := NewChat(Anthropic, AnthropicConfig{APIKey: "k"}, WithHTTPClient(fake))
		_, err := chat.Complete(context.Background(), ChatInput{Model: "m", Messages: []ChatMessage{
			{Role: "user", Content: "q"},
			{Role: "assistant", ToolCalls: []ToolCall{{ID: "toolu_1", Name: "lookup", Arguments: args}}, ProviderMetadata: meta},
			{Role: "tool", ToolCallID: "toolu_1", Content: "ok"},
		}})
		if err != nil {
			t.Fatal(err)
		}
		content := anthropicAssistantContent(t, fake)
		gotVerbatim := strings.Contains(fake.last().Body, `"SIG"`)
		if gotVerbatim != wantVerbatim {
			t.Fatalf("args %s: verbatim=%v, want %v (%v)", args, gotVerbatim, wantVerbatim, content)
		}
		if !wantVerbatim {
			var want map[string]any
			_ = json.Unmarshal([]byte(args), &want)
			if !reflect.DeepEqual(content[0].(map[string]any)["input"], want) {
				t.Fatalf("args %s: rebuilt turn must send the caller's arguments: %v", args, content)
			}
		}
	}
}

func TestAnthropicStreamUsageMergedFieldByField(t *testing.T) {
	// A message_delta that restates some counters (or sends null) keeps the
	// others from message_start.
	sse := strings.Join([]string{
		`data: {"type":"message_start","message":{"usage":{"input_tokens":40,"cache_read_input_tokens":100,"cache_creation_input_tokens":10,"output_tokens":1}}}`,
		``,
		`data: {"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"input_tokens":null,"cache_read_input_tokens":120,"output_tokens":25}}`,
		``,
	}, "\n")
	chat, _ := NewChat(Anthropic, AnthropicConfig{APIKey: "k"}, WithHTTPClient(newFake(sseResp(sse))))
	stream, _ := chat.Stream(context.Background(), ChatInput{Model: "m", Messages: []ChatMessage{{Role: "user", Content: "hi"}}})
	ds := drain(t, stream)
	if u := ds[len(ds)-1].Usage; u == nil || *u != (Usage{InputTokens: 170, OutputTokens: 25, TotalTokens: 195}) {
		t.Fatalf("usage: %+v", u)
	}
}

func TestAnthropicStreamMetadataOnlyOnStopDelta(t *testing.T) {
	sse := strings.Join([]string{
		`data: {"type":"content_block_start","index":0,"content_block":{"type":"tool_use","id":"toolu_1","name":"f","input":{"preset":true}}}`,
		``,
		`data: {"type":"content_block_stop","index":0}`,
		``,
		`data: {"type":"message_delta","delta":{},"usage":{"output_tokens":3}}`,
		``,
		`data: {"type":"message_delta","delta":{"stop_reason":"tool_use"},"usage":{"output_tokens":5}}`,
		``,
	}, "\n")
	chat, _ := NewChat(Anthropic, AnthropicConfig{APIKey: "k"}, WithHTTPClient(newFake(sseResp(sse))))
	stream, _ := chat.Stream(context.Background(), ChatInput{Model: "m", Messages: []ChatMessage{{Role: "user", Content: "hi"}}})
	var withMeta []*ChatStreamDelta
	for _, d := range drain(t, stream) {
		if d.ProviderMetadata != nil {
			withMeta = append(withMeta, d)
		}
	}
	if len(withMeta) != 1 || withMeta[0].FinishReason != "tool_calls" {
		t.Fatalf("metadata must ride only on the stop_reason delta: %+v", withMeta)
	}
	// A start-event input with no input_json_delta after it is kept, not reset to {}.
	blocks := withMeta[0].ProviderMetadata["anthropic"].(map[string]any)["content"].([]any)
	if !reflect.DeepEqual(blocks[0].(map[string]any)["input"], map[string]any{"preset": true}) {
		t.Fatalf("start-event input lost: %v", blocks[0])
	}
}

func TestAnthropicReview5ErrorRules(t *testing.T) {
	// "context window" alone is not an overflow phrase.
	body := `{"type":"error","error":{"type":"invalid_request_error","message":"thinking budget exceeds the context window of this model"}}`
	chat, _ := NewChat(Anthropic, AnthropicConfig{APIKey: "k"}, WithHTTPClient(newFake(resp(400, body))))
	_, err := chat.Complete(context.Background(), ChatInput{Model: "m", Messages: []ChatMessage{{Role: "user", Content: "x"}}})
	var ce *ConnectorError
	if !errors.As(err, &ce) || ce.ProviderCode != CodeInvalidRequest {
		t.Fatalf("context window: got %v", err)
	}
	// Mid-stream frames carry the stream's status and classify by type; an
	// undocumented type is unknown.
	for typ, want := range map[string]ProviderCode{"overloaded_error": CodeProviderUnavailable, "conflict_error": CodeInvalidRequest, "brand_new_error": CodeUnknown, "": CodeUnknown} {
		sse := `data: {"type":"error","error":{"type":"` + typ + `","message":"boom"}}` + "\n\n"
		chat, _ := NewChat(Anthropic, AnthropicConfig{APIKey: "k"}, WithHTTPClient(newFake(sseResp(sse))))
		stream, _ := chat.Stream(context.Background(), ChatInput{Model: "m", Messages: []ChatMessage{{Role: "user", Content: "hi"}}})
		_, err := stream.Recv()
		if !errors.As(err, &ce) || ce.ProviderCode != want || ce.StatusCode == nil || *ce.StatusCode != 200 {
			t.Fatalf("frame %q: got %v (status %v), want %s", typ, err, ce.StatusCode, want)
		}
	}
}
