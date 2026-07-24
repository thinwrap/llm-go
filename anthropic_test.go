package llm

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"strings"
	"testing"
)

func TestAnthropicComplete(t *testing.T) {
	body := `{"model":"claude-3-5-sonnet-20241022","stop_reason":"end_turn","usage":{"input_tokens":10,"output_tokens":7},"content":[{"type":"text","text":"Hello "},{"type":"text","text":"world"}]}`
	fake := newFake(resp(200, body))
	chat, err := NewChat(Anthropic, AnthropicConfig{APIKey: "sk-ant"}, WithHTTPClient(fake))
	if err != nil {
		t.Fatal(err)
	}
	res, err := chat.Complete(context.Background(), ChatInput{
		Model: "claude-3-5-sonnet-20241022",
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
	if res.Model != "claude-3-5-sonnet-20241022" {
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
	if sent["model"] != "claude-3-5-sonnet-20241022" {
		t.Fatalf("bad model in body: %v", sent["model"])
	}
	if sent["system"] != "Be terse" {
		t.Fatalf("system not split to top-level: %v", sent["system"])
	}
	if sent["max_tokens"] != float64(4096) {
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

func TestAnthropicMaxTokensBelowReasoningBudget(t *testing.T) {
	// An explicit MaxOutputTokens below the reasoning budget must ERROR (silently
	// inflating it would spend past the caller's cap), not be rewritten.
	fake := newFake(resp(200, `{"model":"m","stop_reason":"end_turn","usage":{"input_tokens":1,"output_tokens":1},"content":[{"type":"text","text":"ok"}]}`))
	chat, _ := NewChat(Anthropic, AnthropicConfig{APIKey: "sk-ant"}, WithHTTPClient(fake))
	mx := 100
	_, err := chat.Complete(context.Background(), ChatInput{
		Model:           "m",
		Messages:        []ChatMessage{{Role: "user", Content: "hi"}},
		MaxOutputTokens: &mx,
		Reasoning:       &ReasoningOptions{Effort: "low"}, // budget 1024 > 100
	})
	var ce *ConnectorError
	if !errors.As(err, &ce) || ce.ProviderCode != CodeInvalidRequest {
		t.Fatalf("expected invalid_request when max_output_tokens < budget, got %v", err)
	}

	// With NO explicit cap, the default is inflated to keep the request valid.
	fake2 := newFake(resp(200, `{"model":"m","stop_reason":"end_turn","usage":{"input_tokens":1,"output_tokens":1},"content":[{"type":"text","text":"ok"}]}`))
	chat2, _ := NewChat(Anthropic, AnthropicConfig{APIKey: "sk-ant"}, WithHTTPClient(fake2))
	if _, err := chat2.Complete(context.Background(), ChatInput{
		Model:     "m",
		Messages:  []ChatMessage{{Role: "user", Content: "hi"}},
		Reasoning: &ReasoningOptions{Effort: "low"},
	}); err != nil {
		t.Fatalf("default (no explicit cap) should inflate, not error: %v", err)
	}
	var sent map[string]any
	_ = json.Unmarshal([]byte(fake2.last().Body), &sent)
	if sent["max_tokens"].(float64) <= 1024 {
		t.Fatalf("default max_tokens not inflated above budget: %v", sent["max_tokens"])
	}
}

func TestAnthropicToolUse(t *testing.T) {
	body := `{"model":"claude-3-5-sonnet","stop_reason":"tool_use","usage":{"input_tokens":20,"output_tokens":15},"content":[{"type":"text","text":"Let me check"},{"type":"tool_use","id":"toolu_1","name":"get_weather","input":{"city":"Paris"}}]}`
	fake := newFake(resp(200, body))
	chat, _ := NewChat(Anthropic, AnthropicConfig{APIKey: "k"}, WithHTTPClient(fake))
	res, err := chat.Complete(context.Background(), ChatInput{
		Model:    "claude-3-5-sonnet",
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
	var sent map[string]any
	_ = json.Unmarshal([]byte(fake.last().Body), &sent)
	thinking, ok := sent["thinking"].(map[string]any)
	if !ok || thinking["type"] != "enabled" || thinking["budget_tokens"] != float64(12288) {
		t.Fatalf("bad thinking: %v", sent["thinking"])
	}
	// default max_tokens (4096) <= budget (12288) → bumped to budget + 4096.
	if sent["max_tokens"] != float64(12288+4096) {
		t.Fatalf("bad max_tokens bump: %v", sent["max_tokens"])
	}
	if _, ok := sent["temperature"]; ok {
		t.Fatalf("temperature must be omitted when thinking is enabled")
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
