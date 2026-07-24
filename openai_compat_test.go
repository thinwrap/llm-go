package llm

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"strings"
	"testing"
)

func TestCompatComplete(t *testing.T) {
	body := `{"model":"gpt-4o-mini","usage":{"prompt_tokens":9,"completion_tokens":12,"total_tokens":21},"choices":[{"message":{"content":"Hi there"},"finish_reason":"stop"}]}`
	fake := newFake(resp(200, body))
	chat, err := NewChat(OpenAI, OpenAICompatConfig{APIKey: "sk-x"}, WithHTTPClient(fake))
	if err != nil {
		t.Fatal(err)
	}
	res, err := chat.Complete(context.Background(), ChatInput{
		Model:    "gpt-4o-mini",
		Messages: []ChatMessage{{Role: "user", Content: "Hello"}},
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
	req := fake.last()
	if req.URL.String() != "https://api.openai.com/v1/chat/completions" {
		t.Fatalf("bad url: %s", req.URL)
	}
	if req.Header.Get("Authorization") != "Bearer sk-x" {
		t.Fatalf("bad auth: %s", req.Header.Get("Authorization"))
	}
	var sent map[string]any
	_ = json.Unmarshal([]byte(req.Body), &sent)
	if sent["model"] != "gpt-4o-mini" {
		t.Fatalf("bad body: %s", req.Body)
	}
}

func TestCompatBaseURLRequired(t *testing.T) {
	_, err := NewChat(AzureOpenAI, OpenAICompatConfig{APIKey: "k"})
	var ce *ConnectorError
	if !errors.As(err, &ce) || ce.ProviderCode != CodeInvalidRequest {
		t.Fatalf("expected invalid_request for missing baseUrl, got %v", err)
	}
}

func TestCompatStream(t *testing.T) {
	sse := strings.Join([]string{
		`data: {"choices":[{"delta":{"content":"Hel"}}]}`,
		"",
		`data: {"choices":[{"delta":{"content":"lo"}}]}`,
		"",
		`data: {"choices":[{"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":3,"completion_tokens":2,"total_tokens":5}}`,
		"",
		"data: [DONE]",
		"",
	}, "\n")
	fake := newFake(sseResp(sse))
	chat, _ := NewChat(Groq, OpenAICompatConfig{APIKey: "k"}, WithHTTPClient(fake))
	stream, err := chat.Stream(context.Background(), ChatInput{Model: "llama", Messages: []ChatMessage{{Role: "user", Content: "hi"}}})
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
	if text != "Hello" || finish != "stop" {
		t.Fatalf("stream text=%q finish=%q", text, finish)
	}
	if usage == nil || usage.TotalTokens != 5 {
		t.Fatalf("bad stream usage: %+v", usage)
	}
	// stream requested with stream:true
	var sent map[string]any
	_ = json.Unmarshal([]byte(fake.last().Body), &sent)
	if sent["stream"] != true {
		t.Fatalf("expected stream:true in body")
	}
}

func TestCompatStreamMidStreamError(t *testing.T) {
	sse := strings.Join([]string{
		`data: {"choices":[{"delta":{"content":"Par"}}]}`,
		"",
		`data: {"error":{"message":"upstream exploded","type":"server_error"}}`,
		"",
	}, "\n")
	fake := newFake(sseResp(sse))
	chat, _ := NewChat(OpenAI, OpenAICompatConfig{APIKey: "k"}, WithHTTPClient(fake))
	stream, err := chat.Stream(context.Background(), ChatInput{Model: "m", Messages: []ChatMessage{{Role: "user", Content: "hi"}}})
	if err != nil {
		t.Fatal(err)
	}
	defer stream.Close()

	d, err := stream.Recv()
	if err != nil || d.ContentDelta != "Par" {
		t.Fatalf("first delta should arrive: d=%+v err=%v", d, err)
	}
	_, err = stream.Recv()
	if err == io.EOF {
		t.Fatal("mid-stream error frame was swallowed as io.EOF")
	}
	var ce *ConnectorError
	if !errors.As(err, &ce) {
		t.Fatalf("expected *ConnectorError, got %v", err)
	}
	if ce.ProviderCode != CodeProviderUnavailable {
		t.Fatalf("bad code: %s", ce.ProviderCode)
	}
	if ce.ProviderMessage != "upstream exploded" {
		t.Fatalf("bad message: %s", ce.ProviderMessage)
	}
}

func TestCompatStreamMultipleToolCalls(t *testing.T) {
	sse := strings.Join([]string{
		`data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"c0","function":{"name":"a","arguments":"{}"}},{"index":1,"id":"c1","function":{"name":"b","arguments":"{}"}}]}}]}`,
		"",
		`data: {"choices":[{"delta":{},"finish_reason":"tool_calls"}]}`,
		"",
		"data: [DONE]",
		"",
	}, "\n")
	fake := newFake(sseResp(sse))
	chat, _ := NewChat(OpenAI, OpenAICompatConfig{APIKey: "k"}, WithHTTPClient(fake))
	stream, err := chat.Stream(context.Background(), ChatInput{Model: "m", Messages: []ChatMessage{{Role: "user", Content: "hi"}}})
	if err != nil {
		t.Fatal(err)
	}
	defer stream.Close()

	var deltas []*ToolCallDelta
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
			deltas = append(deltas, d.ToolCallDelta)
		}
		if d.FinishReason != "" {
			finish = d.FinishReason
		}
	}
	if len(deltas) != 2 {
		t.Fatalf("expected one delta per tool_call entry (2), got %d", len(deltas))
	}
	if deltas[0].Index != 0 || deltas[0].ID != "c0" || deltas[0].FunctionName != "a" {
		t.Fatalf("bad first tool delta: %+v", deltas[0])
	}
	if deltas[1].Index != 1 || deltas[1].ID != "c1" || deltas[1].FunctionName != "b" {
		t.Fatalf("bad second tool delta: %+v", deltas[1])
	}
	if finish != "tool_calls" {
		t.Fatalf("bad finish: %s", finish)
	}
}

func TestCompatErrorMapping(t *testing.T) {
	fake := newFake(resp(429, `{"error":{"message":"slow down"}}`))
	chat, _ := NewChat(OpenAI, OpenAICompatConfig{APIKey: "k"}, WithHTTPClient(fake))
	_, err := chat.Complete(context.Background(), ChatInput{Model: "m", Messages: []ChatMessage{{Role: "user", Content: "x"}}})
	var ce *ConnectorError
	if !errors.As(err, &ce) || ce.ProviderCode != CodeRateLimited {
		t.Fatalf("expected rate_limited, got %v", err)
	}
	if ce.ProviderMessage != "slow down" {
		t.Fatalf("bad message: %s", ce.ProviderMessage)
	}
}

func TestCompatEmptyBodyOn2xxErrors(t *testing.T) {
	// A 2xx with an empty/non-JSON body (proxy/captive portal, truncated read)
	// must raise, not return an empty "successful" completion.
	fake := newFake(resp(200, ""))
	chat, _ := NewChat(OpenAI, OpenAICompatConfig{APIKey: "k"}, WithHTTPClient(fake))
	_, err := chat.Complete(context.Background(), ChatInput{Model: "m", Messages: []ChatMessage{{Role: "user", Content: "x"}}})
	var ce *ConnectorError
	if !errors.As(err, &ce) || ce.ProviderCode != CodeProviderUnavailable {
		t.Fatalf("expected provider_unavailable for empty 2xx body, got %v", err)
	}
}

func TestCompatContextLengthError(t *testing.T) {
	fake := newFake(resp(400, `{"error":{"message":"This model's maximum context length is 8192 tokens"}}`))
	chat, _ := NewChat(OpenAI, OpenAICompatConfig{APIKey: "k"}, WithHTTPClient(fake))
	_, err := chat.Complete(context.Background(), ChatInput{Model: "m", Messages: []ChatMessage{{Role: "user", Content: "x"}}})
	var ce *ConnectorError
	if !errors.As(err, &ce) || ce.ProviderCode != CodeContextLengthExceeded {
		t.Fatalf("expected context_length_exceeded, got %v", err)
	}
}

func TestCompat413MapsToInvalidRequest(t *testing.T) {
	// HTTP 413 (Payload Too Large) must not fall through to Unknown.
	fake := newFake(resp(413, `{"error":{"message":"request entity too large"}}`))
	chat, _ := NewChat(OpenAI, OpenAICompatConfig{APIKey: "k"}, WithHTTPClient(fake))
	_, err := chat.Complete(context.Background(), ChatInput{Model: "m", Messages: []ChatMessage{{Role: "user", Content: "x"}}})
	var ce *ConnectorError
	if !errors.As(err, &ce) || ce.ProviderCode != CodeInvalidRequest {
		t.Fatalf("expected invalid_request for 413, got %v", err)
	}
}

func TestEmbeddings(t *testing.T) {
	body := `{"model":"text-embedding-3-small","data":[{"index":1,"embedding":[0.3,0.4]},{"index":0,"embedding":[0.1,0.2]}],"usage":{"prompt_tokens":5,"total_tokens":5}}`
	fake := newFake(resp(200, body))
	emb, err := NewEmbeddings(OpenAI, OpenAICompatConfig{APIKey: "k"}, WithHTTPClient(fake))
	if err != nil {
		t.Fatal(err)
	}
	res, err := emb.Create(context.Background(), EmbeddingsInput{Model: "text-embedding-3-small", Input: []string{"a", "b"}})
	if err != nil {
		t.Fatal(err)
	}
	// sorted by index → input order
	if len(res.Embeddings) != 2 || res.Embeddings[0][0] != 0.1 || res.Embeddings[1][0] != 0.3 {
		t.Fatalf("embeddings not in input order: %+v", res.Embeddings)
	}
	if res.Usage.InputTokens != 5 {
		t.Fatalf("bad usage: %+v", res.Usage)
	}
}

func TestEmbeddingsUnsupportedProvider(t *testing.T) {
	_, err := NewEmbeddings(DeepSeek, OpenAICompatConfig{APIKey: "k"})
	var ce *ConnectorError
	if !errors.As(err, &ce) || ce.ProviderCode != CodeInvalidRequest {
		t.Fatalf("expected invalid_request, got %v", err)
	}
}
