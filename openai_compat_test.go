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

func TestCompatMaxTokensField(t *testing.T) {
	// OpenAI deprecated max_tokens (o-series rejects it) and Azure reasoning
	// models accept only max_completion_tokens; other providers keep max_tokens.
	mx := 256
	cases := []struct {
		id        ProviderID
		cfg       OpenAICompatConfig
		want, not string
	}{
		{OpenAI, OpenAICompatConfig{APIKey: "k"}, "max_completion_tokens", "max_tokens"},
		{AzureOpenAI, OpenAICompatConfig{APIKey: "k", BaseURL: "https://r.openai.azure.com/openai/v1"}, "max_completion_tokens", "max_tokens"},
		{Groq, OpenAICompatConfig{APIKey: "k"}, "max_tokens", "max_completion_tokens"},
	}
	for _, c := range cases {
		fake := newFake(resp(200, `{"choices":[{"message":{"content":"ok"},"finish_reason":"stop"}]}`))
		chat, err := NewChat(c.id, c.cfg, WithHTTPClient(fake))
		if err != nil {
			t.Fatal(err)
		}
		if _, err := chat.Complete(context.Background(), ChatInput{Model: "m", Messages: []ChatMessage{{Role: "user", Content: "x"}}, MaxOutputTokens: &mx}); err != nil {
			t.Fatal(err)
		}
		var sent map[string]any
		_ = json.Unmarshal([]byte(fake.last().Body), &sent)
		if sent[c.want] != float64(256) {
			t.Fatalf("%s: expected %s=256, body=%s", c.id, c.want, fake.last().Body)
		}
		if _, ok := sent[c.not]; ok {
			t.Fatalf("%s: unexpected %s in body=%s", c.id, c.not, fake.last().Body)
		}
	}
}

func completeErr(t *testing.T, id ProviderID, status int, body string) *ConnectorError {
	t.Helper()
	fake := newFake(resp(status, body))
	chat, _ := NewChat(id, OpenAICompatConfig{APIKey: "k"}, WithHTTPClient(fake))
	res, err := chat.Complete(context.Background(), ChatInput{Model: "m", Messages: []ChatMessage{{Role: "user", Content: "x"}}})
	var ce *ConnectorError
	if !errors.As(err, &ce) {
		t.Fatalf("%s: expected *ConnectorError for %s, got res=%+v err=%v", id, body, res, err)
	}
	return ce
}

func TestCompat200WithTopLevelErrorRaises(t *testing.T) {
	// OpenRouter: an error after processing started can come back as HTTP 200
	// carrying only an error object and no choices. error.code (the upstream
	// HTTP status) classifies it; the error keeps the 200 it arrived with.
	ce := completeErr(t, OpenRouter, 200, `{"id":"gen-1","error":{"code":429,"message":"Rate limit exceeded upstream"}}`)
	if ce.ProviderCode != CodeRateLimited || ce.StatusCode == nil || *ce.StatusCode != 200 || ce.ProviderMessage != "Rate limit exceeded upstream" {
		t.Fatalf("bad error: code=%s status=%v msg=%q", ce.ProviderCode, ce.StatusCode, ce.ProviderMessage)
	}
	if raw, _ := ce.Cause.(map[string]any)["raw"].(map[string]any); raw["id"] != "gen-1" {
		t.Fatalf("raw body not on Cause: %+v", ce.Cause)
	}

	// A named code classifies by the chat rules (server_error → 500).
	ce = completeErr(t, OpenRouter, 200, `{"error":{"code":"server_error","message":"Provider disconnected"}}`)
	if ce.ProviderCode != CodeProviderUnavailable || ce.StatusCode == nil || *ce.StatusCode != 200 || ce.ProviderMessage != "Provider disconnected" {
		t.Fatalf("bad fallback error: code=%s status=%v msg=%q", ce.ProviderCode, ce.StatusCode, ce.ProviderMessage)
	}
	// No usable code (out of range, unknown name, absent) → unknown.
	for _, body := range []string{`{"error":{"code":200,"message":"odd"}}`, `{"error":{"code":"weird","message":"odd"}}`, `{"error":{"message":"odd"}}`} {
		ce = completeErr(t, OpenRouter, 200, body)
		if ce.ProviderCode != CodeUnknown || *ce.StatusCode != 200 {
			t.Fatalf("%s: code=%s status=%v", body, ce.ProviderCode, *ce.StatusCode)
		}
	}
}

func TestCompat200WithChoiceErrorRaises(t *testing.T) {
	// OpenRouter: choices[0] with finish_reason "error" and an error object
	// alongside partial content.
	ce := completeErr(t, OpenRouter, 200, `{"id":"gen-2","model":"openai/gpt-4o","choices":[{"index":0,"message":{"role":"assistant","content":"Partial answ"},"finish_reason":"error","error":{"code":502,"message":"Provider disconnected"}}]}`)
	if ce.ProviderCode != CodeProviderUnavailable || ce.StatusCode == nil || *ce.StatusCode != 200 || ce.ProviderMessage != "Provider disconnected" {
		t.Fatalf("bad error: code=%s status=%v msg=%q", ce.ProviderCode, ce.StatusCode, ce.ProviderMessage)
	}

	// finish_reason "error" with no error object still fails, as unknown.
	ce = completeErr(t, OpenRouter, 200, `{"choices":[{"message":{"content":"Part"},"finish_reason":"error"}]}`)
	if ce.ProviderCode != CodeUnknown || ce.ProviderMessage == "" || *ce.StatusCode != 200 {
		t.Fatalf("bad bare finish_reason error: code=%s msg=%q", ce.ProviderCode, ce.ProviderMessage)
	}

	// A choice error without finish_reason "error" is still an error; a 4xx code
	// runs through the normal classifier.
	ce = completeErr(t, OpenRouter, 200, `{"choices":[{"message":{"content":""},"finish_reason":"stop","error":{"code":400,"message":"This model's maximum context length is 8192 tokens"}}]}`)
	if ce.ProviderCode != CodeContextLengthExceeded || *ce.StatusCode != 200 {
		t.Fatalf("bad classified choice error: code=%s status=%v", ce.ProviderCode, *ce.StatusCode)
	}
}

func TestCompat200WithoutChoicesRaises(t *testing.T) {
	for _, body := range []string{`{"id":"x","model":"m"}`, `{"choices":[]}`, `{"choices":"nope"}`} {
		ce := completeErr(t, OpenAI, 200, body)
		if ce.ProviderCode != CodeProviderUnavailable || ce.StatusCode == nil || *ce.StatusCode != 200 {
			t.Fatalf("%s: bad error: code=%s status=%v", body, ce.ProviderCode, ce.StatusCode)
		}
	}
}

func TestCompatMistralReasoningChunkContent(t *testing.T) {
	// Mistral reasoning: message.content is a list of ThinkChunk + TextChunk.
	body := `{"model":"magistral-medium-latest","choices":[{"index":0,"message":{"role":"assistant","content":[` +
		`{"type":"thinking","thinking":[{"type":"text","text":"Let me work this out."}]},` +
		`{"type":"text","text":"The answer "},{"type":"text","text":"is 4."}]},"finish_reason":"stop"}],` +
		`"usage":{"prompt_tokens":10,"completion_tokens":20,"total_tokens":30}}`
	fake := newFake(resp(200, body))
	chat, _ := NewChat(Mistral, OpenAICompatConfig{APIKey: "k"}, WithHTTPClient(fake))
	res, err := chat.Complete(context.Background(), ChatInput{Model: "magistral-medium-latest", Messages: []ChatMessage{{Role: "user", Content: "2+2?"}}, Reasoning: &ReasoningOptions{Effort: "high"}})
	if err != nil {
		t.Fatal(err)
	}
	if res.Message.Content != "The answer is 4." || res.FinishReason != "stop" {
		t.Fatalf("bad content: %q finish=%q", res.Message.Content, res.FinishReason)
	}
}

func TestCompatMistralReasoningChunkStream(t *testing.T) {
	// Streaming: a ThinkChunk list while thinking, then closing ThinkChunk + first
	// TextChunk at the transition, then plain strings.
	sse := strings.Join([]string{
		`data: {"choices":[{"index":0,"delta":{"role":"assistant","content":[{"type":"thinking","thinking":[{"type":"text","text":"Adding 2 and 2"}]}]}}]}`,
		"",
		`data: {"choices":[{"index":0,"delta":{"content":[{"type":"thinking","thinking":[{"type":"text","text":"."}]},{"type":"text","text":"The answer"}]}}]}`,
		"",
		`data: {"choices":[{"index":0,"delta":{"content":" is 4."}}]}`,
		"",
		`data: {"choices":[{"index":0,"delta":{"content":""},"finish_reason":"stop"}]}`,
		"",
		"data: [DONE]",
		"",
	}, "\n")
	fake := newFake(sseResp(sse))
	chat, _ := NewChat(Mistral, OpenAICompatConfig{APIKey: "k"}, WithHTTPClient(fake))
	stream, err := chat.Stream(context.Background(), ChatInput{Model: "magistral-medium-latest", Messages: []ChatMessage{{Role: "user", Content: "2+2?"}}})
	if err != nil {
		t.Fatal(err)
	}
	defer stream.Close()
	var text string
	for {
		d, err := stream.Recv()
		if err == io.EOF {
			break
		}
		if err != nil {
			t.Fatal(err)
		}
		text += d.ContentDelta
	}
	if text != "The answer is 4." {
		t.Fatalf("stream text=%q", text)
	}
}

func embeddingsErr(t *testing.T, input []string, body string) *ConnectorError {
	t.Helper()
	fake := newFake(resp(200, body))
	emb, _ := NewEmbeddings(OpenAI, OpenAICompatConfig{APIKey: "k"}, WithHTTPClient(fake))
	res, err := emb.Create(context.Background(), EmbeddingsInput{Model: "text-embedding-3-small", Input: input})
	var ce *ConnectorError
	if !errors.As(err, &ce) {
		t.Fatalf("expected *ConnectorError for %s, got res=%+v err=%v", body, res, err)
	}
	return ce
}

func TestEmbeddingsMalformed2xxBodiesRaise(t *testing.T) {
	two := []string{"a", "b"}
	// Count mismatch: one vector for two inputs.
	ce := embeddingsErr(t, two, `{"data":[{"index":0,"embedding":[0.1,0.2]}],"usage":{"prompt_tokens":2,"total_tokens":2}}`)
	if ce.ProviderCode != CodeProviderUnavailable || *ce.StatusCode != 200 {
		t.Fatalf("count mismatch: code=%s", ce.ProviderCode)
	}
	if _, ok := ce.Cause.(map[string]any)["raw"]; !ok {
		t.Fatalf("raw body not on Cause: %+v", ce.Cause)
	}
	// A null cell must not become 0.
	ce = embeddingsErr(t, two, `{"data":[{"index":0,"embedding":[0.1,null]},{"index":1,"embedding":[0.3,0.4]}]}`)
	if ce.ProviderCode != CodeProviderUnavailable {
		t.Fatalf("null cell: code=%s", ce.ProviderCode)
	}
	// A string cell / base64 row is not a float vector.
	ce = embeddingsErr(t, []string{"a"}, `{"data":[{"index":0,"embedding":["0.1"]}]}`)
	if ce.ProviderCode != CodeProviderUnavailable {
		t.Fatalf("string cell: code=%s", ce.ProviderCode)
	}
	ce = embeddingsErr(t, []string{"a"}, `{"data":[{"index":0,"embedding":"AAAAAA=="}]}`)
	if ce.ProviderCode != CodeProviderUnavailable {
		t.Fatalf("base64 row: code=%s", ce.ProviderCode)
	}
	// data missing / not an array.
	for _, body := range []string{`{"model":"m"}`, `{"data":{}}`} {
		ce = embeddingsErr(t, []string{"a"}, body)
		if ce.ProviderCode != CodeProviderUnavailable || *ce.StatusCode != 200 {
			t.Fatalf("%s: code=%s", body, ce.ProviderCode)
		}
	}
	// Top-level error on a 200 maps through the embeddings classifier.
	ce = embeddingsErr(t, []string{"a"}, `{"error":{"code":401,"message":"No auth credentials found"}}`)
	if ce.ProviderCode != CodeAuthFailed || *ce.StatusCode != 200 || ce.ProviderMessage != "No auth credentials found" {
		t.Fatalf("top-level error: code=%s status=%d msg=%q", ce.ProviderCode, *ce.StatusCode, ce.ProviderMessage)
	}
}

func TestEmbeddingsDimensionsField(t *testing.T) {
	// Mistral names the size output_dimension; OpenAI-style providers use dimensions.
	dims := 256
	for id, want := range map[ProviderID]string{Mistral: "output_dimension", OpenAI: "dimensions"} {
		fake := newFake(resp(200, `{"data":[{"index":0,"embedding":[0.1,0.2]}],"usage":{"prompt_tokens":1,"total_tokens":1}}`))
		emb, _ := NewEmbeddings(id, OpenAICompatConfig{APIKey: "k"}, WithHTTPClient(fake))
		if _, err := emb.Create(context.Background(), EmbeddingsInput{Model: "codestral-embed", Input: []string{"a"}, Dimensions: &dims}); err != nil {
			t.Fatal(err)
		}
		sent := sentBody(t, fake)
		if sent[want] != float64(256) {
			t.Fatalf("%s: expected %s=256, body=%s", id, want, fake.last().Body)
		}
		other := map[string]string{"output_dimension": "dimensions", "dimensions": "output_dimension"}[want]
		if _, ok := sent[other]; ok {
			t.Fatalf("%s: unexpected %s, body=%s", id, other, fake.last().Body)
		}
	}
}

func TestEmbeddingsPassthroughQuery(t *testing.T) {
	fake := newFake(resp(200, `{"data":[{"index":0,"embedding":[0.1]}]}`))
	emb, _ := NewEmbeddings(AzureOpenAI, OpenAICompatConfig{APIKey: "k", BaseURL: "https://r.openai.azure.com/openai/v1"}, WithHTTPClient(fake))
	if _, err := emb.Create(context.Background(), EmbeddingsInput{Model: "m", Input: []string{"a"}, Passthrough: &Passthrough{Query: map[string]string{"api-version": "preview"}}}); err != nil {
		t.Fatal(err)
	}
	if got := fake.last().URL.String(); got != "https://r.openai.azure.com/openai/v1/embeddings?api-version=preview" {
		t.Fatalf("passthrough query not appended: %s", got)
	}
}

func TestCompatEmptyReasoningEffortOmitted(t *testing.T) {
	for effort, wantSent := range map[string]bool{"": false, "high": true} {
		fake := newFake(resp(200, `{"choices":[{"message":{"content":"ok"},"finish_reason":"stop"}]}`))
		chat, _ := NewChat(OpenAI, OpenAICompatConfig{APIKey: "k"}, WithHTTPClient(fake))
		if _, err := chat.Complete(context.Background(), ChatInput{Model: "m", Messages: []ChatMessage{{Role: "user", Content: "x"}}, Reasoning: &ReasoningOptions{Effort: effort}}); err != nil {
			t.Fatal(err)
		}
		v, ok := sentBody(t, fake)["reasoning_effort"]
		if ok != wantSent || (ok && v != effort) {
			t.Fatalf("effort %q: reasoning_effort=%v present=%v", effort, v, ok)
		}
	}
}

func TestCompatMidStreamErrorClassified(t *testing.T) {
	cases := []struct {
		frame string
		want  ProviderCode
	}{
		// OpenRouter puts the upstream HTTP status in error.code.
		{`{"id":"gen-1","error":{"code":429,"message":"Rate limited upstream"},"choices":[{"index":0,"delta":{"content":""},"finish_reason":"error"}]}`, CodeRateLimited},
		{`{"id":"gen-1","error":{"code":"server_error","message":"Provider disconnected"},"choices":[{"index":0,"delta":{"content":""},"finish_reason":"error"}]}`, CodeProviderUnavailable},
		{`{"id":"gen-1","choices":[{"index":0,"delta":{"content":""},"finish_reason":"error"}]}`, CodeUnknown},
	}
	for _, c := range cases {
		sse := `data: {"choices":[{"index":0,"delta":{"content":"Part"}}]}` + "\n\n" + "data: " + c.frame + "\n\n"
		chat, _ := NewChat(OpenRouter, OpenAICompatConfig{APIKey: "k"}, WithHTTPClient(newFake(sseResp(sse))))
		stream, err := chat.Stream(context.Background(), ChatInput{Model: "m", Messages: []ChatMessage{{Role: "user", Content: "x"}}})
		if err != nil {
			t.Fatal(err)
		}
		if d, err := stream.Recv(); err != nil || d.ContentDelta != "Part" {
			t.Fatalf("first delta: %+v %v", d, err)
		}
		_, err = stream.Recv()
		var ce *ConnectorError
		if !errors.As(err, &ce) || ce.ProviderCode != c.want || ce.StatusCode == nil || *ce.StatusCode != 200 {
			t.Fatalf("%s: got %v, want %s", c.frame, err, c.want)
		}
		stream.Close()
	}
}

func TestCompat402IsRateLimited(t *testing.T) {
	ce := completeErr(t, OpenRouter, 402, `{"error":{"code":402,"message":"Insufficient credits"}}`)
	if ce.ProviderCode != CodeRateLimited {
		t.Fatalf("402: %s", ce.ProviderCode)
	}
	fake := newFake(resp(413, `{"error":{"message":"too many inputs"}}`))
	emb, _ := NewEmbeddings(OpenAI, OpenAICompatConfig{APIKey: "k"}, WithHTTPClient(fake))
	_, err := emb.Create(context.Background(), EmbeddingsInput{Model: "m", Input: []string{"a"}})
	if !errors.As(err, &ce) || ce.ProviderCode != CodeInvalidRequest {
		t.Fatalf("embeddings 413: %v", err)
	}
}

func TestCompatReview5ErrorRules(t *testing.T) {
	for _, c := range []struct {
		status int
		body   string
		want   ProviderCode
	}{
		{400, `{"error":{"code":"context_length_exceeded","message":"Too many tokens in the prompt."}}`, CodeContextLengthExceeded},
		{400, `{"error":{"message":"Your request was rejected as a result of our safety system: content_policy_violation"}}`, CodeContentFiltered},
		{400, `{"error":{"message":"Input flagged by moderation."}}`, CodeContentFiltered},
		{400, `{"error":{"message":"Blocked by the content filter."}}`, CodeContentFiltered},
		{408, `{"error":{"message":"timeout"}}`, CodeProviderUnavailable},
		{409, `{"error":{"message":"conflict"}}`, CodeInvalidRequest},
	} {
		if ce := completeErr(t, OpenAI, c.status, c.body); ce.ProviderCode != c.want {
			t.Fatalf("%d %s: got %s, want %s", c.status, c.body, ce.ProviderCode, c.want)
		}
	}
}

func TestEmbeddingsNamedInBodyErrorCode(t *testing.T) {
	// A named code classifies like chat (insufficient_quota → rate_limited) and
	// the error keeps the status it arrived with.
	ce := embeddingsErr(t, []string{"a"}, `{"error":{"code":"insufficient_quota","type":"insufficient_quota","message":"You exceeded your current quota."}}`)
	if ce.ProviderCode != CodeRateLimited || *ce.StatusCode != 200 {
		t.Fatalf("named code: %s status %d", ce.ProviderCode, *ce.StatusCode)
	}
	ce = completeErr(t, OpenRouter, 200, `{"error":{"code":"invalid_api_key","message":"bad key"}}`)
	if ce.ProviderCode != CodeAuthFailed || *ce.StatusCode != 200 {
		t.Fatalf("chat named code: %s status %d", ce.ProviderCode, *ce.StatusCode)
	}
	ce = completeErr(t, OpenRouter, 200, `{"error":{"code":"503","message":"busy"}}`)
	if ce.ProviderCode != CodeProviderUnavailable {
		t.Fatalf("numeric-string code: %s", ce.ProviderCode)
	}
}
