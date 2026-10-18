package llm

import (
	"context"
	"errors"
	"testing"
	"time"
)

func TestConfigMarkersAndConstructors(t *testing.T) {
	OpenAICompatConfig{}.isChatConfig()
	AnthropicConfig{}.isChatConfig()
	BedrockConfig{}.isChatConfig()
	GeminiConfig{}.isChatConfig()

	_ = TextPart("hi")
	_ = ImagePart("aGk=", "image/png")

	chat, _ := NewChat(OpenAI, OpenAICompatConfig{APIKey: "k"})
	if chat.ProviderID() != OpenAI {
		t.Fatal("chat provider id")
	}
	emb, _ := NewEmbeddings(OpenAI, OpenAICompatConfig{APIKey: "k"})
	if emb.ProviderID() != OpenAI {
		t.Fatal("emb provider id")
	}
}

func TestErrorHelpers(t *testing.T) {
	if (&ConnectorError{Message: "m"}).Error() != "m" {
		t.Fatal("Error message")
	}
	if (&ConnectorError{ProviderMessage: "pm"}).Error() != "pm" {
		t.Fatal("Error providerMessage fallback")
	}
	if (&ConnectorError{}).Error() != "Connector error" {
		t.Fatal("Error default")
	}
	sentinel := errors.New("boom")
	ce := transportError(sentinel)
	if !errors.Is(ce, sentinel) || ce.ProviderCode != CodeProviderUnavailable || ce.StatusCode != nil {
		t.Fatalf("transportError shape: %+v", ce)
	}
	if n, ok := parseRetryAfter("30"); !ok || n != 30 {
		t.Fatal("parseRetryAfter numeric")
	}
	if _, ok := parseRetryAfter("garbage"); ok {
		t.Fatal("parseRetryAfter garbage")
	}
	future := time.Now().Add(20 * time.Second).UTC().Format("Mon, 02 Jan 2006 15:04:05 GMT")
	if n, ok := parseRetryAfter(future); !ok || n < 1 || n > 21 {
		t.Fatalf("parseRetryAfter date: %d %v", n, ok)
	}
}

func TestSigV4QueryHelpers(t *testing.T) {
	if q := buildCanonicalQuery(map[string]string{"b": "2", "a": "1 x"}); q != "a=1%20x&b=2" {
		t.Fatalf("canonical query: %q", q)
	}
	if buildCanonicalQuery(nil) != "" {
		t.Fatal("empty query")
	}
	if redactCredentials("x?api_key=SECRET") == "x?api_key=SECRET" {
		t.Fatal("redactCredentials should mask")
	}
}

func kitchenSink(mode, effort string) ChatInput {
	temp, tp, mx := 0.5, 0.9, 32000
	in := ChatInput{
		Model: "m",
		Messages: []ChatMessage{
			{Role: "system", Content: "be brief"},
			{Role: "user", ContentParts: []ContentPart{TextPart("hello"), ImagePart("aGk=", "image/png")}},
			{Role: "assistant", ToolCalls: []ToolCall{{ID: "t1", Name: "f", Arguments: `{"x":1}`}}},
			{Role: "tool", ToolCallID: "t1", Content: "result", Name: "f"},
			{Role: "user", Content: "thanks"},
		},
		Tools:       []ToolDefinition{{Name: "f", Description: "d", Parameters: map[string]any{"type": "object"}}},
		Temperature: &temp, TopP: &tp, MaxOutputTokens: &mx,
		Stop:        []string{"STOP"},
		Passthrough: &Passthrough{Body: map[string]any{"x": 1}, Headers: map[string]string{"H": "1"}},
	}
	if mode != "" {
		in.ToolChoice = &ToolChoice{Mode: mode, FunctionName: "f"}
	}
	if effort != "" {
		in.Reasoning = &ReasoningOptions{Effort: effort}
	}
	return in
}

// TestNativeBuildRequestBranches drives each native's Complete over the
// tool-choice modes and reasoning efforts to exercise buildRequest, message
// splitting, content mapping, and tool-choice/effort helpers. Each connector
// gets the smallest body its wire contract calls a completion.
func TestNativeBuildRequestBranches(t *testing.T) {
	cfgs := []struct {
		id   ProviderID
		cfg  ChatConfig
		body string
	}{
		{Anthropic, AnthropicConfig{APIKey: "k"}, `{"content":[]}`},
		{Bedrock, BedrockConfig{Region: "us-east-1", AccessKeyID: "a", SecretAccessKey: "s"}, `{"output":{"message":{"role":"assistant","content":[]}}}`},
		{Gemini, GeminiConfig{APIKey: "k"}, `{"candidates":[]}`},
		{OpenAI, OpenAICompatConfig{APIKey: "k"}, `{"choices":[{"message":{"content":""}}]}`},
	}
	modes := []string{"auto", "none", "required", "function", ""}
	efforts := []string{"low", "medium", "high", ""}
	for _, c := range cfgs {
		for _, mode := range modes {
			for _, effort := range efforts {
				fake := newFake(resp(200, c.body))
				chat, err := NewChat(c.id, c.cfg, WithHTTPClient(fake))
				if err != nil {
					t.Fatalf("%s NewChat: %v", c.id, err)
				}
				if _, err := chat.Complete(context.Background(), kitchenSink(mode, effort)); err != nil {
					t.Fatalf("%s complete(mode=%q,effort=%q): %v", c.id, mode, effort, err)
				}
			}
		}
	}
}

// TestEmptyObjectOn2xxErrors: a 2xx `{}` decodes as JSON but carries none of the
// payload a completion requires, so every connector must fail rather than
// return an empty "successful" result.
func TestEmptyObjectOn2xxErrors(t *testing.T) {
	cfgs := []struct {
		id  ProviderID
		cfg ChatConfig
	}{
		{Anthropic, AnthropicConfig{APIKey: "k"}},
		{Bedrock, BedrockConfig{Region: "us-east-1", AccessKeyID: "a", SecretAccessKey: "s"}},
		{Gemini, GeminiConfig{APIKey: "k"}},
		{OpenAI, OpenAICompatConfig{APIKey: "k"}},
		{OpenRouter, OpenAICompatConfig{APIKey: "k"}},
		{Mistral, OpenAICompatConfig{APIKey: "k"}},
	}
	for _, c := range cfgs {
		fake := newFake(resp(200, "{}"))
		chat, _ := NewChat(c.id, c.cfg, WithHTTPClient(fake))
		_, err := chat.Complete(context.Background(), ChatInput{Model: "m", Messages: []ChatMessage{{Role: "user", Content: "x"}}})
		var ce *ConnectorError
		if !errors.As(err, &ce) || ce.ProviderCode != CodeProviderUnavailable || ce.StatusCode == nil || *ce.StatusCode != 200 {
			t.Fatalf("%s: expected provider_unavailable for 200 {}, got %v", c.id, err)
		}
	}
	fake := newFake(resp(200, "{}"))
	emb, _ := NewEmbeddings(OpenAI, OpenAICompatConfig{APIKey: "k"}, WithHTTPClient(fake))
	_, err := emb.Create(context.Background(), EmbeddingsInput{Model: "m", Input: []string{"a"}})
	var ce *ConnectorError
	if !errors.As(err, &ce) || ce.ProviderCode != CodeProviderUnavailable {
		t.Fatalf("embeddings: expected provider_unavailable for 200 {}, got %v", err)
	}
}

// TestNativeErrorMatrix pins each connector's status→code classification for a
// body whose message names no specific condition.
func TestNativeErrorMatrix(t *testing.T) {
	const (
		inv  = CodeInvalidRequest
		auth = CodeAuthFailed
		rate = CodeRateLimited
		prov = CodeProviderUnavailable
		unk  = CodeUnknown
	)
	cfgs := []struct {
		id   ProviderID
		cfg  ChatConfig
		want map[int]ProviderCode
	}{
		{Anthropic, AnthropicConfig{APIKey: "k"}, map[int]ProviderCode{402: rate, 408: prov, 424: inv}},
		{Bedrock, BedrockConfig{Region: "us-east-1", AccessKeyID: "a", SecretAccessKey: "s"}, map[int]ProviderCode{402: rate, 408: prov, 424: prov}},
		{Gemini, GeminiConfig{APIKey: "k"}, map[int]ProviderCode{402: rate, 408: prov, 424: inv}},
		{OpenAI, OpenAICompatConfig{APIKey: "k"}, map[int]ProviderCode{402: rate, 408: prov, 424: inv}},
	}
	common := map[int]ProviderCode{400: inv, 401: auth, 403: auth, 404: inv, 413: inv, 422: inv, 429: rate,
		500: prov, 502: prov, 503: prov, 504: prov, 529: prov, 418: inv, 409: inv, 451: inv, 302: unk}
	body := `{"error":{"message":"request rejected"}}`
	for _, c := range cfgs {
		for s, want := range common {
			if _, ok := c.want[s]; !ok {
				c.want[s] = want
			}
		}
		for s, want := range c.want {
			fake := newFake(resp(s, body))
			chat, _ := NewChat(c.id, c.cfg, WithHTTPClient(fake))
			_, err := chat.Complete(context.Background(), ChatInput{Model: "m", Messages: []ChatMessage{{Role: "user", Content: "x"}}})
			var ce *ConnectorError
			if !errors.As(err, &ce) || ce.ProviderCode != want || ce.StatusCode == nil || *ce.StatusCode != s {
				t.Fatalf("%s status %d: got %v, want %s", c.id, s, err, want)
			}
		}
	}
}

func TestEmbeddingsErrorMatrix(t *testing.T) {
	valid := map[ProviderCode]bool{
		CodeRateLimited: true, CodeAuthFailed: true, CodeProviderUnavailable: true,
		CodeInvalidRequest: true, CodeUnknown: true,
	}
	want := map[int]ProviderCode{400: CodeInvalidRequest, 401: CodeAuthFailed, 402: CodeRateLimited, 403: CodeAuthFailed,
		404: CodeInvalidRequest, 413: CodeInvalidRequest, 422: CodeInvalidRequest, 429: CodeRateLimited,
		500: CodeProviderUnavailable, 418: CodeInvalidRequest, 408: CodeProviderUnavailable, 302: CodeUnknown}
	for s, code := range want {
		fake := newFake(resp(s, `{"error":{"message":"nope"}}`))
		emb, _ := NewEmbeddings(OpenAI, OpenAICompatConfig{APIKey: "k"}, WithHTTPClient(fake))
		_, err := emb.Create(context.Background(), EmbeddingsInput{Model: "m", Input: []string{"a"}})
		var ce *ConnectorError
		if !errors.As(err, &ce) || !valid[ce.ProviderCode] || ce.ProviderCode != code {
			t.Fatalf("embeddings status %d: got %v, want %s", s, err, code)
		}
	}
}
