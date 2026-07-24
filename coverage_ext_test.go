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
	// mx must exceed the largest reasoning budget (Gemini "high" = 24576) so the
	// reasoning combos below don't trip the max_output_tokens<=budget guard.
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
// splitting, content mapping, and tool-choice/effort helpers. A tolerant empty
// 200 body means parseResult succeeds without an error.
func TestNativeBuildRequestBranches(t *testing.T) {
	cfgs := []struct {
		id  ProviderID
		cfg ChatConfig
	}{
		{Anthropic, AnthropicConfig{APIKey: "k"}},
		{Bedrock, BedrockConfig{Region: "us-east-1", AccessKeyID: "a", SecretAccessKey: "s"}},
		{Gemini, GeminiConfig{APIKey: "k"}},
		{OpenAI, OpenAICompatConfig{APIKey: "k"}},
	}
	modes := []string{"auto", "none", "required", "function", ""}
	efforts := []string{"low", "medium", "high", ""}
	for _, c := range cfgs {
		for _, mode := range modes {
			for _, effort := range efforts {
				fake := newFake(resp(200, "{}"))
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

// TestNativeErrorMatrix exercises each connector's status→code classifier.
func TestNativeErrorMatrix(t *testing.T) {
	valid := map[ProviderCode]bool{
		CodeRateLimited: true, CodeAuthFailed: true, CodeProviderUnavailable: true,
		CodeInvalidRequest: true, CodeContextLengthExceeded: true, CodeContentFiltered: true, CodeUnknown: true,
	}
	cfgs := []struct {
		id  ProviderID
		cfg ChatConfig
	}{
		{Anthropic, AnthropicConfig{APIKey: "k"}},
		{Bedrock, BedrockConfig{Region: "us-east-1", AccessKeyID: "a", SecretAccessKey: "s"}},
		{Gemini, GeminiConfig{APIKey: "k"}},
		{OpenAI, OpenAICompatConfig{APIKey: "k"}},
	}
	statuses := []int{400, 401, 403, 404, 413, 422, 429, 500, 502, 503, 529, 418}
	body := `{"error":{"message":"maximum context length exceeded"}}`
	for _, c := range cfgs {
		for _, s := range statuses {
			fake := newFake(resp(s, body))
			chat, _ := NewChat(c.id, c.cfg, WithHTTPClient(fake))
			_, err := chat.Complete(context.Background(), ChatInput{Model: "m", Messages: []ChatMessage{{Role: "user", Content: "x"}}})
			var ce *ConnectorError
			if !errors.As(err, &ce) {
				t.Fatalf("%s status %d: expected *ConnectorError, got %v", c.id, s, err)
			}
			if !valid[ce.ProviderCode] {
				t.Fatalf("%s status %d: non-canonical code %q", c.id, s, ce.ProviderCode)
			}
		}
	}
}

func TestEmbeddingsErrorMatrix(t *testing.T) {
	valid := map[ProviderCode]bool{
		CodeRateLimited: true, CodeAuthFailed: true, CodeProviderUnavailable: true,
		CodeInvalidRequest: true, CodeUnknown: true,
	}
	for _, s := range []int{400, 401, 403, 404, 422, 429, 500, 418} {
		fake := newFake(resp(s, `{"error":{"message":"nope"}}`))
		emb, _ := NewEmbeddings(OpenAI, OpenAICompatConfig{APIKey: "k"}, WithHTTPClient(fake))
		_, err := emb.Create(context.Background(), EmbeddingsInput{Model: "m", Input: []string{"a"}})
		var ce *ConnectorError
		if !errors.As(err, &ce) || !valid[ce.ProviderCode] {
			t.Fatalf("embeddings status %d: %v", s, err)
		}
	}
}
