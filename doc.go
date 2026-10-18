// Package llm is a lightweight, SDK-free, zero-dependency Go wrapper for chat
// completions and embeddings across 18 LLM providers behind one normalized
// surface.
//
// It is the Go sibling of @thinwrap/llm (npm) and thinwrap/llm (Packagist),
// sharing the same normalized surface: one ChatInput / ChatResult /
// ChatStreamDelta shape across every provider, an Embeddings operation, and a
// single typed *ConnectorError.
//
// Fifteen first-class providers speak OpenAI Chat Completions and share one
// connector (parameterized per provider); three natives — Anthropic, Bedrock,
// Gemini — have bespoke wire adapters that emit the identical normalized shapes,
// so they stay drop-in swappable.
//
//	chat, err := llm.NewChat(llm.OpenAI, llm.OpenAICompatConfig{APIKey: key})
//	if err != nil { ... }
//	res, err := chat.Complete(ctx, llm.ChatInput{
//		Model:    "gpt-5.5",
//		Messages: []llm.ChatMessage{{Role: "user", Content: "Hello"}},
//	})
//	if err != nil {
//		var ce *llm.ConnectorError
//		if errors.As(err, &ce) { /* ce.ProviderCode, ce.StatusCode */ }
//	}
//	fmt.Println(res.Message.Content, res.Usage.TotalTokens)
//
// Streaming:
//
//	stream, err := chat.Stream(ctx, in)
//	if err != nil { ... }
//	defer stream.Close()
//	for {
//		delta, err := stream.Recv()
//		if err == io.EOF { break }
//		if err != nil { ... }
//		fmt.Print(delta.ContentDelta)
//	}
//
// Switching vendor is a change of provider id + model only. The wrapper holds no
// state (no caching, retries, or conversation memory); provider-specific request
// fields flow through Passthrough and are never emulated. Zero third-party
// dependencies: Bedrock's SigV4 is hand-rolled on the standard library.
package llm
