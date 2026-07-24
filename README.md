# thinwrap/llm (Go)

Unified, **SDK-free**, **zero-dependency** Go wrapper for **chat completions**
(with streaming) and **embeddings** across 18 LLM providers. Switch vendor by
changing the provider id + model; the input and output shapes stay identical.
Stateless — bring your own `*http.Client`, no vendor SDKs.

```go
import "github.com/thinwrap/llm-go"
```

Requires Go ≥ 1.18. No third-party dependencies — Bedrock's SigV4 is hand-rolled
on `crypto/*`, SSE streaming on `bufio`.

## Two-minute chat

```go
chat, err := llm.NewChat(llm.OpenAI, llm.OpenAICompatConfig{APIKey: os.Getenv("OPENAI_API_KEY")})
if err != nil { log.Fatal(err) }

res, err := chat.Complete(ctx, llm.ChatInput{
    Model:    "gpt-4o-mini",
    Messages: []llm.ChatMessage{{Role: "user", Content: "Say hi in one word."}},
})
if err != nil {
    var ce *llm.ConnectorError
    if errors.As(err, &ce) { log.Printf("%s: %s", ce.ProviderCode, ce.ProviderMessage) }
    return
}
fmt.Println(res.Message.Content, res.Usage.TotalTokens)
```

## Streaming

```go
stream, err := chat.Stream(ctx, in)
if err != nil { return }
defer stream.Close()
for {
    delta, err := stream.Recv()
    if err == io.EOF { break }
    if err != nil { return }
    fmt.Print(delta.ContentDelta)
}
```

## Providers

The **provider id selects the connector**; `ChatInput` / `ChatResult` are
identical across all of them.

**Chat (18):** the 15 first-class OpenAI-compatible providers — `OpenAI`,
`AzureOpenAI`, `OpenRouter`, `Groq`, `Together`, `Fireworks`, `DeepSeek`, `XAI`,
`Mistral`, `Perplexity`, `DeepInfra`, `Cloudflare`, `VLLM`, `Ollama`, `LMStudio`
— all take `OpenAICompatConfig`; plus 3 natives: `Anthropic` (`AnthropicConfig`),
`Bedrock` (`BedrockConfig`), `Gemini` (`GeminiConfig`).

**Embeddings (11):** `OpenAI`, `AzureOpenAI`, `OpenRouter`, `Together`,
`Fireworks`, `Mistral`, `DeepInfra`, `Cloudflare`, `VLLM`, `Ollama`, `LMStudio`
(the OpenAI-float subset).

```go
emb, _ := llm.NewEmbeddings(llm.OpenAI, llm.OpenAICompatConfig{APIKey: key})
out, _ := emb.Create(ctx, llm.EmbeddingsInput{Model: "text-embedding-3-small", Input: []string{"hello", "world"}})
// out.Embeddings is [][]float64 in input order
```

Per-provider details (endpoint, auth, quirks, passthrough keys) live in
[`docs/providers/`](docs/providers/) — one page per provider.

## Switching providers

```go
// Same Complete(ctx, in) call, same ChatResult — id + model change only.
chat, _ := llm.NewChat(llm.Anthropic, llm.AnthropicConfig{APIKey: key})
// then in.Model = "claude-sonnet-4-5", etc.
```

`OpenAICompatConfig` covers all 15 first-class providers (`APIKey`, `BaseURL`,
`Headers`); `BaseURL` is required for `AzureOpenAI` / `Cloudflare` and any
self-host provider on a non-default host.

## The ≥90% baseline & passthrough

Only fields ≥90% of providers support normalizably are first-class on
`ChatInput` / `ChatResult`. Everything else rides through `Passthrough` (request)
/ `Raw` (response) and is **never emulated**:

```go
res, _ := chat.Complete(ctx, llm.ChatInput{
    Model:    "...",
    Messages: msgs,
    Passthrough: &llm.Passthrough{Body: map[string]any{"logprobs": true}},
})
_ = res.Raw // verbatim vendor body — reasoning CoT, cached-token counts, cost, ...
```

## Errors

Every failure is a `*ConnectorError` (`errors.As`): `ProviderCode` is one of
`rate_limited`, `auth_failed`, `provider_unavailable`, `invalid_request`,
`context_length_exceeded`, `content_filtered`, `unknown`. The raw `Retry-After`
rides in `Cause["retryAfter"]` (+ parsed `Cause["retryAfterSeconds"]`) — there is
no top-level retry field.

## Relationship to the siblings

One of four thinwrap llm libraries sharing a normalized surface and byte-identical
provider ids / error codes: TypeScript (`@thinwrap/llm`), PHP (`thinwrap/llm`),
Go (this package), Python (`thinwrap-llm`).

## License

MIT © Dmitry Polyanovsky & contributors
