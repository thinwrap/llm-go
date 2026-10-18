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
    Model:    "gpt-5.5",
    Messages: []llm.ChatMessage{{Role: "user", Content: "Say hi in one word."}},
})
if err != nil {
    var ce *llm.ConnectorError
    if errors.As(err, &ce) { log.Printf("%s: %s", ce.ProviderCode, ce.ProviderMessage) }
    return
}
fmt.Println(res.Message.Content, res.Usage.TotalTokens)
```

`Usage.InputTokens` counts every billed prompt token, cache reads and writes
included; `OutputTokens` includes reasoning tokens; `TotalTokens` is their sum
(or the vendor's total). Streamed `Usage` is cumulative, so keep the last one.

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

`io.EOF` comes only after the provider's terminal event (`[DONE]` or a finish
reason). A 2xx that is not `text/event-stream`, or a body cut off before that
event, ends with a `provider_unavailable` `*ConnectorError` instead.

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
// then in.Model = "claude-opus-5-5", etc.
```

`OpenAICompatConfig` covers all 15 first-class providers (`APIKey`, `BaseURL`,
`Headers`); `BaseURL` is required for `AzureOpenAI` / `Cloudflare` and any
self-host provider on a non-default host.

## Multi-turn and tool loops: `ProviderMetadata`

Some vendors need opaque state from one turn sent back on the next: Gemini's
`thoughtSignature` on each function call, and Anthropic's thinking blocks.
`ToolCall.ProviderMetadata` and `ChatMessage.ProviderMetadata` carry it, keyed by
provider id; a connector reads only its own key, so a history survives a
provider switch. When you build the next request from a result, copy it along
with the content and tool calls:

```go
msgs = append(msgs, llm.ChatMessage{
    Role:             "assistant",
    Content:          res.Message.Content,
    ToolCalls:        res.Message.ToolCalls,        // each keeps its ProviderMetadata
    ProviderMetadata: res.Message.ProviderMetadata,
})
```

When streaming, collect `ToolCallDelta.ProviderMetadata` per call and the
terminal delta's `ProviderMetadata` for the message. Copy the values as they
came back (or JSON-decode a stored history into `any`): connectors read only the
JSON-decoded shapes (`map[string]any`, `[]any`), so a hand-built typed value
such as `map[string]string` is ignored. Dropping it is not fatal:
Anthropic then loses earlier reasoning, and Gemini falls back to Google's
documented signature placeholder.

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
`context_length_exceeded`, `content_filtered`, `unknown`. By HTTP status:
401/403 → `auth_failed`, 402/429 → `rate_limited`, 408 and 5xx →
`provider_unavailable`, any other 4xx → `invalid_request` unless the body names
a context overflow or a content filter. The raw `Retry-After` rides in
`Cause["retryAfter"]` (+ parsed `Cause["retryAfterSeconds"]`: whole seconds, or
an HTTP-date as seconds from now with a past date giving 0; absent for anything
else) — there is no top-level retry field. A 2xx whose body carries a vendor
error, or lacks the payload a result requires (`choices`, `content`, `output`,
`candidates`, one finite float vector per embeddings input), is a
`*ConnectorError` too — never an empty success. An error that arrives inside a
stream carries the stream's own HTTP status (normally 200) and is classified by
the vendor's error type or code (`unknown` when the frame has neither).

The wrapper sets no timeout of its own: bound each call with the `ctx` you pass
(or your own `*http.Client`).

## Relationship to the siblings

One of four thinwrap llm libraries sharing a normalized surface and byte-identical
provider ids / error codes: TypeScript (`@thinwrap/llm`), PHP (`thinwrap/llm`),
Go (this package), Python (`thinwrap-llm`).

## Security

Report vulnerabilities **privately** — please do not open a public issue. Preferred: a
[private security advisory](https://github.com/thinwrap/llm-go/security/advisories/new)
on this repository. Alternatively, email **security@thinwrap.dev**. Include the affected
versions and a minimal reproduction if you have one.

A vulnerability in a *provider's* own API or service belongs to that vendor rather than
to this wrapper — please report those upstream.

## License

MIT © Dmitry Polyanovsky & contributors
