# Groq (Go)

Chat via the shared OpenAI-compatible connector (spec row, not a per-provider type). Switching to/from Groq is just the provider const + `Model`; `ChatInput`/`ChatResult` are identical to every other provider.

## Quick start

```go
import "github.com/thinwrap/llm-go"

chat, err := llm.NewChat(llm.Groq, llm.OpenAICompatConfig{APIKey: os.Getenv("GROQ_API_KEY")})
res, err := chat.Complete(ctx, llm.ChatInput{
	Model:    "llama-3.3-70b-versatile",
	Messages: []llm.ChatMessage{{Role: "user", Content: "Hi"}},
})
```

## Configuration

`llm.OpenAICompatConfig`:

| Field | Type | Required | Notes |
|---|---|---|---|
| `APIKey` | `string` | Yes | Sent as `Authorization: Bearer <APIKey>`. |
| `BaseURL` | `string` | No | Overrides the default `https://api.groq.com/openai/v1`. |
| `Headers` | `map[string]string` | No | Merged onto every request. |

Inject a custom transport with `llm.WithHTTPClient(...)`.

## Auth setup

Create a key at <https://console.groq.com/keys>. It is sent as `Authorization: Bearer <APIKey>`.

## Chat

### Endpoint

`POST https://api.groq.com/openai/v1/chat/completions`

### Notes / quirks

- **Streaming usage quirk (handled):** in streaming, token usage arrives on `x_groq.usage` of the final chunk, not top-level `usage`. The connector relocates it, so `ChatStreamDelta.Usage` is populated the same as everywhere else (spec `streamUsagePath: "x_groq"`).
- Strict JSON-schema structured output is limited to `openai/gpt-oss-*` models and can't combine with streaming or tools; other models are best-effort.
- Model ids are provider-namespaced (`openai/…`, `meta-llama/…`, `qwen/…`).
- `MaxOutputTokens` maps to `max_tokens`. `Reasoning.Effort` maps to `reasoning_effort`.
- Streaming via `chat.Stream(ctx, in)` → `Recv()` until `io.EOF`, then `Close()`. `stream_options.include_usage` is set.

## Embeddings

No OpenAI-float embeddings surface — `llm.NewEmbeddings(llm.Groq, ...)` returns an `invalid_request` `*ConnectorError` ("Provider 'groq' has no OpenAI-compatible embeddings surface").

## Error mapping & passthrough

- Non-2xx returns a `*llm.ConnectorError` with `ProviderCode` in `rate_limited` / `auth_failed` / `provider_unavailable` / `invalid_request` / `context_length_exceeded` / `content_filtered` / `unknown`.
- `Retry-After` rides on `Cause["retryAfter"]` / `Cause["retryAfterSeconds"]`; raw body on `Cause["raw"]`.
- Sub-baseline request fields go via `Passthrough.Body` / `.Headers` / `.Query`; response extras on `ChatResult.Raw`.
