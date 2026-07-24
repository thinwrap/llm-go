# DeepSeek (Go)

Chat via the shared OpenAI-compatible connector (spec row, not a per-provider type). Switching to/from DeepSeek is just the provider const + `Model`; `ChatInput`/`ChatResult` are identical to every other provider.

## Quick start

```go
import "github.com/thinwrap/llm-go"

chat, err := llm.NewChat(llm.DeepSeek, llm.OpenAICompatConfig{APIKey: os.Getenv("DEEPSEEK_API_KEY")})
res, err := chat.Complete(ctx, llm.ChatInput{
	Model:    "deepseek-chat",
	Messages: []llm.ChatMessage{{Role: "user", Content: "Hi"}},
})
```

## Configuration

`llm.OpenAICompatConfig`:

| Field | Type | Required | Notes |
|---|---|---|---|
| `APIKey` | `string` | Yes | Sent as `Authorization: Bearer <APIKey>`. |
| `BaseURL` | `string` | No | Overrides the default `https://api.deepseek.com/v1`. |
| `Headers` | `map[string]string` | No | Merged onto every request. |

Inject a custom transport with `llm.WithHTTPClient(...)`.

## Auth setup

Create a key at <https://platform.deepseek.com/api_keys>. It is sent as `Authorization: Bearer <APIKey>`.

## Chat

### Endpoint

`POST https://api.deepseek.com/v1/chat/completions`

### Notes / quirks

- The OpenAI-compatible endpoint is first-class (the native API).
- **No strict `json_schema`** — `response_format` supports only `text` / `json_object`; a `json_schema` request degrades to best-effort JSON.
- `frequency_penalty` / `presence_penalty` are silently ignored; reasoning is exposed via `reasoning_content` — read it from `ChatResult.Raw`.
- `MaxOutputTokens` maps to `max_tokens`. `Reasoning.Effort` maps to `reasoning_effort`.
- Streaming via `chat.Stream(ctx, in)` → `Recv()` until `io.EOF`, then `Close()`. `stream_options.include_usage` is set.

## Embeddings

No OpenAI-float embeddings surface — `llm.NewEmbeddings(llm.DeepSeek, ...)` returns an `invalid_request` `*ConnectorError` ("Provider 'deepseek' has no OpenAI-compatible embeddings surface").

## Error mapping & passthrough

- Non-2xx returns a `*llm.ConnectorError` with `ProviderCode` in `rate_limited` / `auth_failed` / `provider_unavailable` / `invalid_request` / `context_length_exceeded` / `content_filtered` / `unknown`.
- `Retry-After` rides on `Cause["retryAfter"]` / `Cause["retryAfterSeconds"]`; raw body on `Cause["raw"]`.
- Sub-baseline request fields go via `Passthrough.Body` / `.Headers` / `.Query`; response extras on `ChatResult.Raw`.
