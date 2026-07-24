# xAI (Grok) (Go)

Chat via the shared OpenAI-compatible connector (spec row, not a per-provider type). Switching to/from xAI is just the provider const + `Model`; `ChatInput`/`ChatResult` are identical to every other provider.

## Quick start

```go
import "github.com/thinwrap/llm-go"

chat, err := llm.NewChat(llm.XAI, llm.OpenAICompatConfig{APIKey: os.Getenv("XAI_API_KEY")})
res, err := chat.Complete(ctx, llm.ChatInput{
	Model:    "grok-4",
	Messages: []llm.ChatMessage{{Role: "user", Content: "Hi"}},
})
```

## Configuration

`llm.OpenAICompatConfig`:

| Field | Type | Required | Notes |
|---|---|---|---|
| `APIKey` | `string` | Yes | Sent as `Authorization: Bearer <APIKey>`. |
| `BaseURL` | `string` | No | Overrides the default `https://api.x.ai/v1`. |
| `Headers` | `map[string]string` | No | Merged onto every request. |

Inject a custom transport with `llm.WithHTTPClient(...)`.

## Auth setup

Create a key at <https://console.x.ai>. It is sent as `Authorization: Bearer <APIKey>`.

## Chat

### Endpoint

`POST https://api.x.ai/v1/chat/completions`

### Notes / quirks

- Model ids are `grok-*`.
- xAI positions Chat Completions as a **legacy** endpoint (it steers new work to its own Responses API); this connector targets Chat Completions.
- In streaming, a tool/function call arrives **whole in a single chunk**, not as incremental argument deltas — so `ChatStreamDelta.ToolCallDelta.ArgumentsDelta` carries the full arguments at once.
- `MaxOutputTokens` maps to `max_tokens`. `Reasoning.Effort` maps to `reasoning_effort`.
- Streaming via `chat.Stream(ctx, in)` → `Recv()` until `io.EOF`, then `Close()`. `stream_options.include_usage` is set.

## Embeddings

No OpenAI-float embeddings surface — `llm.NewEmbeddings(llm.XAI, ...)` returns an `invalid_request` `*ConnectorError` ("Provider 'xai' has no OpenAI-compatible embeddings surface").

## Error mapping & passthrough

- Non-2xx returns a `*llm.ConnectorError` with `ProviderCode` in `rate_limited` / `auth_failed` / `provider_unavailable` / `invalid_request` / `context_length_exceeded` / `content_filtered` / `unknown`.
- `Retry-After` rides on `Cause["retryAfter"]` / `Cause["retryAfterSeconds"]`; raw body on `Cause["raw"]`.
- Sub-baseline request fields go via `Passthrough.Body` / `.Headers` / `.Query`; response extras on `ChatResult.Raw`.
