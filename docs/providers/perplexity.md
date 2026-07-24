# Perplexity (Sonar) (Go)

Chat via the shared OpenAI-compatible connector (spec row, not a per-provider type). Switching to/from Perplexity is just the provider const + `Model`; `ChatInput`/`ChatResult` are identical to every other provider.

## Quick start

```go
import "github.com/thinwrap/llm-go"

chat, err := llm.NewChat(llm.Perplexity, llm.OpenAICompatConfig{APIKey: os.Getenv("PERPLEXITY_API_KEY")})
res, err := chat.Complete(ctx, llm.ChatInput{
	Model:    "sonar",
	Messages: []llm.ChatMessage{{Role: "user", Content: "Hi"}},
})
```

## Configuration

`llm.OpenAICompatConfig`:

| Field | Type | Required | Notes |
|---|---|---|---|
| `APIKey` | `string` | Yes | Sent as `Authorization: Bearer <APIKey>`. |
| `BaseURL` | `string` | No | Overrides the default `https://api.perplexity.ai` (no `/v1` segment). |
| `Headers` | `map[string]string` | No | Merged onto every request. |

Inject a custom transport with `llm.WithHTTPClient(...)`.

## Auth setup

Create a key at <https://www.perplexity.ai/settings/api>. It is sent as `Authorization: Bearer <APIKey>`.

## Chat

### Endpoint

`POST https://api.perplexity.ai/chat/completions` (the default base URL has no `/v1` prefix).

### Notes / quirks

- **No tool calling (handled):** the Sonar chat surface has no tool/function calling, so the connector sets `supportsTools: false` — `Tools` and `ToolChoice` on `ChatInput` are **not sent** (tools live on Perplexity's separate Agent API).
- Sampling surface is stripped upstream (no `top_k` / penalties / `seed` / `n` on Sonar).
- Model ids are `sonar` / `sonar-pro` / `sonar-reasoning-pro` / `sonar-deep-research`.
- `MaxOutputTokens` maps to `max_tokens`. `Reasoning.Effort` maps to `reasoning_effort`.
- Streaming via `chat.Stream(ctx, in)` → `Recv()` until `io.EOF`, then `Close()`. `stream_options.include_usage` is set.

## Embeddings

No OpenAI-float embeddings surface — `llm.NewEmbeddings(llm.Perplexity, ...)` returns an `invalid_request` `*ConnectorError` ("Provider 'perplexity' has no OpenAI-compatible embeddings surface").

## Error mapping & passthrough

- Non-2xx returns a `*llm.ConnectorError` with `ProviderCode` in `rate_limited` / `auth_failed` / `provider_unavailable` / `invalid_request` / `context_length_exceeded` / `content_filtered` / `unknown`.
- `Retry-After` rides on `Cause["retryAfter"]` / `Cause["retryAfterSeconds"]`; raw body on `Cause["raw"]`.
- Search/citation controls + sub-baseline fields go via `Passthrough.Body` / `.Headers` / `.Query`; citations and other vendor extras on `ChatResult.Raw`.
