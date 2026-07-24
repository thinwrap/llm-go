# Fireworks AI (Go)

Chat + embeddings via the shared OpenAI-compatible connector (spec row, not a per-provider type). Switching to/from Fireworks is just the provider const + `Model`; `ChatInput`/`ChatResult` are identical to every other provider.

## Quick start

```go
import "github.com/thinwrap/llm-go"

chat, err := llm.NewChat(llm.Fireworks, llm.OpenAICompatConfig{APIKey: os.Getenv("FIREWORKS_API_KEY")})
res, err := chat.Complete(ctx, llm.ChatInput{
	Model:    "accounts/fireworks/models/llama-v3p3-70b-instruct",
	Messages: []llm.ChatMessage{{Role: "user", Content: "Hi"}},
})
```

## Configuration

`llm.OpenAICompatConfig`:

| Field | Type | Required | Notes |
|---|---|---|---|
| `APIKey` | `string` | Yes | Sent as `Authorization: Bearer <APIKey>`. |
| `BaseURL` | `string` | No | Overrides the default `https://api.fireworks.ai/inference/v1`. |
| `Headers` | `map[string]string` | No | Merged onto every request. |

Inject a custom transport with `llm.WithHTTPClient(...)`.

## Auth setup

Create a key at <https://fireworks.ai/account/api-keys>. It is sent as `Authorization: Bearer <APIKey>`.

## Chat

### Endpoint

`POST https://api.fireworks.ai/inference/v1/chat/completions`

### Notes / quirks

- The OpenAI-compatible endpoint **is** the native API (a superset).
- Model ids are account-namespaced (`accounts/fireworks/models/…`); bare OpenAI names won't resolve.
- Reasoning chain-of-thought is exposed via `reasoning_content` — read it from `ChatResult.Raw`.
- `MaxOutputTokens` maps to `max_tokens`. `Reasoning.Effort` maps to `reasoning_effort`.
- Streaming via `chat.Stream(ctx, in)` → `Recv()` until `io.EOF`, then `Close()`. `stream_options.include_usage` is set.

## Embeddings

Supported.

```go
emb, err := llm.NewEmbeddings(llm.Fireworks, llm.OpenAICompatConfig{APIKey: os.Getenv("FIREWORKS_API_KEY")})
out, err := emb.Create(ctx, llm.EmbeddingsInput{Model: "nomic-ai/nomic-embed-text-v1.5", Input: []string{"hello"}})
```

`POST https://api.fireworks.ai/inference/v1/embeddings` with `encoding_format: "float"`; `Dimensions` maps to `dimensions` when set.

## Error mapping & passthrough

- Non-2xx returns a `*llm.ConnectorError` with `ProviderCode` in `rate_limited` / `auth_failed` / `provider_unavailable` / `invalid_request` / `context_length_exceeded` / `content_filtered` / `unknown`.
- `Retry-After` rides on `Cause["retryAfter"]` / `Cause["retryAfterSeconds"]`; raw body on `Cause["raw"]`.
- Sub-baseline request fields go via `Passthrough.Body` / `.Headers` / `.Query`; response extras on `ChatResult.Raw`.
