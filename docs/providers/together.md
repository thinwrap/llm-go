# Together AI (Go)

Chat + embeddings via the shared OpenAI-compatible connector (spec row, not a per-provider type). Switching to/from Together is just the provider const + `Model`; `ChatInput`/`ChatResult` are identical to every other provider.

## Quick start

```go
import "github.com/thinwrap/llm-go"

chat, err := llm.NewChat(llm.Together, llm.OpenAICompatConfig{APIKey: os.Getenv("TOGETHER_API_KEY")})
res, err := chat.Complete(ctx, llm.ChatInput{
	Model:    "meta-llama/Llama-3.3-70B-Instruct-Turbo",
	Messages: []llm.ChatMessage{{Role: "user", Content: "Hi"}},
})
```

## Configuration

`llm.OpenAICompatConfig`:

| Field | Type | Required | Notes |
|---|---|---|---|
| `APIKey` | `string` | Yes | Sent as `Authorization: Bearer <APIKey>`. |
| `BaseURL` | `string` | No | Overrides the default `https://api.together.ai/v1`. |
| `Headers` | `map[string]string` | No | Merged onto every request. |

Inject a custom transport with `llm.WithHTTPClient(...)`.

## Auth setup

Create a key at <https://api.together.ai/settings/api-keys>. It is sent as `Authorization: Bearer <APIKey>`.

## Chat

### Endpoint

`POST https://api.together.ai/v1/chat/completions`

### Notes / quirks

- Model ids are namespaced `org/model` slugs — **OpenAI model strings like `gpt-4o` return 404**; use Together's ids.
- `service_tier`, `store`, `metadata`, `prediction`, and the vision `detail` field are accepted but ignored; `seed` is best-effort.
- `MaxOutputTokens` maps to `max_tokens`. `Reasoning.Effort` maps to `reasoning_effort`.
- Streaming via `chat.Stream(ctx, in)` → `Recv()` until `io.EOF`, then `Close()`. `stream_options.include_usage` is set.

## Embeddings

Supported.

```go
emb, err := llm.NewEmbeddings(llm.Together, llm.OpenAICompatConfig{APIKey: os.Getenv("TOGETHER_API_KEY")})
out, err := emb.Create(ctx, llm.EmbeddingsInput{Model: "BAAI/bge-large-en-v1.5", Input: []string{"hello"}})
```

`POST https://api.together.ai/v1/embeddings` with `encoding_format: "float"`; `Dimensions` maps to `dimensions` when set.

## Error mapping & passthrough

- Non-2xx returns a `*llm.ConnectorError` with `ProviderCode` in `rate_limited` / `auth_failed` / `provider_unavailable` / `invalid_request` / `context_length_exceeded` / `content_filtered` / `unknown`.
- `Retry-After` rides on `Cause["retryAfter"]` / `Cause["retryAfterSeconds"]`; raw body on `Cause["raw"]`.
- Sub-baseline request fields go via `Passthrough.Body` / `.Headers` / `.Query`; response extras on `ChatResult.Raw`.
