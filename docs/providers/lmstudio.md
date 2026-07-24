# LM Studio (self-host) (Go)

Chat + embeddings via the shared OpenAI-compatible connector (spec row, not a per-provider type). Switching to/from LM Studio is just the provider const + `Model`; `ChatInput`/`ChatResult` are identical to every other provider.

## Quick start

```go
import "github.com/thinwrap/llm-go"

chat, err := llm.NewChat(llm.LMStudio, llm.OpenAICompatConfig{}) // APIKey optional; defaults to localhost:1234
res, err := chat.Complete(ctx, llm.ChatInput{
	Model:    "<loaded-model>",
	Messages: []llm.ChatMessage{{Role: "user", Content: "Hi"}},
})
```

## Configuration

`llm.OpenAICompatConfig`:

| Field | Type | Required | Notes |
|---|---|---|---|
| `APIKey` | `string` | No | Optional — sent as `Authorization: Bearer <APIKey>` only if set (API-token auth since v0.4.0; off by default). |
| `BaseURL` | `string` | No* | Overrides the default `http://localhost:1234/v1`. *Required if not on `:1234`. |
| `Headers` | `map[string]string` | No | Merged onto every request. |

Inject a custom transport with `llm.WithHTTPClient(...)`.

## Auth setup

Local single-user server; auth is off by default (leave `APIKey` empty → no auth header). If you enabled API-token auth, pass the token as `APIKey`; it is sent as `Authorization: Bearer <APIKey>`.

## Chat

### Endpoint

`POST http://localhost:1234/v1/chat/completions` (override `BaseURL` if not on `:1234`).

### Notes / quirks

- Self-host: with a single model loaded, the server ignores the requested `Model` id and serves whatever is loaded.
- Structured output is grammar-enforced (llama.cpp / MLX); small models can be unreliable.
- `MaxOutputTokens` maps to `max_tokens`. `Reasoning.Effort` maps to `reasoning_effort`.
- Streaming via `chat.Stream(ctx, in)` → `Recv()` until `io.EOF`, then `Close()`. `stream_options.include_usage` is set.

## Embeddings

Supported (an embedding model must be loaded).

```go
emb, err := llm.NewEmbeddings(llm.LMStudio, llm.OpenAICompatConfig{})
out, err := emb.Create(ctx, llm.EmbeddingsInput{Model: "<loaded-embedding-model>", Input: []string{"hello"}})
```

`POST http://localhost:1234/v1/embeddings` with `encoding_format: "float"`; `Dimensions` maps to `dimensions` when set.

## Error mapping & passthrough

- Non-2xx returns a `*llm.ConnectorError` with `ProviderCode` in `rate_limited` / `auth_failed` / `provider_unavailable` / `invalid_request` / `context_length_exceeded` / `content_filtered` / `unknown`.
- `Retry-After` rides on `Cause["retryAfter"]` / `Cause["retryAfterSeconds"]`; raw body on `Cause["raw"]`.
- Sub-baseline request fields go via `Passthrough.Body` / `.Headers` / `.Query`; response extras on `ChatResult.Raw`.
