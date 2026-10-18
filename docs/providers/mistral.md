# Mistral (Go)

Chat + embeddings via the shared OpenAI-compatible connector (spec row, not a per-provider type). Switching to/from Mistral is just the provider const + `Model`; `ChatInput`/`ChatResult` are identical to every other provider.

## Quick start

```go
import "github.com/thinwrap/llm-go"

chat, err := llm.NewChat(llm.Mistral, llm.OpenAICompatConfig{APIKey: os.Getenv("MISTRAL_API_KEY")})
res, err := chat.Complete(ctx, llm.ChatInput{
	Model:    "mistral-large-latest",
	Messages: []llm.ChatMessage{{Role: "user", Content: "Hi"}},
})
```

## Configuration

`llm.OpenAICompatConfig`:

| Field | Type | Required | Notes |
|---|---|---|---|
| `APIKey` | `string` | Yes | Sent as `Authorization: Bearer <APIKey>`. |
| `BaseURL` | `string` | No | Overrides the default `https://api.mistral.ai/v1`. |
| `Headers` | `map[string]string` | No | Merged onto every request. |

Inject a custom transport with `llm.WithHTTPClient(...)`.

## Auth setup

Create a key at <https://console.mistral.ai/api-keys>. It is sent as `Authorization: Bearer <APIKey>`.

## Chat

### Endpoint

`POST https://api.mistral.ai/v1/chat/completions`

### Notes / quirks

- **Strict-params quirk (handled):** Mistral 422s on unknown params, so the connector omits OpenAI-only extras — notably `stream_options` is **not** sent on streaming requests (spec `strictParams: true`). Keep `Passthrough.Body` to known Mistral fields.
- Because `stream_options.include_usage` is not sent, streaming `Usage` is only present if Mistral includes it natively on the final chunk.
- Model ids are date-stamped (e.g. `-YYMM`).
- `MaxOutputTokens` maps to `max_tokens`. `Reasoning.Effort` maps to `reasoning_effort`.
- With reasoning on, Mistral returns `message.content` (and streaming `delta.content`) as a list of chunks; `Message.Content` / `ContentDelta` carry only the `text` chunks — the `thinking` chunks stay on `Raw`.
- Streaming via `chat.Stream(ctx, in)` → `Recv()` until `io.EOF`, then `Close()`.

## Embeddings

Supported.

```go
emb, err := llm.NewEmbeddings(llm.Mistral, llm.OpenAICompatConfig{APIKey: os.Getenv("MISTRAL_API_KEY")})
out, err := emb.Create(ctx, llm.EmbeddingsInput{Model: "mistral-embed", Input: []string{"hello"}})
```

`POST https://api.mistral.ai/v1/embeddings` with `encoding_format: "float"`; `Dimensions` maps to Mistral's `output_dimension` when set (models that support it, e.g. `codestral-embed`).

## Error mapping & passthrough

- Non-2xx returns a `*llm.ConnectorError` with `ProviderCode` in `rate_limited` / `auth_failed` / `provider_unavailable` / `invalid_request` / `context_length_exceeded` / `content_filtered` / `unknown`.
- `Retry-After` rides on `Cause["retryAfter"]` / `Cause["retryAfterSeconds"]`; raw body on `Cause["raw"]`.
- Sub-baseline request fields go via `Passthrough.Body` / `.Headers` / `.Query`; response extras on `ChatResult.Raw`.
