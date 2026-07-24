# Cloudflare Workers AI (Go)

Chat + embeddings via the shared OpenAI-compatible connector (spec row, not a per-provider type). Switching to/from Cloudflare is just the provider const + `Model`; `ChatInput`/`ChatResult` are identical to every other provider.

## Quick start

```go
import "github.com/thinwrap/llm-go"

chat, err := llm.NewChat(llm.Cloudflare, llm.OpenAICompatConfig{
	APIKey:  os.Getenv("CLOUDFLARE_API_TOKEN"),
	BaseURL: "https://api.cloudflare.com/client/v4/accounts/<accountId>/ai/v1", // required
})
res, err := chat.Complete(ctx, llm.ChatInput{
	Model:    "@cf/meta/llama-3.1-8b-instruct",
	Messages: []llm.ChatMessage{{Role: "user", Content: "Hi"}},
})
```

## Configuration

`llm.OpenAICompatConfig`:

| Field | Type | Required | Notes |
|---|---|---|---|
| `APIKey` | `string` | Yes | Sent as `Authorization: Bearer <APIKey>`. |
| `BaseURL` | `string` | **Yes** | No public default — construction fails without it. |
| `Headers` | `map[string]string` | No | Merged onto every request. |

Inject a custom transport with `llm.WithHTTPClient(...)`.

## Auth setup

Create an API token (Workers AI scope) in the Cloudflare dashboard. It is sent as `Authorization: Bearer <APIKey>`.

## Chat

### Endpoint

`POST <BaseURL>/chat/completions` — `BaseURL` is `https://api.cloudflare.com/client/v4/accounts/<accountId>/ai/v1`. Omitting `BaseURL` returns an `invalid_request` `*ConnectorError`.

### Notes / quirks

- **`BaseURL` is required** (no fixed public default — it embeds your account id).
- Model ids are `@cf/{provider}/{model}` (some `@hf/…`), required verbatim.
- JSON mode is **not strict** (no schema-adherence guarantee) and can't stream.
- `MaxOutputTokens` maps to `max_tokens`. `Reasoning.Effort` maps to `reasoning_effort`.
- Streaming via `chat.Stream(ctx, in)` → `Recv()` until `io.EOF`, then `Close()`. `stream_options.include_usage` is set.

## Embeddings

Supported.

```go
emb, err := llm.NewEmbeddings(llm.Cloudflare, llm.OpenAICompatConfig{
	APIKey:  os.Getenv("CLOUDFLARE_API_TOKEN"),
	BaseURL: "https://api.cloudflare.com/client/v4/accounts/<accountId>/ai/v1",
})
out, err := emb.Create(ctx, llm.EmbeddingsInput{Model: "@cf/baai/bge-large-en-v1.5", Input: []string{"hello"}})
```

`POST <BaseURL>/embeddings` with `encoding_format: "float"`; `Dimensions` maps to `dimensions` when set.

## Error mapping & passthrough

- Non-2xx returns a `*llm.ConnectorError` with `ProviderCode` in `rate_limited` / `auth_failed` / `provider_unavailable` / `invalid_request` / `context_length_exceeded` / `content_filtered` / `unknown`.
- `Retry-After` rides on `Cause["retryAfter"]` / `Cause["retryAfterSeconds"]`; raw body on `Cause["raw"]`.
- Sub-baseline request fields go via `Passthrough.Body` / `.Headers` / `.Query`; response extras on `ChatResult.Raw`.
