# OpenRouter (Go)

Chat + embeddings via the shared OpenAI-compatible connector (spec row, not a per-provider type). Switching to/from OpenRouter is just the provider const + `Model`; `ChatInput`/`ChatResult` are identical to every other provider.

## Quick start

```go
import "github.com/thinwrap/llm-go"

chat, err := llm.NewChat(llm.OpenRouter, llm.OpenAICompatConfig{APIKey: os.Getenv("OPENROUTER_API_KEY")})
res, err := chat.Complete(ctx, llm.ChatInput{
	Model:    "openai/gpt-4o",
	Messages: []llm.ChatMessage{{Role: "user", Content: "Hi"}},
})
```

## Configuration

`llm.OpenAICompatConfig`:

| Field | Type | Required | Notes |
|---|---|---|---|
| `APIKey` | `string` | Yes | Sent as `Authorization: Bearer <APIKey>`. |
| `BaseURL` | `string` | No | Overrides the default `https://openrouter.ai/api/v1`. |
| `Headers` | `map[string]string` | No | Merged onto every request — set `HTTP-Referer` / `X-Title` here for app attribution. |

Inject a custom transport with `llm.WithHTTPClient(...)`.

## Auth setup

Create a key at <https://openrouter.ai/keys>. It is sent as `Authorization: Bearer <APIKey>`.

## Chat

### Endpoint

`POST https://openrouter.ai/api/v1/chat/completions`

### Notes / quirks

- Aggregator: model ids are `{provider}/{model}` (e.g. `anthropic/claude-sonnet-4.5`), with optional `:free` / `:nitro` / `:floor` suffixes.
- Set `HTTP-Referer` / `X-Title` via `Headers` for attribution.
- Drops params an underlying provider lacks (unless `provider.require_parameters` is set via `Passthrough.Body`); `n` is unsupported.
- `MaxOutputTokens` maps to `max_tokens`. `Reasoning.Effort` maps to `reasoning_effort`.
- Streaming via `chat.Stream(ctx, in)` → `Recv()` until `io.EOF`, then `Close()`. `stream_options.include_usage` is set.

## Embeddings

Supported.

```go
emb, err := llm.NewEmbeddings(llm.OpenRouter, llm.OpenAICompatConfig{APIKey: os.Getenv("OPENROUTER_API_KEY")})
out, err := emb.Create(ctx, llm.EmbeddingsInput{Model: "openai/text-embedding-3-small", Input: []string{"hello"}})
```

`POST https://openrouter.ai/api/v1/embeddings` with `encoding_format: "float"`; `Dimensions` maps to `dimensions` when set.

## Error mapping & passthrough

- A 200 whose body carries an `error` object (no `choices`), or whose `choices[0]` has `finish_reason: "error"` / an `error`, also returns a `*llm.ConnectorError`. `StatusCode` is the status the response arrived with (the 2xx). `ProviderCode` comes from the embedded error: an integer (or numeric-string) `error.code` 400–599 is classified as if it were that HTTP status, a named code or type (`insufficient_quota`, `invalid_api_key`, `server_error`, …) by the same rules, and an error with no usable code is `unknown`.
- Non-2xx returns a `*llm.ConnectorError` with `ProviderCode` in `rate_limited` / `auth_failed` / `provider_unavailable` / `invalid_request` / `context_length_exceeded` / `content_filtered` / `unknown`. `402` (insufficient credits) is `rate_limited`.
- A mid-stream error chunk (a top-level `error`, or `finish_reason: "error"`) ends the stream with a `*llm.ConnectorError` the same way: the stream's own status (normally 200), classified by `error.code` / `type`, `unknown` without one.
- `Retry-After` rides on `Cause["retryAfter"]` / `Cause["retryAfterSeconds"]`; raw body on `Cause["raw"]`.
- Provider-routing controls + sub-baseline fields go via `Passthrough.Body` / `.Headers` / `.Query`; response extras on `ChatResult.Raw`.
