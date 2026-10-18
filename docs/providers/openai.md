# OpenAI (Go)

Chat + embeddings via the shared OpenAI-compatible connector (spec row, not a per-provider type). Switching to/from OpenAI is just the provider const + `Model`; `ChatInput`/`ChatResult` are identical to every other provider.

## Quick start

```go
import "github.com/thinwrap/llm-go"

chat, err := llm.NewChat(llm.OpenAI, llm.OpenAICompatConfig{APIKey: os.Getenv("OPENAI_API_KEY")})
res, err := chat.Complete(ctx, llm.ChatInput{
	Model:    "gpt-5.5",
	Messages: []llm.ChatMessage{{Role: "user", Content: "Hi"}},
})
```

## Configuration

`llm.OpenAICompatConfig`:

| Field | Type | Required | Notes |
|---|---|---|---|
| `APIKey` | `string` | Yes | Sent as `Authorization: Bearer <APIKey>`. |
| `BaseURL` | `string` | No | Overrides the default `https://api.openai.com/v1`. |
| `Headers` | `map[string]string` | No | Merged onto every request. |

Inject a custom `http.Client`-style transport with `llm.WithHTTPClient(...)`.

## Auth setup

Create an API key at <https://platform.openai.com/api-keys>. It is sent verbatim as `Authorization: Bearer <APIKey>`.

## Chat

### Endpoint

`POST https://api.openai.com/v1/chat/completions`

### Notes / quirks

- The reference provider — its Chat Completions shape is the canonical facade shape.
- `MaxOutputTokens` maps to `max_completion_tokens` (OpenAI deprecated `max_tokens`, which o-series models reject; the cap includes reasoning tokens). `Reasoning.Effort` maps to `reasoning_effort`.
- Reasoning models (o-series / GPT-5-class) reject `temperature`/`top_p`/penalties with HTTP 400; guard by model family and leave those normalized sampling fields unset.
- **Function tools with reasoning (Chat Completions):** current models (gpt-5.4, gpt-5.4-mini, gpt-5.5, the gpt-5.6 family) reject `Tools` combined with reasoning with a 400, which passes through as `invalid_request`; gpt-5.6 does so even with `Reasoning` unset, because it reasons by default. To call tools on Chat Completions, send `Passthrough.Body["reasoning_effort"] = "none"` (accepted on gpt-5.4 and later; gpt-5-mini and o4-mini reject `"none"`). Moving this connector to the Responses API, which combines both, is planned.
- OpenAI also ships the newer Responses API; this connector targets Chat Completions.
- Streaming via `chat.Stream(ctx, in)` → read `*ChatStream` with `Recv()` until `io.EOF`, then `Close()`. `stream_options.include_usage` is set so the final chunk carries `Usage`.

## Embeddings

Supported.

```go
emb, err := llm.NewEmbeddings(llm.OpenAI, llm.OpenAICompatConfig{APIKey: os.Getenv("OPENAI_API_KEY")})
out, err := emb.Create(ctx, llm.EmbeddingsInput{
	Model: "text-embedding-3-small",
	Input: []string{"hello", "world"},
})
```

`POST https://api.openai.com/v1/embeddings` with `encoding_format: "float"`. `Dimensions` maps to `dimensions` (Matryoshka) when set. Vectors are returned in input order on `out.Embeddings`.

## Error mapping & passthrough

- Non-2xx returns a `*llm.ConnectorError` (reachable via `errors.As`) with `ProviderCode` in `rate_limited` / `auth_failed` / `provider_unavailable` / `invalid_request` / `context_length_exceeded` / `content_filtered` / `unknown`.
- `Retry-After` (when present) is on `Cause["retryAfter"]` (raw) and `Cause["retryAfterSeconds"]` (parsed); the raw vendor body is on `Cause["raw"]`.
- Sub-baseline request fields (`seed`, `frequency`/`presence_penalty`, `logit_bias`, `n`, prompt caching) go via `Passthrough.Body` / `.Headers` / `.Query`; vendor extras on the response are on `ChatResult.Raw`.
