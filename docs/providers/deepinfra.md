# DeepInfra (Go)

Chat + embeddings via the shared OpenAI-compatible connector (spec row, not a per-provider type). Switching to/from DeepInfra is just the provider const + `Model`; `ChatInput`/`ChatResult` are identical to every other provider.

## Quick start

```go
import "github.com/thinwrap/llm-go"

chat, err := llm.NewChat(llm.DeepInfra, llm.OpenAICompatConfig{APIKey: os.Getenv("DEEPINFRA_API_KEY")})
res, err := chat.Complete(ctx, llm.ChatInput{
	Model:    "meta-llama/Meta-Llama-3.1-70B-Instruct",
	Messages: []llm.ChatMessage{{Role: "user", Content: "Hi"}},
})
```

## Configuration

`llm.OpenAICompatConfig`:

| Field | Type | Required | Notes |
|---|---|---|---|
| `APIKey` | `string` | Yes | Sent as `Authorization: Bearer <APIKey>`. |
| `BaseURL` | `string` | No | Overrides the default `https://api.deepinfra.com/v1/openai`. |
| `Headers` | `map[string]string` | No | Merged onto every request. |

Inject a custom transport with `llm.WithHTTPClient(...)`.

## Auth setup

Create a token at <https://deepinfra.com/dash/api_keys>. It is sent as `Authorization: Bearer <APIKey>`.

## Chat

### Endpoint

`POST https://api.deepinfra.com/v1/openai/chat/completions`

### Notes / quirks

- Model ids are HuggingFace-style `org/model` slugs (case-sensitive).
- Runs on a vLLM backend (superset of OpenAI); `usage` adds a non-standard `estimated_cost` (read it from `ChatResult.Raw`); the vision `detail` field is ignored.
- `MaxOutputTokens` maps to `max_tokens`. `Reasoning.Effort` maps to `reasoning_effort`.
- Streaming via `chat.Stream(ctx, in)` → `Recv()` until `io.EOF`, then `Close()`. `stream_options.include_usage` is set.

## Embeddings

Supported.

```go
emb, err := llm.NewEmbeddings(llm.DeepInfra, llm.OpenAICompatConfig{APIKey: os.Getenv("DEEPINFRA_API_KEY")})
out, err := emb.Create(ctx, llm.EmbeddingsInput{Model: "BAAI/bge-large-en-v1.5", Input: []string{"hello"}})
```

`POST https://api.deepinfra.com/v1/openai/embeddings` with `encoding_format: "float"`; `Dimensions` maps to `dimensions` when set.

## Error mapping & passthrough

- Non-2xx returns a `*llm.ConnectorError` with `ProviderCode` in `rate_limited` / `auth_failed` / `provider_unavailable` / `invalid_request` / `context_length_exceeded` / `content_filtered` / `unknown`.
- `Retry-After` rides on `Cause["retryAfter"]` / `Cause["retryAfterSeconds"]`; raw body on `Cause["raw"]`.
- Sub-baseline request fields go via `Passthrough.Body` / `.Headers` / `.Query`; response extras on `ChatResult.Raw`.
