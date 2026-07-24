# Ollama (self-host) (Go)

Chat + embeddings via the shared OpenAI-compatible connector (spec row, not a per-provider type). Switching to/from Ollama is just the provider const + `Model`; `ChatInput`/`ChatResult` are identical to every other provider.

## Quick start

```go
import "github.com/thinwrap/llm-go"

chat, err := llm.NewChat(llm.Ollama, llm.OpenAICompatConfig{}) // no auth; defaults to localhost:11434
res, err := chat.Complete(ctx, llm.ChatInput{
	Model:    "llama3.1:8b",
	Messages: []llm.ChatMessage{{Role: "user", Content: "Hi"}},
})
```

## Configuration

`llm.OpenAICompatConfig`:

| Field | Type | Required | Notes |
|---|---|---|---|
| `APIKey` | `string` | No | None by default (local server); if set, sent as `Authorization: Bearer <APIKey>`. |
| `BaseURL` | `string` | No* | Overrides the default `http://localhost:11434/v1`. *Required for a non-default host. |
| `Headers` | `map[string]string` | No | Merged onto every request. |

Inject a custom transport with `llm.WithHTTPClient(...)`.

## Auth setup

No auth by default — leave `APIKey` empty and no auth header is sent. Override `BaseURL` for a remote host.

## Chat

### Endpoint

`POST http://localhost:11434/v1/chat/completions` (override `BaseURL` for a non-default host).

### Notes / quirks

- Self-host: model ids are host-local `name:tag` (e.g. `llama3.1:8b`) — a 404 means the model isn't pulled (`ollama pull …`).
- The `/v1` compat layer is documented-partial (drops `tool_choice` / `logit_bias` / `n`; base64-only images).
- `MaxOutputTokens` maps to `max_tokens`. `Reasoning.Effort` maps to `reasoning_effort`.
- Streaming via `chat.Stream(ctx, in)` → `Recv()` until `io.EOF`, then `Close()`. `stream_options.include_usage` is set.

## Embeddings

Supported.

```go
emb, err := llm.NewEmbeddings(llm.Ollama, llm.OpenAICompatConfig{})
out, err := emb.Create(ctx, llm.EmbeddingsInput{Model: "nomic-embed-text", Input: []string{"hello"}})
```

`POST http://localhost:11434/v1/embeddings` with `encoding_format: "float"`; `Dimensions` maps to `dimensions` when set.

## Error mapping & passthrough

- Non-2xx returns a `*llm.ConnectorError` with `ProviderCode` in `rate_limited` / `auth_failed` / `provider_unavailable` / `invalid_request` / `context_length_exceeded` / `content_filtered` / `unknown`.
- `Retry-After` rides on `Cause["retryAfter"]` / `Cause["retryAfterSeconds"]`; raw body on `Cause["raw"]`.
- Sub-baseline request fields go via `Passthrough.Body` / `.Headers` / `.Query`; response extras on `ChatResult.Raw`.
