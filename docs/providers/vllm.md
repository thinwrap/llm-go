# vLLM (self-host) (Go)

Chat + embeddings via the shared OpenAI-compatible connector (spec row, not a per-provider type). Switching to/from vLLM is just the provider const + `Model`; `ChatInput`/`ChatResult` are identical to every other provider.

## Quick start

```go
import "github.com/thinwrap/llm-go"

chat, err := llm.NewChat(llm.VLLM, llm.OpenAICompatConfig{}) // APIKey optional; defaults to localhost:8000
res, err := chat.Complete(ctx, llm.ChatInput{
	Model:    "<served-model-name>",
	Messages: []llm.ChatMessage{{Role: "user", Content: "Hi"}},
})
```

## Configuration

`llm.OpenAICompatConfig`:

| Field | Type | Required | Notes |
|---|---|---|---|
| `APIKey` | `string` | No | Optional — sent as `Authorization: Bearer <APIKey>` only if set (omit if the server has no `--api-key`). |
| `BaseURL` | `string` | No* | Overrides the default `http://localhost:8000/v1`. *Required for a remote/non-default host. |
| `Headers` | `map[string]string` | No | Merged onto every request. |

Inject a custom transport with `llm.WithHTTPClient(...)`.

## Auth setup

Open by default. If the server was launched with `--api-key`, pass that value as `APIKey`; it is sent as `Authorization: Bearer <APIKey>`. When `APIKey` is empty, no auth header is sent.

## Chat

### Endpoint

`POST http://localhost:8000/v1/chat/completions` (override `BaseURL` for a remote server).

### Notes / quirks

- Self-host: the OpenAI-compatible server is vLLM's first-class interface.
- Model ids are operator-defined (the HF repo id or `--served-model-name`).
- `ToolChoice` `"auto"` requires the server launched with `--enable-auto-tool-choice` + a model-matched `--tool-call-parser`.
- `MaxOutputTokens` maps to `max_tokens`. `Reasoning.Effort` maps to `reasoning_effort`.
- Streaming via `chat.Stream(ctx, in)` → `Recv()` until `io.EOF`, then `Close()`. `stream_options.include_usage` is set.

## Embeddings

Supported (the model must be served with an embeddings task, e.g. `--task embed`).

```go
emb, err := llm.NewEmbeddings(llm.VLLM, llm.OpenAICompatConfig{})
out, err := emb.Create(ctx, llm.EmbeddingsInput{Model: "<served-embedding-model>", Input: []string{"hello"}})
```

`POST http://localhost:8000/v1/embeddings` with `encoding_format: "float"`; `Dimensions` maps to `dimensions` when set.

## Error mapping & passthrough

- Non-2xx returns a `*llm.ConnectorError` with `ProviderCode` in `rate_limited` / `auth_failed` / `provider_unavailable` / `invalid_request` / `context_length_exceeded` / `content_filtered` / `unknown`.
- `Retry-After` rides on `Cause["retryAfter"]` / `Cause["retryAfterSeconds"]`; raw body on `Cause["raw"]`.
- Sub-baseline request fields go via `Passthrough.Body` / `.Headers` / `.Query`; response extras on `ChatResult.Raw`.
