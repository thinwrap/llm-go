# Azure OpenAI (Go)

Chat + embeddings via the shared OpenAI-compatible connector (spec row, not a per-provider type). Switching to/from Azure OpenAI is just the provider const + `Model`; `ChatInput`/`ChatResult` are identical to every other provider.

## Quick start

```go
import "github.com/thinwrap/llm-go"

chat, err := llm.NewChat(llm.AzureOpenAI, llm.OpenAICompatConfig{
	APIKey:  os.Getenv("AZURE_OPENAI_API_KEY"),
	BaseURL: "https://<resource>.openai.azure.com/openai/v1", // required
})
res, err := chat.Complete(ctx, llm.ChatInput{
	Model:    "<your-deployment-name>",
	Messages: []llm.ChatMessage{{Role: "user", Content: "Hi"}},
})
```

## Configuration

`llm.OpenAICompatConfig`:

| Field | Type | Required | Notes |
|---|---|---|---|
| `APIKey` | `string` | Yes | Sent as the `api-key: <APIKey>` header (not Bearer). |
| `BaseURL` | `string` | **Yes** | No public default — construction fails without it. |
| `Headers` | `map[string]string` | No | Merged onto every request (put an Entra bearer token here for OAuth mode). |

Inject a custom transport with `llm.WithHTTPClient(...)`.

## Auth setup

Get the key + resource endpoint from the Azure portal. The connector sends it as the `api-key` header. Microsoft Entra OAuth is also accepted on the v1 GA API — supply that bearer token via `Headers` instead.

## Chat

### Endpoint

`POST <BaseURL>/chat/completions` — `BaseURL` is `https://<resource>.openai.azure.com/openai/v1` (v1 GA API, opt-in from Aug 2025; no `api-version` query needed). Omitting `BaseURL` returns an `invalid_request` `*ConnectorError`.

### Notes / quirks

- **`BaseURL` is required** (no fixed public default).
- `Model` is the per-resource **deployment name**, not a portable global model id.
- `MaxOutputTokens` maps to `max_tokens`. `Reasoning.Effort` maps to `reasoning_effort`.
- Streaming via `chat.Stream(ctx, in)` → `Recv()` until `io.EOF`, then `Close()`. `stream_options.include_usage` is set.

## Embeddings

Supported.

```go
emb, err := llm.NewEmbeddings(llm.AzureOpenAI, llm.OpenAICompatConfig{
	APIKey:  os.Getenv("AZURE_OPENAI_API_KEY"),
	BaseURL: "https://<resource>.openai.azure.com/openai/v1",
})
out, err := emb.Create(ctx, llm.EmbeddingsInput{Model: "<embeddings-deployment>", Input: []string{"hello"}})
```

`POST <BaseURL>/embeddings` with `encoding_format: "float"`; `Dimensions` maps to `dimensions` when set.

## Error mapping & passthrough

- Non-2xx returns a `*llm.ConnectorError` with `ProviderCode` in `rate_limited` / `auth_failed` / `provider_unavailable` / `invalid_request` / `context_length_exceeded` / `content_filtered` / `unknown`.
- `Retry-After` rides on `Cause["retryAfter"]` / `Cause["retryAfterSeconds"]`; raw body on `Cause["raw"]`.
- Sub-baseline request fields (`seed`, penalties, `logit_bias`, `n`, prompt caching) go via `Passthrough`; response extras on `ChatResult.Raw`.
