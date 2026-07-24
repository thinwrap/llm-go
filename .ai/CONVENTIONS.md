# Conventions — thinwrap/llm (Go)

## File layout

- One file per **shared concern**: `provider.go` (ids), `errors.go`
  (`ConnectorError` + `ProviderCode` + `vendorError`), `config.go` (all config
  structs), `chat.go` (chat DTOs + `Chat` facade + `ChatStream` + native
  registry), `embeddings.go` (embeddings DTOs + `Embeddings` facade),
  `passthrough.go`, `httpclient.go`, `base.go` (transport), `json.go`,
  `jsonpath.go`, `sse.go` (SSE + `newChatStream`), `awssigv4.go`, `options.go`,
  `spec.go` (the OpenAI-compat `specs` registry).
- Shared connectors: `openai_compat.go` (chat), `openai_compat_embeddings.go`.
- One file per **native**: `<id>.go` holds its connector, config→connector
  factory, `init()` registration, and private helpers (`<id>MapCode`, wire
  structs). Tests are `<id>_test.go`.

## Naming

- Exported public surface: `Chat`, `Embeddings`, `NewChat`, `NewEmbeddings`, the
  DTO structs (`ChatInput`, `ChatMessage`, `ChatResult`, `ChatStreamDelta`,
  `ToolCall`, `ToolDefinition`, `ToolChoice`, `ResponseFormat`,
  `ReasoningOptions`, `Usage`, `EmbeddingsInput`/`Result`/`Usage`,
  `AssistantMessage`, `ContentPart`, `TextPart`/`ImagePart`), `Passthrough`,
  `ConnectorError`, `ProviderCode` + `Code*`, `ProviderID` + provider consts,
  `ChatConfig`, the `<Name>Config` structs, `HTTPClient`, `WithHTTPClient`,
  `ChatStream`.
- Everything else is unexported: connectors, `baseConnector`, `chatConnector` /
  `embeddingsConnector`, `specs`, the SSE machinery, all vendor helpers.
- **Single-package collision rule.** Every package-level identifier a native file
  introduces MUST be provider-prefixed (`anthropicDefaultBaseURL`,
  `bedrockMapCode`, `geminiMapParts`). Connector type `<id>Connector`.

## Error mapping

Each connector has a private `<id>MapCode(status int, ...) ProviderCode` porting
the sibling truth table:

| HTTP | ProviderCode |
|---|---|
| 401 / 403 | `auth_failed` |
| 429 | `rate_limited` |
| 5xx | `provider_unavailable` |
| 400 / 404 / 422 | `invalid_request` (or `context_length_exceeded` / `content_filtered` on a body signal) |
| transport | `provider_unavailable` (status nil) |

Build the vendor error with `vendorError(status, code, providerMessage, rawBody, resp.Header)`
so Retry-After surfacing stays uniform. Pre-flight: `invalidReq(msg)`.

## Streaming

`stream()` returns `newChatStream(resp, mapChunk)`. `mapChunk(chunk any) (*ChatStreamDelta, error)`
maps one decoded SSE event; return `(nil, nil)` to skip. Keep stream state in the
closure. Non-2xx before the stream starts → map immediately and return the error
from `stream()`.

## Testing

- White-box (`package llm`), standard `testing`.
- Inject the shared `fakeClient` (in `testutil_test.go`) via `WithHTTPClient`;
  `resp`/`respH` for JSON, `sseResp` for streams.
- Assert the captured request (URL/headers/body), the normalized result
  (content, finishReason, usage, model), streaming accumulation, and
  `*ConnectorError` (`ProviderCode`, `StatusCode`, `ProviderMessage`) via
  `errors.As`. Never assert a top-level retry field.

## Go / build

- `go build`, `go vet`, `gofmt -l .` (empty), `go test -cover` (≥80%).
- Go ≥ 1.18; no third-party imports. Avoid 1.21+ builtins and `slices`/`maps`.
