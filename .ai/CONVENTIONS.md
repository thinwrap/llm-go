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
| 429 / 402 | `rate_limited` (402 is out of credit) |
| 5xx / 408 | `provider_unavailable` (Bedrock also 424) |
| any other 4xx | `invalid_request` (or `context_length_exceeded` / `content_filtered` on a body signal) |
| transport | `provider_unavailable` (status nil) |
| mid-stream frame | the stream's own status; code from the frame's type / code, `unknown` without one |

A `context_length_exceeded` body signal is the vendor's documented overflow
phrase or code, never a loose substring ("context", "token", "max"): a consumer
that trims history on that code loops forever on a misclassified 400.

Build the vendor error with `vendorError(status, code, providerMessage, rawBody, resp.Header)`
so Retry-After surfacing stays uniform. Pre-flight: `invalidReq(msg)`.

## Streaming

`stream()` calls `requireEventStream(resp)` and returns
`newChatStream(resp, mapChunk, terminal)`. `mapChunk(chunk any) ([]*ChatStreamDelta, error)`
maps one decoded SSE event; return an empty slice to skip. Keep stream state in a
per-call struct captured by the closure. A clean EOF is accepted only after a
terminal event (`[DONE]`, a delta with a FinishReason, or `terminal()` reporting
the connector's own end event); otherwise the stream fails with `streamEndedEarlyError`.
Non-2xx before the stream starts → map immediately and return the error
from `stream()`.

## Provider metadata

`ProviderMetadata` maps are keyed by provider id. A connector reads and writes
only its own key, never sends another provider's key, and treats a missing or
malformed value as absent (rebuild from the normalized fields). Values are read
as JSON-decoded shapes (`map[string]any`, `[]any`, `string`), the same shapes
the connector writes; a typed value such as `map[string]string` is treated as
absent.

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
