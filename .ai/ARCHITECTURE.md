# Architecture — thinwrap/llm (Go)

## Facade → dispatch → connector

```
llm.NewChat(OpenAI, OpenAICompatConfig{...})   →  *Chat facade
   .Complete(ctx, in) / .Stream(ctx, in)            │  specs[id] → openAICompatConnector
                                                    ▼        (or nativeChatRegistry[id] → native)
                              connector (embeds baseConnector)
                                 .complete/.stream → sendPostJSON → *http.Client → Vendor
```

- **One `Chat` facade** (`Complete` + pull-based `Stream`) and a separate
  `Embeddings` facade (`Create`). Both dispatch by provider id.
- **Two dispatch paths in `NewChat`:** a native registered in
  `nativeChatRegistry` (populated by each native file's `init()`), else a row in
  the `specs` map → the shared `openAICompatConnector`. An unknown id → a typed
  `invalid_request` error.
- **`baseConnector`** centralizes the HTTP seam: the injectable `HTTPClient`
  (default non-redirect-following), `sendPostJSON` / `post` (returning the raw
  `*http.Response` so streams read incrementally), and transport-error →
  `provider_unavailable`. No JSON parsing, no error mapping.

## Two connector tiers

1. **OpenAI-compat (`openai_compat.go`, `openai_compat_embeddings.go`).** 15
   providers share it; `spec.go` parameterizes base URL, auth headers, and quirk
   flags. Chat builds `/chat/completions`; embeddings builds `/embeddings`.
2. **Native adapters (`anthropic.go`, `bedrock.go`, `gemini.go`).** Full wire
   translation to Messages / Converse / generateContent, emitting the identical
   normalized shapes. Self-register via `init()`.

## Streaming

`Chat.Stream` returns a `*ChatStream` — a pull-based reader whose `Recv` returns
the next `*ChatStreamDelta` or `io.EOF`. `sse.go` provides `newSSEScanner` (a
`bufio`-based SSE event reader, bounded to 16 MB per event, tolerant of `\n\n`
and `\r\n\r\n`, stopping at `[DONE]`), `requireEventStream` (a 2xx that is not
`text/event-stream` is an error), and `newChatStream(resp, mapChunk, terminal)`,
where `mapChunk` maps one decoded vendor chunk to zero or more deltas. A clean EOF
before the provider's terminal event is an error, not the end of the answer.
Stateful streams (Anthropic tracks input tokens across events) capture state in
the `mapChunk` closure.

## Invariants

- **Canonical shape = OpenAI Chat Completions.** Every connector emits the same
  `ChatInput`/`ChatResult`/`ChatStreamDelta`/`*ConnectorError`. That parity is the product.
- **≥90% baseline coverage.** Sub-baseline / provider-specific request fields flow
  through `Passthrough` (never emulated); vendor response extras live in `Raw`.
  The one deliberate sub-90% carve-out is `Reasoning.Effort`.
- **Per-connector locality.** Wire translation + `<id>MapCode` live in the
  connector; `vendorError` (Retry-After surfacing), JSON helpers, `signAWSv4`, and
  the SSE machinery are building blocks connectors call.
- **`ProviderCode`** — 7 values (`rate_limited`, `auth_failed`,
  `provider_unavailable`, `invalid_request`, `context_length_exceeded`,
  `content_filtered`, `unknown`), byte-identical across siblings. No top-level
  retry field.
- **Stateless / zero-dep.** No caching, retries, or conversation memory. SigV4 and
  SSE are hand-rolled on the standard library.
- **Go idioms:** errors are values (`*ConnectorError` via `errors.As`);
  `context.Context` threaded; streams are `*ChatStream` with `Recv`/`Close`.
