# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [1.1.0] — 2026-10-18

### Added

- `ProviderMetadata` on `ToolCall`, `ChatMessage`, `AssistantMessage`, `ToolCallDelta` and `ChatStreamDelta` carries opaque per-provider state across turns.

### Changed

- **Breaking:** `ToolCall`, `ToolCallDelta` and `ChatStreamDelta` hold a map (`ProviderMetadata`), so they can no longer be compared with `==` or used as map keys.
- **Breaking:** Anthropic `Reasoning` sends adaptive thinking plus `output_config.effort` instead of a `budget_tokens` budget; models before Claude 4.6 (e.g. Haiku 4.5) now return the vendor's 400 and take budget thinking via `Passthrough.Body`.
- **Breaking:** Bedrock `ToolChoice` `none` omits `toolConfig` instead of sending the tools, so a history with tool blocks gets Bedrock's 400.
- **Breaking:** OpenAI and Azure OpenAI send `MaxOutputTokens` as `max_completion_tokens`, which Azure api-versions older than `2024-09-01-preview` / `2024-10-21` do not accept.
- Anthropic's default `max_tokens` is 16000 instead of 4096, since thinking counts toward it.
- An unset Anthropic `Reasoning.Effort` sends `medium`; any other value is sent as given.
- Usage counts Anthropic and Bedrock cache tokens and Gemini tool-use prompt tokens as input, Gemini thinking tokens as output, and totals input + output.
- Gemini synthetic tool-call ids are index-based (`call_0`, `call_1`, …) and Gemini's own ids are echoed back.
- HTTP `402` maps to `rate_limited` and `408` to `provider_unavailable` on every connector; any other unlisted `4xx` maps to `invalid_request`.
- In-body and mid-stream errors carry the HTTP status they arrived with and are classified by their own type or code (`unknown` without one).

### Fixed

- Gemini tool loops on Gemini 3 replay each call's `thoughtSignature` (or Google's placeholder) instead of failing with a 400.
- Anthropic tool loops replay the turn's thinking blocks verbatim while they still match the message, tool inputs included.
- Once an Anthropic assistant turn is rebuilt, every later turn is rebuilt too, so no thinking block follows an edited turn.
- Gemini tools send `parametersJsonSchema` and JSON-schema output sends `responseJsonSchema`, so `additionalProperties` / `$ref` no longer 400.
- Gemini streams report the cumulative usage exactly once, on the finish delta.
- Anthropic streams merge usage counters field by field and attach the content array only to the `stop_reason` delta.
- Gemini `API_KEY_INVALID` maps to `auth_failed`.
- Ordinary 400s that mention "context", "token" or "max" are no longer reported as `context_length_exceeded` (Anthropic, Gemini, Bedrock).
- OpenAI-compatible errors with code `context_length_exceeded` map to `context_length_exceeded`, and `content_policy` / moderation errors to `content_filtered`.
- Embeddings in-body errors with a named code (`insufficient_quota`, …) are classified like chat errors.
- Anthropic `model_context_window_exceeded`, Bedrock `model_context_window_exceeded` and Gemini `CONTINUATION` finish as `length`; more Gemini safety reasons finish as `content_filter`.
- Bedrock `408` / `424` map to `provider_unavailable`, and `413` maps to `invalid_request` on every connector.
- Bedrock signs a `BaseURL` path prefix, so gateways mounted under a path no longer fail with `SignatureDoesNotMatch`.
- Mistral embeddings send `Dimensions` as `output_dimension`.
- Embeddings honour `Passthrough.Query`.
- A stream with a non-SSE 2xx body, or one that ends before the provider's terminal event, returns a `provider_unavailable` error instead of ending cleanly.
- Bedrock rejects a `Region` that is not a single DNS label instead of building the host from it.
- A 2xx whose body read fails returns a transport error instead of an "empty body" error.
- An unsupported `ContentPart.Type` returns `invalid_request` instead of being sent as an empty image.
- Empty text parts are dropped for Anthropic and Bedrock, which reject empty text blocks.
- An empty `Reasoning.Effort` is no longer sent as `reasoning_effort: ""`.
- `Retry-After` accepts only whole seconds or an HTTP-date; a sign or a fraction is ignored.
- A 2xx carrying an `error` (OpenRouter's mid-generation failures) or missing its required payload raises `*ConnectorError` instead of an empty success.
- Embeddings reject a response whose vector count or cells don't match the input instead of misaligning or zero-filling.
- Mistral reasoning responses (array-shaped `content`) yield their answer text instead of an empty string.

## [1.0.1] — 2026-08-08

### Changed

- `HTTPClient` documents its contract: a non-2xx must be RETURNED, never reported as an error.

## [1.0.0] — 2026-07-24

### Added

- Initial Go port of `@thinwrap/llm` and `thinwrap/llm`: `Chat` (`Complete` + pull-based
  `Stream`) and `Embeddings` facades over 18 providers, with a typed `*ConnectorError`.
- Fifteen OpenAI-compatible providers share one connector driven by a spec registry; Anthropic,
  Bedrock and Gemini have bespoke wire adapters emitting identical shapes.
- Zero third-party dependencies: Bedrock SigV4 hand-rolled on the standard library, SSE parsed
  on `bufio`.
