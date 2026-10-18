# AWS Bedrock (Go)

Chat via the native adapter for Amazon Bedrock's unified **Converse API**.
Requests are signed with **hand-rolled AWS SigV4** (no `aws-sdk`, zero
dependencies — same approach as the notifications SES/SNS connectors). Converse
is structurally different from OpenAI, so this is a full translation layer that
still emits the identical normalized `ChatResult` / `ChatStreamDelta` /
`*ConnectorError`.

## Quick start

```go
import "github.com/thinwrap/llm-go"

chat, err := llm.NewChat(llm.Bedrock, llm.BedrockConfig{
	Region:          "us-east-1",
	AccessKeyID:     os.Getenv("AWS_ACCESS_KEY_ID"),
	SecretAccessKey: os.Getenv("AWS_SECRET_ACCESS_KEY"),
	// SessionToken: os.Getenv("AWS_SESSION_TOKEN"), // for temporary creds
})
res, err := chat.Complete(ctx, llm.ChatInput{
	// The US geo inference-profile id from AWS's Claude Sonnet 5.5 model card;
	// Claude models on Bedrock are served through inference profiles.
	Model:    "us.anthropic.claude-sonnet-5-5",
	Messages: []llm.ChatMessage{{Role: "user", Content: "Hi"}},
})
```

## Configuration

`llm.BedrockConfig` (implements `ChatConfig`):

| Field | Type | Required | Notes |
|---|---|---|---|
| `Region` | `string` | yes | e.g. `us-east-1`; folded into the SigV4 credential scope and the default host, so it must be a single lowercase DNS label (`NewChat` returns `invalid_request` otherwise). No environment inference. |
| `AccessKeyID` | `string` | yes | SigV4 credential. |
| `SecretAccessKey` | `string` | yes | SigV4 credential. |
| `SessionToken` | `string` | no | STS temporary-credential token → `X-Amz-Security-Token`. |
| `BaseURL` | `string` | no | Overrides `https://bedrock-runtime.<region>.amazonaws.com` (VPC endpoint / proxy). Must yield a host or `NewChat` errors `invalid_request`. A path prefix (a gateway mount point such as `/proxy/bedrock`) is kept on the wire and covered by the signature. |
| `DefaultMaxTokens` | `int` | no | Converse `inferenceConfig.maxTokens` when `MaxOutputTokens` is unset; default `4096` (Converse requires `maxTokens`). |
| `Headers` | `map[string]string` | no | Merged onto the request **and folded into the SigV4 signature**. |

## Auth setup

Uses standard AWS SigV4 (service `bedrock`) over the stdlib — no AWS SDK. Supply
`AccessKeyID` / `SecretAccessKey` (and `SessionToken` for temporary credentials).
The body is serialized once and those exact bytes are signed together with the
canonical path so URL, body, and signature all agree.

## Chat

### Endpoint

`POST <origin>/model/{modelId}/converse` where `origin` defaults to
`https://bedrock-runtime.<region>.amazonaws.com`. The model id is
percent-encoded for the path (e.g. `...-v1:0` → `...-v1%3A0`) so the SigV4
canonical URI matches the wire URL exactly.

### Notes / quirks

- **System messages** → top-level `system: [{text}]`.
- **`Role: "tool"` messages** → `toolResult` blocks inside a `user` turn; consecutive tool results coalesce into one turn.
- **Assistant `ToolCalls`** → `toolUse` blocks; **`Tools`** → `toolConfig.tools[].toolSpec` with the JSON Schema under `inputSchema.json`.
- **`ToolChoice`** → `auto`→`{auto:{}}`, `required`→`{any:{}}`, `function`→`{tool:{name}}`. Converse has no "none", so `none` omits `toolConfig` entirely (sending the tools with no choice would let the model call one). Converse rejects a history holding `toolUse` / `toolResult` blocks without a `toolConfig`; that 400 passes through as `invalid_request`, so use `auto` there.
- **Images** (`ContentParts`) → `{image:{format, source:{bytes}}}` (format derived from `MediaType`). Empty text parts are dropped (Converse rejects blank text blocks).
- **Sampling** → `inferenceConfig` `{maxTokens, temperature, topP, stopSequences}`.
- **`ResponseFormat` is NOT mapped** — use tool-based structured output, or pass raw fields via `Passthrough.Body`.
- **Reasoning is NOT auto-mapped** — reasoning control is model-dependent on Bedrock; pass `additionalModelRequestFields` via `Passthrough.Body`.
- **Prompt caching** — `cachePoint` blocks via `Passthrough.Body`. `cacheReadInputTokens` and `cacheWriteInputTokens` are billed input, so they are included in `Usage.InputTokens`.
- **Streaming** (`chat.Stream`) is a **non-incremental fallback.** Bedrock's `converse-stream` uses AWS's binary event-stream framing (`application/vnd.amazon.eventstream`), not SSE, and this port does not parse it. `Stream` issues one `/converse` call and yields the whole completion as deltas: an optional content delta, one `ToolCallDelta` per tool call, then a terminal delta carrying `FinishReason` + `Usage` + `Raw`. A response without a `stopReason` is a `provider_unavailable` error rather than a stream that looks complete. A true binary-event-stream parser is a documented follow-up.
- **`stopReason`** → `FinishReason`: `end_turn`/`stop_sequence`→`stop`, `max_tokens`/`model_context_window_exceeded`→`length`, `tool_use`→`tool_calls`, `content_filtered`/`guardrail_intervened`→`content_filter`, else (`malformed_model_output`, `malformed_tool_use`) `unknown`. `TotalTokens` is input (cache included) + output.

## Embeddings

No OpenAI-float embeddings surface — `llm.NewEmbeddings(llm.Bedrock, ...)`
returns a `*ConnectorError` with `ProviderCode` `invalid_request`. (Amazon Titan
embeddings are not a Converse/OpenAI-float surface.)

## Error mapping & passthrough

`ConnectorError.ProviderCode`: `401`/`403`→`auth_failed`, `429` and `402`→`rate_limited`,
`>=500`, `408` (`ModelTimeoutException`) and `424` (`ModelErrorException`)→`provider_unavailable`;
any other `4xx`→`invalid_request`, except when a 400/404/413/422 message mentions
"throttl" (→`rate_limited`) or is an input overflow. Bedrock relays the model
provider's text, so the overflow phrases are "input is too long", "prompt is too
long", "too many input tokens", "maximum context length" and "exceed context
limit" (→`context_length_exceeded`); anything else → `unknown`. The raw
decoded body is on `Cause["raw"]`; a `Retry-After` header (when present) on
`Cause["retryAfter"]` / `Cause["retryAfterSeconds"]`. Retrieve with
`errors.As(err, &ce)`. Provider-specific request fields go through
`Passthrough.Body` / `.Headers` / `.Query`; vendor response extras stay in
`ChatResult.Raw`.
