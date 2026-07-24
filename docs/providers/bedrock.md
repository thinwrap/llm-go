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
	Model:    "anthropic.claude-3-5-sonnet-20241022-v2:0",
	Messages: []llm.ChatMessage{{Role: "user", Content: "Hi"}},
})
```

## Configuration

`llm.BedrockConfig` (implements `ChatConfig`):

| Field | Type | Required | Notes |
|---|---|---|---|
| `Region` | `string` | yes | e.g. `us-east-1`; folded into the SigV4 credential scope and the default host. No environment inference. |
| `AccessKeyID` | `string` | yes | SigV4 credential. |
| `SecretAccessKey` | `string` | yes | SigV4 credential. |
| `SessionToken` | `string` | no | STS temporary-credential token → `X-Amz-Security-Token`. |
| `BaseURL` | `string` | no | Overrides `https://bedrock-runtime.<region>.amazonaws.com` (VPC endpoint / proxy). Must yield a host or `NewChat` errors `invalid_request`. |
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
percent-encoded for the path (e.g. `...-v2:0` → `...-v2%3A0`) so the SigV4
canonical URI matches the wire URL exactly.

### Notes / quirks

- **System messages** → top-level `system: [{text}]`.
- **`Role: "tool"` messages** → `toolResult` blocks inside a `user` turn; consecutive tool results coalesce into one turn.
- **Assistant `ToolCalls`** → `toolUse` blocks; **`Tools`** → `toolConfig.tools[].toolSpec` with the JSON Schema under `inputSchema.json`.
- **`ToolChoice`** → `auto`→`{auto:{}}`, `required`→`{any:{}}`, `function`→`{tool:{name}}`; `none` → `toolChoice` omitted (Converse has no "none").
- **Images** (`ContentParts`) → `{image:{format, source:{bytes}}}` (format derived from `MediaType`).
- **Sampling** → `inferenceConfig` `{maxTokens, temperature, topP, stopSequences}`.
- **`ResponseFormat` is NOT mapped** — use tool-based structured output, or pass raw fields via `Passthrough.Body`.
- **Reasoning is NOT auto-mapped** — reasoning control is model-dependent on Bedrock; pass `additionalModelRequestFields` via `Passthrough.Body`.
- **Prompt caching** — `cachePoint` blocks via `Passthrough.Body`.
- **Streaming** (`chat.Stream`) is a **non-incremental fallback.** Bedrock's `converse-stream` uses AWS's binary event-stream framing (`application/vnd.amazon.eventstream`), not SSE, and this port does not parse it. `Stream` issues one `/converse` call and yields the whole completion as deltas: an optional content delta, one `ToolCallDelta` per tool call, then a terminal delta carrying `FinishReason` + `Usage` + `Raw`. A true binary-event-stream parser is a documented follow-up.
- **`stopReason`** → `FinishReason`: `end_turn`/`stop_sequence`→`stop`, `max_tokens`→`length`, `tool_use`→`tool_calls`, `content_filtered`/`guardrail_intervened`→`content_filter`, else `unknown`. Converse `usage` maps directly; `TotalTokens` falls back to input+output when the vendor omits it.

## Embeddings

No OpenAI-float embeddings surface — `llm.NewEmbeddings(llm.Bedrock, ...)`
returns a `*ConnectorError` with `ProviderCode` `invalid_request`. (Amazon Titan
embeddings are not a Converse/OpenAI-float surface.)

## Error mapping & passthrough

`ConnectorError.ProviderCode`: `401`/`403`→`auth_failed`, `429`→`rate_limited`,
`>=500`→`provider_unavailable`; `400`/`404`/`422`→`invalid_request`, except when
the message mentions "throttl" (→`rate_limited`) or "too long" / "context" /
"token" / "max" (→`context_length_exceeded`); anything else → `unknown`. The raw
decoded body is on `Cause["raw"]`; a `Retry-After` header (when present) on
`Cause["retryAfter"]` / `Cause["retryAfterSeconds"]`. Retrieve with
`errors.As(err, &ce)`. Provider-specific request fields go through
`Passthrough.Body` / `.Headers` / `.Query`; vendor response extras stay in
`ChatResult.Raw`.
