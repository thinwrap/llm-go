# Anthropic (Go)

Chat via the native Anthropic **Messages API** adapter (Claude models). Anthropic
does not speak OpenAI's Chat Completions shape, so this is a full translation
layer that still emits the identical normalized `ChatResult` / `ChatStreamDelta`
/ `*ConnectorError` — switching to/from `anthropic` is only the id + `Model`.

## Quick start

```go
import "github.com/thinwrap/llm-go"

chat, err := llm.NewChat(llm.Anthropic, llm.AnthropicConfig{
	APIKey: os.Getenv("ANTHROPIC_API_KEY"),
})
res, err := chat.Complete(ctx, llm.ChatInput{
	Model:    "claude-sonnet-4-5",
	Messages: []llm.ChatMessage{{Role: "user", Content: "Hi"}},
})
```

## Configuration

`llm.AnthropicConfig` (implements `ChatConfig`):

| Field | Type | Required | Notes |
|---|---|---|---|
| `APIKey` | `string` | yes | Sent as `x-api-key` (not `Authorization: Bearer`). |
| `BaseURL` | `string` | no | Overrides `https://api.anthropic.com/v1`. Trailing slash trimmed; `/messages` is appended. |
| `AnthropicVersion` | `string` | no | `anthropic-version` header; default `2023-06-01`. |
| `DefaultMaxTokens` | `int` | no | Used when `ChatInput.MaxOutputTokens` is unset; default `4096`. Anthropic **requires** `max_tokens`. |
| `Headers` | `map[string]string` | no | Merged onto every request. |

## Auth setup

Create an API key in the Anthropic Console. It is sent as the `x-api-key` header
along with `anthropic-version`. No environment inference — pass `APIKey`
explicitly (or read it from your own env as above).

## Chat

### Endpoint

`POST <baseURL>/messages` (default `https://api.anthropic.com/v1/messages`).

### Notes / quirks

- **System messages** → hoisted into the top-level `system` field (multiple are concatenated with blank lines).
- **`Role: "tool"` messages** → carried inside a `user` turn as `tool_result` blocks; consecutive tool results coalesce into one turn.
- **Assistant `ToolCalls`** → `tool_use` blocks; **`Tools`** → `{name, description, input_schema}` (JSON Schema goes directly in `input_schema`).
- **`ToolChoice`** → `auto`→`{type:"auto"}`, `required`→`{type:"any"}`, `none`→`{type:"none"}`, `function`→`{type:"tool", name}`.
- **Images** (`ContentParts`) → `{type:"image", source:{type:"base64", media_type, data}}`.
- **`Stop`** → `stop_sequences`.
- **`Reasoning.Effort`** → `thinking:{type:"enabled", budget_tokens}` with budget `low`=1024, `medium`/unset=4096, `high`=12288. While thinking is enabled, `Temperature` / `TopP` are **omitted** (Anthropic requires temperature=1 and disallows top_p/top_k with extended thinking), and `max_tokens` is bumped to `budget + 4096` if it would otherwise be ≤ the budget.
- **`ResponseFormat` is intentionally NOT mapped** — Anthropic has no `response_format`. Use tool-based structured output, supplying the tool via `Tools` or raw fields via `Passthrough.Body`.
- **Reasoning CoT output** — `thinking` blocks are not normalized; read them from `ChatResult.Raw`.
- **Prompt caching** — pass `cache_control` markers via `Passthrough.Body`; cached-token counts appear in `Raw`.
- **Streaming** (`chat.Stream`) is real Anthropic SSE — genuinely incremental. Text arrives as `ContentDelta`, tool calls as `ToolCallDelta` (concatenate `ArgumentsDelta` per `Index`); the terminal `message_delta` carries `FinishReason` + `Usage`.
- **`stop_reason`** → `FinishReason`: `end_turn`/`stop_sequence`→`stop`, `max_tokens`→`length`, `tool_use`→`tool_calls`, `refusal`→`content_filter`, else `unknown`. Anthropic reports no total token count; `Usage.TotalTokens` is summed.

## Embeddings

No OpenAI-float embeddings surface — `llm.NewEmbeddings(llm.Anthropic, ...)`
returns a `*ConnectorError` with `ProviderCode` `invalid_request`.

## Error mapping & passthrough

`ConnectorError.ProviderCode`: `401`/`403`→`auth_failed`, `429`→`rate_limited`,
`500`/`502`/`503`/`529`→`provider_unavailable`, `400`/`404`/`413`/`422`→`invalid_request`
(or `context_length_exceeded` when the message mentions "too long" / "context" /
"max_tokens"); anything else → `unknown`. The raw decoded body is on
`Cause["raw"]`; a `Retry-After` header (when present) on `Cause["retryAfter"]`
with parsed seconds on `Cause["retryAfterSeconds"]`. Retrieve the error with
`errors.As(err, &ce)`. Provider-specific request fields go through
`Passthrough.Body` / `.Headers` / `.Query`; vendor response extras stay in
`ChatResult.Raw`.
