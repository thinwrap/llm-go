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
	Model:    "claude-opus-5-5",
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
| `DefaultMaxTokens` | `int` | no | Used when `ChatInput.MaxOutputTokens` is unset; default `16000` (Anthropic's recommended non-streaming default). Anthropic **requires** `max_tokens`, and thinking counts toward it: current models think by default, so a small cap can end at `length` with no text. It is a ceiling, not a charge. |
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
- **`ToolChoice`** → `auto`→`{type:"auto"}`, `required`→`{type:"any"}`, `none`→`{type:"none"}`, `function`→`{type:"tool", name}`. Claude Opus 5.5, Sonnet 5.5 and Fable 5.1 reject forced tool use (`required` or a named tool) with a 400; that is the vendor's rule, so the error passes through as `invalid_request` and is never emulated.
- **Images** (`ContentParts`) → `{type:"image", source:{type:"base64", media_type, data}}`. Empty text parts are dropped (Anthropic rejects empty text blocks).
- **`Stop`** → `stop_sequences`.
- **`Reasoning.Effort`** → `thinking:{type:"adaptive"}` plus `output_config:{effort}` on every model, with no budget and no model-id matching. An unset effort sends `medium`; any other value is sent as given (so Anthropic's own levels such as `max` work, and an invalid one gets the vendor's 400). `max_tokens` stays what you set (or the default). While `Reasoning` is set, `Temperature` / `TopP` are **omitted**, because thinking does not allow changing them.
- A `Passthrough.Body["output_config"]` replaces the connector's whole `output_config`, effort included (the merge is shallow), so put `effort` in it yourself when you pass one.
- **Models before Claude 4.6** do not support adaptive thinking (verified on Haiku 4.5, which also rejects `output_config.effort`); the 400 passes through as `invalid_request`. For those models leave `Reasoning` unset and send `Passthrough.Body["thinking"] = map[string]any{"type": "enabled", "budget_tokens": N}` (with `MaxOutputTokens` above `N`).
- **`Temperature` / `TopP`** return a 400 on Claude Opus 4.7 and later, every 5.x model and Fable (on Sonnet 5.5 and Haiku 5.5, any non-default value). That passes through as `invalid_request`; leave them unset for those models.
- **`ResponseFormat` is intentionally NOT mapped** — Anthropic has no `response_format`. Use tool-based structured output, supplying the tool via `Tools` or raw fields via `Passthrough.Body`.
- **Reasoning CoT output** — `thinking` blocks are not normalized into `Content`; read them from `ChatResult.Raw`.
- **`ProviderMetadata["anthropic"]`** — `ChatResult.Message.ProviderMetadata` is `{"anthropic": {"content": [...]}}`, the response's `content` array exactly as received (thinking, redacted_thinking, text, tool_use). Copy it onto the assistant `ChatMessage` you send back, and that turn is replayed verbatim, thinking included, as long as its `tool_use` ids (in order) still equal the message's `ToolCalls` ids, each `tool_use.input` still equals that call's parsed `Arguments`, and its text blocks still join to the message's text. Otherwise the turn is rebuilt from `Content` / `ToolCalls` without its thinking, and so loses its earlier reasoning; edited fields always win over the stored blocks. Because Anthropic invalidates every thinking block after an edited or stripped turn (a 400 on accounts that enforce preserved thinking), once one assistant turn is rebuilt — including one with no metadata — every later assistant turn in the request is rebuilt too. Other providers' keys are ignored and never sent.
- **Prompt caching** — pass `cache_control` markers via `Passthrough.Body`. `cache_read_input_tokens` and `cache_creation_input_tokens` are billed input, so they are included in `Usage.InputTokens`.
- **Streaming** (`chat.Stream`) is real Anthropic SSE — genuinely incremental. Text arrives as `ContentDelta`, tool calls as `ToolCallDelta` (concatenate `ArgumentsDelta` per `Index`, which is the tool call's ordinal, not the content-block index); every `message_delta` carries the cumulative `Usage` (counters merged field by field across events), and the one with `stop_reason` also carries `FinishReason` and `ProviderMetadata` rebuilt from the events (the same array a non-stream call returns).
- **`stop_reason`** → `FinishReason`: `end_turn`/`stop_sequence`→`stop`, `max_tokens`/`model_context_window_exceeded`→`length`, `tool_use`→`tool_calls`, `refusal`→`content_filter`, else (including `pause_turn`) `unknown`. Anthropic reports no total token count; `Usage.TotalTokens` is summed.

## Embeddings

No OpenAI-float embeddings surface — `llm.NewEmbeddings(llm.Anthropic, ...)`
returns a `*ConnectorError` with `ProviderCode` `invalid_request`.

## Error mapping & passthrough

`ConnectorError.ProviderCode`: `401`/`403`→`auth_failed`, `429` and `402`
(`billing_error`)→`rate_limited`, `408` and any `5xx` (incl. `529`)→`provider_unavailable`,
`413` (`request_too_large`, the byte limit)→`invalid_request`, any other `4xx`→`invalid_request`,
or `context_length_exceeded` when a 400/404/422 message is a context-window
overflow ("prompt is too long", "exceed context limit"); anything else →
`unknown`. A mid-stream `error` event keeps the stream's status (normally 200)
and is classified by its `error.type` (`unknown` for a type Anthropic has not
documented). The raw decoded body is on
`Cause["raw"]`; a `Retry-After` header (when present) on `Cause["retryAfter"]`
with parsed seconds on `Cause["retryAfterSeconds"]`. Retrieve the error with
`errors.As(err, &ce)`. Provider-specific request fields go through
`Passthrough.Body` / `.Headers` / `.Query`; vendor response extras stay in
`ChatResult.Raw`.
