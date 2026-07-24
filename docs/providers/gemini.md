# Google Gemini (Go)

Chat via the native adapter for the Google Gemini (**AI Studio**)
`generateContent` API. Gemini uses `contents[]`/`parts[]` (roles `user`/`model`),
a top-level `systemInstruction`, and `functionDeclarations`, so this is a full
translation layer that still emits the identical normalized `ChatResult` /
`ChatStreamDelta` / `*ConnectorError`. Unlike the Bedrock fallback, Gemini
streaming is **real SSE** — genuinely incremental.

> This is the AI Studio Gemini API (`gemini` id). Vertex AI Gemini is a separate
> product with different auth (GCP OAuth/ADC) and endpoints — not this connector.

## Quick start

```go
import "github.com/thinwrap/llm-go"

chat, err := llm.NewChat(llm.Gemini, llm.GeminiConfig{
	APIKey: os.Getenv("GEMINI_API_KEY"),
})
res, err := chat.Complete(ctx, llm.ChatInput{
	Model:    "gemini-2.5-flash",
	Messages: []llm.ChatMessage{{Role: "user", Content: "Hi"}},
})
```

## Configuration

`llm.GeminiConfig` (implements `ChatConfig`):

| Field | Type | Required | Notes |
|---|---|---|---|
| `APIKey` | `string` | yes | Sent as the `x-goog-api-key` header. |
| `BaseURL` | `string` | no | Overrides `https://generativelanguage.googleapis.com/v1beta`. Trailing slash trimmed. |
| `DefaultMaxTokens` | `int` | no | `generationConfig.maxOutputTokens` when `MaxOutputTokens` is unset. No hard default — if both are zero, `maxOutputTokens` is simply omitted. |
| `Headers` | `map[string]string` | no | Merged onto every request. |

## Auth setup

Create an API key in Google AI Studio. It is sent as the `x-goog-api-key`
header (the key is not placed in the query string). No environment inference —
pass `APIKey` explicitly.

## Chat

### Endpoint

`POST <baseURL>/models/{model}:generateContent` (non-streaming) and
`POST <baseURL>/models/{model}:streamGenerateContent?alt=sse` (streaming). The
model is path-escaped. Default base URL
`https://generativelanguage.googleapis.com/v1beta`.

### Notes / quirks

- **System messages** → top-level `systemInstruction` (multiple concatenated with blank lines).
- **Roles** → `assistant`→`model`, everything else → `user`.
- **`Role: "tool"` messages** → `functionResponse` parts inside a `user` turn, keyed by function **NAME** (Gemini has no tool-call ids). The name is resolved from the matching prior assistant tool call (`ToolCallID`→name), else `Name`, else `ToolCallID`, else `"tool"`. Consecutive tool results coalesce into one turn.
- **Assistant `ToolCalls`** → `functionCall` parts; **`Tools`** → `[{functionDeclarations: [...]}]`.
- **`ToolChoice`** → `toolConfig.functionCallingConfig.mode`: `auto`→`AUTO`, `required`→`ANY`, `none`→`NONE`, `function`→`ANY` + `allowedFunctionNames`.
- **Images** (`ContentParts`) → `{inlineData:{mimeType, data}}`.
- **Sampling** → `generationConfig` `{maxOutputTokens, temperature, topP, stopSequences}`.
- **`ResponseFormat` IS mapped** (unlike Anthropic/Bedrock): `json_object`→`responseMimeType: application/json`; `json_schema`→ that plus `responseSchema` (when `JSONSchema` is set).
- **`Reasoning.Effort`** → `generationConfig.thinkingConfig.thinkingBudget`: `low`=1024, `medium`=8192, `high`=24576 (any other value → budget 0).
- **Prompt caching** (`cachedContent`), safety settings, and other model params → `Passthrough.Body`; reasoning CoT / `thought_signature` round-trip are not persisted by the wrapper (available via `Raw`).
- **Streaming** (`chat.Stream`) is real SSE. Each chunk collapses to one delta: concatenated text as `ContentDelta`, a `functionCall` as `ToolCallDelta`, plus the trailing `FinishReason` / `Usage` when present.
- **`finishReason`** → `FinishReason`: `STOP`→`stop`, `MAX_TOKENS`→`length`, `SAFETY`/`RECITATION`/`BLOCKLIST`/`PROHIBITED_CONTENT`/`SPII`→`content_filter`, else `unknown`; a `functionCall` part forces `tool_calls`. Usage maps from `usageMetadata` (`promptTokenCount`/`candidatesTokenCount`/`totalTokenCount`), summing the total when omitted.

## Embeddings

No OpenAI-float embeddings surface — `llm.NewEmbeddings(llm.Gemini, ...)` returns
a `*ConnectorError` with `ProviderCode` `invalid_request`.

## Error mapping & passthrough

`ConnectorError.ProviderCode`: `401`/`403`→`auth_failed`, `429`→`rate_limited`,
`>=500`→`provider_unavailable`, `400`/`404`/`422`→`invalid_request` (or
`context_length_exceeded` when the message mentions "too long" / "context" /
"token count"); anything else → `unknown`. Gemini emits **no rate-limit
headers** — retry timing is parsed from the 429 body's `error.details`
`RetryInfo.retryDelay` (e.g. `"12s"`) into `Cause["retryAfterSeconds"]` when a
`Retry-After` header did not already populate it. The raw decoded body is on
`Cause["raw"]`. Retrieve with `errors.As(err, &ce)`. Provider-specific request
fields go through `Passthrough.Body` / `.Headers` / `.Query`; vendor response
extras stay in `ChatResult.Raw`.
