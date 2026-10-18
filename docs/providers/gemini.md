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
	Model:    "gemini-3.8-flash",
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
- **`Role: "tool"` messages** → `functionResponse` parts inside a `user` turn, keyed by function **NAME**. The name is resolved from the matching tool call of the assistant turn before it (`ToolCallID`→name), else `Name`, else `ToolCallID`, else `"tool"`. Consecutive tool results coalesce into one turn.
- **Tool-call ids** — Gemini's own `functionCall.id`, when it sends one, becomes `ToolCall.ID` and is echoed back as the `functionCall.id` / `functionResponse.id`. When it sends none, the id is synthesized from the call's position (`call_0`, `call_1`, …, the same in stream and non-stream), so parallel calls to one function stay distinct; synthesized ids are never sent back to Gemini.
- **Assistant `ToolCalls`** → `functionCall` parts; **`Tools`** → `[{functionDeclarations: [...]}]` with the JSON Schema in `parametersJsonSchema` (standard JSON Schema: `additionalProperties`, `$ref`, … are accepted, unlike the OpenAPI-subset `parameters`).
- **Thought signatures** — Gemini 3 puts a `thoughtSignature` on the function-call part and rejects a tool loop that drops it (400 "Function call is missing a thought_signature"). Each `ToolCall` (and streamed `ToolCallDelta`) carries it as `ProviderMetadata: {"gemini": {"thoughtSignature": "…"}}`; keep it on the `ToolCall` you send back and it is replayed on its `functionCall` part. If no call of an assistant turn carries one (a hand-built history, or one from another provider), the first `functionCall` gets Google's documented placeholder `skip_thought_signature_validator`. Other providers' keys are ignored and never sent.
- **`ToolChoice`** → `toolConfig.functionCallingConfig.mode`: `auto`→`AUTO`, `required`→`ANY`, `none`→`NONE`, `function`→`ANY` + `allowedFunctionNames`.
- **Images** (`ContentParts`) → `{inlineData:{mimeType, data}}`.
- **Sampling** → `generationConfig` `{maxOutputTokens, temperature, topP, stopSequences}`.
- **`ResponseFormat` IS mapped** (unlike Anthropic/Bedrock): `json_object`→`responseMimeType: application/json`; `json_schema`→ that plus `responseJsonSchema` (when `JSONSchema` is set; standard JSON Schema, not the deprecated OpenAPI-subset `responseSchema`).
- **`Reasoning.Effort`** → `generationConfig.thinkingConfig.thinkingBudget`: `low`=1024, `medium`=8192, `high`=24576 (unset or unknown → 8192; a budget of 0 would turn thinking off).
- **Prompt caching** (`cachedContent`), safety settings, and other model params → `Passthrough.Body`; reasoning CoT (`thought: true` parts) is not normalized into `Content` (available via `Raw`).
- **Streaming** (`chat.Stream`) is real SSE. Each chunk yields concatenated text as one `ContentDelta`, one `ToolCallDelta` per `functionCall` part, and the `FinishReason` on a trailing delta. Gemini repeats the cumulative `usageMetadata` on every chunk, so `Usage` is reported exactly once, on the finish delta (a chunk after the finish adds nothing).
- **`finishReason`** (the google-genai `FinishReason` enum) → `FinishReason`: `STOP`→`stop`, `MAX_TOKENS`/`CONTINUATION`→`length`, `SAFETY`/`RECITATION`/`BLOCKLIST`/`PROHIBITED_CONTENT`/`SPII`/`IMAGE_SAFETY`/`IMAGE_PROHIBITED_CONTENT`/`IMAGE_RECITATION`→`content_filter`, else (`LANGUAGE`, `MALFORMED_FUNCTION_CALL`, `UNEXPECTED_TOOL_CALL`, `TOO_MANY_TOOL_CALLS`, `OTHER`, …) `unknown`; a `functionCall` part forces `tool_calls`. Usage maps from `usageMetadata`: `InputTokens`=`promptTokenCount`+`toolUsePromptTokenCount`, `OutputTokens`=`candidatesTokenCount`+`thoughtsTokenCount` (thinking is billed as output), `TotalTokens`=input+output (what `totalTokenCount` documents).

## Embeddings

No OpenAI-float embeddings surface — `llm.NewEmbeddings(llm.Gemini, ...)` returns
a `*ConnectorError` with `ProviderCode` `invalid_request`.

## Error mapping & passthrough

`ConnectorError.ProviderCode`: `401`/`403`→`auth_failed`, `429` and `402`
(`RESOURCE_EXHAUSTED`, prepay credit depleted)→`rate_limited`, `408` (transient;
Google says to retry it) and `>=500`→`provider_unavailable`, any other
`4xx`→`invalid_request`, except a 400/404/413/422 whose `ErrorInfo` reason is
`API_KEY_INVALID` (→`auth_failed`) or whose message is the input overflow
("… exceeds the maximum number of tokens allowed", →`context_length_exceeded`);
anything else → `unknown`. A mid-stream `{"error": …}` frame keeps the stream's
status (normally 200) and is classified by its `code`, else its google.rpc
`status` name, else `unknown`. Gemini emits **no rate-limit
headers** — retry timing is parsed from the 429 body's `error.details`
`RetryInfo.retryDelay` (e.g. `"12s"`) into `Cause["retryAfterSeconds"]` when a
`Retry-After` header did not already populate it. The raw decoded body is on
`Cause["raw"]`. Retrieve with `errors.As(err, &ce)`. Provider-specific request
fields go through `Passthrough.Body` / `.Headers` / `.Query`; vendor response
extras stay in `ChatResult.Raw`.
