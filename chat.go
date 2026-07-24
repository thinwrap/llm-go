package llm

import "context"

// ---- normalized chat types (identical surface across every provider) ----

// ChatRole is one of "system", "user", "assistant", "tool".
type ChatRole = string

// ContentPart is one part of a multimodal message. Type is "text" or "image".
type ContentPart struct {
	Type string
	// Text is set when Type == "text".
	Text string
	// Base64 is the raw base64 image payload (no data: prefix) when Type == "image".
	Base64 string
	// MediaType e.g. "image/png" when Type == "image".
	MediaType string
}

// TextPart / ImagePart constructors for ergonomics.
func TextPart(text string) ContentPart { return ContentPart{Type: "text", Text: text} }
func ImagePart(base64, mediaType string) ContentPart {
	return ContentPart{Type: "image", Base64: base64, MediaType: mediaType}
}

// ToolCall is a function call the assistant emitted.
type ToolCall struct {
	ID   string
	Name string
	// Arguments is a JSON-encoded arguments string.
	Arguments string
}

// ChatMessage is one turn. Use Content for plain text, or ContentParts for
// multimodal (ContentParts wins when non-empty).
type ChatMessage struct {
	Role         ChatRole
	Content      string
	ContentParts []ContentPart
	// ToolCalls is present on assistant turns that invoked tools.
	ToolCalls []ToolCall
	// ToolCallID is required on role:"tool" messages.
	ToolCallID string
	// Name is an optional author / tool name.
	Name string
}

// ToolDefinition describes a callable function (type is implicitly "function").
type ToolDefinition struct {
	Name        string
	Description string
	// Parameters is a JSON Schema object describing the arguments.
	Parameters map[string]any
}

// ToolChoice controls tool selection. Mode is "auto", "none", "required", or
// "function" (with FunctionName set).
type ToolChoice struct {
	Mode         string
	FunctionName string
}

// ResponseFormat controls structured output. Type is "text", "json_object", or
// "json_schema" (with JSONSchemaName + JSONSchema set).
type ResponseFormat struct {
	Type           string
	JSONSchemaName string
	JSONSchema     map[string]any
}

// ReasoningOptions is the minimal normalized reasoning control. Effort is "low",
// "medium", or "high". CoT output is not normalized — read it from ChatResult.Raw.
type ReasoningOptions struct {
	Effort string
}

// ChatInput is the normalized request. Pointer fields distinguish unset from zero.
type ChatInput struct {
	Model           string
	Messages        []ChatMessage
	Tools           []ToolDefinition
	ToolChoice      *ToolChoice
	ResponseFormat  *ResponseFormat
	Temperature     *float64
	TopP            *float64
	MaxOutputTokens *int
	Stop            []string
	Reasoning       *ReasoningOptions
	Passthrough     *Passthrough
}

// FinishReason is one of "stop", "length", "tool_calls", "content_filter", "unknown".
type FinishReason = string

// Usage is normalized token accounting.
type Usage struct {
	InputTokens  int
	OutputTokens int
	TotalTokens  int
}

// AssistantMessage is the assistant turn on a ChatResult.
type AssistantMessage struct {
	Role      ChatRole
	Content   string
	ToolCalls []ToolCall
}

// ChatResult is the normalized non-streaming completion.
type ChatResult struct {
	Message      AssistantMessage
	FinishReason FinishReason
	Usage        Usage
	// Model is the model id the provider reported serving.
	Model string
	// Raw is the verbatim decoded vendor response body.
	Raw any
}

// ToolCallDelta is an incremental tool-call fragment (concatenate ArgumentsDelta per Index).
type ToolCallDelta struct {
	Index          int
	ID             string
	FunctionName   string
	ArgumentsDelta string
}

// ChatStreamDelta is one incremental streaming chunk.
type ChatStreamDelta struct {
	ContentDelta  string
	ToolCallDelta *ToolCallDelta
	FinishReason  FinishReason
	Usage         *Usage
	Raw           any
}

// ---- dispatch ----

// ChatConfig is implemented by every chat-provider config value
// (OpenAICompatConfig for the 15 first-class providers; AnthropicConfig /
// BedrockConfig / GeminiConfig for the natives).
type ChatConfig interface{ isChatConfig() }

type chatConnector interface {
	complete(ctx context.Context, in ChatInput) (*ChatResult, error)
	stream(ctx context.Context, in ChatInput) (*ChatStream, error)
}

// nativeChatFactory builds a native (non-OpenAI-compat) chat connector. Native
// provider files register one in their init().
type nativeChatFactory func(cfg ChatConfig, o facadeOptions) (chatConnector, error)

var nativeChatRegistry = map[ProviderID]nativeChatFactory{}

func registerNativeChat(id ProviderID, f nativeChatFactory) { nativeChatRegistry[id] = f }

// Chat is the chat-completion facade. Construct it with NewChat.
type Chat struct {
	id        ProviderID
	connector chatConnector
}

// NewChat builds a chat facade for the given provider id. For the 15 first-class
// providers pass an OpenAICompatConfig; for a native pass its config. Switching
// vendor is a change of id + model only; ChatInput/ChatResult are identical.
func NewChat(id ProviderID, cfg ChatConfig, opts ...Option) (*Chat, error) {
	o := applyOptions(opts)
	if factory, ok := nativeChatRegistry[id]; ok {
		c, err := factory(cfg, o)
		if err != nil {
			return nil, err
		}
		return &Chat{id: id, connector: c}, nil
	}
	spec, ok := specs[id]
	if !ok {
		return nil, invalidReq("Unsupported LLM provider: " + string(id))
	}
	compatCfg, ok := cfg.(OpenAICompatConfig)
	if !ok {
		return nil, invalidReq("Provider '" + string(id) + "' requires an OpenAICompatConfig")
	}
	c, err := newOpenAICompatChat(spec, compatCfg, o)
	if err != nil {
		return nil, err
	}
	return &Chat{id: id, connector: c}, nil
}

// ProviderID returns the provider this facade dispatches to.
func (c *Chat) ProviderID() ProviderID { return c.id }

// Complete performs a non-streaming chat completion.
func (c *Chat) Complete(ctx context.Context, in ChatInput) (*ChatResult, error) {
	return c.connector.complete(ctx, in)
}

// Stream performs a streaming chat completion. Read deltas from the returned
// *ChatStream with Recv until it returns io.EOF, then Close it.
func (c *Chat) Stream(ctx context.Context, in ChatInput) (*ChatStream, error) {
	return c.connector.stream(ctx, in)
}

// ChatStream is a pull-based stream of ChatStreamDelta. Recv returns io.EOF when
// the stream is exhausted. Always Close it (safe to defer).
type ChatStream struct {
	recv   func() (*ChatStreamDelta, error)
	closer func() error
}

// Recv returns the next delta, or io.EOF at end of stream.
func (s *ChatStream) Recv() (*ChatStreamDelta, error) { return s.recv() }

// Close releases the underlying response body.
func (s *ChatStream) Close() error {
	if s.closer != nil {
		return s.closer()
	}
	return nil
}
