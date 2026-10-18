package llm

// OpenAICompatConfig configures any of the 15 first-class OpenAI-compatible
// providers (the provider id selects which). APIKey is optional for self-host
// providers (vLLM / Ollama / LM Studio) that run without auth. BaseURL is
// required for providers with no fixed public default (azure-openai, cloudflare).
type OpenAICompatConfig struct {
	APIKey  string
	BaseURL string
	// Headers are merged onto every request (e.g. OpenRouter HTTP-Referer / X-Title).
	Headers map[string]string
}

func (OpenAICompatConfig) isChatConfig() {}

// AnthropicConfig configures the Anthropic Messages API native adapter.
type AnthropicConfig struct {
	// APIKey is sent as the x-api-key header (not Bearer).
	APIKey string
	// BaseURL overrides the default https://api.anthropic.com/v1.
	BaseURL string
	// AnthropicVersion is the anthropic-version header (default 2023-06-01).
	AnthropicVersion string
	// DefaultMaxTokens is used when ChatInput.MaxOutputTokens is unset (default
	// 16000; thinking counts toward it).
	DefaultMaxTokens int
	Headers          map[string]string
}

func (AnthropicConfig) isChatConfig() {}

// BedrockConfig configures the AWS Bedrock Converse native adapter (SigV4).
type BedrockConfig struct {
	// Region (e.g. "us-east-1") must be a single DNS label: it is part of the
	// default host and the SigV4 scope.
	Region          string
	AccessKeyID     string
	SecretAccessKey string
	SessionToken    string
	// BaseURL overrides the default https://bedrock-runtime.<region>.amazonaws.com.
	BaseURL string
	// DefaultMaxTokens is used when ChatInput.MaxOutputTokens is unset (default 4096).
	DefaultMaxTokens int
	// Headers participate in the SigV4 signature.
	Headers map[string]string
}

func (BedrockConfig) isChatConfig() {}

// GeminiConfig configures the Google Gemini generateContent native adapter.
type GeminiConfig struct {
	// APIKey is sent as the x-goog-api-key header.
	APIKey string
	// BaseURL overrides the default https://generativelanguage.googleapis.com/v1beta.
	BaseURL          string
	DefaultMaxTokens int
	Headers          map[string]string
}

func (GeminiConfig) isChatConfig() {}
