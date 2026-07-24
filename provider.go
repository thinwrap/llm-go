package llm

// ProviderID is the stable, user-facing identifier for an LLM provider. The
// string values are byte-identical across the thinwrap llm siblings
// (TypeScript, PHP, Go).
type ProviderID string

const (
	// First-class OpenAI-compatible providers (one shared connector).
	OpenAI      ProviderID = "openai"
	AzureOpenAI ProviderID = "azure-openai"
	OpenRouter  ProviderID = "openrouter"
	Groq        ProviderID = "groq"
	Together    ProviderID = "together"
	Fireworks   ProviderID = "fireworks"
	DeepSeek    ProviderID = "deepseek"
	XAI         ProviderID = "xai"
	Mistral     ProviderID = "mistral"
	Perplexity  ProviderID = "perplexity"
	DeepInfra   ProviderID = "deepinfra"
	Cloudflare  ProviderID = "cloudflare"
	VLLM        ProviderID = "vllm"
	Ollama      ProviderID = "ollama"
	LMStudio    ProviderID = "lmstudio"

	// Native adapters (structurally different wire, identical normalized surface).
	Anthropic ProviderID = "anthropic"
	Bedrock   ProviderID = "bedrock"
	Gemini    ProviderID = "gemini"
)

// ProviderIDs is the full list of implemented provider ids.
var ProviderIDs = []ProviderID{
	OpenAI, AzureOpenAI, OpenRouter, Groq, Together, Fireworks, DeepSeek, XAI,
	Mistral, Perplexity, DeepInfra, Cloudflare, VLLM, Ollama, LMStudio,
	Anthropic, Bedrock, Gemini,
}
