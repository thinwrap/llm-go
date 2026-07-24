package llm

// openAICompatSpec parameterizes the one shared OpenAI-compatible connector per
// provider. Adding a first-class provider is a spec row, not a new type.
type openAICompatSpec struct {
	id ProviderID
	// defaultBaseURL is the current-as-of-2026 default (no trailing slash); "" ⇒
	// the consumer must supply config.BaseURL.
	defaultBaseURL string
	// buildAuthHeaders returns the provider's auth headers for the config.
	buildAuthHeaders func(cfg OpenAICompatConfig) map[string]string
	// supportsTools is false for providers with no tool support (Perplexity Sonar).
	supportsTools *bool
	// maxTokensField overrides the default "max_tokens".
	maxTokensField string
	// streamUsagePath == "x_groq" relocates streaming usage under x_groq.usage (Groq).
	streamUsagePath string
	// strictParams skips OpenAI-only extras like stream_options (Mistral 422s on unknown params).
	strictParams bool
}

func bearer(cfg OpenAICompatConfig) map[string]string {
	if cfg.APIKey == "" {
		return map[string]string{}
	}
	return map[string]string{"Authorization": "Bearer " + cfg.APIKey}
}

func boolPtr(b bool) *bool { return &b }

var specs = map[ProviderID]openAICompatSpec{
	OpenAI: {id: OpenAI, defaultBaseURL: "https://api.openai.com/v1", buildAuthHeaders: bearer},
	AzureOpenAI: {id: AzureOpenAI, buildAuthHeaders: func(cfg OpenAICompatConfig) map[string]string {
		if cfg.APIKey == "" {
			return map[string]string{}
		}
		return map[string]string{"api-key": cfg.APIKey}
	}},
	OpenRouter: {id: OpenRouter, defaultBaseURL: "https://openrouter.ai/api/v1", buildAuthHeaders: bearer},
	Groq:       {id: Groq, defaultBaseURL: "https://api.groq.com/openai/v1", buildAuthHeaders: bearer, streamUsagePath: "x_groq"},
	Together:   {id: Together, defaultBaseURL: "https://api.together.ai/v1", buildAuthHeaders: bearer},
	Fireworks:  {id: Fireworks, defaultBaseURL: "https://api.fireworks.ai/inference/v1", buildAuthHeaders: bearer},
	DeepSeek:   {id: DeepSeek, defaultBaseURL: "https://api.deepseek.com/v1", buildAuthHeaders: bearer},
	XAI:        {id: XAI, defaultBaseURL: "https://api.x.ai/v1", buildAuthHeaders: bearer},
	Mistral:    {id: Mistral, defaultBaseURL: "https://api.mistral.ai/v1", buildAuthHeaders: bearer, strictParams: true},
	Perplexity: {id: Perplexity, defaultBaseURL: "https://api.perplexity.ai", buildAuthHeaders: bearer, supportsTools: boolPtr(false)},
	DeepInfra:  {id: DeepInfra, defaultBaseURL: "https://api.deepinfra.com/v1/openai", buildAuthHeaders: bearer},
	Cloudflare: {id: Cloudflare, buildAuthHeaders: bearer},
	VLLM:       {id: VLLM, defaultBaseURL: "http://localhost:8000/v1", buildAuthHeaders: bearer},
	Ollama:     {id: Ollama, defaultBaseURL: "http://localhost:11434/v1", buildAuthHeaders: bearer},
	LMStudio:   {id: LMStudio, defaultBaseURL: "http://localhost:1234/v1", buildAuthHeaders: bearer},
}

// embeddingsProviderIDs are the first-class providers exposing an OpenAI-float
// /embeddings surface (excludes providers with no native / non-float embeddings).
var embeddingsProviderIDs = map[ProviderID]bool{
	OpenAI: true, AzureOpenAI: true, OpenRouter: true, Together: true, Fireworks: true,
	Mistral: true, DeepInfra: true, Cloudflare: true, VLLM: true, Ollama: true, LMStudio: true,
}
