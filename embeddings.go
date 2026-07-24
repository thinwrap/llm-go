package llm

import "context"

// EmbeddingsInput is the normalized embeddings request. Input is one or more
// strings (embedded in input order).
type EmbeddingsInput struct {
	Model string
	Input []string
	// Dimensions requests a specific output dimensionality (Matryoshka), when supported.
	Dimensions  *int
	Passthrough *Passthrough
}

// EmbeddingsUsage is normalized token accounting for embeddings.
type EmbeddingsUsage struct {
	InputTokens int
	TotalTokens int
}

// EmbeddingsResult is the normalized embeddings response.
type EmbeddingsResult struct {
	// Embeddings has one float vector per input, in input order.
	Embeddings [][]float64
	Usage      EmbeddingsUsage
	Model      string
	Raw        any
}

type embeddingsConnector interface {
	create(ctx context.Context, in EmbeddingsInput) (*EmbeddingsResult, error)
}

// Embeddings is the embeddings facade — a separate operation from Chat. Only the
// OpenAI-float subset of providers is supported.
type Embeddings struct {
	id        ProviderID
	connector embeddingsConnector
}

// NewEmbeddings builds an embeddings facade for a first-class provider that
// exposes an OpenAI-float /embeddings surface.
func NewEmbeddings(id ProviderID, cfg OpenAICompatConfig, opts ...Option) (*Embeddings, error) {
	if !embeddingsProviderIDs[id] {
		return nil, invalidReq("Provider '" + string(id) + "' has no OpenAI-compatible embeddings surface")
	}
	o := applyOptions(opts)
	c, err := newOpenAICompatEmbeddings(specs[id], cfg, o)
	if err != nil {
		return nil, err
	}
	return &Embeddings{id: id, connector: c}, nil
}

// ProviderID returns the provider this facade dispatches to.
func (e *Embeddings) ProviderID() ProviderID { return e.id }

// Create computes embeddings for the input(s).
func (e *Embeddings) Create(ctx context.Context, in EmbeddingsInput) (*EmbeddingsResult, error) {
	return e.connector.create(ctx, in)
}
