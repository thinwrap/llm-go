package llm

import (
	"context"
	"sort"
	"strings"
)

type openAICompatEmbeddingsConnector struct {
	baseConnector
	spec    openAICompatSpec
	cfg     OpenAICompatConfig
	baseURL string
}

func newOpenAICompatEmbeddings(spec openAICompatSpec, cfg OpenAICompatConfig, o facadeOptions) (*openAICompatEmbeddingsConnector, error) {
	base := cfg.BaseURL
	if base == "" {
		base = spec.defaultBaseURL
	}
	if base == "" {
		return nil, invalidReq("Provider '" + string(spec.id) + "' requires an explicit baseUrl in config")
	}
	return &openAICompatEmbeddingsConnector{
		baseConnector: baseConnector{o.httpClient},
		spec:          spec,
		cfg:           cfg,
		baseURL:       strings.TrimRight(base, "/"),
	}, nil
}

func (c *openAICompatEmbeddingsConnector) create(ctx context.Context, in EmbeddingsInput) (*EmbeddingsResult, error) {
	body := map[string]any{
		"model":           in.Model,
		"input":           in.Input,
		"encoding_format": "float",
	}
	if in.Dimensions != nil {
		body["dimensions"] = *in.Dimensions
	}
	applyBodyPassthrough(body, in.Passthrough)

	headers := map[string]string{}
	for k, v := range c.spec.buildAuthHeaders(c.cfg) {
		headers[k] = v
	}
	for k, v := range c.cfg.Headers {
		headers[k] = v
	}
	headers = mergeHeaders(headers, in.Passthrough)

	resp, err := c.sendPostJSON(ctx, c.baseURL+"/embeddings", body, headers)
	if err != nil {
		return nil, err
	}
	raw := decodeAny(readBody(resp))
	if !okStatus(resp.StatusCode) {
		return nil, vendorError(resp.StatusCode, embeddingsMapCode(resp.StatusCode), extractErrorMessage(raw), raw, resp.Header)
	}

	if raw == nil {
		return nil, emptyBodyError(resp.StatusCode)
	}
	rows, _ := jsonSliceVal(jsonGet(raw, "data"))
	sorted := make([]any, len(rows))
	copy(sorted, rows)
	sort.SliceStable(sorted, func(i, j int) bool {
		return jsonNum(jsonGet(sorted[i], "index")) < jsonNum(jsonGet(sorted[j], "index"))
	})
	embeddings := make([][]float64, 0, len(sorted))
	for _, row := range sorted {
		vecRaw, _ := jsonSliceVal(jsonGet(row, "embedding"))
		vec := make([]float64, 0, len(vecRaw))
		for _, f := range vecRaw {
			vec = append(vec, jsonNum(f))
		}
		embeddings = append(embeddings, vec)
	}

	usage := jsonGet(raw, "usage")
	inTok := int(jsonNum(jsonGet(usage, "prompt_tokens")))
	total := int(jsonNum(jsonGet(usage, "total_tokens")))
	if total == 0 {
		total = inTok
	}
	return &EmbeddingsResult{
		Embeddings: embeddings,
		Usage:      EmbeddingsUsage{InputTokens: inTok, TotalTokens: total},
		Model:      firstStr(jsonStr(jsonGet(raw, "model")), in.Model),
		Raw:        raw,
	}, nil
}

func embeddingsMapCode(status int) ProviderCode {
	switch {
	case status == 401 || status == 403:
		return CodeAuthFailed
	case status == 429:
		return CodeRateLimited
	case status >= 500:
		return CodeProviderUnavailable
	case status == 400 || status == 404 || status == 422:
		return CodeInvalidRequest
	}
	return CodeUnknown
}
