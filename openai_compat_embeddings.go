package llm

import (
	"context"
	"sort"
	"strconv"
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
		field := c.spec.embeddingsDimensionsField
		if field == "" {
			field = "dimensions"
		}
		body[field] = *in.Dimensions
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

	url := c.baseURL + "/embeddings"
	if in.Passthrough != nil && len(in.Passthrough.Query) > 0 {
		url += "?" + buildCanonicalQuery(in.Passthrough.Query)
	}
	resp, err := c.sendPostJSON(ctx, url, body, headers)
	if err != nil {
		return nil, err
	}
	raw, readErr := readJSON(resp)
	if !okStatus(resp.StatusCode) {
		return nil, vendorError(resp.StatusCode, embeddingsMapCode(resp.StatusCode), extractErrorMessage(raw), raw, resp.Header)
	}
	if readErr != nil {
		return nil, transportError(readErr)
	}

	if raw == nil {
		return nil, emptyBodyError(resp.StatusCode)
	}
	if errObj := jsonGet(raw, "error"); errObj != nil {
		return nil, compatEmbeddedError(resp, errObj, raw, func(status int, _ any) ProviderCode { return embeddingsMapCode(status) })
	}
	rows, ok := jsonSliceVal(jsonGet(raw, "data"))
	if !ok {
		return nil, emptyBodyError(resp.StatusCode)
	}
	// Embeddings[i] must be the vector for Input[i]; a short/long data array or
	// a null/non-numeric cell would silently misalign or corrupt vectors, so it
	// is a failed call rather than a result.
	if len(rows) != len(in.Input) {
		return nil, malformedBodyError(resp.StatusCode,
			"provider returned "+strconv.Itoa(len(rows))+" embeddings for "+strconv.Itoa(len(in.Input))+" inputs", raw)
	}
	sorted := make([]any, len(rows))
	copy(sorted, rows)
	sort.SliceStable(sorted, func(i, j int) bool {
		return jsonNum(jsonGet(sorted[i], "index")) < jsonNum(jsonGet(sorted[j], "index"))
	})
	embeddings := make([][]float64, 0, len(sorted))
	for _, row := range sorted {
		vecRaw, ok := jsonSliceVal(jsonGet(row, "embedding"))
		if !ok {
			return nil, malformedBodyError(resp.StatusCode, "provider returned an embedding that is not a float array", raw)
		}
		vec := make([]float64, 0, len(vecRaw))
		for _, f := range vecRaw {
			// encoding/json only yields finite float64s, so the type check alone
			// rejects null and string cells.
			n, ok := f.(float64)
			if !ok {
				return nil, malformedBodyError(resp.StatusCode, "provider returned a non-numeric embedding value", raw)
			}
			vec = append(vec, n)
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
	case status == 429 || status == 402:
		return CodeRateLimited
	// 408 is a transient timeout.
	case status >= 500 || status == 408:
		return CodeProviderUnavailable
	// 404, 413 (the batch is too large to send), 422, …: a request problem, as in chat.
	case status >= 400:
		return CodeInvalidRequest
	}
	return CodeUnknown
}
