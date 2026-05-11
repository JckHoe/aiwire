package aiwire

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"strings"

	"github.com/openai/openai-go/v3"
	"github.com/openai/openai-go/v3/option"
	"github.com/openai/openai-go/v3/packages/pagination"
	"github.com/openai/openai-go/v3/packages/respjson"
)

type Config struct {
	ApiKey  string `mask:"first=3,last=4"`
	BaseUrl string
}

type Service struct {
	client openai.Client
}

func NewOpenAIService(apiKey string, baseUrl string) *Service {
	client := openai.NewClient(
		option.WithAPIKey(apiKey),
		option.WithBaseURL(baseUrl),
	)
	return &Service{
		client: client,
	}
}

func buildRequestOptions(provider *ProviderOption, reasoning *ReasoningOption) []option.RequestOption {
	var opts []option.RequestOption

	if provider != nil {
		providerMap := map[string]any{
			"allow_fallbacks": provider.AllowFallbacks,
		}
		if provider.RequireParameters {
			providerMap["require_parameters"] = true
		}
		if provider.DataCollection != "" {
			providerMap["data_collection"] = provider.DataCollection
		}
		if provider.Sort != "" {
			providerMap["sort"] = provider.Sort
		}
		if len(provider.Order) > 0 {
			providerMap["order"] = provider.Order
		}
		if len(provider.Ignore) > 0 {
			providerMap["ignore"] = provider.Ignore
		}
		opts = append(opts, option.WithJSONSet("provider", providerMap))
	}

	if reasoning != nil {
		reasoningMap := map[string]any{}
		if reasoning.Effort != "" {
			reasoningMap["effort"] = string(reasoning.Effort)
		}
		if reasoning.MaxTokens != nil {
			reasoningMap["max_tokens"] = *reasoning.MaxTokens
		}
		if reasoning.Exclude {
			reasoningMap["exclude"] = reasoning.Exclude
		}
		if reasoning.Summary != "" {
			reasoningMap["summary"] = reasoning.Summary
		}
		opts = append(opts, option.WithJSONSet("reasoning", reasoningMap))
	}

	return opts
}

func extractRawStringFromExtraFields(extraFields map[string]respjson.Field, keys ...string) string {
	for _, key := range keys {
		field, ok := extraFields[key]
		if !ok {
			continue
		}

		raw := strings.TrimSpace(field.Raw())
		if raw == "" || raw == "null" {
			continue
		}

		var value string
		if err := json.Unmarshal([]byte(raw), &value); err != nil {
			continue
		}
		if value == "" {
			continue
		}
		return value
	}

	return ""
}

func extractStringFromExtraFields(extraFields map[string]respjson.Field, keys ...string) string {
	return strings.TrimSpace(extractRawStringFromExtraFields(extraFields, keys...))
}

func extractProviderFromHeader(response *http.Response) string {
	if response == nil {
		return ""
	}

	return strings.TrimSpace(response.Header.Get("X-OpenRouter-Provider"))
}

func extractReasoning(extraFields map[string]respjson.Field) string {
	return extractStringFromExtraFields(extraFields, "reasoning", "reasoning_content")
}

func extractReasoningDetails(extraFields map[string]respjson.Field) []ReasoningDetail {
	field, ok := extraFields["reasoning_details"]
	if !ok {
		return nil
	}
	raw := strings.TrimSpace(field.Raw())
	if raw == "" || raw == "null" {
		return nil
	}
	var rawItems []json.RawMessage
	if err := json.Unmarshal([]byte(raw), &rawItems); err != nil {
		return nil
	}
	if len(rawItems) == 0 {
		return nil
	}
	out := make([]ReasoningDetail, 0, len(rawItems))
	for _, item := range rawItems {
		var d ReasoningDetail
		if err := json.Unmarshal(item, &d); err != nil {
			continue
		}
		d.Raw = append(json.RawMessage(nil), item...)
		out = append(out, d)
	}
	if len(out) == 0 {
		return nil
	}
	return out
}

func mergeReasoningDetailFragments(acc map[int]*ReasoningDetail, order *[]int, fragments []ReasoningDetail) {
	for _, frag := range fragments {
		idx := frag.Index
		existing, ok := acc[idx]
		if !ok {
			copy := frag
			acc[idx] = &copy
			*order = append(*order, idx)
			continue
		}
		if frag.Type != "" {
			existing.Type = frag.Type
		}
		if frag.Format != "" {
			existing.Format = frag.Format
		}
		if frag.ID != "" {
			existing.ID = frag.ID
		}
		existing.Text += frag.Text
		existing.Data += frag.Data
		if len(frag.Raw) > 0 {
			existing.Raw = append(json.RawMessage(nil), frag.Raw...)
		}
	}
}

// finalizeMergedReasoningDetails regenerates each Raw from the merged typed
// fields so replay sends the full assembled object, not the last fragment.
func finalizeMergedReasoningDetails(acc map[int]*ReasoningDetail, order []int) []ReasoningDetail {
	if len(acc) == 0 {
		return nil
	}
	out := make([]ReasoningDetail, 0, len(acc))
	seen := make(map[int]bool, len(acc))
	for _, idx := range order {
		if seen[idx] {
			continue
		}
		seen[idx] = true
		d := *acc[idx]
		d.Raw = nil
		bytes, err := json.Marshal(d)
		if err == nil {
			d.Raw = bytes
		}
		out = append(out, d)
	}
	return out
}

func (s *Service) ParamsCompletions(ctx context.Context, params openai.ChatCompletionNewParams, provider *ProviderOption, reasoning *ReasoningOption) (CompletionResponse, error) {
	var response *http.Response
	var err error
	var completion *openai.ChatCompletion

	opts := buildRequestOptions(provider, reasoning)
	opts = append(opts, option.WithResponseInto(&response))
	completion, err = s.client.Chat.Completions.New(ctx, params, opts...)

	if err != nil {
		return CompletionResponse{}, err
	}
	if len(completion.Choices) == 0 {
		return CompletionResponse{}, errors.New("no completion choices returned from OpenAI API")
	}

	message := completion.Choices[0].Message
	reasoningContent := extractReasoning(message.JSON.ExtraFields)
	reasoningDetails := extractReasoningDetails(message.JSON.ExtraFields)

	routedProvider := extractStringFromExtraFields(completion.JSON.ExtraFields, "provider", "provider_name")
	if routedProvider == "" {
		routedProvider = extractStringFromExtraFields(message.JSON.ExtraFields, "provider", "provider_name")
	}
	if routedProvider == "" {
		routedProvider = extractProviderFromHeader(response)
	}

	return CompletionResponse{
		Message:          message,
		Reasoning:        reasoningContent,
		ReasoningDetails: reasoningDetails,
		Provider:         routedProvider,
		Usage:            UsageFromOpenAI(completion.Usage),
	}, nil
}

func (s *Service) Completions(
	ctx context.Context,
	messages []openai.ChatCompletionMessageParamUnion,
	tools []openai.ChatCompletionToolUnionParam,
	option CompletionOption,
) (CompletionResponse, error) {
	params := openai.ChatCompletionNewParams{
		Messages:       messages,
		Model:          option.Model,
		Temperature:    openai.Float(option.Temperature),
		ResponseFormat: option.ResponseFormat,
	}

	if option.MaxTokens != nil {
		params.MaxTokens = openai.Int(int64(*option.MaxTokens))
	}

	if len(tools) > 0 {
		params.Tools = tools
	}

	return s.ParamsCompletions(ctx, params, option.Provider, option.Reasoning)
}

// ParamsCompletionsStream initiates a streaming completion request
func (s *Service) ParamsCompletionsStream(ctx context.Context, params openai.ChatCompletionNewParams, provider *ProviderOption, reasoning *ReasoningOption, callback StreamCallback) error {
	var response *http.Response
	var finalUsage Usage
	hasFinalUsage := false
	var stream interface {
		Next() bool
		Current() openai.ChatCompletionChunk
		Err() error
		Close() error
	}

	opts := buildRequestOptions(provider, reasoning)
	opts = append(opts, option.WithResponseInto(&response))
	stream = s.client.Chat.Completions.NewStreaming(ctx, params, opts...)
	defer stream.Close()

	routedProvider := ""
	headerChecked := false

	// Accumulate tool calls across chunks
	toolCallsMap := make(map[int]*openai.ChatCompletionMessageToolCallUnion)

	reasoningDetailsAcc := make(map[int]*ReasoningDetail)
	var reasoningDetailsOrder []int

	for stream.Next() {
		chunk := stream.Current()

		// Extract provider: check header once, then prefer body fields
		if !headerChecked {
			routedProvider = extractProviderFromHeader(response)
			headerChecked = true
		}
		if p := extractStringFromExtraFields(chunk.JSON.ExtraFields, "provider", "provider_name"); p != "" {
			routedProvider = p
		}

		if chunk.JSON.Usage.Valid() {
			finalUsage = UsageFromOpenAI(chunk.Usage)
			hasFinalUsage = true
		}

		if len(chunk.Choices) == 0 {
			continue
		}

		choice := chunk.Choices[0]
		delta := choice.Delta

		streamChunk := StreamChunk{
			Content:      delta.Content,
			Provider:     routedProvider,
			Role:         delta.Role,
			FinishReason: choice.FinishReason,
			Done:         false,
		}

		if providerFromChoice := extractStringFromExtraFields(choice.JSON.ExtraFields, "provider", "provider_name"); providerFromChoice != "" {
			routedProvider = providerFromChoice
			streamChunk.Provider = routedProvider
		}

		streamChunk.Reasoning = extractRawStringFromExtraFields(delta.JSON.ExtraFields, "reasoning", "reasoning_content")

		if fragments := extractReasoningDetails(delta.JSON.ExtraFields); len(fragments) > 0 {
			streamChunk.ReasoningDetails = fragments
			mergeReasoningDetailFragments(reasoningDetailsAcc, &reasoningDetailsOrder, fragments)
		}

		// Handle tool calls if present
		if len(delta.ToolCalls) > 0 {
			for _, toolCall := range delta.ToolCalls {
				index := int(toolCall.Index)

				// Initialize tool call if not exists
				if _, exists := toolCallsMap[index]; !exists {
					toolCallsMap[index] = &openai.ChatCompletionMessageToolCallUnion{
						ID:   toolCall.ID,
						Type: toolCall.Type,
						Function: openai.ChatCompletionMessageFunctionToolCallFunction{
							Name:      toolCall.Function.Name,
							Arguments: "",
						},
					}
				}

				// Append function arguments
				if toolCall.Function.Arguments != "" {
					tc := toolCallsMap[index]
					tc.Function.Arguments += toolCall.Function.Arguments
				}
			}

			// Convert map to slice for the chunk
			var toolCalls []openai.ChatCompletionMessageToolCallUnion
			for _, tc := range toolCallsMap {
				toolCalls = append(toolCalls, *tc)
			}
			streamChunk.ToolCalls = toolCalls
		}

		// Call the callback with the chunk
		if err := callback(streamChunk); err != nil {
			return err
		}
	}

	// Check for stream errors
	if err := stream.Err(); err != nil {
		return err
	}

	// Send final chunk with usage information if available
	// Note: OpenAI streaming API may not always provide usage information
	finalChunk := StreamChunk{
		Done:             true,
		Provider:         routedProvider,
		ReasoningDetails: finalizeMergedReasoningDetails(reasoningDetailsAcc, reasoningDetailsOrder),
	}
	if hasFinalUsage {
		finalChunk.Usage = &finalUsage
	}

	if err := callback(finalChunk); err != nil {
		return err
	}

	return nil
}

// CompletionsStream performs a streaming completion request
func (s *Service) CompletionsStream(
	ctx context.Context,
	messages []openai.ChatCompletionMessageParamUnion,
	tools []openai.ChatCompletionToolUnionParam,
	option CompletionOption,
	callback StreamCallback,
) error {
	params := openai.ChatCompletionNewParams{
		Messages:       messages,
		Model:          option.Model,
		Temperature:    openai.Float(option.Temperature),
		ResponseFormat: option.ResponseFormat,
	}

	if option.MaxTokens != nil {
		params.MaxTokens = openai.Int(int64(*option.MaxTokens))
	}

	if len(tools) > 0 {
		params.Tools = tools
	}

	return s.ParamsCompletionsStream(ctx, params, option.Provider, option.Reasoning, callback)
}

// Models retrieves the list of available models
func (s *Service) Models(ctx context.Context) (*pagination.Page[openai.Model], error) {
	return s.client.Models.List(ctx)
}

func (s *Service) Embedding(ctx context.Context, input string, model string) ([]float32, error) {
	embedding, err := s.client.Embeddings.New(ctx, openai.EmbeddingNewParams{
		Model: openai.EmbeddingModel(model),
		Input: openai.EmbeddingNewParamsInputUnion{
			OfString: openai.String(input),
		},
	})
	if err != nil {
		return nil, err
	}

	if len(embedding.Data) == 0 {
		return nil, errors.New("no embedding data returned from OpenAI")
	}

	f64 := embedding.Data[0].Embedding
	f32 := make([]float32, len(f64))
	for i, v := range f64 {
		f32[i] = float32(v)
	}

	return f32, nil
}
