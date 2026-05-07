//go:build usage

package aiwire

import (
	"context"
	"fmt"
	"os"
	"strings"
	"testing"
	"time"

	"github.com/openai/openai-go/v3"
	"github.com/stretchr/testify/assert"
)

// Anthropic requires >=1024 tokens to cache; the nonce forces a cold cache per run.
func buildLongSystemPrompt() string {
	var b strings.Builder
	fmt.Fprintf(&b, "Session id: %d\n", time.Now().UnixNano())
	b.WriteString("You are a meticulous assistant. Follow these instructions carefully.\n\n")
	for i := 0; i < 200; i++ {
		b.WriteString("Rule: be concise, accurate, and never invent facts. Cite sources when possible. ")
		b.WriteString("Prefer short answers. Avoid filler. Respect user's time. Use simple English. ")
		b.WriteString("If unsure, say so. Do not speculate beyond what is asked. ")
	}
	return b.String()
}

func systemMessageWithCacheControl(text string) openai.ChatCompletionMessageParamUnion {
	part := openai.ChatCompletionContentPartTextParam{Text: text}
	part.SetExtraFields(map[string]any{
		"cache_control": map[string]any{"type": "ephemeral"},
	})
	return openai.SystemMessage([]openai.ChatCompletionContentPartTextParam{part})
}

func runUsageCacheTestWithProvider(t *testing.T, model string, provider *ProviderOption, expectCacheWrite bool) {
	t.Helper()
	apiKey := os.Getenv("OPENROUTER_API_KEY")
	assert.NotEmpty(t, apiKey)

	service := NewOpenAIService(apiKey, "https://openrouter.ai/api/v1")
	system := buildLongSystemPrompt()
	messages := []openai.ChatCompletionMessageParamUnion{
		systemMessageWithCacheControl(system),
		openai.UserMessage("Reply with the single word: hello."),
	}

	opts := CompletionOption{
		Model:       model,
		Temperature: 0.0,
		Provider:    provider,
	}

	ctx := context.Background()

	first, err := service.Completions(ctx, messages, nil, opts)
	assert.NoError(t, err)
	assert.NotEmpty(t, first.Message.Content)
	t.Logf("[%s] first call provider=%s", model, first.Provider)
	logUsage(t, first.Usage)

	second, err := service.Completions(ctx, messages, nil, opts)
	assert.NoError(t, err)
	assert.NotEmpty(t, second.Message.Content)
	t.Logf("[%s] second call provider=%s", model, second.Provider)
	logUsage(t, second.Usage)

	if expectCacheWrite {
		assert.Greater(t, first.Usage.PromptTokensDetails.CacheCreationTokens, int64(0),
			"expected cache_write tokens on first call for %s", model)
	}
	assert.Greater(t, second.Usage.PromptTokensDetails.CachedTokens, int64(0),
		"expected cache_read tokens on second call for %s", model)
}

func TestUsage_OpenRouter_AnthropicSonnet46(t *testing.T) {
	runUsageCacheTestWithProvider(t, "anthropic/claude-sonnet-4.6", &ProviderOption{
		Order:          []string{"anthropic"},
		AllowFallbacks: false,
	}, true)
}

func TestUsage_OpenRouter_KimiK25(t *testing.T) {
	runUsageCacheTestWithProvider(t, "moonshotai/kimi-k2.5", &ProviderOption{
		Sort:           "latency",
		AllowFallbacks: true,
	}, false)
}

func TestUsage_OpenRouter_GLM47(t *testing.T) {
	runUsageCacheTestWithProvider(t, "z-ai/glm-4.7", &ProviderOption{
		Sort:           "latency",
		AllowFallbacks: true,
	}, false)
}

func TestUsage_OpenRouter_OpenAIGPT5Mini(t *testing.T) {
	apiKey := os.Getenv("OPENROUTER_API_KEY")
	assert.NotEmpty(t, apiKey)

	service := NewOpenAIService(apiKey, "https://openrouter.ai/api/v1")
	system := buildLongSystemPrompt()
	messages := []openai.ChatCompletionMessageParamUnion{
		openai.SystemMessage(system),
		openai.UserMessage("Reply with the single word: hello."),
	}

	opts := CompletionOption{
		Model:       "openai/gpt-5-mini",
		Temperature: 0.0,
		Provider: &ProviderOption{
			Order:          []string{"openai"},
			AllowFallbacks: false,
		},
	}

	ctx := context.Background()

	first, err := service.Completions(ctx, messages, nil, opts)
	assert.NoError(t, err)
	assert.NotEmpty(t, first.Message.Content)
	t.Logf("[openai/gpt-5-mini] first call provider=%s", first.Provider)
	logUsage(t, first.Usage)

	second, err := service.Completions(ctx, messages, nil, opts)
	assert.NoError(t, err)
	assert.NotEmpty(t, second.Message.Content)
	t.Logf("[openai/gpt-5-mini] second call provider=%s", second.Provider)
	logUsage(t, second.Usage)

	assert.Greater(t, second.Usage.PromptTokensDetails.CachedTokens, int64(0),
		"expected cache_read tokens on second OpenRouter->OpenAI call")
	assert.Equal(t, int64(0), first.Usage.PromptTokensDetails.CacheCreationTokens,
		"OpenAI has no cache write tier")
	assert.Equal(t, int64(0), second.Usage.PromptTokensDetails.CacheCreationTokens,
		"OpenAI has no cache write tier")
}

type streamResult struct {
	content  string
	provider string
	usage    *Usage
}

func collectStream(t *testing.T, service *Service, ctx context.Context, params openai.ChatCompletionNewParams, provider *ProviderOption) (streamResult, error) {
	t.Helper()
	var result streamResult
	err := service.ParamsCompletionsStream(ctx, params, provider, nil, func(chunk StreamChunk) error {
		if chunk.Provider != "" {
			result.provider = chunk.Provider
		}
		if chunk.Done {
			result.usage = chunk.Usage
			return nil
		}
		result.content += chunk.Content
		return nil
	})
	return result, err
}

func runUsageCacheStreamTestWithProvider(t *testing.T, model string, provider *ProviderOption, expectCacheWrite bool) {
	t.Helper()
	apiKey := os.Getenv("OPENROUTER_API_KEY")
	assert.NotEmpty(t, apiKey)

	service := NewOpenAIService(apiKey, "https://openrouter.ai/api/v1")
	system := buildLongSystemPrompt()
	messages := []openai.ChatCompletionMessageParamUnion{
		systemMessageWithCacheControl(system),
		openai.UserMessage("Reply with the single word: hello."),
	}

	params := openai.ChatCompletionNewParams{
		Messages:    messages,
		Model:       model,
		Temperature: openai.Float(0.0),
		StreamOptions: openai.ChatCompletionStreamOptionsParam{
			IncludeUsage: openai.Bool(true),
		},
	}

	ctx := context.Background()

	first, err := collectStream(t, service, ctx, params, provider)
	assert.NoError(t, err)
	assert.NotEmpty(t, first.content)
	t.Logf("[%s stream] first call provider=%s", model, first.provider)
	if assert.NotNil(t, first.usage, "expected usage on first stream call for %s", model) {
		logUsage(t, *first.usage)
	}

	second, err := collectStream(t, service, ctx, params, provider)
	assert.NoError(t, err)
	assert.NotEmpty(t, second.content)
	t.Logf("[%s stream] second call provider=%s", model, second.provider)
	if assert.NotNil(t, second.usage, "expected usage on second stream call for %s", model) {
		logUsage(t, *second.usage)
	}

	if expectCacheWrite && first.usage != nil {
		assert.Greater(t, first.usage.PromptTokensDetails.CacheCreationTokens, int64(0),
			"expected cache_write tokens on first stream call for %s", model)
	}
	if second.usage != nil {
		assert.Greater(t, second.usage.PromptTokensDetails.CachedTokens, int64(0),
			"expected cache_read tokens on second stream call for %s", model)
	}
}

func TestUsage_OpenRouter_AnthropicSonnet46_Stream(t *testing.T) {
	runUsageCacheStreamTestWithProvider(t, "anthropic/claude-sonnet-4.6", &ProviderOption{
		Order:          []string{"anthropic"},
		AllowFallbacks: false,
	}, true)
}

func TestUsage_OpenRouter_KimiK25_Stream(t *testing.T) {
	runUsageCacheStreamTestWithProvider(t, "moonshotai/kimi-k2.5", &ProviderOption{
		Sort:           "latency",
		AllowFallbacks: true,
	}, false)
}

func TestUsage_OpenRouter_GLM47_Stream(t *testing.T) {
	runUsageCacheStreamTestWithProvider(t, "z-ai/glm-4.7", &ProviderOption{
		Sort:           "latency",
		AllowFallbacks: true,
	}, false)
}

func TestUsage_OpenRouter_OpenAIGPT5Mini_Stream(t *testing.T) {
	apiKey := os.Getenv("OPENROUTER_API_KEY")
	assert.NotEmpty(t, apiKey)

	service := NewOpenAIService(apiKey, "https://openrouter.ai/api/v1")
	system := buildLongSystemPrompt()
	messages := []openai.ChatCompletionMessageParamUnion{
		openai.SystemMessage(system),
		openai.UserMessage("Reply with the single word: hello."),
	}

	params := openai.ChatCompletionNewParams{
		Messages:    messages,
		Model:       "openai/gpt-5-mini",
		Temperature: openai.Float(0.0),
		StreamOptions: openai.ChatCompletionStreamOptionsParam{
			IncludeUsage: openai.Bool(true),
		},
	}

	provider := &ProviderOption{
		Order:          []string{"openai"},
		AllowFallbacks: false,
	}

	ctx := context.Background()

	first, err := collectStream(t, service, ctx, params, provider)
	assert.NoError(t, err)
	assert.NotEmpty(t, first.content)
	t.Logf("[openai/gpt-5-mini stream] first call provider=%s", first.provider)
	if assert.NotNil(t, first.usage, "expected usage on first stream call") {
		logUsage(t, *first.usage)
	}

	second, err := collectStream(t, service, ctx, params, provider)
	assert.NoError(t, err)
	assert.NotEmpty(t, second.content)
	t.Logf("[openai/gpt-5-mini stream] second call provider=%s", second.provider)
	if assert.NotNil(t, second.usage, "expected usage on second stream call") {
		logUsage(t, *second.usage)
	}

	if first.usage == nil || second.usage == nil {
		return
	}
	assert.Greater(t, second.usage.PromptTokensDetails.CachedTokens, int64(0),
		"expected cache_read tokens on second OpenRouter->OpenAI stream call")
	assert.Equal(t, int64(0), first.usage.PromptTokensDetails.CacheCreationTokens,
		"OpenAI has no cache write tier")
	assert.Equal(t, int64(0), second.usage.PromptTokensDetails.CacheCreationTokens,
		"OpenAI has no cache write tier")
}

func TestUsage_OpenAI_Direct(t *testing.T) {
	apiKey := os.Getenv("OPENAI_API_KEY")
	assert.NotEmpty(t, apiKey)

	service := NewOpenAIService(apiKey, "https://api.openai.com/v1")
	system := buildLongSystemPrompt()
	messages := []openai.ChatCompletionMessageParamUnion{
		openai.SystemMessage(system),
		openai.UserMessage("Reply with the single word: hello."),
	}

	opts := CompletionOption{
		Model:       "gpt-5.4-mini",
		Temperature: 0.0,
	}

	ctx := context.Background()

	first, err := service.Completions(ctx, messages, nil, opts)
	assert.NoError(t, err)
	assert.NotEmpty(t, first.Message.Content)
	t.Logf("[openai/gpt-4.1-mini] first call")
	logUsage(t, first.Usage)

	second, err := service.Completions(ctx, messages, nil, opts)
	assert.NoError(t, err)
	assert.NotEmpty(t, second.Message.Content)
	t.Logf("[openai/gpt-4.1-mini] second call")
	logUsage(t, second.Usage)

	assert.Greater(t, second.Usage.PromptTokensDetails.CachedTokens, int64(0),
		"expected cache_read tokens on second OpenAI call")
	assert.Equal(t, int64(0), first.Usage.PromptTokensDetails.CacheCreationTokens,
		"OpenAI has no cache write tier")
	assert.Equal(t, int64(0), second.Usage.PromptTokensDetails.CacheCreationTokens,
		"OpenAI has no cache write tier")
	assert.Equal(t, float64(0), first.Usage.Cost, "OpenAI direct does not return cost")
}

func TestUsage_OpenAI_Direct_Stream(t *testing.T) {
	apiKey := os.Getenv("OPENAI_API_KEY")
	assert.NotEmpty(t, apiKey)

	service := NewOpenAIService(apiKey, "https://api.openai.com/v1")
	system := buildLongSystemPrompt()
	messages := []openai.ChatCompletionMessageParamUnion{
		openai.SystemMessage(system),
		openai.UserMessage("Reply with the single word: hello."),
	}

	params := openai.ChatCompletionNewParams{
		Messages:    messages,
		Model:       "gpt-5.4-mini",
		Temperature: openai.Float(0.0),
		StreamOptions: openai.ChatCompletionStreamOptionsParam{
			IncludeUsage: openai.Bool(true),
		},
	}

	ctx := context.Background()

	first, err := collectStream(t, service, ctx, params, nil)
	assert.NoError(t, err)
	assert.NotEmpty(t, first.content)
	t.Logf("[openai/gpt-4.1-mini stream] first call")
	if assert.NotNil(t, first.usage, "expected usage on first stream call") {
		logUsage(t, *first.usage)
	}

	second, err := collectStream(t, service, ctx, params, nil)
	assert.NoError(t, err)
	assert.NotEmpty(t, second.content)
	t.Logf("[openai/gpt-4.1-mini stream] second call")
	if assert.NotNil(t, second.usage, "expected usage on second stream call") {
		logUsage(t, *second.usage)
	}

	if first.usage == nil || second.usage == nil {
		return
	}
	assert.Greater(t, second.usage.PromptTokensDetails.CachedTokens, int64(0),
		"expected cache_read tokens on second OpenAI stream call")
	assert.Equal(t, int64(0), first.usage.PromptTokensDetails.CacheCreationTokens,
		"OpenAI has no cache write tier")
	assert.Equal(t, int64(0), second.usage.PromptTokensDetails.CacheCreationTokens,
		"OpenAI has no cache write tier")
	assert.Equal(t, float64(0), first.usage.Cost, "OpenAI direct does not return cost")
}
