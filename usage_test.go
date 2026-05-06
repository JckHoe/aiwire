package aiwire

import (
	"encoding/json"
	"testing"

	"github.com/openai/openai-go/v3"
	"github.com/stretchr/testify/assert"
)

func TestUsageFromOpenAI_AnthropicViaOpenRouter(t *testing.T) {
	raw := []byte(`{
		"prompt_tokens": 1500,
		"completion_tokens": 200,
		"total_tokens": 1700,
		"cache_creation_input_tokens": 800,
		"cache_read_input_tokens": 500,
		"cache_discount": 0.0012,
		"cost": 0.0345,
		"prompt_tokens_details": {"cached_tokens": 500},
		"completion_tokens_details": {"reasoning_tokens": 50}
	}`)

	var u openai.CompletionUsage
	assert.NoError(t, json.Unmarshal(raw, &u))

	got := UsageFromOpenAI(u)
	assert.Equal(t, int64(1500), got.PromptTokens)
	assert.Equal(t, int64(200), got.CompletionTokens)
	assert.Equal(t, int64(1700), got.TotalTokens)
	assert.Equal(t, int64(500), got.PromptTokensDetails.CachedTokens)
	assert.Equal(t, int64(800), got.PromptTokensDetails.CacheCreationTokens)
	assert.Equal(t, int64(50), got.CompletionTokensDetails.ReasoningTokens)
	assert.InDelta(t, 0.0012, got.CacheDiscount, 1e-9)
	assert.InDelta(t, 0.0345, got.Cost, 1e-9)
}

func TestUsageFromOpenAI_FallbackCacheReadFromRootField(t *testing.T) {
	raw := []byte(`{
		"prompt_tokens": 1000,
		"completion_tokens": 100,
		"total_tokens": 1100,
		"cache_read_input_tokens": 600
	}`)

	var u openai.CompletionUsage
	assert.NoError(t, json.Unmarshal(raw, &u))

	got := UsageFromOpenAI(u)
	assert.Equal(t, int64(600), got.PromptTokensDetails.CachedTokens)
	assert.Equal(t, int64(0), got.PromptTokensDetails.CacheCreationTokens)
}

func TestUsage_Add(t *testing.T) {
	a := Usage{
		PromptTokens:     10,
		CompletionTokens: 5,
		TotalTokens:      15,
		PromptTokensDetails: PromptTokensDetails{
			CachedTokens:        4,
			CacheCreationTokens: 3,
		},
		Cost: 0.01,
	}
	b := Usage{
		PromptTokens:     20,
		CompletionTokens: 7,
		TotalTokens:      27,
		PromptTokensDetails: PromptTokensDetails{
			CachedTokens:        2,
			CacheCreationTokens: 1,
		},
		Cost: 0.02,
	}
	a.Add(b)
	assert.Equal(t, int64(30), a.PromptTokens)
	assert.Equal(t, int64(12), a.CompletionTokens)
	assert.Equal(t, int64(42), a.TotalTokens)
	assert.Equal(t, int64(6), a.PromptTokensDetails.CachedTokens)
	assert.Equal(t, int64(4), a.PromptTokensDetails.CacheCreationTokens)
	assert.InDelta(t, 0.03, a.Cost, 1e-9)
}
