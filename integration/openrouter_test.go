//go:build openrouter

package integration

import (
	"context"
	"encoding/json"
	"os"
	"testing"

	"github.com/lwlee2608/aiwire"
	"github.com/openai/openai-go/v3"
	"github.com/openai/openai-go/v3/shared"
	"github.com/stretchr/testify/assert"
)

func TestOpenAI_Completion_OpenRouter(t *testing.T) {
	apiKey := os.Getenv("OPENROUTER_API_KEY")
	assert.NotEmpty(t, apiKey)

	service := aiwire.NewOpenAIService(apiKey, "https://openrouter.ai/api/v1")
	messages := []openai.ChatCompletionMessageParamUnion{
		openai.UserMessage("Hello, can you tell me a joke?"),
	}

	runCompletionTest(t, service, messages, aiwire.CompletionOption{
		Model:       "z-ai/glm-4.7",
		Temperature: 0.7,
		Provider: &aiwire.ProviderOption{
			AllowFallbacks: true,
			Sort:           "throughput",
		},
		Reasoning: &aiwire.ReasoningOption{
			Effort: aiwire.ReasoningEffortLow,
		},
	})
}

func TestOpenAI_ResponseFormat_OpenRouter(t *testing.T) {
	apiKey := os.Getenv("OPENROUTER_API_KEY")
	assert.NotEmpty(t, apiKey)

	type person struct {
		Name       string   `json:"name" jsonschema:"required"`
		Age        int      `json:"age" jsonschema:"required"`
		Email      string   `json:"email" jsonschema:"required"`
		City       string   `json:"city" jsonschema:"required"`
		Occupation string   `json:"occupation" jsonschema:"required"`
		Hobbies    []string `json:"hobbies" jsonschema:"required"`
	}

	service := aiwire.NewOpenAIService(apiKey, "https://openrouter.ai/api/v1")
	messages := []openai.ChatCompletionMessageParamUnion{
		openai.UserMessage("Return a person named Alice who is 30 years old, email alice@example.com, lives in Paris, works as a software engineer, and enjoys hiking and painting."),
	}

	response, err := service.Completions(context.Background(), messages, nil, aiwire.CompletionOption{
		Model:       "z-ai/glm-4.7",
		Temperature: 0.0,
		Provider: &aiwire.ProviderOption{
			AllowFallbacks: true,
			Order:          []string{"parasail/fp8", "google-vertex", "cerebras/fp16"},
		},
		ResponseFormat: openai.ChatCompletionNewParamsResponseFormatUnion{
			OfJSONSchema: &shared.ResponseFormatJSONSchemaParam{
				JSONSchema: shared.ResponseFormatJSONSchemaJSONSchemaParam{
					Name:   "person",
					Strict: openai.Bool(true),
					Schema: aiwire.GenerateSchema[person](),
				},
			},
		},
	})

	assert.NoError(t, err)
	assert.NotEmpty(t, response.Message.Content)

	var got person
	err = json.Unmarshal([]byte(response.Message.Content), &got)
	assert.NoError(t, err)
	assert.Equal(t, "Alice", got.Name)
	assert.Equal(t, 30, got.Age)
	assert.Equal(t, "alice@example.com", got.Email)
	assert.Equal(t, "Paris", got.City)
	assert.NotEmpty(t, got.Occupation)
	assert.NotEmpty(t, got.Hobbies)

	t.Logf("Response: %s", response.Message.Content)
	t.Logf("Provider: %s", response.Provider)
	logUsage(t, response.Usage)
}

func TestOpenAI_ProviderIgnore_OpenRouter(t *testing.T) {
	apiKey := os.Getenv("OPENROUTER_API_KEY")
	assert.NotEmpty(t, apiKey)

	service := aiwire.NewOpenAIService(apiKey, "https://openrouter.ai/api/v1")
	messages := []openai.ChatCompletionMessageParamUnion{
		openai.UserMessage("Hello, can you tell me a joke?"),
	}

	response, err := service.Completions(context.Background(), messages, nil, aiwire.CompletionOption{
		Model:       "moonshotai/kimi-k2.6",
		Temperature: 0.7,
		Provider: &aiwire.ProviderOption{
			AllowFallbacks: true,
			Sort:           "latency",
			Ignore:         []string{"together"},
		},
	})

	assert.NoError(t, err)
	assert.NotEmpty(t, response.Message.Content)
	assert.NotEmpty(t, response.Provider)
	t.Logf("Routed provider: %s", response.Provider)
}

func TestOpenAI_ProviderOrder_OpenRouter(t *testing.T) {
	apiKey := os.Getenv("OPENROUTER_API_KEY")
	assert.NotEmpty(t, apiKey)

	service := aiwire.NewOpenAIService(apiKey, "https://openrouter.ai/api/v1")
	messages := []openai.ChatCompletionMessageParamUnion{
		openai.UserMessage("Hello, can you tell me a joke?"),
	}

	response, err := service.Completions(context.Background(), messages, nil, aiwire.CompletionOption{
		Model:       "moonshotai/kimi-k2.6",
		Temperature: 0.7,
		Provider: &aiwire.ProviderOption{
			AllowFallbacks: true,
			Order:          []string{"moonshotai/int4"},
		},
	})

	assert.NoError(t, err)
	assert.NotEmpty(t, response.Message.Content)
	assert.NotEmpty(t, response.Provider)
	t.Logf("Routed provider: %s", response.Provider)
}

func TestOpenAI_Streaming_OpenRouter(t *testing.T) {
	apiKey := os.Getenv("OPENROUTER_API_KEY")
	assert.NotEmpty(t, apiKey)

	service := aiwire.NewOpenAIService(apiKey, "https://openrouter.ai/api/v1")
	messages := []openai.ChatCompletionMessageParamUnion{
		openai.UserMessage("Hello, can you tell me a short joke?"),
	}

	runStreamingTest(t, service, messages, aiwire.CompletionOption{
		Model:       "z-ai/glm-4.7",
		Temperature: 0.7,
		Provider: &aiwire.ProviderOption{
			AllowFallbacks: true,
			Sort:           "throughput",
		},
		Reasoning: &aiwire.ReasoningOption{
			Effort: aiwire.ReasoningEffortLow,
		},
	})
}
