//go:build cerebras

package aiwire

import (
	"os"
	"testing"

	"github.com/openai/openai-go/v3"
	"github.com/stretchr/testify/assert"
)

func TestCerebras_Completion(t *testing.T) {
	apiKey := os.Getenv("CEREBRAS_API_KEY")
	assert.NotEmpty(t, apiKey)

	service := NewOpenAIService(apiKey, "https://api.cerebras.ai/v1")
	messages := []openai.ChatCompletionMessageParamUnion{
		openai.UserMessage("Hello, can you tell me a joke?"),
	}

	runCompletionTest(t, service, messages, CompletionOption{
		Model:       "qwen-3-235b-a22b-instruct-2507",
		Temperature: 0.7,
	})
}

func TestCerebras_Streaming(t *testing.T) {
	apiKey := os.Getenv("CEREBRAS_API_KEY")
	assert.NotEmpty(t, apiKey)

	service := NewOpenAIService(apiKey, "https://api.cerebras.ai/v1")
	messages := []openai.ChatCompletionMessageParamUnion{
		openai.UserMessage("Hello, can you tell me a short joke?"),
	}

	runStreamingTest(t, service, messages, CompletionOption{
		Model:       "qwen-3-235b-a22b-instruct-2507",
		Temperature: 0.7,
	})
}
