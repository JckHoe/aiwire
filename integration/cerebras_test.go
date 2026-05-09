//go:build integration

package integration

import (
	"os"
	"testing"

	"github.com/lwlee2608/aiwire"
	"github.com/openai/openai-go/v3"
)

func cerebrasKeyOrSkip(t *testing.T) string {
	t.Helper()
	apiKey := os.Getenv("CEREBRAS_API_KEY")
	if apiKey == "" {
		t.Skip("CEREBRAS_API_KEY not set")
	}
	return apiKey
}

func TestCerebras_Completion(t *testing.T) {
	service := aiwire.NewOpenAIService(cerebrasKeyOrSkip(t), "https://api.cerebras.ai/v1")
	messages := []openai.ChatCompletionMessageParamUnion{
		openai.UserMessage("Hello, can you tell me a joke?"),
	}

	runCompletionTest(t, service, messages, aiwire.CompletionOption{
		Model:       "qwen-3-235b-a22b-instruct-2507",
		Temperature: 0.7,
	})
}

func TestCerebras_Streaming(t *testing.T) {
	service := aiwire.NewOpenAIService(cerebrasKeyOrSkip(t), "https://api.cerebras.ai/v1")
	messages := []openai.ChatCompletionMessageParamUnion{
		openai.UserMessage("Hello, can you tell me a short joke?"),
	}

	runStreamingTest(t, service, messages, aiwire.CompletionOption{
		Model:       "qwen-3-235b-a22b-instruct-2507",
		Temperature: 0.7,
	})
}
