//go:build integration

package integration

import (
	"os"
	"testing"

	"github.com/lwlee2608/aiwire"
	"github.com/openai/openai-go/v3"
)

func zaiKeyOrSkip(t *testing.T) string {
	t.Helper()
	apiKey := os.Getenv("ZAI_API_KEY")
	if apiKey == "" {
		t.Skip("ZAI_API_KEY not set")
	}
	return apiKey
}

func TestZAI_Completion(t *testing.T) {
	service := aiwire.NewOpenAIService(zaiKeyOrSkip(t), "https://api.z.ai/api/paas/v4")
	messages := []openai.ChatCompletionMessageParamUnion{
		openai.UserMessage("Hello, can you tell me a joke?"),
	}

	runCompletionTest(t, service, messages, aiwire.CompletionOption{
		Model:       "glm-5.1",
		Temperature: 0.7,
	})
}

func TestZAI_Streaming(t *testing.T) {
	service := aiwire.NewOpenAIService(zaiKeyOrSkip(t), "https://api.z.ai/api/paas/v4")
	messages := []openai.ChatCompletionMessageParamUnion{
		openai.UserMessage("Hello, can you tell me a short joke?"),
	}

	runStreamingTest(t, service, messages, aiwire.CompletionOption{
		Model:       "glm-5.1",
		Temperature: 0.7,
	})
}
