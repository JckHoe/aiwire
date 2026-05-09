//go:build zai

package integration

import (
	"os"
	"testing"

	"github.com/lwlee2608/aiwire"
	"github.com/openai/openai-go/v3"
	"github.com/stretchr/testify/assert"
)

func TestZAI_Completion(t *testing.T) {
	apiKey := os.Getenv("ZAI_API_KEY")
	assert.NotEmpty(t, apiKey)

	service := aiwire.NewOpenAIService(apiKey, "https://api.z.ai/api/paas/v4")
	messages := []openai.ChatCompletionMessageParamUnion{
		openai.UserMessage("Hello, can you tell me a joke?"),
	}

	runCompletionTest(t, service, messages, aiwire.CompletionOption{
		Model:       "glm-5.1",
		Temperature: 0.7,
	})
}

func TestZAI_Streaming(t *testing.T) {
	apiKey := os.Getenv("ZAI_API_KEY")
	assert.NotEmpty(t, apiKey)

	service := aiwire.NewOpenAIService(apiKey, "https://api.z.ai/api/paas/v4")
	messages := []openai.ChatCompletionMessageParamUnion{
		openai.UserMessage("Hello, can you tell me a short joke?"),
	}

	runStreamingTest(t, service, messages, aiwire.CompletionOption{
		Model:       "glm-5.1",
		Temperature: 0.7,
	})
}
