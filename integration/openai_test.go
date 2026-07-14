//go:build integration

package integration

import (
	"testing"

	"github.com/lwlee2608/aiwire"
	"github.com/openai/openai-go/v3"
	"github.com/stretchr/testify/assert"
)

func TestOpenAI_Completion(t *testing.T) {
	service := aiwire.NewOpenAIService(keyOrSkip(t, "OPENAI_API_KEY"), "https://api.openai.com/v1")
	messages := []openai.ChatCompletionMessageParamUnion{
		openai.UserMessage("Hello, can you tell me a joke?"),
	}

	runCompletionTest(t, service, messages, aiwire.CompletionOption{
		Model:       "gpt-4.1-nano",
		Temperature: openai.Float(0.7),
	})
}

func TestOpenAI_Embedding(t *testing.T) {
	service := aiwire.NewOpenAIService(keyOrSkip(t, "OPENAI_API_KEY"), "https://api.openai.com/v1")

	embedding, err := service.Embedding(t.Context(), "Hello, world!", "text-embedding-3-small")
	assert.NoError(t, err)
	assert.NotEmpty(t, embedding)
	assert.Equal(t, 1536, len(embedding))
}

func TestOpenAI_Streaming(t *testing.T) {
	service := aiwire.NewOpenAIService(keyOrSkip(t, "OPENAI_API_KEY"), "https://api.openai.com/v1")
	messages := []openai.ChatCompletionMessageParamUnion{
		openai.UserMessage("Hello, can you tell me a short joke?"),
	}

	runStreamingTest(t, service, messages, aiwire.CompletionOption{
		Model:       "gpt-4.1-nano",
		Temperature: openai.Float(0.7),
	})
}
