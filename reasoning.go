package aiwire

import (
	"encoding/json"

	"github.com/openai/openai-go/v3"
)

type ReasoningEffort string

const (
	ReasoningEffortXHigh   ReasoningEffort = "xhigh"
	ReasoningEffortHigh    ReasoningEffort = "high"
	ReasoningEffortMedium  ReasoningEffort = "medium"
	ReasoningEffortLow     ReasoningEffort = "low"
	ReasoningEffortMinimal ReasoningEffort = "minimal"
	ReasoningEffortNone    ReasoningEffort = "none"
)

// ReasoningOption controls reasoning behavior. Forwarded as the `reasoning`
// field on the request body for OpenRouter-style endpoints.
type ReasoningOption struct {
	Effort    ReasoningEffort
	MaxTokens *int
	Exclude   bool
	Summary   string // OpenAI gpt-5 reasoning summary verbosity: "auto" | "concise" | "detailed"
}

// ReasoningDetail is one entry of OpenRouter's reasoning_details array.
// Raw is the verbatim wire bytes — Type/Index are surfaced for filtering
// and slot keying; everything else lives in Raw and round-trips opaquely.
type ReasoningDetail struct {
	Type  string `json:"type,omitempty"`
	Index int    `json:"index"`

	Raw json.RawMessage `json:"-"`
}

func (r ReasoningDetail) MarshalJSON() ([]byte, error) {
	if len(r.Raw) > 0 {
		return r.Raw, nil
	}
	type alias ReasoningDetail
	return json.Marshal(alias(r))
}

// AssistantMessageWithReasoning builds an assistant message that carries
// reasoning_details alongside content and tool calls, so the model can replay
// its prior reasoning on the next loop iteration. Empty details yields a plain message.
func AssistantMessageWithReasoning(
	content string,
	toolCalls []openai.ChatCompletionMessageToolCallUnion,
	details []ReasoningDetail,
) openai.ChatCompletionMessageParamUnion {
	msg := openai.ChatCompletionMessage{
		Role:      "assistant",
		Content:   content,
		ToolCalls: toolCalls,
	}
	if len(details) == 0 {
		return msg.ToParam()
	}
	p := msg.ToAssistantMessageParam()
	p.SetExtraFields(map[string]any{
		"reasoning_details": details,
	})
	return openai.ChatCompletionMessageParamUnion{OfAssistant: &p}
}
