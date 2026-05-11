package aiwire

import (
	"encoding/json"
	"strings"
	"testing"

	"github.com/openai/openai-go/v3/packages/respjson"
)

func fieldsFromJSON(t *testing.T, raw string) map[string]respjson.Field {
	t.Helper()
	var obj map[string]json.RawMessage
	if err := json.Unmarshal([]byte(raw), &obj); err != nil {
		t.Fatalf("invalid test JSON: %v", err)
	}
	out := make(map[string]respjson.Field, len(obj))
	for k, v := range obj {
		out[k] = respjson.NewField(string(v))
	}
	return out
}

func TestExtractReasoningDetails_EncryptedBlock(t *testing.T) {
	fields := fieldsFromJSON(t, `{
		"reasoning_details": [
			{"type":"reasoning.encrypted","data":"opaque-blob","format":"openai-responses-v1","id":"rs_1","index":0}
		]
	}`)

	got := extractReasoningDetails(fields)
	if len(got) != 1 {
		t.Fatalf("expected 1 detail, got %d", len(got))
	}
	d := got[0]
	if d.Type != "reasoning.encrypted" || d.Data != "opaque-blob" || d.Format != "openai-responses-v1" || d.ID != "rs_1" {
		t.Fatalf("typed fields not parsed: %+v", d)
	}
	if len(d.Raw) == 0 {
		t.Fatalf("raw bytes should be preserved")
	}
}

func TestExtractReasoningDetails_SummaryAndEncryptedMix(t *testing.T) {
	fields := fieldsFromJSON(t, `{
		"reasoning_details": [
			{"type":"reasoning.summary","text":"step one","index":0},
			{"type":"reasoning.encrypted","data":"abc","index":1}
		]
	}`)
	got := extractReasoningDetails(fields)
	if len(got) != 2 {
		t.Fatalf("expected 2, got %d", len(got))
	}
	if got[0].Text != "step one" || got[1].Data != "abc" {
		t.Fatalf("unexpected parse: %+v", got)
	}
}

func TestExtractReasoningDetails_MissingOrEmpty(t *testing.T) {
	cases := map[string]string{
		"missing":  `{"other":"x"}`,
		"null":     `{"reasoning_details": null}`,
		"empty":    `{"reasoning_details": []}`,
		"notArray": `{"reasoning_details": {"foo":"bar"}}`,
	}
	for name, raw := range cases {
		t.Run(name, func(t *testing.T) {
			if got := extractReasoningDetails(fieldsFromJSON(t, raw)); got != nil {
				t.Fatalf("expected nil, got %+v", got)
			}
		})
	}
}

func TestReasoningDetail_MarshalUsesRaw(t *testing.T) {
	d := ReasoningDetail{
		Type: "reasoning.encrypted",
		Raw:  json.RawMessage(`{"type":"reasoning.encrypted","data":"X","unknown_future_field":42}`),
	}
	bytes, err := json.Marshal(d)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(bytes), "unknown_future_field") {
		t.Fatalf("Raw should round-trip unknown fields; got %s", bytes)
	}
}

func TestReasoningDetail_MarshalFallsBackToFields(t *testing.T) {
	d := ReasoningDetail{Type: "reasoning.summary", Text: "hi"}
	bytes, err := json.Marshal(d)
	if err != nil {
		t.Fatal(err)
	}
	want := `{"type":"reasoning.summary","text":"hi"}`
	if string(bytes) != want {
		t.Fatalf("got %s want %s", bytes, want)
	}
}

func TestMergeReasoningDetailFragments_AccumulatesByIndex(t *testing.T) {
	acc := make(map[int]*ReasoningDetail)
	var order []int

	mergeReasoningDetailFragments(acc, &order, []ReasoningDetail{
		{Type: "reasoning.summary", Text: "step ", Index: 0},
	})
	mergeReasoningDetailFragments(acc, &order, []ReasoningDetail{
		{Text: "one", Index: 0},
		{Type: "reasoning.encrypted", Data: "abc", Index: 1},
	})
	mergeReasoningDetailFragments(acc, &order, []ReasoningDetail{
		{Data: "def", Index: 1},
	})

	out := finalizeMergedReasoningDetails(acc, order)
	if len(out) != 2 {
		t.Fatalf("expected 2 details, got %d", len(out))
	}
	if out[0].Text != "step one" || out[0].Type != "reasoning.summary" {
		t.Fatalf("idx0 not merged: %+v", out[0])
	}
	if out[1].Data != "def" || out[1].Type != "reasoning.encrypted" {
		t.Fatalf("idx1 should last-write-wins on Data: %+v", out[1])
	}
	if !strings.Contains(string(out[1].Raw), "def") {
		t.Fatalf("raw should reflect merged fields: %s", out[1].Raw)
	}
}

func TestAssistantMessageWithReasoning_EmptyDetailsIsPlain(t *testing.T) {
	msg := AssistantMessageWithReasoning("hello", nil, nil)
	bytes, err := json.Marshal(msg)
	if err != nil {
		t.Fatal(err)
	}
	if strings.Contains(string(bytes), "reasoning_details") {
		t.Fatalf("plain message should not include reasoning_details: %s", bytes)
	}
}

func TestAssistantMessageWithReasoning_AttachesDetails(t *testing.T) {
	details := []ReasoningDetail{
		{Type: "reasoning.encrypted", Data: "opaque", Format: "openai-responses-v1", ID: "rs_1"},
	}
	msg := AssistantMessageWithReasoning("done", nil, details)
	bytes, err := json.Marshal(msg)
	if err != nil {
		t.Fatal(err)
	}
	s := string(bytes)
	if !strings.Contains(s, `"reasoning_details"`) {
		t.Fatalf("missing reasoning_details: %s", s)
	}
	if !strings.Contains(s, `"opaque"`) || !strings.Contains(s, `"rs_1"`) {
		t.Fatalf("payload missing detail content: %s", s)
	}
	if !strings.Contains(s, `"role":"assistant"`) {
		t.Fatalf("missing assistant role: %s", s)
	}
}

func TestAssistantMessageWithReasoning_RawTakesPrecedence(t *testing.T) {
	details := []ReasoningDetail{{
		Type: "reasoning.encrypted",
		Raw:  json.RawMessage(`{"type":"reasoning.encrypted","data":"X","extra":"keep-me"}`),
	}}
	msg := AssistantMessageWithReasoning("", nil, details)
	bytes, err := json.Marshal(msg)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(bytes), "keep-me") {
		t.Fatalf("raw passthrough lost: %s", bytes)
	}
}
