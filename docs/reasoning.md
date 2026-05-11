# Reasoning support

aiwire's reasoning capture and replay is built on OpenRouter's `reasoning_details` wire format. Calling reasoning models through any other gateway works for the answer, but reasoning content is not captured.

## Provider matrix

Verified by integration tests against live providers, May 2026. "Replay" means an assistant turn carrying reasoning was successfully echoed back into a follow-up call.

### Via OpenRouter (`https://openrouter.ai/api/v1`)

| Model                      | Tokens counted | Reasoning text | Encrypted blob | Streaming | Replay across loop |
| -------------------------- | -------------- | -------------- | -------------- | --------- | ------------------ |
| `anthropic/claude-sonnet-4.6` | yes         | yes            | n/a            | yes       | yes                |
| `anthropic/claude-opus-4.6`   | yes         | yes            | n/a            | yes       | yes                |
| `moonshotai/kimi-k2.6`        | yes         | yes            | n/a            | yes       | yes                |
| `openai/gpt-5.5`              | yes         | summary only*  | yes (~1KB)     | yes       | yes (encrypted)    |

\* gpt-5.5 emits a reasoning summary, not the full chain. The replay-relevant payload is the encrypted blob.

### Direct providers

| Endpoint                                        | Tokens counted | Reasoning text | Encrypted blob | Replay |
| ----------------------------------------------- | -------------- | -------------- | -------------- | ------ |
| OpenAI `https://api.openai.com/v1/chat/completions` | yes        | **no**         | **no**         | n/a    |
| OpenAI `https://api.openai.com/v1/responses`        | yes (out-of-scope, see below) |
| Anthropic native (`https://api.anthropic.com/v1/messages`) | not supported by aiwire (uses Messages API, not chat completions) |

## Why OpenRouter only

OpenRouter normalises every upstream's reasoning payload into a single
`reasoning_details: [{type, text, data, format, id, index}, ...]` array attached
to the assistant message in `/chat/completions`. aiwire reads that array
verbatim and re-attaches it on replay.

Direct OpenAI exposes reasoning via:

- `/v1/chat/completions` — bills `reasoning_tokens` but **never returns content**
- `/v1/responses` — returns reasoning items with `summary` text and (with `include=reasoning.encrypted_content` + `store=false`) an opaque encrypted blob, but uses a completely different request/response shape

Wiring `/v1/responses` would require a parallel `Service` implementation. It's
intentionally out of scope; if you need direct-OpenAI reasoning replay, use
`openai-go`'s `responses` package directly.

## What gets captured

For supported providers, every successful completion (streaming or not) populates:

- `CompletionResponse.Reasoning` — concatenated reasoning text (when the
  provider exposes it)
- `CompletionResponse.ReasoningDetails` — structured array with raw wire bytes
  preserved for verbatim replay
- `Usage.CompletionTokensDetails.ReasoningTokens` — billed reasoning tokens

For streaming, `StreamChunk.ReasoningDetails` carries fragments per chunk; the
final `Done` chunk carries the merged result with `Raw` regenerated from the
merged typed fields.

## Replay across the agent loop

`Agent.Execute` and `Agent.ExecuteStream` thread captured `ReasoningDetails`
into the assistant message via `AssistantMessageWithReasoning`. This satisfies
provider requirements that prior reasoning be echoed back on subsequent
tool-call turns (notably gpt-5.5 at `effort: high`, which can otherwise reject
follow-up calls).

## Known caveats

- **Streaming Raw round-trip is best-effort for unknown fields.** Fragments are
  merged from typed fields and `Raw` is regenerated at finalize. Unknown
  future fields carried in fragment Raw are dropped. Non-streaming preserves
  unknowns verbatim.
- **`Data` is last-write-wins on merge.** Encrypted blobs arrive whole, not in
  concat-able chunks; concatenating two complete blobs would corrupt them.
  `Text` still concatenates as expected for streamed reasoning summaries.
- **Indexless fragments get a synthetic slot.** OpenRouter always sends
  `index`; if a provider ever omits it, distinct fragments won't collapse into
  one entry.
