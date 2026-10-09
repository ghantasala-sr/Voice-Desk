# 04 — Provider interfaces

Three interfaces, each with at least two implementations. The second implementation is not
decoration: it is what proves the interface is an abstraction rather than a wrapper around
one vendor's SDK.

| Interface | v1 implementations | Also |
|---|---|---|
| `LLMProvider` | `anthropic` (default), `openai` | `replay` — deterministic, for tests |
| `STTProvider` | `deepgram` (default), `faster_whisper` | `null` — for text mode |
| `TTSProvider` | `cartesia` | `null` — for text mode |

---

## 1. `LLMProvider`

The hard part is that streaming and tool calling interact: the model may emit text, then a
tool call, then more text, and the harness must be able to cancel mid-stream on barge-in
without leaking the connection.

```python
# providers/llm/base.py
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import AsyncIterator, Literal, Any

@dataclass(frozen=True)
class TextDelta:
    text: str

@dataclass(frozen=True)
class ToolCallDelta:
    """A complete tool call. Providers buffer partial JSON internally and emit
    only when the arguments parse — the harness never sees half a tool call."""
    call_id: str
    name: str
    arguments: dict[str, Any]

@dataclass(frozen=True)
class StreamEnd:
    stop_reason: Literal["end_turn", "tool_use", "max_tokens", "cancelled", "error"]
    input_tokens: int
    output_tokens: int
    error: str | None = None

LLMEvent = TextDelta | ToolCallDelta | StreamEnd

@dataclass(frozen=True)
class LLMRequest:
    system: str
    messages: list[dict]            # provider-neutral: {"role", "content"}
    tools: list[dict]               # JSON Schema, provider-neutral
    max_tokens: int = 512
    temperature: float = 0.2
    timeout_ms: int = 10_000

class LLMProvider(ABC):
    name: str
    model: str

    @abstractmethod
    def stream(self, req: LLMRequest) -> AsyncIterator[LLMEvent]:
        """Yield events until StreamEnd. MUST be cancellable: on
        `asyncio.CancelledError` the provider closes the upstream connection
        and does not swallow the exception."""

    @abstractmethod
    async def count_tokens(self, req: LLMRequest) -> int: ...
```

### Rules every implementation obeys

1. **Partial tool-call JSON never escapes.** Both Anthropic and OpenAI stream tool
   arguments as incremental JSON fragments. The provider accumulates and parses; the
   harness receives a `ToolCallDelta` only when arguments are valid. A fragment that never
   parses becomes `StreamEnd(stop_reason="error")`.
2. **Cancellation is real.** `await stream.aclose()` must terminate the HTTP connection,
   not just stop iterating. Barge-in correctness depends on this, and a leaked stream shows
   up as mysterious latency three turns later. Tested in
   `eval/fault_injection/test_interruption.py`.
3. **Token counts are always reported**, including on error, because cost-per-call appears
   in the results table.
4. **No retries inside the provider.** Retry policy is the harness's decision
   ([06-harness.md](06-harness.md) §6) because only the harness knows the turn's remaining
   latency budget.

### `anthropic.py`

- Model from `VD_LLM_MODEL`, default `claude-sonnet-5`.
- Maps the neutral message list onto the Messages API; tool results become
  `tool_result` content blocks.
- System prompt is passed as the top-level `system` parameter, and the static portion
  (persona, guardrails, tool descriptions) is marked for **prompt caching** — it is
  identical on every turn of every call, and caching it cuts time-to-first-token
  measurably. The dynamic portion (retrieved chunks) goes in the user message, after the
  cache breakpoint.
- Streams `content_block_delta` → `TextDelta`, accumulates `input_json_delta` for tool
  calls.

### `openai.py`

Same contract against Chat Completions with `stream=True` and `tool_choice="auto"`.
Exists so the interface stays honest; exercised by `tests/test_harness_loop.py::test_provider_parity`,
which runs one fixed scenario through both and asserts the same tool was called with the
same arguments. It does **not** assert identical prose — that would be testing the model,
not the interface.

### `replay.py`

Deterministic provider that replays a recorded event sequence from a YAML fixture.

```yaml
# tests/fixtures/llm/transfer_with_interruption.yaml
- text: "I can help with that. "
- text: "Let me pull up the order."
- tool_call: {name: lookup_order, arguments: {order_id: "TK-88213"}}
- text: "I have two tickets for Saturday. "
- tool_call: {name: transfer_tickets, arguments: {order_id: "TK-88213", quantity: 2, recipient_email: "sam@example.com"}}
- end: {stop_reason: tool_use}
```

It supports a `delay_ms` per event so interruption timing can be made deterministic — you
can assert that barge-in at 150 ms into the second text delta trims context exactly where
expected. This is what makes the whole harness test suite fast, free, and repeatable.

---

## 2. `STTProvider`

```python
# providers/stt/base.py
@dataclass(frozen=True)
class Transcript:
    text: str
    is_final: bool
    confidence: float              # 0..1, provider-normalized
    word_confidences: list[tuple[str, float]]   # for entity-level confidence
    started_at_ms: int             # offset from stream start
    ended_at_ms: int
    provider: str

class STTProvider(ABC):
    name: str
    streaming: bool                # True = true streaming model, False = chunked batch

    @abstractmethod
    async def stream(self, audio: AsyncIterator[bytes]) -> AsyncIterator[Transcript]: ...

    @abstractmethod
    async def transcribe_file(self, path: str) -> Transcript:
        """Used by the replay harness and WER measurement."""
```

`word_confidences` is load-bearing: the handoff signal is *low confidence on key entities*,
not low average confidence over the utterance. An order number heard at 0.4 confidence
inside an otherwise clean sentence must be catchable, and sentence-mean confidence hides it
([09-handoff.md](09-handoff.md) §3).

### `deepgram.py`

True streaming model over a WebSocket. Interim results arrive continuously; `is_final`
and the endpointing signal come from the provider. Config: `smart_format=true`,
`numerals=true` (so "eighty-eight two thirteen" surfaces as digits for entity matching),
`endpointing` aligned with the VAD setting in [10-audio-pipeline.md](10-audio-pipeline.md).

### `faster_whisper.py`

**Whisper is a batch model.** There is no honest way to make it a streaming model, and
pretending otherwise is the methodological trap flagged in
[00-decisions.md](00-decisions.md) D14. This implementation is explicit about what it does:

- `streaming = False`.
- Buffers audio using Silero VAD; on an endpoint, runs the full buffered utterance through
  the model and emits one final transcript.
- Optionally emits a provisional transcript every `N` ms over the partial buffer, for
  responsiveness — but these re-transcriptions are inconsistent at chunk boundaries, which
  is exactly where entity errors concentrate. Off by default.
- Records `model_size`, `compute_type`, and device in the config snapshot, because
  `small.en` on CPU and `distil-large-v3` on GPU are different results.

The comparison report measures the consequences of this difference. It does not pretend the
difference is not there.

### `null.py`

Returns nothing; audio input is never connected. Used by the text driver.

---

## 3. `TTSProvider`

```python
# providers/tts/base.py
@dataclass(frozen=True)
class AudioChunk:
    pcm: bytes
    sample_rate: int
    is_final: bool
    char_offset: int    # how far into the submitted text this chunk corresponds to

class TTSProvider(ABC):
    name: str
    sample_rate: int

    @abstractmethod
    async def synthesize(self, text: str, *, context_id: str) -> AsyncIterator[AudioChunk]: ...

    @abstractmethod
    async def cancel(self, context_id: str) -> int:
        """Stop synthesis and playout. Returns the character offset actually
        played, for spoken_text reconstruction."""
```

### `char_offset` and the honesty of `spoken_text`

`cancel()` returning a character offset is how `spoken_text` gets reconstructed on barge-in.
It is **approximate**, and the spec says so rather than overclaiming:

- The offset reflects audio *handed to the transport*, not audio that reached the caller's
  ear. Jitter buffer and playout latency mean the caller heard slightly less — typically
  50–150 ms, a word or two.
- The harness therefore applies a configurable `VD_PLAYOUT_TRIM_MS` (default 120 ms) and
  truncates `spoken_text` to the nearest **word boundary before** the computed offset. It
  errs toward assuming the caller heard *less*, which is the safe direction: the agent may
  repeat itself, but it will never assume the caller heard something they did not.

This is written down because the alternative — claiming exact knowledge of what was heard —
is a claim the system cannot support, and an interviewer will ask.

### `cartesia.py`

- Sonic over WebSocket, continuations keyed by `context_id` so consecutive approved
  sentences join into one prosodic stream instead of sounding like separate utterances.
- Emits the first chunk as soon as it is available; TTFB is recorded as the `tts_ttfb_ms`
  stage.
- `cancel()` closes the context and reports offset.

---

## 4. Provider selection and the config snapshot

```python
# providers/__init__.py
def build_llm(settings) -> LLMProvider: ...
def build_stt(settings) -> STTProvider: ...
def build_tts(settings) -> TTSProvider: ...

def config_snapshot(settings) -> dict:
    """Written to calls.config_snapshot and eval_runs.config_snapshot.
    Every field that could change a number must appear here."""
    return {
        "git_sha": ...,
        "llm":   {"provider": ..., "model": ..., "temperature": ...},
        "stt":   {"provider": ..., "streaming": ..., "model_detail": {...}},
        "tts":   {"provider": ..., "voice_id": ...},
        "embed": {"model": ..., "dim": ...},
        "retrieval": {"top_k": ..., "rrf_k": ...},
        "vad":   {"stop_secs": ..., "start_secs": ..., "confidence": ...},
        "budgets": {...},
        "domain": ..., "corpus_hash": ...,
        "handoff_thresholds": {...},
    }
```

A number without its snapshot is not a result. See
[15-results-and-claims.md](15-results-and-claims.md).
