# 06 — The harness

The harness is the project. Everything else is plumbing around it.

It is a plain async Python module with no audio dependency, driven by events in and
commands out. The same code runs under a microphone, under an audio replay, and under a
pytest with typed strings.

---

## 1. State

```python
# harness/state.py
from dataclasses import dataclass, field
from typing import Literal

Trigger    = Literal["transcript", "barge_in", "silence", "tool_result"]
StopReason = Literal["answered", "handoff", "budget_exceeded", "interrupted", "error"]
ExecStatus = Literal["pending", "dispatched", "committed", "failed", "cancelled", "unknown"]

@dataclass
class ToolExecution:
    execution_id: str
    tool_name: str
    side_effecting: bool
    idempotency_key: str
    args_canonical: dict
    status: ExecStatus = "pending"
    attempt: int = 1
    response: dict | None = None
    error_type: str | None = None
    dispatched_at_ms: int | None = None
    resolved_at_ms: int | None = None

@dataclass
class CheckerVerdict:
    segment_index: int
    segment_text: str
    verdict: Literal["pass", "no_checkable_content",
                     "fail_ungrounded_entity", "fail_unsupported_claim"]
    spans: list[dict]
    action_taken: Literal["released", "replaced", "handoff"]
    checker_ms: int

@dataclass
class TurnState:
    turn_id: str
    seq: int
    trigger: Trigger
    user_text: str = ""
    stt_confidence: float | None = None
    entity_confidences: dict[str, float] = field(default_factory=dict)

    retrieved_chunk_ids: list[str] = field(default_factory=list)
    retrieval_top_score: float | None = None
    retrieval_margin: float | None = None

    llm_iterations: int = 0
    generated_text: str = ""        # everything the model produced
    approved_text: str = ""         # everything sent to TTS
    spoken_text: str = ""           # what the caller actually heard (approximate)

    checker_verdicts: list[CheckerVerdict] = field(default_factory=list)
    tool_executions: list[ToolExecution] = field(default_factory=list)
    handoff_signals: list[str] = field(default_factory=list)
    handoff_score: float = 0.0

    stop_reason: StopReason | None = None
    latency_budget_ms: int = 2500
    stages_ms: dict[str, int] = field(default_factory=dict)   # the [A]..[H] table
    trace_id: str | None = None

@dataclass
class CallState:
    call_id: str
    domain_config: str
    source: Literal["browser", "phone", "replay", "simulation", "text"]
    caller_verified: bool = False
    order_id: str | None = None
    turns: list[TurnState] = field(default_factory=list)
    messages: list[dict] = field(default_factory=list)   # LLM conversation, trimmed
    tool_executions: list[ToolExecution] = field(default_factory=list)
    handoff_signals: list[str] = field(default_factory=list)
    rephrase_count: int = 0
    pending_readback: "PendingReadback | None" = None
    outcome: str | None = None
```

### Three distinct texts, and why

`generated_text` ⊇ `approved_text` ⊇ `spoken_text`.

- `generated_text` — everything the model produced, including segments the checker
  rejected. Needed for the checker catch-rate metric.
- `approved_text` — what passed the checker and was handed to TTS.
- `spoken_text` — what the caller actually heard, after barge-in truncation.

Conflating these is the classic bug: trim context to `approved_text` after a barge-in and
the model believes it said a sentence the caller never heard, then refers back to it. The
three-way split is the fix.

---

## 2. The turn loop

```python
# harness/loop.py
class Harness:
    async def handle(self, call: CallState, event: InEvent) -> AsyncIterator[OutCommand]:
        ...
```

Commands out: `Speak(text, segment_id)`, `CancelSpeech()`, `EndCall(reason)`,
`EmitEvent(type, payload)`. The harness never touches an audio device.

### Sequence

```
 0. assemble TurnState, stamp t0, open OTel span "turn"
 1. PRE-GATE   hard handoff rules on raw transcript ───────────► handoff, done
 2. REPHRASE   near-duplicate of previous user turn? bump counter, add signal
 3. RETRIEVE   hybrid search ─────────────────────────────────► top_score, margin
 4. SIGNALS    soft handoff signals from retrieval + STT confidence
               score ≥ threshold ────────────────────────────► handoff, done
 5. ITERATE    up to max_tool_iterations:
                 stream LLM
                   TextDelta      → segmenter → checker → Speak
                   ToolCallDelta  → read-only : execute, append, continue
                                  → side-effecting : enter read-back, break
                   StreamEnd      → done or next iteration
 6. STOP       set exactly one stop_reason
 7. PERSIST    write turn + events, close span
```

Steps 1 and 2 are deliberately before any model call. The cheapest exit wins: a caller
saying "let me talk to a person" should never cost an LLM round trip.

### The iteration cap, precisely

`VD_MAX_TOOL_ITERATIONS = 3` counts **LLM calls**, not tool calls. A turn that does
`verify_caller → lookup_order → answer` uses three iterations and is the realistic worst
case. On hitting the cap with an unresolved tool loop, the harness does not call the model
again: it raises `iteration_cap_reached` as a handoff signal. A model stuck in a tool loop
is a model that does not know what to do, which is the definition of a handoff.

---

## 3. Prompt assembly

`harness/prompts.py` builds the system prompt in two parts, split at the prompt-cache
breakpoint ([04-providers.md](04-providers.md) §1).

**Static** (identical every turn, cached): persona, the spoken-style rules, the grounding
rules, the safety rules, tool descriptions.

```
You are a phone support agent for Tide & Tempo Tickets. You are speaking out loud.

STYLE
- One or two sentences. Never more than three.
- No lists, no markdown, no URLs read aloud.
- Say numbers the way a person says them.
- If you need a moment, say so briefly rather than going silent.

GROUNDING
- Answer only from <knowledge>. Reference the tag you used, e.g. [K2].
- If <knowledge> does not contain the answer, say you'll get a colleague who can help.
  Do not guess, infer, or generalize from a related policy.

AUTHORITY
- You may explain policy and file a refund request. You may NEVER approve, promise,
  or estimate a refund, credit, or any other money decision.
- Before any action that changes an order, read the details back and wait for a
  clear yes.

VERIFICATION
- Do not disclose order details until verify_caller has succeeded in this call.
```

**Dynamic** (per turn, after the breakpoint): the `<knowledge>` block, verification state,
and a one-line call context.

The authority rule is the design rule from the original spec, enforced in three places —
prompt, tool schema (`request_refund` has no amount field it could promise), and the
checker's unsupported-claim rule. Prompt-only enforcement of a money rule is not
enforcement.

---

## 4. Segmentation

`harness/segmenter.py` turns a token stream into checkable units.

```python
class Segmenter:
    def feed(self, delta: str) -> list[str]:
        """Return zero or more complete segments."""
    def flush(self) -> str | None:
        """Final partial segment at stream end."""
```

Boundaries, in priority order:

1. Sentence-final punctuation `. ! ?` followed by whitespace — but not after an
   abbreviation (`Mr.`, `No.`, `approx.`) or inside a decimal (`5.50`).
2. `;` and `:`.
3. Clause boundaries: `, and` / `, but` / `, so` / `, which` — **only** once the buffer
   exceeds 12 words. This is the latency mitigation from
   [01-architecture.md](01-architecture.md) §5: long sentences get checked in pieces
   instead of holding audio hostage.
4. Hard cap: 25 words forces a boundary regardless.

A segment shorter than 3 words is held and merged forward, so TTS never receives a
fragment that prosodically dangles.

### Acknowledgment bypass

If a turn's first LLM iteration is going to involve a tool call, or if time-to-first-token
exceeds `VD_ACK_AFTER_MS` (default 350 ms), the harness emits one acknowledgment line from
a fixed template list before the model's own text:

```yaml
acknowledgments:
  - "Let me check that for you."
  - "One moment, I'll look that up."
  - "Sure — checking now."
```

These are fixed strings with no checkable content by construction, so they bypass the
checker (`verdict = no_checkable_content`, `ack_bypass_used = TRUE`). They are chosen
round-robin with no immediate repeat. Latency is reported both with and without the bypass.

---

## 5. Context hygiene

Executed on every barge-in, before the next turn is constructed.

```python
async def trim_on_interrupt(call: CallState, turn: TurnState, played_chars: int) -> None:
```

1. **Finalize `spoken_text`** from the TTS cancel offset, minus `VD_PLAYOUT_TRIM_MS`,
   truncated to a word boundary, erring toward *less*
   ([04-providers.md](04-providers.md) §3).
2. **Rewrite the assistant message** in `call.messages` to `spoken_text`. If `spoken_text`
   is empty, remove the message entirely rather than leaving an empty assistant turn.
3. **Drop stale tool results.** Any `tool_result` message whose matching tool call is no
   longer referenced by a surviving assistant message is removed, along with its tool-call
   block. An orphaned `tool_result` is a hard API error on most providers, and a silent
   confusion on the rest.
4. **Resolve in-flight executions** ([07-tools.md](07-tools.md) §3): `pending` → `cancelled`;
   `dispatched` is left alone to resolve or reconcile. Never cancel what you have already
   sent.
5. **Clear `pending_readback`** if one was open, and record `readback_rejected`. An
   interruption during a read-back is a *no*, not a timeout — treat it as the caller
   objecting, because that is overwhelmingly what it is.
6. Emit `context_trimmed` with before/after message counts and character counts.

Invariant asserted after every trim: `call.messages` alternates roles correctly, every
`tool_use` has a matching `tool_result`, and no message has empty content. Violations raise
in tests rather than failing opaquely at the provider.

---

## 6. Error handling and retry

Retry policy lives here, not in providers, because only the harness knows the remaining
budget.

| Failure | Action |
|---|---|
| LLM timeout / 5xx | Retry once with a context trimmed to the last 4 messages, **only if** remaining budget > 1200 ms. Else fallback line + handoff. |
| LLM malformed tool call | Retry once with a corrective system note. Second failure → handoff. |
| Tool timeout | Status `unknown`, reconcile ([07-tools.md](07-tools.md) §3.4), raise `tool_failure` signal. |
| Retrieval error | Hard handoff — answering without grounding is the one thing the system must not do. |
| TTS error | Fallback line through the alternate provider if configured, else handoff. |
| Checker failure (exception, not verdict) | Fail closed: suppress the segment, raise handoff. |

The invariant from [01-architecture.md](01-architecture.md) §4 is testable and is tested:
grep the harness for any path where an exception handler produces caller-facing content
that was not drawn from a fixed template.

---

## 7. Stop rules

Exactly one `stop_reason` per turn, asserted before persistence.

| Reason | Condition |
|---|---|
| `answered` | Stream ended, all segments released, nothing pending |
| `handoff` | Hard rule, or soft score ≥ threshold |
| `budget_exceeded` | `latency_budget_ms` elapsed before first audio |
| `interrupted` | Barge-in |
| `error` | Unhandled exception; the call ends |

`budget_exceeded` speaks a fixed fallback (`"Sorry — give me just a second."`) and the turn
re-enters rather than abandoning the caller. Two consecutive `budget_exceeded` turns become
a handoff signal; a system that is this slow is not serving the caller.

The stop-reason distribution is a first-class health metric
([03-data-model.md](03-data-model.md) §5). A healthy demo build sits around 70–85%
`answered`, 10–25% `handoff`, under 5% `interrupted`, and essentially zero
`budget_exceeded`. A spike in `interrupted` usually means endpointing is too aggressive,
not that callers got rude.

---

## 8. Triggers other than transcripts

- **`silence`** — `VD_SILENCE_TIMEOUT_MS` (8 s) with no speech. First occurrence
  reprompts ("Are you still there?"); second ends the call as `dropped`.
- **`tool_result`** — a long-running tool completing after filler speech was spoken.
  Re-enters the loop with the result already in context.
- **`barge_in`** — handled by §5, then the caller's new speech produces a `transcript`
  turn.

---

## 9. The text driver

`eval/text_driver.py` is the harness's primary test surface, not an afterthought.

```bash
make chat                                  # interactive REPL
uv run python -m eval.text_driver --script tests/scripts/transfer.yaml
```

```yaml
# tests/scripts/transfer_interrupted.yaml
llm_fixture: tests/fixtures/llm/transfer_with_interruption.yaml
turns:
  - user: "I need to move two tickets to my friend"
  - user: "Order TK-88213, it's sam at example dot com"
  - interrupt_at_ms: 150        # barge-in 150ms into the read-back
    user: "no wait, the other order"
expect:
  stop_reasons: [answered, interrupted, answered]
  committed_tools: []                  # nothing committed — the read-back was rejected
  cancelled_tools: [transfer_tickets]
  context_invariant: ok
```

Scripts run in milliseconds with the replay LLM provider and null STT/TTS. The interruption
and idempotency test suites are built entirely on this, which is why they can run on every
commit instead of only before a demo.
