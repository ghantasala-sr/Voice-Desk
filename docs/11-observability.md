# 11 — Observability

> Live view of all of this: the console at `/console` — see [16-console.md](16-console.md).

Two independent systems, on purpose:

- **OpenTelemetry → Jaeger** for *debugging one call*. Rich, hierarchical, ephemeral.
- **Postgres `call_events` + `turns`** for *measuring many calls*. Flat, queryable,
  permanent.

Traces answer "why did this call go wrong." SQL answers "how often does this go wrong."
Trying to make one system do both produces something that does neither.

---

## 1. Span tree

One trace per call. `trace_id` is written to `turns.trace_id`, which is the join between
the two systems: find a bad turn in SQL, open its trace in Jaeger.

```
call                                    [call_id, domain, source]
├── turn (seq=0)                         [turn_id, trigger, stop_reason]
│   ├── pre_gate                          [hard_rule_hit]
│   ├── retrieval                         [top_score, margin, chunk_ids, n_candidates]
│   ├── llm_iteration (0)                 [model, input_tokens, output_tokens, ttft_ms]
│   │   └── tool_call: lookup_order       [execution_id, idem_key, status, attempt]
│   ├── llm_iteration (1)
│   │   ├── segment (0)                   [text, n_chars]
│   │   │   ├── checker                   [verdict, n_spans, n_ungrounded]
│   │   │   ├── normalize                 [n_substitutions]
│   │   │   └── tts                       [ttfb_ms, provider, voice_id]
│   │   └── segment (1) …
│   ├── handoff_eval                      [score, signals, decision]
│   └── persist
├── turn (seq=1)
│   ├── barge_in                          [played_chars, trim_ms]
│   └── context_trim                      [messages_before, messages_after, chars_dropped]
└── call_end                              [outcome, duration_ms, turn_count]
```

### Rules

- **Span per harness step, not per function.** A span for every call stack frame produces a
  trace nobody reads.
- **Every span carries `call_id` and `turn_id`** as attributes, so a span is interpretable
  in isolation.
- **Errors set span status and record an exception event** — a failed span that looks
  successful is worse than no span.
- **Cancellation is explicit.** A span terminated by barge-in sets
  `voicedesk.cancelled = true` rather than just ending. Otherwise an interrupted turn looks
  identical to a fast one in the trace view, and the latency histogram quietly fills with
  truncated turns.

---

## 2. Attribute namespace

All custom attributes use the `voicedesk.` prefix so they are filterable in Jaeger.

| Attribute | Type | On |
|---|---|---|
| `voicedesk.call_id` | string | every span |
| `voicedesk.turn_id` | string | every span under `turn` |
| `voicedesk.domain` | string | `call` |
| `voicedesk.trigger` | string | `turn` |
| `voicedesk.stop_reason` | string | `turn` |
| `voicedesk.cancelled` | bool | any span |
| `voicedesk.retrieval.top_score` | double | `retrieval` |
| `voicedesk.retrieval.margin` | double | `retrieval` |
| `voicedesk.retrieval.chunk_ids` | string[] | `retrieval` |
| `voicedesk.llm.model` | string | `llm_iteration` |
| `voicedesk.llm.ttft_ms` | int | `llm_iteration` |
| `voicedesk.llm.input_tokens` | int | `llm_iteration` |
| `voicedesk.llm.output_tokens` | int | `llm_iteration` |
| `voicedesk.llm.cache_read_tokens` | int | `llm_iteration` |
| `voicedesk.tool.name` | string | `tool_call` |
| `voicedesk.tool.status` | string | `tool_call` |
| `voicedesk.tool.idempotency_key` | string | `tool_call` |
| `voicedesk.tool.attempt` | int | `tool_call` |
| `voicedesk.checker.verdict` | string | `checker` |
| `voicedesk.checker.ungrounded_spans` | int | `checker` |
| `voicedesk.handoff.score` | double | `handoff_eval` |
| `voicedesk.handoff.signals` | string[] | `handoff_eval` |
| `voicedesk.ack_bypass` | bool | `turn` |

`cache_read_tokens` is tracked because prompt caching is supposed to be reducing
time-to-first-token ([04-providers.md](04-providers.md) §1). If cache reads are zero, the
cache breakpoint is misplaced and the latency work is being wasted — a bug that is
otherwise invisible.

---

## 3. What is NOT in traces

- **No caller PII.** Emails and order IDs are hashed in span attributes. Full values live
  in Postgres, which is access-controlled and deletable; Jaeger is not.
- **No full transcripts.** Character counts and the first 64 characters only. Full text is
  in `call_events`.
- **No audio.**

A tracing backend is the easiest place to accidentally build an unmanaged copy of customer
data. Decide this once, at the start, rather than retrofitting redaction.

---

## 4. The events table as a trace

`call_events` ([03-data-model.md](03-data-model.md) §2) duplicates some span information on
purpose. Traces expire; the call record must not. Every eval failure must be replayable
from Postgres alone, months later, with Jaeger long since restarted.

`analytics/events.py` is the only writer:

```python
class EventWriter:
    async def emit(self, call_id: str, event_type: str,
                   payload: dict, turn_id: str | None = None) -> None: ...
```

- `event_type` is validated against the closed vocabulary; an unknown type raises in tests
  and logs a warning in production. An open vocabulary becomes unqueryable within a week.
- Writes are buffered and flushed per turn, not per event — one round trip per turn rather
  than fifteen.
- Flush is in a `finally` block. A crashed call must still have its events.

---

## 5. Structured logs

`structlog` to stdout as JSON, with `call_id` and `turn_id` bound via contextvars so every
log line in a turn carries them without being passed around.

Logs are for *development*. They are not a metrics source, and no number in `results/` may
be derived from a log line — that is what the tables are for.

---

## 6. Metric definitions

Precise definitions, so a number computed in Week 4 means the same thing as one from Week 1.

| Metric | Definition | Source |
|---|---|---|
| **Voice-to-voice latency** | `UserStoppedSpeakingFrame` timestamp → first TTS audio sample handed to transport. Includes VAD endpointing. | `turns.voice_to_voice_ms` |
| **Harness-side latency** | Final transcript received → first TTS audio sample handed to transport. Excludes endpointing. | `turns.first_audio_ms` |
| **Barge-in latency** | Speech onset (VAD confidence crossing) → last non-silent output sample. | measured in replay |
| **WER** | `jiwer` with the standard normalizer, on the final transcript vs the reference. | `eval_results.wer` |
| **Entity accuracy** | Exact match after normalization on labelled key entities only. Per-entity, not per-utterance. | `eval_results.entity_correct / entity_total` |
| **Answer accuracy** | Judge `correctness == 2`, median of 3 runs. | `eval_results.judge_correctness` |
| **Citation correctness** | Judge `citation == 2`: the cited chunk actually supports the claim. | `eval_results.judge_citation` |
| **Handoff precision** | `TP / (TP + FP)` against `should_escalate`, Wilson 95% CI. | `eval_results` |
| **Handoff recall** | `TP / (TP + FN)`, Wilson 95% CI. | `eval_results` |
| **Containment** | Calls with `outcome = 'contained'` ÷ all non-`error` calls. | `calls.outcome` |
| **Wrong-answer rate** | Calls containing ≥1 turn with judge `correctness == 0` ÷ contained calls. **Denominator is contained calls**, because a handed-off call did not give a wrong answer. | joined |
| **Checker catch rate** | Per `seeded_class`: seeded segments receiving a `fail*` verdict ÷ seeded segments. | `checker_verdicts` |
| **Checker false-positive rate** | Known-good segments receiving a `fail*` verdict ÷ known-good segments. | `checker_verdicts` |
| **Duplicate side effects** | Count of `idempotency_key` values with >1 `committed` row. Target 0. | `tool_executions` |
| **Tool calls per turn** | Mean `count(tool_executions) / count(turns)`. | joined |
| **Time saved per escalation** | Mean human handling time without a summary minus with one. Estimated, and labelled as estimated. | manual |

Two of these have a trap worth naming:

- **Wrong-answer rate denominator.** Using all calls makes a system that escalates
  everything look safe. The denominator is contained calls only, so the metric measures
  what it claims: when the agent chose to answer, how often was it wrong.
- **Entity accuracy is per entity.** Per-utterance scoring hides that one utterance had
  three entities and two were wrong, which is the common case at 5 dB.
