# VoiceDesk — project summary

## What it is

A real-time **voice support agent** for a (fictional) event-ticketing company.
A caller phones in; the agent answers from the company's policy documents,
takes actions like transferring tickets, and hands off to a human with a
written summary when it isn't confident.

The engine is domain-agnostic. Ticketing is just the first configuration.

## What makes it more than a chatbot

| Problem | What VoiceDesk does |
|---|---|
| Voice agents make things up | Every sentence is checked against the retrieved documents **before** it is spoken. An invented price is never said out loud. |
| Callers interrupt | The agent stops talking and rewrites its memory to what the caller *actually heard*, not what it generated. |
| Interruptions can double-charge | Every action has an idempotency key enforced by a database constraint. An interrupted, retried, or repeated transfer happens exactly once. |
| Actions can be misheard | Changes to an order are read back from a template built from the exact values that will be executed, and run only on a clear "yes". |
| Agents never admit defeat | Rule-based handoff to a human, with a summary stating what was done and what was **not** promised. |
| Nobody can see what it's doing | A live console shows every call, turn, retrieval score, checker verdict, tool call and latency stage as it happens. |

## How it's built

```
caller audio ─► Pipecat (VAD, STT, TTS) ─► Harness ─► Claude
                                             │
                    hybrid retrieval ◄───────┤  (pgvector + BM25)
                    grounding checker ◄──────┤  (per sentence, no LLM call)
                    tool executor ◄──────────┤  (idempotency ledger in Postgres)
                    handoff policy ◄─────────┘
                                             │
                    Postgres call log ─► live console
```

- **Pipecat** handles audio only. The **harness** — a plain Python module I
  wrote — owns everything about the conversation, and is forbidden by an
  automated check from importing any audio code. That's why the whole system
  can be tested with typed text in under 20 seconds.
- Python 3.12, FastAPI, PostgreSQL + pgvector, BGE embeddings, Claude,
  OpenTelemetry, Docker.

## Progress

| Week | Scope | Status |
|---|---|---|
| 1 | Voice loop, harness, barge-in, tracing | Done |
| 2 | Retrieval, grounding checker, tools, read-back, handoff, fault suite, console | Done |
| 3 | 50-case eval set, noise/phone audio, STT benchmark, threshold calibration | In progress |
| 4 | Simulated callers, gap report, one measured improvement cycle | Planned |

---

## Measured numbers (usable today)

Every number below came from a test or a run directory in this repo. **n** is
shown because most of these sets are small — say so if asked. Week-3 runs
replaced several week-2 figures; where a number went DOWN, that is because the
week-3 test set is harder and fairer, and the lower number is the one to use.

| Metric | Result | Source |
|---|---|---|
| Duplicate side effects under fault injection | **0** across 33 fault tests: commit-then-timeout, reconciliation down, concurrent duplicates, interruption mid-read-back | `make faults` |
| Concurrent identical transfers (real Postgres) | **5 → 1** side effect, 1 committed row | `eval/fault_injection/test_postgres_ledger.py` |
| Handoff recall / precision (deterministic policy) | **0.86** [0.72, 0.93] / **0.80** [0.66, 0.89], n=79 non-tool cases, 42 positives — up from recall **0.31** before calibration | `results/2026-10-13-handoff-calibration/` |
| Hard-rule handoffs | **0** LLM calls before handing off | same |
| Grounding checker, invented numbers / dates / entities | **23/23, 5/5, 4/4** caught | `results/2026-10-13-checker-catch-rate/` |
| Grounding checker, fabricated policy claims | **7/8** caught (from 1/8 before calibration) | same |
| Grounding checker false positives | **2/46** correct answers rejected | same |
| Grounding checker latency | **p95 0.07 ms** per sentence | `tests/test_checker.py` |
| Retrieval, top-k document accuracy | **12/14** = 0.86 [0.60, 0.96] on caller-phrased questions | `results/2026-10-13-retrieval-calibration/` |
| Retrieval, top-1 document accuracy | **9/14** = 0.64 [0.39, 0.84] (week 2 said 12/12 on a flattering set) | same |
| Retrieval latency | **p95 ~20 ms** (budget 40 ms) | `tests/test_retrieval.py` |
| STT key-entity accuracy, clean → 5 dB | **0.83 → 0.62** (WER only 0.025 → 0.052); emails **0.67 → 0.43** | `results/2026-10-13-stt-faster-whisper-smallen/` |
| Test suite | **~310** tests, ~20 s | `make test` |

### Findings worth talking about in interviews

1. **The planned retrieval-confidence signal was a coin toss.** The fused RRF
   margin scored AUC **0.53**; raw dense similarity **0.91** (n=64).
2. **The planned handoff policy caught 31% of cases that needed a human.** Weak
   retrieval could never hand off on its own (0.45 < 0.6). A second, stricter
   level took recall to 0.86 without sending everyday refund questions to a
   human — and the ablation shows that level is now the single biggest
   contributor.
3. **A real production bug found by measurement:** ISO dates from tool results
   were read as number ranges, so the agent could never confirm an event date
   it had just looked up. No unit test had covered it.
4. **The week-2 retrieval number was flattered by its own test set** — written
   by the corpus author in corpus phrasing. Re-measured on caller phrasing.

## Resume bullets

**Use now:**

- Built VoiceDesk, a real-time voice support agent (Python, FastAPI, Pipecat,
  PostgreSQL/pgvector, Claude) with a custom agent harness that owns retrieval,
  tool calls, sentence-level grounding checks and human handoff, independent
  of the audio stack.
- Designed an idempotent tool-execution ledger enforced by a Postgres partial
  unique index, with reconciliation for commit-then-timeout failures: **0
  duplicate side effects** across a 26-case fault-injection suite covering
  interruptions, timeouts, garbage responses and concurrent duplicates.
- Built a deterministic per-sentence grounding checker that blocks invented
  prices, dates and venues before speech at **0.07 ms p95** — **32/32** seeded
  entity hallucinations caught, **7/8** fabricated policy claims, **2/46**
  false positives — with its two blind spots (contradictions, wrong-fact
  numbers) measured and published rather than hidden.
- Calibrated handoff on a 94-case labelled set: raised recall on cases needing
  a human from **0.31 to 0.86** (precision 0.80) by replacing the planned
  RRF-margin signal (AUC 0.53) with raw dense similarity (AUC 0.91) at two
  thresholds; hard-rule handoffs cost **0** LLM calls.
- Built a live observability console (FastAPI + Server-Sent Events) showing
  per-turn retrieval scores, checker verdicts, tool state transitions and
  latency stages in real time.

- Benchmarked self-hosted Whisper on 900 synthesized call recordings across
  noise levels and 8 kHz phone audio: WER stayed under 6% while key-entity
  accuracy fell from **83% to 62%** at 5 dB — spoken emails worst at **43%** —
  showing WER alone hides the failures that break a support call.

**Fill in after Week 4 / with API keys (leave the blanks until measured):**

- Achieved **X ms p50 / Y ms p95** voice-to-voice latency, with VAD
  endpointing reported separately from harness latency.
- Built a call-analytics feedback loop that raised containment from **X% to
  Y%** over N simulated calls without increasing the wrong-answer rate.

**Which to lead with:** harness + idempotency for AI/ML engineering and
forward-deployed roles; the console and retrieval finding for data roles.
