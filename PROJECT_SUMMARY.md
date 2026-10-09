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
| 3 | 50-case eval set, noise/phone audio, STT comparison, threshold calibration | Next |
| 4 | Simulated callers, gap report, one measured improvement cycle | Planned |

---

## Measured numbers (usable today)

Every number below came from a test or run in this repo. **n** is shown
because most of these sets are small; say so if asked.

| Metric | Result | Source |
|---|---|---|
| Duplicate side effects under fault injection | **0** across 26 fault tests, incl. commit-then-timeout and reconciliation-down | `make faults` |
| Concurrent identical transfers | **5 → 1** side effect, 1 committed row (real Postgres) | `eval/fault_injection/test_postgres_ledger.py` |
| Grounding checker false positives | **0 / 22** realistic answers against real document chunks | `tests/test_checker.py` |
| Grounding checker catch, seeded fakes | **8 / 8** invented numbers, prices, percentages, venues, order IDs | `tests/test_checker.py` |
| Grounding checker latency | **p95 0.07 ms** per sentence (budget 5 ms) | `tests/test_checker.py` |
| Retrieval accuracy | **12 / 12** top-1 document (small set) | `tests/test_retrieval.py` |
| Retrieval latency | **p95 ~20 ms** (budget 40 ms) | `tests/test_retrieval.py` |
| Hard-rule handoffs ("get me a manager") | **0** LLM calls before handing off | `tests/test_handoff.py` |
| Test suite | **269** tests, ~19 s, no network | `make test` |
| Knowledge base | 12 documents, 40 chunks, 3 held out for later | `make ingest` |

### A finding worth talking about in interviews

The original design used the fused RRF score margin to decide whether
retrieval found anything relevant. Measured, it was **inverted** — irrelevant
queries scored *higher* (0.00099 vs 0.00064) — because RRF margins are rank
arithmetic, not relevance. Raw dense similarity separated cleanly: **0.795 vs
0.555, no overlap** (n = 16). Written up in
`results/2026-10-08-retrieval-signal-selection/`.

## Not measured yet — do not put on a resume

| Metric | Why not yet | Comes in |
|---|---|---|
| Voice-to-voice latency (ms) | No live audio run yet; replay numbers are not real latency | needs API keys + `make dev` |
| STT accuracy under noise | Eval audio not built | Week 3 |
| Handoff precision / recall | Thresholds still provisional; no labelled set | Week 3 |
| Containment improvement | Needs the caller simulator and a feedback cycle | Week 4 |

---

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
- Built a deterministic per-sentence grounding checker that blocks ungrounded
  prices, dates and entities before speech at **0.07 ms p95**, with **0
  false positives** on 22 realistic answers and declared, tested coverage gaps.
- Measured hybrid retrieval signals and found the planned RRF-margin
  confidence signal was inverted; switched handoff decisions to raw dense
  similarity, which separated covered from uncovered queries with no overlap.
- Built a live observability console (FastAPI + Server-Sent Events) showing
  per-turn retrieval scores, checker verdicts, tool state transitions and
  latency stages in real time.

**Fill in after Weeks 3–4 (leave the blanks until measured):**

- Achieved **X ms p50 / Y ms p95** voice-to-voice latency, with VAD
  endpointing reported separately from harness latency.
- Benchmarked self-hosted Whisper vs. streaming Deepgram across 4 noise
  levels and 8 kHz phone audio; entity accuracy fell from **A% to B%** at 5 dB.
- Built a call-analytics feedback loop that raised containment from **X% to
  Y%** over N simulated calls without increasing the wrong-answer rate.

**Which to lead with:** harness + idempotency for AI/ML engineering and
forward-deployed roles; the console and retrieval finding for data roles.
