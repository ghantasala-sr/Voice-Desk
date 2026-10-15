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
| Nobody can see what it's doing | A live console shows every call, turn, retrieval score, checker verdict, tool call and latency stage as it happens. A web app lets you talk to it and see *why* it said each sentence. |
| Nobody knows what to fix next | A gap report reads the call log, attributes each failed call to the document that would fix it, and ranks the fixes. One feedback cycle measured the result. |

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
| 3 | 94-case eval set, noise/phone audio, STT benchmark, threshold calibration | Done |
| 4 | Simulated callers, gap report, one measured improvement cycle, static report, web app | Done |

---

## Measured numbers (usable today)

Every number below came from a test or a run directory in this repo. **n** is
shown because most of these sets are small — say so if asked. Week-3 runs
replaced several week-2 figures; where a number went DOWN, that is because the
week-3 test set is harder and fairer, and the lower number is the one to use.
Week 4 found and fixed a bug in the retrieval signal, and the affected runs
were re-done (2026-10-15); those are the numbers below.

**No API key was available.** Every "simulated calls" number used an offline
extractive agent (no language model). Say "simulated calls, offline agent"
whenever you quote one.

| Metric | Result | Source |
|---|---|---|
| Duplicate side effects under fault injection | **0** across 34 fault tests: commit-then-timeout, reconciliation down, concurrent duplicates, interruption mid-read-back | `make faults` |
| Concurrent identical transfers (real Postgres) | **5 → 1** side effect, 1 committed row | `eval/fault_injection/test_postgres_ledger.py` |
| Handoff recall / precision (deterministic policy) | **0.86** [0.72, 0.93] / **0.86** [0.72, 0.93] on the 79 non-tool cases (0.82 on all 94), 42 positives — up from recall **0.31** before calibration | `results/2026-10-15-handoff-calibration/` |
| Hard-rule handoffs | **0** LLM calls before handing off | same |
| Grounding checker, invented numbers / dates / entities | **23/23, 5/5, 4/4** caught | `results/2026-10-15-checker-catch-rate/` |
| Grounding checker, fabricated policy claims | **7/8** caught (from 1/8 before calibration) | same |
| Grounding checker false positives | **2/46** correct answers rejected | same |
| Grounding checker latency | **p95 0.07 ms** per sentence | `tests/test_checker.py` |
| Retrieval, top-k document accuracy | **14/14** [0.78, 1.00] on caller-phrased questions | `results/2026-10-15-retrieval-calibration/` |
| Retrieval, top-1 document accuracy | **11/14** = 0.79 [0.52, 0.92] (week 2 said 12/12 on a flattering set) | same |
| Retrieval sufficiency signal | dense cosine AUC **0.93** vs fused RRF margin **0.51** (n=64) | same |
| Retrieval latency | **p95 ~20 ms** (budget 40 ms) | `tests/test_retrieval.py` |
| STT key-entity accuracy, clean → 5 dB | **0.83 → 0.62** (WER only 0.025 → 0.052); emails **0.67 → 0.43** | `results/2026-10-13-stt-faster-whisper-smallen/` |
| Feedback cycle: containment *(300 simulated calls, offline agent)* | **0.737 → 0.793**; paired **+17 / −0**, McNemar p = 1.5 × 10⁻⁵ | `results/2026-10-15-feedback-cycle/` |
| Feedback cycle: wrong-answer rate (guardrail) | **0.136 → 0.113** — did not rise | same |
| Gap report vs hidden labels | **3/3** planted gaps found (ranks 2, 4, 5); 1/3 in the top 3; cluster purity 0.985 | `results/2026-10-15-gap-report-cycle-before/` |
| Feedback cycle with **Gemini 2.5 Flash** (live model) | target intent paired **+16 / −0** (p = 3 × 10⁻⁵); weather helped 0/20 → 14/20; gap report found the same 3 gaps at the same ranks | `results/2026-10-15-feedback-cycle-vertex-gemini/` |
| Test suite | **381** tests, ~25 s | `make test` |

### Findings worth talking about in interviews

1. **The planned retrieval-confidence signal was a coin toss.** The fused RRF
   margin scored AUC **0.51**; raw dense similarity **0.93** (n=64).
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
5. **The handoff signal silently depended on tie order.** `top_vec_score` took the
   cosine of the *fused* leader, and lexical ties were broken by physical row
   order — so re-ingesting one document changed the same question's score from
   0.670 to 0.621, across a handoff threshold. Found because the feedback cycle
   changed answers it should not have touched. Fixing it raised handoff
   precision 0.75 → 0.82 at the same recall.
6. **A real race in the idempotency ledger**, caught by a test failing 3 times in
   5 under load: two concurrent duplicates could both move one `pending` row to
   `dispatched`. Now a compare-and-set, with a deterministic regression test.
7. **The gap report's first design found 0 of 3 planted gaps** — one topic split
   across phrasings lost to tidier clusters. Ranking *fixes* (documents) instead
   of clusters found all three.

## Resume bullets

**Use now:**

- Built VoiceDesk, a real-time voice support agent (Python, FastAPI, Pipecat,
  PostgreSQL/pgvector, Claude) with a custom agent harness that owns retrieval,
  tool calls, sentence-level grounding checks and human handoff, independent
  of the audio stack.
- Designed an idempotent tool-execution ledger enforced by a Postgres partial
  unique index and a compare-and-set dispatch transition, with reconciliation
  for commit-then-timeout failures: **0 duplicate side effects** across a
  34-case fault-injection suite covering
  interruptions, timeouts, garbage responses and concurrent duplicates.
- Built a deterministic per-sentence grounding checker that blocks invented
  prices, dates and venues before speech at **0.07 ms p95** — **32/32** seeded
  entity hallucinations caught, **7/8** fabricated policy claims, **2/46**
  false positives — with its two blind spots (contradictions, wrong-fact
  numbers) measured and published rather than hidden.
- Calibrated handoff on a 94-case labelled set: raised recall on cases needing
  a human from **0.31 to 0.86** (precision 0.82) by replacing the planned
  RRF-margin signal (AUC 0.51) with raw dense similarity (AUC 0.93) at two
  thresholds; hard-rule handoffs cost **0** LLM calls.
- Built a live observability console (FastAPI + Server-Sent Events) and a web
  app that shows, under every reply, the retrieved chunks, grounding verdicts,
  tool state transitions and latency stages behind it.
- Built a call-analytics feedback loop — simulated callers, HDBSCAN clustering
  and per-call failure attribution — that found all 3 knowledge-base gaps
  planted in the corpus; applying its top automatable fix raised containment
  from **73.7% to 79.3%** over 300 simulated calls (paired +17/−0, p < 0.001)
  while the wrong-answer rate fell (13.6% → 11.3%). *(Offline agent, simulated
  callers — say so.)*
- Found and fixed a concurrency race in the idempotency ledger and a
  tie-ordering bug in the retrieval confidence signal through measurement, not
  code review — the latter raised handoff precision from 0.75 to 0.82.

- Benchmarked self-hosted Whisper on 900 synthesized call recordings across
  noise levels and 8 kHz phone audio: WER stayed under 6% while key-entity
  accuracy fell from **83% to 62%** at 5 dB — spoken emails worst at **43%** —
  showing WER alone hides the failures that break a support call.

- Ran the same feedback cycle with a live LLM (Gemini 2.5 Flash on Vertex AI):
  the gap report named the same three gaps at the same ranks, and the fix moved
  its target intent +16/−0 (p < 0.001) while the guardrails held — showing the
  loop is model-independent, and quantifying the live model's own failure
  modes (29% citation rate, 3.3% of calls lost to provider timeouts).

**Fill in with API keys (leave the blanks until measured):**

- Achieved **X ms p50 / Y ms p95** voice-to-voice latency, with VAD
  endpointing reported separately from harness latency.
- The containment bullet above, re-run with Claude as the agent (`make cycle`
  with `ANTHROPIC_API_KEY` set) — then the "offline agent" caveat goes away.

**Which to lead with:** harness + idempotency for AI/ML engineering and
forward-deployed roles; the console and retrieval finding for data roles.
