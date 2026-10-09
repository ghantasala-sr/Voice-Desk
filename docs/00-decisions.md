# 00 — Locked decisions

Every decision here is locked for v1. Each records why, what it costs, and how expensive
it is to reverse. Revisit only with a reason, and record the reversal in this file.

---

## D1. Spring Boot is not in v1

**Decision.** The voice path is Python-only. Tools are Python mocks that match the request
and response shape of the intended Spring Boot ticketing services.

**Why.** The tool registry already defines the integration seam. A mock behind that seam
proves the design as well as a real service does, and a second language on the critical
path costs a week that is budgeted for evaluation. The resume claim is "designed for a
real backend behind the registry," which the mocks substantiate.

**Consequence.** Mocks must be adversarial, not convenient — see D9.

**Reversal cost.** Low. Swap the `endpoint` field in `tools.yaml` from `mock://` to
`http://`, implement one HTTP executor. No harness change.

---

## D2. TTS provider is Cartesia

**Decision.** Cartesia Sonic as the single v1 TTS, behind `TTSProvider`.

**Why.** Time-to-first-byte sits directly inside the voice-to-voice latency you will
report, and Cartesia is consistently the lowest-TTFB mainstream option with a
first-class Pipecat service. ElevenLabs Flash is the documented fallback if quota or
voice quality becomes a problem.

**Consequence.** One more API key and one more vendor dependency in the demo path. Cache
TTS audio for eval replays so the eval suite does not burn quota.

**Reversal cost.** Low. One new `TTSProvider` implementation.

---

## D3. Default LLM is Claude Sonnet 5

**Decision.** `claude-sonnet-5` is the default. `providers/llm/openai.py` exists and is
exercised by at least one test so the interface stays honest.

**Why.** Strong tool-calling, fast enough for the hot path, streaming with tool use.
Keeping a second implementation working is what makes `LLMProvider` a real abstraction
rather than a wrapper around one vendor.

**Consequence.** Prompt tuning is model-specific; do not claim provider parity you have
not measured.

**Reversal cost.** Low by design. `--llm openai` flag.

---

## D4. The corpus is written with held-out documents

**Decision.** Author roughly twelve knowledge-base documents. Ingest nine. Three are
held out at build time and are not in the index when the baseline is measured.

**Why.** The Week-4 containment improvement is circular if you delete a document you know
is needed and then restore it. Holding documents out *before* you know which gaps matter
makes the gap report's discovery genuine: the analytics job tells you which held-out topic
is actually costing calls, and you ingest that one.

**Consequence.** Baseline containment will look worse than it could. That is the point —
the delta is the result, and it is honestly earned. Disclose the construction in
`results/`.

**Reversal cost.** High after the fact. This decision must be made before the corpus is
written, because it determines what you write.

---

## D5. Eval audio is synthesized, not self-recorded

**Decision.** Base utterances are rendered with 3 TTS voices (distinct gender, pace, and
accent). A 10-utterance human-recorded calibration subset is recorded separately.

**Why.** Recording 50 utterances × 5 variants yourself is the single largest schedule
risk in the original plan, and a single-speaker test set confounds every STT finding with
one accent. Synthesis is free, reproducible, multi-speaker, and regenerable whenever the
test set changes.

**Consequence.** TTS audio has no disfluencies, no breath, no room tone — it is
out-of-distribution for ASR and will inflate accuracy. Mitigated by the human calibration
subset: report synthesized and human WER side by side, and state the gap.

**Reversal cost.** Low. Recording more human samples later only improves the calibration.

---

## D6. Retrieval is hybrid (BM25 + dense), not pure vector

**Decision.** Postgres full-text search and pgvector cosine, fused with Reciprocal Rank
Fusion.

**Why.** The corpus is a handful of policy documents where exact terms carry the meaning —
"will-call", "postponement", "resale", "ADA". Pure dense retrieval over a tiny corpus is
the weakest available option, and your first soft handoff signal depends on retrieval
score quality.

**Consequence.** Slightly more ingestion and query code; one extra index. Negligible
latency (both run in one SQL statement over a few hundred rows).

**Reversal cost.** Low.

---

## D7. Handoff thresholds are calibrated from the eval score distribution

**Decision.** No hand-picked constants. Every threshold in `handoff.yaml` is derived from
the eval set's score distribution and the derivation is committed as a plot plus a table
in `results/`.

**Why.** Bi-encoder cosine scores are poorly calibrated and compress into a narrow band; a
relevant hit and an irrelevant one may be 0.61 and 0.52. An arbitrary threshold is both
fragile and indefensible in an interview. A calibrated one is a result.

**Consequence.** Thresholds cannot be finalized until the eval set exists (Week 3). Week 2
uses provisional values explicitly labelled `PROVISIONAL` in `handoff.yaml`.

**Reversal cost.** None — recalibration is a script.

---

## D8. A synthetic caller simulator is a first-class component

**Decision.** `eval/caller_sim/` is built in Week 4 **before** anything else that week. It
drives LLM personas through the text-mode driver to produce a few hundred calls.

**Why.** The feedback loop in the original plan has no source of call volume. Gap
clustering over the ~30 calls you generate by hand-testing produces nothing. The
containment-improvement result depends entirely on this existing.

**Consequence.** Synthetic calls are not real traffic. Every analytics result must say so.
What the loop demonstrates is that the *mechanism* finds real gaps in the knowledge base,
which is the claim worth making.

**Reversal cost.** N/A — without it Week 4 has no deliverable.

---

## D9. Tool mocks are adversarial

**Decision.** Mocks inject configurable latency, duplicate delivery, and commit-then-timeout
ambiguity. The default profile is benign; the fault-injection profile is not.

**Why.** "Zero duplicate side effects" measured against a dictionary write you control is
near-tautological. The genuinely hard case is: the call was dispatched, it timed out, and
you do not know whether it committed. That case is what the idempotency protocol exists to
survive, so it must be in the test suite.

**Reversal cost.** N/A.

---

## D10. Checker is deterministic, with a declared coverage boundary

**Decision.** No per-sentence LLM call in v1. Entity/number/date grounding plus a
deterministic unsupported-policy-claim rule (see [08-checker.md](08-checker.md)). Catch
rate is reported **by hallucination class**, with the uncovered class visible in the table.

**Why.** Entity matching catches invented prices and dates. It cannot catch "you can
transfer tickets any time before the event" — no entities, entirely wrong, and a more
common failure than invented numbers. Reporting a single blended catch rate would hide
that. A metric with a labelled hole is stronger than one with an unknown hole.

**Reversal cost.** Low. An LLM second-pass check can be added behind the same interface,
at a latency cost that must then be measured.

---

## D11. Latency is reported in two numbers, always

**Decision.** Every latency claim reports **voice-to-voice** (includes VAD endpointing) and
**harness-side** (end of endpointing → first TTS audio byte) separately, plus the
endpointing constant in use.

**Why.** The largest single term in perceived latency is usually how long you wait in
silence before deciding the caller stopped — typically 500–800 ms, a Pipecat setting, and
nothing to do with the harness. A single blended number invites a knowledgeable reader to
assume you do not know what is in it.

**Reversal cost.** N/A.

---

## D12. p95 is computed over turns, with ≥150 samples

**Decision.** Latency runs are 50 cases × 3 repetitions minimum. Sample size and the
repetition count appear in every latency table.

**Why.** A p95 over 50 samples is the second-worst observation. Latency is measured
per-turn, so repetitions are cheap and fix the problem outright.

---

## D13. Handoff precision/recall reports a confidence interval

**Decision.** Grow the `should_escalate` subset to ≥40 labelled cases (candidates generated
by the caller simulator, labelled by hand). Report Wilson score intervals alongside point
estimates.

**Why.** With 15 positive labels the interval is roughly ±20 points, which makes the point
estimate meaningless. Either widen the set or show the interval; this does both.

---

## D14. The STT comparison is framed as self-hosting cost

**Decision.** The report is titled *what self-hosting ASR costs on a real-time path*, not
*faster-whisper vs Deepgram WER*.

**Why.** faster-whisper is a batch model. Streaming it means VAD-chunking and re-running,
which costs latency and hurts accuracy at chunk boundaries — precisely where entity
accuracy is measured. Deepgram is a true streaming model. A head-to-head WER table invites
a conclusion about model quality the setup cannot support.

**Consequence.** The honest finding — latency delta, entity accuracy delta, dollars saved —
is a better result anyway.

---

## D15. Static HTML report, not a dashboard app

**Decision.** `analytics/report.py` renders a single self-contained HTML file from the gap
report and the latest eval run. No server, no JS framework.

**Why.** The demo value is identical and it costs a fraction of the Week-4 budget, which is
already the tightest week.

---

## D16. Tracing backend is Jaeger in docker-compose

**Decision.** Local Jaeger all-in-one. No SaaS tracing account.

**Why.** No vendor on the critical path, no signup in the demo, works offline.

---

## D17. Demo is local + recorded video; Cloud Run is optional

**Decision.** Build the Dockerfile and the Cloud Run config, but do not run a
min-instances=1 service continuously. Deploy near application time if a live link is
needed.

**Why.** min-instances=1 bills around the clock for a portfolio project. A recorded demo
video plus a reproducible `make demo` is equally credible and free.

---

## D18. Turn-taking tuning is budgeted work, not a config afterthought

**Decision.** Week 1 reserves explicit time (task W1-7) for endpointing and interruption
tuning, with a written tuning log.

**Why.** Endpointing is the largest latency term and the dominant driver of perceived
quality. A technically correct agent that talks over people demos badly, and no amount of
harness correctness compensates.
