# BUILD — week by week

Every task has an id, its dependencies, and an acceptance criterion you can actually check.
A task is done when its criterion passes, not when the code is written.

Estimates assume focused days. They are deliberately not padded; §6 is the triage list for
when they turn out to be wrong.

**The apply point is the end of Week 2.** Week 2 ships a demoable system with two of the
four resume bullets already measured. Start applying there, and keep building.

---

## Week 1 — voice loop and harness skeleton

Goal: audio in, audio out, through a harness you own, with a real latency number and clean
barge-in.

### W1-1 · Repo scaffold · ½ day · no deps

`pyproject.toml` with `uv`, `Makefile`, `docker-compose.yml` (postgres + jaeger),
`.env.example`, `pydantic-settings` config object, `ruff`/`mypy` config.
→ [02-repo-and-tooling.md](docs/02-repo-and-tooling.md)

**Accept:** `make up && make test` passes on a clean clone with an empty test suite.
`Settings()` loads from `.env` and every non-secret has a default.

### W1-2 · Architecture contract · ½ day · W1-1

`import-linter` contract forbidding `harness`/`retrieval` from importing
`pipecat`/`app`/`eval`, plus `tests/test_no_pipecat_in_harness.py` as a backstop.

**Accept:** adding `import pipecat` to any file under `harness/` makes `make test` fail.
Verify by actually doing it once, then reverting.

> Do this on day one. Retrofitting the boundary after the pipeline exists is a rewrite,
> and the boundary is what the entire test strategy rests on.

### W1-3 · Migrations · ½ day · W1-1

`001_init.sql`, `002_corpus.sql`, `003_eval.sql`.
→ [03-data-model.md](docs/03-data-model.md)

**Accept:** `docker compose down -v && make up` applies all three cleanly. Every query in
[03-data-model.md](docs/03-data-model.md) §5 runs against the empty schema without error.

### W1-4 · Provider interfaces · 1 day · W1-1

ABCs for `LLMProvider`/`STTProvider`/`TTSProvider`, `null` implementations, and
`providers/llm/replay.py` with its YAML fixture format.
→ [04-providers.md](docs/04-providers.md)

**Accept:** a test drives `replay.py` through a fixture containing text, a tool call, and
`StreamEnd`, asserting event order and that cancellation mid-stream raises
`CancelledError` out of the iterator.

### W1-5 · Harness skeleton + text driver · 1½ days · W1-3, W1-4

`state.py`, `loop.py` (no retrieval, no tools yet), `events.py`, stop rules, persistence,
`eval/text_driver.py` with YAML scripts.
→ [06-harness.md](docs/06-harness.md)

**Accept:** `make chat` holds a multi-turn conversation using the replay provider. Every
turn persists a `turns` row with exactly one non-null `stop_reason`. A scripted run
asserting a stop-reason sequence passes in under 200 ms.

### W1-6 · Real providers · 1 day · W1-4

`anthropic.py` (streaming + tool calls + prompt-cache breakpoint), `deepgram.py`,
`cartesia.py`, and `openai.py`.

**Accept:** `make chat` works against the live Anthropic API. The provider-parity test
drives one fixed scenario through Anthropic and OpenAI and asserts both called the same
tool with the same arguments. `cache_read_tokens > 0` on turn 2 of any call.

### W1-7 · Pipecat pipeline + browser client · 1½ days · W1-5, W1-6

`app/pipeline.py`, `app/bridge.py`, `app/main.py`, `app/static/`. All three UI panes —
transcript, trace, latency strip.
→ [10-audio-pipeline.md](docs/10-audio-pipeline.md)

**Accept:** speak into a browser, hear a reply. The trace pane shows tool and checker
events (empty for now) and the latency strip shows per-stage numbers. `harness/` still has
no Pipecat import (W1-2 still passes).

> Build all three panes now. They are the demo, and in Week 4 there will be no time.

### W1-8 · Barge-in and context hygiene · 1 day · W1-7

Cancellation ordering, `spoken_text` reconstruction from the TTS offset, `trim_on_interrupt`
with the invariant assertion.
→ [06-harness.md](docs/06-harness.md) §5, [10-audio-pipeline.md](docs/10-audio-pipeline.md) §3

**Accept:** interrupting mid-sentence stops audio in **under 150 ms** (measured, not
felt). `spoken_text ⊂ approved_text`. The context invariant holds after a double
interrupt. `tests/test_context_hygiene.py` covers both.

### W1-9 · Endpointing tuning · ½ day · W1-7

The five-setting sweep with pause utterances and a written log.
→ [10-audio-pipeline.md](docs/10-audio-pipeline.md) §2

**Accept:** `results/<date>-endpointing/README.md` contains the table (premature-endpoint
count, mean `endpoint_ms`, naturalness 1–5 per setting) and the chosen `stop_secs` is
written into `.env.example` with a comment naming the run.

### W1-10 · OTel + first latency number · ½ day · W1-7

Span tree, attributes, Jaeger export, latency stages persisted to `turns`.
→ [11-observability.md](docs/11-observability.md)

**Accept:** a call appears in Jaeger at `:16686` with the full span tree. `results/<date>-w1-latency/`
holds p50/p95 for **both** voice-to-voice and harness-side over ≥150 turns, with the
per-stage breakdown and the endpointing constant.

**End of Week 1:** latency bullet is measurable.

---

## Week 2 — grounding, tools, handoff · the apply point

### W2-1 · Corpus · 1 day · W1-3

Twelve markdown documents following the writing rules, plus `corpus_manifest.yaml` with
three held out.
→ [05-retrieval.md](docs/05-retrieval.md) §1, [00-decisions.md](docs/00-decisions.md) D4

**Accept:** every section is 60–120 words and answers one question. A linter script asserts
section word counts and that no section contains a table or nested list. The three held-out
documents cover topics genuinely absent from the other nine.

> Write this before the retrieval code. A corpus written to satisfy a retriever you already
> built will flatter it.

### W2-2 · Ingestion · 1 day · W2-1

`chunk.py`, `embed.py` (asymmetric BGE prefixes), `ingest.py`, corpus hash.

**Accept:** `make ingest` is idempotent — a second run writes nothing and prints the same
corpus hash. Held-out docs appear in `documents` with `ingested = FALSE` and zero chunks.
A test asserts `embed_query` and `embed_passage` produce different vectors for the same
text (the prefix is actually applied).

### W2-3 · Hybrid search · 1 day · W2-2

The RRF CTE, conversational query construction, `RetrievalResult`.
→ [05-retrieval.md](docs/05-retrieval.md) §5

**Accept:** p95 under 30 ms. A test set of 10 question→document pairs achieves top-1
document accuracy ≥ 0.8. Follow-up queries ("what about parking?") retrieve correctly when
the previous utterance is supplied, and demonstrably worse without it.

### W2-4 · Prompt assembly · ½ day · W2-3

`prompts.py`, the `<knowledge>` block, the cache breakpoint, `domains/*/prompts.yaml`.
→ [06-harness.md](docs/06-harness.md) §3

**Accept:** answers are 1–2 spoken sentences, cite a `[K*]` tag, and the static prefix is
byte-identical across turns (assert it, don't assume it).

### W2-5 · Segmenter + acknowledgment bypass · ½ day · W1-5

Abbreviation and decimal handling, clause boundaries past 12 words, 25-word hard cap,
short-segment merge, ack templates.
→ [06-harness.md](docs/06-harness.md) §4

**Accept:** a 30-case table test covering `Mr.`, `5.50`, `No. 4`, clause splits, and the
hard cap. `ack_bypass_used` is recorded on `turns`.

### W2-6 · Checker rule 1 · 1 day · W2-3, W2-5

Span extraction, normalization, grounding against chunks → tool results → user utterance.
→ [08-checker.md](docs/08-checker.md) §2

**Accept:** `checker_ms` p95 under 5 ms. On 20 known-good answers the false-positive rate
is **0**. Hand-corrupted numbers, dates, and entities are all caught.

> Build the normalizer before the extractor and test it hardest. A checker that rejects
> correct sentences is worse than none — the fallback line fires constantly and the system
> looks broken.

### W2-7 · Checker rule 2 + lexicon · ½ day · W2-6

Policy-modal patterns, lexical-overlap condition, lexicon built at ingest.
→ [08-checker.md](docs/08-checker.md) §3–4

**Accept:** a fabricated policy claim in an uncovered topic is caught; a correct policy
claim that paraphrases its chunk is not. False-positive rate on the 20 known-good answers
stays at 0.

### W2-8 · Spoken-form normalization · ½ day · W2-5

→ [08-checker.md](docs/08-checker.md) §7

**Accept:** a ~40-pair table test. Listen to `TK-88213` and `sam@example.com` spoken aloud
and confirm they are intelligible. This one is judged by ear, not by assertion.

### W2-9 · Tool registry + mocks · 1 day · W1-5

`tools.yaml`, registry with startup assertions, the six mocks, the seeded dataset,
`faults.py` profiles.
→ [07-tools.md](docs/07-tools.md) §1, §6

**Accept:** startup assertions fail loudly on a tool missing `idempotency_fields` or a
`readback_template` with an unresolvable placeholder. The seeded dataset includes the five
awkward orders.

### W2-10 · Idempotency ledger · 1½ days · W2-9

The state machine, the partial unique index, dispatch-before-call commit, reconciliation,
startup sweep.
→ [07-tools.md](docs/07-tools.md) §3

**Accept:** all five tests in [12-eval.md](docs/12-eval.md) §7.1 pass, including
`commit_then_timeout` and `reconcile_unavailable`. The duplicate query in
[03-data-model.md](docs/03-data-model.md) §5 returns zero rows after the full suite.

### W2-11 · Read-back · 1 day · W2-10

Template rendering from validated args, deterministic affirm/deny, LLM fallback for
`correction`/`unclear`, interrupt-as-deny, timeout.
→ [07-tools.md](docs/07-tools.md) §4

**Accept:** "no, three tickets" is handled as a correction with a fresh read-back, not a
rejection. Interrupting the read-back commits nothing. An ambiguous reply re-asks once,
then hands off.

### W2-12 · Handoff · 1 day · W2-3, W2-11

Hard rules pre-model, weighted soft scoring with decay, the combined classifier call near
threshold, the summary schema and its rendering.
→ [09-handoff.md](docs/09-handoff.md)

**Accept:** "get me a human" hands off with **zero LLM calls** (assert the call count).
The rendered summary contains the "has NOT promised anything" line. Thresholds are tagged
`PROVISIONAL`.

### W2-13 · Interruption + idempotency fault suite · 1 day · W2-10, W2-11

Pull [12-eval.md](docs/12-eval.md) §7.1–7.2 forward. They need only the text driver and the
adversarial mocks — no audio eval set.
→ [00-decisions.md](docs/00-decisions.md) — this is the reordering that makes Week 2
applicable

**Accept:** `make faults` is green and prints
`duplicate side effects observed: 0 / N`. Record `results/<date>-w2-faults/`.

### W2-14 · Demo pass · ½ day · all of Week 2

Run the four-beat demo script end to end. Record it.

**Accept:** a recorded video showing (1) a cited policy answer, (2) a transfer interrupted
mid-read-back with clean recovery and nothing committed, (3) an out-of-policy refund demand
producing a handoff summary, (4) the trace pane showing retrieval scores and checker
verdicts live.

**End of Week 2: apply.** Latency bullet (W1-10) and harness/safety bullet (W2-13) are both
measured and backed by run directories.

---

## Week 3 — evaluation

### W3-1 · Case set · 1 day · W2-1

50 cases in the composition table.
→ [12-eval.md](docs/12-eval.md) §1

**Accept:** composition matches the table. Every case has `should_escalate`. Every
answer case has `citation_docs` **and** `must_not_contain`. A schema validator passes.

### W3-2 · Audio synthesis + human calibration · ½ day · W3-1

Three voices, manifest, plus 10 human recordings of the hardest entity cases.
→ [12-eval.md](docs/12-eval.md) §2

**Accept:** `make synth` regenerates 150 clean files deterministically from the manifest.
`audio_human/` holds 10 recordings of the same text.

### W3-3 · Noise + telephony · ½ day · W3-2

SNR mixing over **active speech frames only**, seeded offsets, the μ-law chain.
→ [12-eval.md](docs/12-eval.md) §3–4

**Accept:** measured SNR of each mixed file is within 0.5 dB of its label — verify with an
independent measurement, since this is the detail that silently invalidates the whole
table. 900 files total. `NOISE_SOURCES.md` records licences.

### W3-4 · Replay runner · 1 day · W3-3

Real-time feed, `--fast` with `timing_valid: false`, `--sample`, `--interrupt-at`,
`--resume`, run-directory output.
→ [12-eval.md](docs/12-eval.md) §5

**Accept:** a 10-case run produces a complete run directory. Replay latency is within 15%
of the W1-10 live number — a larger gap means the real-time feed is wrong.

### W3-5 · metrics.py · ½ day · W3-4

WER, per-entity accuracy, Wilson intervals, percentiles, `summarize()`.

**Accept:** `metrics.json` matches the schema in
[15-results-and-claims.md](docs/15-results-and-claims.md) §3. Every proportion has `n` and
a CI. Wilson is unit-tested against known values.

### W3-6 · Judge + calibration · 1 day · W3-5

Three-dimension rubric, 3 runs with shuffled dimension order, median, inter-run agreement,
`must_not_contain` hard-fail bypass.
→ [12-eval.md](docs/12-eval.md) §6

**Accept:** inter-run agreement ≥ 0.85. You hand-score 20 cases and judge-human agreement
is recorded in the run README. Below 0.85 on either, tighten the rubric before using any
of its numbers.

### W3-7 · Retrieval calibration · ½ day · W3-4

Histograms, threshold sweep, operating point at recall ≥ 0.90 on misses.
→ [05-retrieval.md](docs/05-retrieval.md) §6

**Accept:** `retrieval_calibration.png` committed to the run directory. `handoff.yaml` has
real values with a comment naming the run. All `PROVISIONAL` tags removed.

### W3-8 · Handoff calibration + ablation · 1 day · W3-7

Grow the escalation set to ≥40 positives, PR sweep, per-signal ablation.
→ [09-handoff.md](docs/09-handoff.md) §5

**Accept:** `handoff_pr_curve.png`, chosen threshold at recall ≥ 0.90, the ablation table
(ΔF1 per signal) in the run README, and **every false negative enumerated individually**.

### W3-9 · Checker measurement · ½ day · W3-6

`seed_hallucinations.py` across six classes; catch rate and false-positive rate.
→ [08-checker.md](docs/08-checker.md) §6

**Accept:** the by-class table, including the near-zero `negation` row. Catch rate ≥ 90% on
numeric/date/entity; false positives < 3%.

### W3-10 · faster-whisper + comparison · 1 day · W3-4

`faster_whisper.py` with `streaming = False`, full 900-sample run per provider.
→ [04-providers.md](docs/04-providers.md) §2, [00-decisions.md](docs/00-decisions.md) D14

**Accept:** two run directories. The comparison section is framed as self-hosting cost —
latency delta, entity-accuracy delta, dollars saved — not as a model-quality head-to-head.

### W3-11 · Remaining fault injection · ½ day · W3-4

§7.3 STT faults and §7.4 retrieval/verification faults.

**Accept:** `make faults` green across all four groups, with seeds printed.

### W3-12 · Week 3 write-up · ½ day · all of Week 3

**Accept:** every run directory has a hand-written README answering the three questions in
[15-results-and-claims.md](docs/15-results-and-claims.md) §2.

**End of Week 3:** STT bullet measured; latency and safety bullets now have CIs.

---

## Week 4 — feedback loop

### W4-1 · Caller simulator · 1½ days · W2-14 · **do this first**

Personas, weighted intent distribution, turn loop with probabilistic interruption, seeding,
hidden `simulation_label` events.
→ [13-caller-sim.md](docs/13-caller-sim.md)

**Accept:** `make simulate N=300` produces 300 calls in roughly 20 minutes. Rerunning with
the same seed reproduces identical transcripts. Interruption share matches the persona
weights within 20%. Cost recorded.

> Nothing else in Week 4 works without this. If it slips, everything slips.

### W4-2 · Event completeness audit · ½ day · W4-1

Verify every event type in the closed vocabulary is actually emitted somewhere.

**Accept:** a test asserts that a full simulated call emits every expected event type for
its path. Unknown types raise.

### W4-3 · Intent clustering · ½ day · W4-1

Embeddings → HDBSCAN → one naming call per cluster. Noise bucket reported.

**Accept:** on 300 simulated calls, clusters broadly correspond to the planted intents
(checked **after** clustering, against the hidden labels). Noise bucket under 15%.

### W4-4 · Gap report · 1 day · W4-3

The per-cluster fields, the five diagnoses, `priority = n × failure_rate × confidence`,
the text output.
→ [14-analytics.md](docs/14-analytics.md) §3

**Accept:** the report distinguishes `no_coverage` from `held_out_coverage` from
`threshold_too_strict`. Suggested actions name a specific file or threshold, not a
generality.

### W4-5 · Gap report validation · ½ day · W4-4

Score the report against the hidden labels.

**Accept:** the accuracy block (planted gaps, their ranks, false positives in top 3) is in
the run README. If the planted gaps do **not** surface in the top 3, fix the ranking — that
is a real finding and fixing it is the work.

### W4-6 · Static HTML report · ½ day · W4-4, W3-12

Eight sections, inline SVG, provenance section mandatory.
→ [14-analytics.md](docs/14-analytics.md) §4

**Accept:** `make report` renders one self-contained file. Section 8 lists the run directory
for every number above it. It refuses to render headline latency from a
`timing_valid: false` run.

### W4-7 · The feedback cycle · 1 day · W4-5

The six-step protocol, exactly once, one change.
→ [14-analytics.md](docs/14-analytics.md) §5

**Accept:** before/after table with both guardrails (wrong-answer rate, handoff recall),
identical seeds, both corpus hashes recorded. If containment did not improve, the diagnosis
is written up — that is a complete result, not a failure.

### W4-8 · Final README + demo · 1 day · everything

Architecture, results tables, demo video, the claims-vs-cannot-support table.

**Accept:** every number in the README traces to a run directory. The resume bullets are
filled from `metrics.json` only. A stranger can run `make up && make ingest && make demo`
from the README alone.

---

## 5. Dependency spine

```
W1-1 ─┬─ W1-2
      ├─ W1-3 ──────────────┬── W1-5 ── W1-7 ─┬─ W1-8
      └─ W1-4 ── W1-6 ──────┘                 ├─ W1-9
                                              └─ W1-10
W2-1 ── W2-2 ── W2-3 ─┬─ W2-4
                      ├─ W2-6 ── W2-7
                      └─ W2-12
W1-5 ─┬─ W2-5 ── W2-8
      └─ W2-9 ── W2-10 ── W2-11 ─┬─ W2-12
                                 └─ W2-13 ── W2-14  ◄── APPLY HERE
W2-1 ── W3-1 ── W3-2 ── W3-3 ── W3-4 ─┬─ W3-5 ─┬─ W3-6 ── W3-9
                                      │        └─ W3-7 ── W3-8
                                      ├─ W3-10
                                      └─ W3-11
W2-14 ── W4-1 ─┬─ W4-2
               └─ W4-3 ── W4-4 ── W4-5 ─┬─ W4-6
                                        └─ W4-7 ── W4-8
```

Critical path: **W1-1 → W1-3 → W1-5 → W1-7 → W2-9 → W2-10 → W2-11 → W2-13 → W4-1 → W4-4 → W4-7.**
Anything not on it can slip without moving the end date.

---

## 6. If you are behind

Cut in this order. Each line says what the cut costs.

1. **`openai.py` beyond the parity test.** Costs: a weaker claim about the provider
   interface. Cheapest cut available.
2. **Telephony variant (W3-3 partial).** Costs: two columns of the STT table. The 8 kHz
   result is a nice-to-have, not a headline.
3. **Checker rule 2 (W2-7).** Costs: the `policy_claim` row becomes a declared gap too.
   Honest, just less impressive.
4. **Three voices → one (W3-2).** Costs: the multi-speaker claim. Keep the human
   calibration subset; it is doing more work than voices two and three.
5. **900 samples → 450 (`--sample`).** Costs: wider CIs. Report them.
6. **Static report (W4-6).** Costs: visual polish. The numbers live in `metrics.json` and
   the run READMEs regardless.
7. **The ablation table (W3-8 partial).** Costs: a nice interview answer.

**Never cut:** W1-2 (the boundary), W2-10 (idempotency), W2-13 (the fault suite),
W4-1 (the simulator), W4-7 (the cycle). Those five are the project. Everything else is
supporting evidence.

### Scope that must stay out

Twilio, Kafka, the second domain, the second TTS provider, the agent-assist screen, and
Spring Boot. All of them are listed as stretch goals precisely so that they remain
stretches when Week 3 runs long. Adding one mid-build is how the four weeks becomes seven.

---

## 7. Standing checks

Run before every commit:

```bash
make test        # unit + architecture contract
make faults      # fault injection (from W2-13 onward)
```

Run weekly, and record the output in that week's run directory:

```sql
-- must always return zero rows
SELECT idempotency_key, count(*) FROM tool_executions
WHERE side_effecting AND status='committed' GROUP BY 1 HAVING count(*) > 1;
```

```sql
-- harness health; a spike in 'interrupted' means endpointing, not rude callers
SELECT stop_reason, count(*), round(100.0*count(*)/sum(count(*)) OVER (),1) AS pct
FROM turns GROUP BY 1 ORDER BY 2 DESC;
```
