# Week 4: status

**Simulated callers, the gap report, one measured feedback cycle, the static report, and
the web app.** As in week 3, nothing called a paid API. There is no key in this
environment, so the agent in every simulated call is the **offline extractive agent**
(`providers/llm/extractive.py`). It quotes the best-matching retrieved sentence and has no
language model, and every result says so.

```
366 tests passing · 34 fault-injection tests · ruff clean
```

## Run directories

| Run | What it measured |
|---|---|
| `results/2026-10-15-feedback-cycle/` | baseline → gap report → one change → identical seeds; both guardrails; paired test |
| `results/2026-10-15-gap-report-cycle-before/` | the gap report on the baseline, scored against hidden labels afterwards |
| `results/2026-10-15-retrieval-calibration/` | re-run after the `top_vec_score` fix (supersedes 2026-10-13) |
| `results/2026-10-15-handoff-calibration/` | re-run after the same fix (supersedes 2026-10-13) |
| `results/2026-10-15-checker-catch-rate/` | re-run after the week-4 checker fix; identical numbers |
| `*-superseded/` | the first cycle and gap report, run on the pre-fix signal. Kept for the record, not quoted |

---

## What each task delivered

| Task | State | Evidence |
|---|---|---|
| **W4-1** caller simulator | done, **deviation** | 12 weighted intents × 6 personas (interrupts, rephrases, misstated order ids, vague openings). Template callers, not an LLM (no key). Hidden labels go into a `simulation_label` event. Same seed → identical transcripts: 266/280 non-target calls identical across the two cycle runs. The 14 that differ are explained under *Retrieval determinism* below |
| **W4-2** event-completeness audit | done | `tests/test_event_completeness.py`: 10 call paths together emit every type in the closed vocabulary. It found 3 types nothing emitted, 1 nothing *could* emit, and a bug that meant unknown types had never raised in any test |
| **W4-3** intent clustering | done, **short of the bar** | BGE query embeddings → HDBSCAN → TF-IDF names (no LLM naming call; no key). Purity **0.985**, but the noise bucket is **31%** against a < 15% target |
| **W4-4** gap report | done | six diagnoses; actions ranked by troubled calls per (diagnosis, document); each action names a file |
| **W4-5** gap report validation | done, **short of the bar** | all **3/3** planted gaps found with the right diagnosis (ranks 2, 4, 5), **1/3** in the top 3. The two actions above them are real, unplanted problems. Version 1 found 0/3 |
| **W4-6** static report | done | `make report` → `reports/index.html`: one self-contained file, inline SVG, 8 sections. Provenance is generated from the same lookups as the numbers. It refuses latency from any run without `timing_valid: true` |
| **W4-7** feedback cycle | done | containment **0.737 → 0.793**, wrong-answer rate **0.136 → 0.113**, paired **+17 / −0** (McNemar p = 1.5 × 10⁻⁵) |
| **W4-8** final README + demo | README done; **no demo video** | a demo video needs live voice, which needs keys |
| *extra* web app | done | `/`: Talk, Voice, Console, Insights, Results views over the real harness (docs/17) |

---

## The feedback cycle

The gap report ranked 11 actions. #1, rewriting the transfers document, needs a human
author, so the script listed it and moved on. It applied #2: ingest the held-out weather
document.

| n = 300, seed 20261015 | before | after |
|---|---|---|
| containment | 0.737 [0.68, 0.78] | **0.793** [0.74, 0.84] |
| handoff rate | 0.253 | 0.200 |
| wrong-answer rate (guardrail) | 0.136 | **0.113** |
| resolved (hidden label) | 0.637 | 0.703 |
| duplicate side effects | 0 | 0 |

- weather_outdoor went from **0/20 to 19/20** helped.
- One call outside that intent changed, in the helpful direction. A caller asked whether
  "the summer opener" was cancelled, and a weather chunk was relevant to the question. It is
  counted, not hidden.
- The 95% intervals overlap, which is why the paired test is the one that matters.

## What the measurements changed

Like week 3, the most valuable output of the week was the bugs that measurement found.

1. **The sufficiency signal was not what the docs said it was.** `top_vec_score`,
   documented as "the best dense cosine", read the cosine of the chunk the *fused* ranking
   put first. Lexical ties were broken by physical row order, which a re-ingest changes, so
   the same query scored **0.670** in one run and **0.621** in the next. That crossed the
   0.62 handoff threshold.
   - **How it was found:** the first cycle showed answers changing for transfer questions
     that retrieved no weather chunk at all.
   - **Fix:** the true maximum, plus explicit tie-breakers on every ORDER BY.
   - **After the fix:** AUC **0.91 → 0.93**; handoff precision **0.75 → 0.82** with recall
     unchanged at 0.86; top-k document accuracy **12/14 → 14/14**.
   - **Re-runs:** both calibrations and the cycle were re-run. The first cycle is kept as
     `*-superseded`.
2. **A real race in the idempotency ledger.** The Postgres concurrent-duplicates test
   failed **3 times in 5** while the machine was under load.
   - **The race:** the loser of the insert found the row still `pending`, received the same
     execution back, and both coroutines dispatched. The unique index guarded the *insert*;
     nothing guarded the `pending → dispatched` transition.
   - **Fix:** that transition is now a compare-and-set (`UPDATE … WHERE status = 'pending'
     RETURNING`), and the loser waits for the winner's outcome. 15/15 passes.
   - **Regression test:** an in-memory test reproduces the race deterministically. It fails
     without the fix.
   - **Scope:** the "0 duplicate side effects" claim held in every reported run; the race
     needed concurrency that the simulated calls never produce. It is fixed rather than
     argued away.
3. **Unknown event types had never raised in a test.** The strict check read
   `PYTEST_CURRENT_TEST` at import time, before pytest sets it. It now reads it per call.
4. **The vocabulary promised events that nothing emitted.**
   - `llm_iteration_started`, `llm_text_completed` and `stt_final` are now emitted.
   - `stt_partial` was removed: partial transcripts never leave Pipecat.
5. **The gap report's first design found 0/3 planted gaps.** One topic split across five
   phrasings lost to single well-worded clusters. Ranking *actions* (troubled calls per
   document to fix) instead of clusters found all three.
6. **Smaller fixes:**
   - The offline agent re-verified an old, misstated order number instead of the corrected
     one.
   - Its lookup rule swallowed "my tickets" policy questions.
   - The checker read "Could I" as a proper noun. The fix was re-measured, with identical
     catch rates.
   - The console stream sent the rest of a batch to a client that had already disconnected.

## Retrieval determinism

Before the fix, re-ingesting a document could change answers to unrelated questions,
because tie order changed. After the fix, the only transcripts that differ between the two
cycle runs outside the target intent are ones where the newly ingested document legitimately
entered the top 4. `tests/test_retrieval.py::test_search_is_deterministic` holds the line.

## Honest limits

- **The agent is not Claude.** Containment, wrong-answer rate and the gap report's
  diagnoses all describe the offline extractive agent. With Claude, the absolute numbers
  will differ, probably upward, since it can combine sentences. Re-run `make cycle` with a
  key; the script records which agent ran.
- **Callers are templates.** The six personas are written by the same author as the corpus.
  That makes them cleaner and more on-topic than people.
- **"Helped" is judged per document.** It is true if the agent cited a gold document, so a
  weaker sentence from the right document still counts. The LLM judge (written in week 3,
  not run) is the fix.
- **The handoff guardrail is a rate, not recall.** The simulator does not label which calls
  should have reached a human. Recall is measured on the labelled set instead.
- **No latency numbers.** The static report refuses to show any, by design, until a live,
  timing-valid run exists.
- **No demo video, no human calibration audio.** Both need things this environment does not
  have.

## Next

1. Add `ANTHROPIC_API_KEY` and re-run `make cycle`. That gives the same protocol with the
   real agent.
2. Ingest group/VIP (the gap report's #4). Then rewrite the refund section (#1) by hand and
   measure that too.
3. Add Deepgram and Cartesia keys for live voice latency, then record the demo.
