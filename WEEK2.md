# Week 2 — status

**Grounding, tools, handoff, and a live console.** Everything below either
passes a test in this repo or is explicitly marked as not verified.

```
269 passed in ~19s        # 0 skipped — Postgres is up and the corpus is ingested
ruff: all checks passed
duplicate side effects in Postgres (docs/03 §5 query): 0
```

## Watch it

```bash
make up && make ingest
make dev                 # http://localhost:8088/console
make demo-scripts        # in another terminal — four beats, recorded live
make faults              # the fault-injection suite on its own
```

Ports are 5434 / 8088 / 16687 / 4319 because 5432, 8000, 8080 and 4317 are
already taken on this machine (a native Postgres and a SigNoz stack).

---

## What each task delivered

| Task | State | Where | Evidence |
|---|---|---|---|
| **W2-1** corpus | done | `domains/ticketing/corpus/`, `corpus_manifest.yaml` | 12 docs, 40 chunks indexed, 3 docs held out; `retrieval/lint_corpus.py` enforces 60–120 words per section |
| **W2-2** ingestion | done | `retrieval/{chunk,embed,ingest}.py` | idempotent re-run writes 0 chunks; BGE query/passage asymmetry tested |
| **W2-3** hybrid search | done | `retrieval/search.py` | top-1 document accuracy **12/12**, p95 **~20 ms** (budget 40) |
| **W2-4** prompt assembly | done | `harness/prompts.py`, `harness/retriever.py` | static prompt byte-stable across turns (tested); knowledge only after the cache breakpoint |
| **W2-5** segmenter + ack | done | `harness/segmenter.py`, `harness/loop.py` | ack only on a slow first token; `first_content_ms` recorded as the honest headline |
| **W2-6** checker rule 1 | done | `harness/checker.py` | **0/22 false positives** on known-good answers against real chunks; p95 **0.07 ms** |
| **W2-7** checker rule 2 | done | `harness/checker.py` | fabricated claim with weak retrieval caught; a paraphrase is not |
| **W2-8** spoken form | done | `harness/normalize.py` | 64 table tests |
| **W2-9** registry + mocks | done | `domains/ticketing/tools.yaml`, `harness/tools/` | startup assertions; `request_refund` has no `amount` field |
| **W2-10** idempotency ledger | **done + verified against Postgres** | `harness/tools/execution.py` | 5 concurrent identical transfers → 1 effect, 1 committed row; commit-then-timeout reconciles |
| **W2-11** read-back | done | `harness/tools/readback.py` | rendered from the arguments, never the model; ambiguous answers never execute |
| **W2-12** handoff | done | `harness/handoff.py`, `domains/ticketing/handoff.yaml` | hard rules cost **0 LLM calls**; summary states what was NOT promised |
| **W2-13** fault suite | done | `eval/fault_injection/` | 26 tests; **0 duplicate side effects** under the `ambiguous` and `blackout` profiles |
| **W2-14** demo pass | scripted, **video not recorded** | `demo/` | all four beats pass with real retrieval and the checker on; recording a video needs you |
| console | done (built by a parallel agent) | `app/observe.py`, `app/static/console/`, `docs/16-console.md` | verified in the browser against real demo calls |

---

## Things the build found that the spec got wrong

These are the reason to read this file. Each one was found by running the
system rather than by reviewing it, and each now has a test.

1. **The fused RRF margin is an inverted sufficiency signal.** The spec said
   it would be the strongest one. RRF margins are rank arithmetic: ~0.0005
   whether the match is perfect or hopeless. The raw dense cosine separates
   cleanly. → `results/2026-10-08-retrieval-signal-selection/`
2. **The lexical half of hybrid search was dead.** `websearch_to_tsquery`
   ANDs every term, so natural questions matched nothing. Accuracy did not
   drop — it showed up only as `lex_rank IS NULL` everywhere.
3. **Barge-in during playout was ignored.** A turn completes when its text
   reaches TTS, not when playback ends. In live audio most interruptions land
   in that gap, and the Week-1 code treated them as harmless — leaving
   unheard text in history. Week-1 tests missed it because they interrupt
   during generation. → `Harness._barge_in_after_turn`
4. **Weak retrieval on transactional turns accumulated into a handoff.** "yes"
   matches no policy. With decay, three turns of a perfectly good transfer
   summed past the threshold. The signal now applies only to turns that
   answered from knowledge.
5. **The rephrase signal fired on a caller re-issuing a request after cutting
   the agent off.** It now compares only against turns the caller heard in
   full.
6. **"Let me talk to a manager" did not match the human-request rule.** For a
   rule that exists for recall, that is the worst kind of miss.
7. **A provider that raised, instead of returning an error event, crashed the
   call.** Same path as a reported error now: retry once, fixed line, signal.
8. **The checker's Rule 1 is presence-based, not binding-based.** "Refunds take
   10 business days" passes against a chunk mentioning 10 business days in a
   different sentence. Recorded as a second declared gap beside negation.
9. **The console marked every normalized sentence as rejected**, because
   approved text is stored in spoken form. It now shows checker-withheld
   segments from the verdict records.

---

## Deviations from the spec, deliberate

| Change | Why |
|---|---|
| `iteration_cap_reached` is a **hard** handoff | docs/09 gives it a soft weight, but by then the agent has told the caller it is getting help |
| No LLM fallback for read-back classification | Deterministic only. A correction ("no, three tickets") is a deny, and the model proposes the corrected action with a fresh read-back. Anything else unclear re-asks once, then hands off. Never executes on a maybe |
| No combined classifier call (`refund_outside_policy`, `out_of_scope`) | Not built. `distress` and payment disputes are pattern rules. Refund demands reach a human through "talk to a manager" or soft signals |
| Handoff summary has no LLM narrative | Structured fields come from call state, so they cannot be wrong; `suggested_next_step` is a template per reason |
| `approved_text` holds the spoken form | Required for barge-in: the transport's played-character count only means something against what TTS received |
| `harness/assembly.py` | One factory for CLI, scripts and the live pipeline, so they cannot drift |

---

## Not verified

1. **No live model call has been made.** There is no `.env` with an API key.
   Every test and demo uses the replay provider. `make chat` will exercise
   the real stack against Claude once `ANTHROPIC_API_KEY` is set — expect
   prompt tuning, because real models write sentences the fixtures do not.
2. **The Pipecat pipeline has not run.** It now uses the same stack factory,
   but needs `make audio` and Deepgram/Cartesia keys.
3. **Handoff thresholds are PROVISIONAL.** The console already shows why: a
   real paraphrase ("I can't make Saturday and I want my money back")
   retrieved the right policy at 0.564, under the provisional 0.65. That is
   W3-7's job, with the full eval set.
4. **Fixed lines are not in `spoken_text`.** Safe lines, acknowledgments and
   fillers are spoken but excluded (by design, from history). The console's
   "heard N/M chars" therefore omits them. Persisting the full segment list
   would close this.
5. **No demo video.** The four beats are scripted and pass; recording needs
   you, ideally with `make chat` against the live model.

## Next

Week 3 (`BUILD.md`) starts with **W3-1, the 50-case eval set** — and W3-7
calibration is now the most urgent item in it, because the provisional
retrieval threshold is visibly too strict on real paraphrases.
