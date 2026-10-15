> **SUPERSEDED** by `../2026-10-15-feedback-cycle/`. This run used the pre-fix retrieval, where `top_vec_score` was
> the cosine of the *fused* rank-1 chunk instead of the best dense match, so lexical ties
> changed the handoff signal between runs (see WEEK4.md). Kept for the record; do not quote.

# Feedback cycle: one change, measured (W4-7)

**Date:** 2026-10-15 · **Config:** `config.json` · **Metrics:** `metrics.json` · **Raw:** `raw.jsonl` (600 rows, `phase` = before/after)
**Command:** `make cycle` (`.venv/bin/python -m eval.feedback_cycle --n 300`)

> **Read this first: what this does and does not measure.** These are **300 simulated
> callers** (template personas, `eval/caller_sim/`) talking to an **offline extractive
> agent** (`providers/llm/extractive.py`). That agent picks the best-matching sentence from
> the retrieved chunks and has no language model. There is no API key in this environment.
>
> So this run measures **the loop around the model**: retrieval, the grounding checker,
> handoff policy, tools, the gap report, and the protocol itself. It says nothing about
> how Claude would answer, and it says nothing about real callers. Quote it as "simulated
> calls, offline agent", or do not quote it.

## 1. What was this run for

To run the docs/14 §5 protocol exactly once, end to end:

1. Freeze the seed and the agent.
2. Run a baseline.
3. Build a gap report.
4. Apply **one** change.
5. Re-run the identical seeds.
6. Report the outcome and both guardrails.

## 2. What it showed

**The change.** The gap report ranked 12 actions.

- **#1** was `weak_coverage → 02-ticket-transfers.md` (20 calls). That fix means rewriting a document, which needs a human author, so the script listed it and did not attempt it.
- **#2** was the highest-ranked action that can be applied automatically: `held_out_coverage → 11-weather-and-outdoor-events.md` (19 calls).

The script ingested that one document for the re-run only. Corpus hash `98138b54d0e0e5a5 → 0cf91e60e8d8741e`, seed `20261015` for both runs.

| metric (n = 300) | before | after | Δ |
|---|---|---|---|
| containment | **0.737** [0.68, 0.78] | **0.790** [0.74, 0.83] | +0.053 |
| handoff rate | 0.253 [0.21, 0.31] | 0.203 [0.16, 0.25] | −0.050 |
| wrong-answer rate *(guardrail; contained but not helped, of contained)* | 0.131 [0.09, 0.18] | 0.110 [0.08, 0.16] | −0.021 |
| resolved rate *(helped, of all calls)* | 0.640 [0.58, 0.69] | 0.703 [0.65, 0.75] | +0.063 |
| duplicate side effects | 0 | 0 | — |

**Containment rose by handing off less, and the wrong-answer rate did not rise.** That
pairing is the point. Containment alone always improves if you simply stop handing off.

**The intervals overlap, but the comparison is paired.** Identical seeds make every call
a matched pair, so the correct test is McNemar on the discordant pairs. Two Wilson
intervals side by side are the wrong test here.

| paired (same caller, same seed) | gained | lost | exact McNemar p |
|---|---|---|---|
| contained | 16 | **0** | 3.1 × 10⁻⁵ |
| helped (hidden label) | 19 | **0** | 3.8 × 10⁻⁶ |

Calls outside the weather intent that changed outcome: **0**. The effect is attributable
to the one change, which is what running one change buys.

| intent | helped before | helped after |
|---|---|---|
| weather_outdoor | 0/20 | **19/20** |
| every other intent | unchanged | unchanged |

**The handoff guardrail is measured as a rate, not as recall.** docs/14 asks for *handoff
recall*, which needs a label saying which calls *should* have gone to a human. The
simulator does not produce that label. Handoff recall for the deterministic policy is
measured on the labelled set in `../2026-10-13-handoff-calibration/` (0.86).

**How good was the gap report?** It was scored against the simulator's hidden labels, after
it was built, in `../2026-10-15-gap-report-cycle-before/`.

- All 3 planted gaps were surfaced: weather **#2**, group/VIP **#4**, age/ID **#6**.
- Only 1 of the 3 was in the top 3.
- The actions ranked above the planted gaps are **real, unplanted problems**, not false positives:
  - #1 is refund questions retrieving the transfers document (refund_request resolved rate 0.48).
  - #3 is vague callers ("the app isn't working for me") that no document can answer.
- The W4-5 bar was "planted gaps in the top 3". It was met for 1 of 3. The ranking was not
  tuned to pass it, because the higher-ranked items are the more valuable findings.

## 3. What changed as a result

- **Nothing in the repo's corpus.** The script restores the documented baseline afterwards,
  because the held-out documents are part of the experimental design (docs/00 D4) and the
  test suite depends on them. To adopt the change, set `ingested: true` for
  `11-weather-and-outdoor-events.md` in `domains/ticketing/corpus_manifest.yaml`.
- **The next cycle's change** should be action #4 (ingest group/VIP: 18 of 19 calls hand off
  today). Alternatively, a human rewrite of the refund section, which is action #1.
- **Not yet measured:** the same cycle with Claude as the agent. Rerun
  `make cycle` with `ANTHROPIC_API_KEY` set. The script records which agent ran.
