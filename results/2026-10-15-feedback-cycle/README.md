# Feedback cycle: one change, measured (W4-7)

**Date:** 2026-10-15 · **Config:** `config.json` · **Metrics:** `metrics.json` · **Raw:** `raw.jsonl` (600 rows, `phase` = before/after)
**Command:** `make cycle` (`.venv/bin/python -m eval.feedback_cycle --n 300`)
**Supersedes:** `../2026-10-15-feedback-cycle-superseded/`, which ran on the pre-fix retrieval signal (see WEEK4.md).

> **Read this first: what this does and does not measure.** These are **300 simulated
> callers** (template personas, `eval/caller_sim/`) talking to an **offline extractive
> agent** (`providers/llm/extractive.py`). That agent quotes the best-matching sentence from
> the retrieved chunks and has no language model. There is no API key in this environment.
>
> So this run measures **the loop around the model**: retrieval, the grounding checker,
> handoff policy, tools, the gap report, and the protocol itself. It says nothing about how
> Claude would answer, and it says nothing about real callers. Quote it as "simulated calls,
> offline agent", or do not quote it.

## 1. What was this run for

To run the docs/14 §5 protocol exactly once, end to end:

1. Freeze the seed and the agent.
2. Run a baseline.
3. Build a gap report.
4. Apply **one** change.
5. Re-run the identical seeds.
6. Report the outcome and both guardrails.

## 2. What it showed

**The change.** The gap report on the baseline ranked 11 actions.

- **#1** was `weak_coverage → 02-ticket-transfers.md` (19 calls). That fix means rewriting a document, which needs a human author, so the script listed it and did not attempt it.
- **#2** was the highest-ranked action that can be applied automatically: `held_out_coverage → 11-weather-and-outdoor-events.md` (19 calls).

The script ingested that one document for the re-run only. Corpus hash `98138b54d0e0e5a5 → 0cf91e60e8d8741e`, seed `20261015` for both runs.

| metric (n = 300) | before | after | Δ |
|---|---|---|---|
| containment | **0.737** [0.68, 0.78] (221) | **0.793** [0.74, 0.84] (238) | +0.057 |
| handoff rate | 0.253 [0.21, 0.31] (76) | 0.200 [0.16, 0.25] (60) | −0.053 |
| wrong-answer rate *(guardrail; contained but not helped, of contained)* | 0.136 [0.10, 0.19] (30/221) | 0.113 [0.08, 0.16] (27/238) | −0.022 |
| resolved rate *(helped, of all calls)* | 0.637 [0.58, 0.69] (191) | 0.703 [0.65, 0.75] (211) | +0.067 |
| duplicate side effects | 0 | 0 | — |

**Containment rose by handing off less, and the wrong-answer rate did not rise.** That
pairing is the point. Containment alone always improves if you simply stop handing off.

**The intervals overlap, but the comparison is paired.** Identical seeds make every call a
matched pair, so the right test is McNemar on the discordant pairs:

| paired (same caller, same seed) | gained | lost | exact McNemar p |
|---|---|---|---|
| contained | 17 | **0** | 1.5 × 10⁻⁵ |
| helped (hidden label) | 20 | **0** | 1.9 × 10⁻⁶ |

| intent | helped before | helped after |
|---|---|---|
| weather_outdoor *(the target: its gold doc was ingested)* | 0/20 | **19/20** |
| event_status | 23/29 | 24/29 |
| every other intent | unchanged | unchanged |

**One call outside the target intent changed.** It was a caller asking whether "the summer
opener" was cancelled. A weather chunk entered its top 4 and the call went from handoff to
contained. This is real spillover: the new document is relevant to an outdoor-event
question. It is counted, not hidden, and the script now defines the target intents by the
ingested document, not by "whichever improved".

**The handoff guardrail is measured as a rate, not as recall.** docs/14 asks for *handoff
recall*, which needs a label saying which calls *should* have gone to a human. The simulator
does not produce that label. Handoff recall for the deterministic policy is measured on the
labelled set in `../2026-10-15-handoff-calibration/` (0.86, precision 0.82).

**How good was the gap report?** It was scored against the hidden labels, after it was
built, in `../2026-10-15-gap-report-cycle-before/`.

- All 3 planted gaps were surfaced: weather **#2**, group/VIP **#4**, age/ID **#5**.
- Only 1 of the 3 was in the top 3.
- The actions above them are **real, unplanted problems**, not false positives:
  - refund questions retrieving the transfers document (refund_request resolved rate 0.48);
  - vague callers that no document can answer.

## 3. What changed as a result

- **Nothing in the repo's corpus.** The script restores the documented baseline afterwards,
  because the held-out documents are part of the experimental design (docs/00 D4) and the
  test suite depends on them. To adopt the change, set `ingested: true` for
  `11-weather-and-outdoor-events.md` in `domains/ticketing/corpus_manifest.yaml`.
- **Next cycle:** action #4 (ingest group/VIP: 18 of 19 of those calls hand off today), or a
  human rewrite of the refund section (action #1).
- **Not yet measured:** the same cycle with Claude as the agent. Run `make cycle` with
  `ANTHROPIC_API_KEY` set. The script records which agent ran.
- **Known limitation of "helped":** it is judged at the document level (did the agent cite
  a gold document). Within the right document, the extractive agent sometimes picks a less
  relevant sentence, and "helped" cannot see that. An LLM judge (`eval/judge.py`, written
  and not run) is the fix.
