# Retrieval sufficiency calibration: re-run after the top_vec_score fix

**Date:** 2026-10-15 · **Supersedes:** `../2026-10-13-retrieval-calibration/` (same 64 queries, same corpus)
**Config:** `config.json` · **Raw:** `raw.jsonl` · **Command:** `.venv/bin/python -m eval.calibrate_retrieval`

## 1. What was this run for
Week 4 found a bug in the signal this calibration chooses a threshold for.

- **The bug:** `top_vec_score` is documented as the best dense cosine. It was actually the
  cosine of the chunk ranked first by the *fused* (RRF) score.
- **Why it mattered:** that chunk is decided by lexical rank, and lexical ties (equal
  `ts_rank_cd`) were broken by physical row order. A re-ingest changes that order, so the
  same query scored 0.670 in one run and 0.621 in the next.
- **The fix:** `top_vec_score` is now the maximum dense cosine, and every ORDER BY in the
  hybrid query has an explicit tie-breaker.

The 2026-10-13 write-up of the method is still the full account. This README records only
what changed.

## 2. What it showed

| | 2026-10-13 (bug) | **2026-10-15 (fixed)** |
|---|---|---|
| top_vec_score AUC | 0.91 | **0.93** |
| fused-margin AUC | 0.53 | 0.51 |
| threshold at ≥ 90% recall on uncovered | 0.688 | **0.688** (unchanged) |
| recall on uncovered | 26/28 | 26/28 |
| false alarms on covered | 13/36 | **12/36** |
| top-1 document accuracy | 9/14 | **11/14** [0.52, 0.92] |
| top-k document accuracy | 12/14 | **14/14** [0.78, 1.00] |

The headline finding stands, and is slightly stronger: raw dense similarity separates
answerable from unanswerable questions, and the RRF margin is a coin toss. The top-1/top-k
gains come from deterministic tie-breaking, not from any change to the retrievers.

Top-1 misses that remain:
- refund_cannot_attend → transfers doc
- exchange_other_date → transfers doc
- card_keeps_declining → refunds doc

The first two are the same failure the gap report ranks #1: refund questions retrieve the
transfers document.

## 3. What changed as a result
`domains/ticketing/handoff.yaml` now cites this run. The threshold value did not change.
The handoff calibration was re-run on the fixed signal (`../2026-10-15-handoff-calibration/`).
