# Retrieval signal selection

**Date:** 2026-10-08
**Config:** BAAI/bge-small-en-v1.5, RRF k=60, top_k=4, candidates=20,
corpus hash `98138b54d0e0e5a5` (9 ingested documents, 40 chunks, 3 held out)
**Code:** week 2, W2-3

## 1. What was this run for

To pick the signal that the first soft handoff rule thresholds on — "is there
anything relevant in the knowledge base for this question".

The design in docs/05 §6 assumed two signals: the fused RRF score of rank 1,
and the fused margin `fused(1) - fused(2)`, with margin expected to be the
stronger of the two.

## 2. What it showed

**The fused margin is inverted. It must not be thresholded on.**

Median over 12 covered queries (answerable from an ingested document) and
4 uncovered queries (answerable only from a held-out document):

| signal | covered | uncovered | verdict |
|---|---|---|---|
| fused rank-1 score | 0.0327 | 0.0320 | barely separates |
| **fused margin** | **0.00064** | **0.00099** | **INVERTED** |
| raw dense cosine, rank 1 | 0.795 | 0.555 | separates cleanly, no overlap |
| raw BM25, rank 1 | 0.80 | 0.10 | separates cleanly |
| dense margin | 0.077 | 0.012 | separates, one overlap |

The reason is structural rather than incidental. RRF margins are rank
arithmetic: a chunk ranked 1 by both retrievers scores `2/61 = 0.0328`, one
ranked 2 by both scores `2/62 = 0.0323`. The gap is ~0.0005 whether the match
is perfect or hopeless, so what is left is noise from which retriever happened
to rank what.

Raw dense cosine separated with **no overlap at all** on this set: the lowest
covered query scored 0.711, the highest uncovered one 0.563.

### A second finding, from the same run

The lexical half of the hybrid search was silently dead for natural questions.
`websearch_to_tsquery` ANDs every term, so "can I get my money back for a
cancelled show" required all of *get & money & back & cancelled & show* in one
chunk and matched nothing. Only short exact-term queries like "will-call" ever
produced a lexical hit.

Fixed by rewriting the generated tsquery's `&` operators to `|`, which keeps
stemming, stopword removal and phrase handling while letting `ts_rank_cd` rank
by how many query terms a chunk matches. Top-1 document accuracy was 1.00
before and after, so this did not show up as an accuracy regression — it
showed up as `lex_rank IS NULL` on every result.

## 3. What changed as a result

- `RetrievalResult` now exposes four signals, not two. `top_vec_score` is the
  primary sufficiency signal; `margin` is kept for calibration and explicitly
  marked as not-a-sufficiency-signal.
- `migrations/004_retrieval_signals.sql` persists all four per turn, so W3-7
  calibrates from the full eval distribution rather than from these 16 queries.
- `tests/test_retrieval.py::test_fused_margin_is_not_used_as_a_sufficiency_signal`
  guards the finding: if the fused margin ever starts separating correctly,
  the study gets re-run before anything relies on it.
- The lexical query builder ORs its terms.

## 4. Limits of this result

**n = 16.** This is a signal *selection* study, not a calibration. It says
which signal is worth thresholding; it does not say where the threshold goes.
Confidence intervals on 4 uncovered queries are enormous and no threshold
value is quoted here. `handoff.yaml` stays `PROVISIONAL` until W3-7 runs the
sweep over the full ~50-case eval set.

The 4 uncovered queries all come from held-out documents the author wrote, so
they are gaps by construction rather than observed gaps.

## 5. Reproduce

```bash
make up && make ingest
.venv/bin/python -m pytest tests/test_retrieval.py -q -s
```
