# Checker catch rate by hallucination class (W3-9)

**Date:** 2026-10-13 · **Raw:** `raw.jsonl` (every seed, every verdict) · **Script:** `eval/seed_hallucinations.py`

## 1. What was this run for

To measure what the grounding checker catches, class by class, with the
classes it is not designed to catch kept in the same table, and with the
false-positive rate beside it. A checker that rejects everything scores 100%
catch and is useless.

Seeds come from real corpus chunks. The known-good answers are in
`eval/dataset/checker_known_good.yaml`, shared with `tests/test_checker.py`.
Rule 2 cases use **what real retrieval returns** for the question that would
provoke each claim, because that is the only condition Rule 2 operates in.

## 2. Final numbers (after the changes in §3)

| class | caught | rate | 95% CI | note |
|---|---|---|---|---|
| numeric | 23/23 | 1.00 | [0.86, 1.00] | |
| date | 5/5 | 1.00 | [0.57, 1.00] | |
| entity (invented venue) | 4/4 | 1.00 | [0.51, 1.00] | |
| policy claim, uncovered topic | 7/8 | 0.88 | [0.53, 0.98] | Rule 2 |
| contradiction of a covered fact | 0/9 | 0.00 | [0.00, 0.30] | **declared gap** |
| binding (right number, wrong fact) | 1/4 | 0.25 | [0.05, 0.70] | **declared gap** |
| citation (wrong [K] tag) | 0/22 | 0.00 | [0.00, 0.15] | not an enforcement target |
| **false positives on correct answers** | **2/46** | **0.04** | [0.01, 0.15] | |

**Checker latency:** p95 0.07 ms per segment (tests/test_checker.py).

## 3. What the study found and changed

The first run read: policy claims **1/8**, false positives **5/33**. Both were
wrong for interesting reasons.

**A production bug: event dates from tools could never ground.**
`canonicalize()` read the ISO date `2026-11-14` as a *range* and turned it into
"2026 to 11-14". So after a `lookup_order`, the agent saying "your show is on
November 14" would always fail the checker and be replaced by a safe line.
That accounted for all 5 initial false positives, and no unit test had
grounded a date from a tool result. Fixed, with a regression test.

**Most of the first "policy claim" seeds were really contradictions.** Claims
like "you can split one order across many cards", where the corpus says you
can't, share their chunk's vocabulary *by construction*. They belong to the
declared negation gap, not to Rule 2's target. They were moved to a
`contradiction` class and replaced with fabricated rules in topics no ingested
document covers.

**Rule 2's overlap threshold was too permissive.** Retrieval returns
topically *adjacent* chunks because they share words with the question, so a
25% vocabulary overlap is met by fabrications too. A sweep:

| claim_overlap_min | fabricated claims caught | correct weak-retrieval answers rejected |
|---|---|---|
| 0.25 (week 2) | 3/8 | 2/10 |
| 0.35 | 6/8 | 2/10 |
| **0.50 (chosen)** | **7/8** | **2/10** |
| 0.65 | 7/8 | 2/10 |
| 0.80 | 7/8 | 3/10 |

**A grounded venue name shielded fabricated rules.** "You must wear smart
dress for the late session at Lumen Hall" passed, because Lumen Hall is a real
venue, and any grounded span switched Rule 2 off. A real name proves the venue
exists, not the rule. Names the lexicon knows no longer count as anchoring a
claim.

## 4. The two remaining false positives

- *"Tickets aren't refundable if you can't attend, but you can transfer them
  to someone else at no charge."* → `fail_unsupported_claim`. Retrieval for
  that question was weak (0.645) and returned the transfers document first,
  so the refund half of a correct answer looked unsupported. This is the cost
  of Rule 2, and it lands on the most common intent in the plan.
- *"…face value plus 10 percent."* → `fail_ungrounded_entity`. **The checker
  was right**: the price-cap chunk was not among the four retrieved, so "10
  percent" really was absent from the model's context. This is a retrieval
  miss surfacing safely, with a safe line instead of an answer, and not a
  checker defect.

## 5. Limits

- **In-sample.** The 0.50 threshold and the lexicon-anchor rule were chosen by
  looking at these seeds, so 7/8 is optimistic for new fabrications.
- **Programmatic corruptions are easy.** A perturbed number is exactly what
  Rule 1 is built for. Real model hallucinations will be subtler, which is why
  the binding and contradiction rows matter more than the 23/23.
- Small n everywhere, as the CIs show. One author.

## 5b. What changed

- `harness/normalize.py`: ISO dates canonicalized before ranges.
- `harness/checker.py`: lexicon-only spans do not anchor a claim.
- `handoff.yaml`: `claim_overlap_min` 0.25 → 0.50, naming this run.

## 6. Reproduce

```bash
make up && make ingest
.venv/bin/python -m eval.seed_hallucinations
```
