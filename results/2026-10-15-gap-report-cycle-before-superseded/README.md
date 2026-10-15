> **SUPERSEDED** by `../2026-10-15-gap-report-cycle-before/`. This run used the pre-fix retrieval, where `top_vec_score` was
> the cosine of the *fused* rank-1 chunk instead of the best dense match, so lexical ties
> changed the handoff signal between runs (see WEEK4.md). Kept for the record; do not quote.

# Gap report on the cycle baseline (W4-3 · W4-4 · W4-5)

**Date:** 2026-10-15 · **Input:** simulation run `cycle-before-01M5049K` (300 calls, seed 20261015, offline extractive agent)
**Files:** `report.txt` (what an operator reads), `metrics.json` (call ids stripped), `config.json`
**Command:** `make gap-report RUN=cycle-before-01M5049K`

## 1. What was this run for

To build the gap report the feedback cycle acts on. Afterwards, the report was scored
against the simulator's hidden labels (`simulation_label` events), which the report itself
never reads.

## 2. What it showed

**Ranking.** Actions are ranked by the troubled calls they would fix, summed per
(diagnosis, document). Clusters are ranked separately.

| rank | diagnosis → target | troubled calls | planted? |
|---|---|---|---|
| 1 | weak_coverage → 02-ticket-transfers.md | 20 | no: refund questions retrieve the transfers doc |
| 2 | **held_out_coverage → 11-weather-and-outdoor-events.md** | 19 | **yes** |
| 3 | weak_coverage → 04-delivery-mobile-and-willcall.md | 18 | no: vague callers ("the app isn't working") |
| 4 | **held_out_coverage → 10-group-and-vip-packages.md** | 13 | **yes** |
| 5 | weak_coverage → 08-payment-and-billing.md | 9 | no |
| 6 | **held_out_coverage → 12-age-restrictions-and-id.md** | 8 | **yes** |

**Scored against the hidden labels:**

- All 3 planted gaps were found, at ranks 2, 4 and 6, with the right diagnosis: *held out*,
  not *no coverage*.
- 1 of 3 was in the top 3.
- No top-3 action is a false positive. Ranks 1 and 3 are genuine unplanted failures (refund
  resolved rate is 0.48 by the hidden label).
- Mean cluster purity is 0.985.

**What did not meet the bar:**

- **Noise bucket: 94/300 = 31%** against a W4-3 target of under 15%. HDBSCAN leaves vague and
  multi-topic openings ("Hi, um, it's about my order") unclustered. This is why the report
  ranks *actions* (per-call attribution) and not clusters: a call's diagnosis does not depend
  on it landing in a cluster.
- **Planted gaps in the top 3: 1/3** against the W4-5 bar of all three. Version 1 of the report
  ranked clusters and found **0/3**, because one topic split across five phrasings lost to
  single well-worded clusters. Restructuring to rank actions is what made all three surface.
  The remaining gap to the bar comes from real problems outranking them, so the ranking was
  left honest rather than tuned to the planted set.

## 3. What changed as a result

- The feedback cycle (`../2026-10-15-feedback-cycle/`) applied action #2, the
  highest-ranked action that needs no human author.
- Known limitation, written up rather than hidden: the cluster *names* come from TF-IDF
  terms, not from the one LLM naming call per cluster that BUILD.md specifies (no API key).
