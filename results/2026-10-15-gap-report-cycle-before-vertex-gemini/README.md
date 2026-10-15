# Gap report on the Gemini baseline

**Date:** 2026-10-15 · **Input:** simulation run `cycle-before-01M508V3` (300 calls, seed 20261015, Gemini 2.5 Flash on Vertex AI)
**Files:** `report.txt`, `metrics.json` (call ids stripped), `config.json` · **Command:** `make gap-report RUN=cycle-before-01M508V3` (with `--label cycle-before-vertex-gemini`)

## 1. What was this run for
The gap report the Gemini feedback cycle acted on (`../2026-10-15-feedback-cycle-vertex-gemini/`).
It was scored against the hidden labels afterwards.

## 2. What it showed
- **Same planted gaps, same ranks** as the offline-agent baseline: weather **#2**, group/VIP
  **#4**, age/ID **#5**, each diagnosed `held_out_coverage`. 1/3 is in the top 3, with no
  false positives in the top 3. Cluster purity is 0.985, and 94/300 calls are unclustered
  (31%, still above the < 15% target).
- **The diagnosis is model-independent**, because it comes from retrieval signals over the
  caller's words. Swapping the agent from extractive to Gemini changed the *volume* behind
  each action, not which documents were named.
- **Rank 1 changed** to `weak_coverage → 04-delivery-mobile-and-willcall.md` (26 calls). Gemini
  asks follow-up questions on vague delivery calls, and those turns retrieve weakly and hand
  off.

## 3. What changed as a result
The cycle applied action #2 (ingest weather). The gap report CLI now refuses to overwrite an
existing run directory and takes `--label`; without that, this run would have overwritten
the offline agent's gap report.
