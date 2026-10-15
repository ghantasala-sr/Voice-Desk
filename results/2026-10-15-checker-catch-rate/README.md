# Checker catch rate: re-run after the week-4 checker fix

**Date:** 2026-10-15 · **Supersedes:** `../2026-10-13-checker-catch-rate/` (same suite, same cases)

## 1. What was this run for
Week 4 changed the grounding checker. The offline agent's "Could I get your order number…"
had been rejected because "Could I" was read as a proper noun. The fix added sentence-opener
function words to `_NOT_ENTITIES` and strips non-entity words from both ends of a span.
Any checker change re-runs the catch-rate suite before it can ship.

## 2. What it showed
`metrics.json` is **identical** to the 2026-10-13 run apart from the run name:

- numeric / date / entity: 23/23, 5/5, 4/4
- policy claims: 7/8
- false positives: 2/46

The fix removed a false positive without losing any catches.

## 3. What changed as a result
Nothing further. The 2026-10-13 README remains the full write-up; quote either directory.
