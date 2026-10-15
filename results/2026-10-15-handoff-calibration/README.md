# Handoff calibration: re-run after the top_vec_score fix

**Date:** 2026-10-15 · **Supersedes:** `../2026-10-13-handoff-calibration/` (same 94 cases, 42 positives, same scope)
**Config:** `config.json` · **Raw:** `raw.jsonl` · **Command:** `.venv/bin/python -m eval.calibrate_handoff`

## 1. What was this run for
To re-measure the deterministic handoff policy after fixing the signal it thresholds on
(see `../2026-10-15-retrieval-calibration/`). Scope is unchanged from 2026-10-13: the real
harness with the model's reply held constant. It measures the hard rules and the retrieval
signals, not handoffs the model would trigger itself.

## 2. What it showed

| policy | precision (all) | precision (no-tool path) | recall |
|---|---|---|---|
| current, 2026-10-13 (bug) | 36/48 = 0.75 | 36/45 = 0.80 | 36/42 = **0.86** |
| **current, 2026-10-15 (fixed)** | **36/44 = 0.82** [0.68, 0.90] | **36/42 = 0.86** [0.72, 0.93] | 36/42 = **0.86** [0.72, 0.93] |
| one level only (as designed) | 15/15 = 1.00 | 15/15 | 15/42 = 0.36 |
| no hard patterns | 31/39 = 0.79 | 31/37 = 0.84 | 31/42 = 0.74 |

- **Recall** is unchanged: the same 36 cases are caught.
- **False positives** fell from 12 to 8. All four that dropped out had been sent to a human
  by `retrieval_very_weak` (< 0.62) on the cosine of the *fused* leader, while a better
  dense match sat lower in the same results. Old → fixed `top_vec_score`:
  - card_keeps_declining: 0.586 → 0.707
  - neg_guide_dog: 0.575 → 0.647
  - transfer_how: 0.616 → 0.657
  - wrong_email_verification: 0.611 → 0.621
- **The remaining 8:** refund_timing_cancelled, what_seats, restaurant_recommendation,
  arithmetic, disclose_without_verification, neg_hearing_loop, neg_capital_france,
  neg_tell_joke. Most are off-topic questions, where handing off is arguably right.
- **The 6 false negatives** are the same as before (listed in `metrics.json`).
- Hard-rule handoffs still cost **0** LLM calls.

Still in-sample: the thresholds were chosen on these cases.

## 3. What changed as a result
`handoff.yaml` cites this run. No threshold changed.
