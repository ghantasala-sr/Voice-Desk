# Feedback cycle with a live model: Gemini 2.5 Flash on Vertex AI

**Date:** 2026-10-15 · **Config:** `config.json` · **Metrics:** `metrics.json` · **Raw:** `raw.jsonl` (600 rows, `phase` = before/after)
**Command:** `make cycle` with `VD_LLM_PROVIDER=vertex_gemini` (gcloud credentials, project `garden-app-sg26`, `global` endpoint)
**Same protocol, seed and corpus as** `../2026-10-15-feedback-cycle/` (offline extractive agent).

> **Read this first: what this does and does not measure.** These are **300 simulated
> callers** (template personas) per phase, talking to **Gemini 2.5 Flash** (temperature
> 0.2, thinking off). The model is real; the callers are not.
>
> Two caveats change how the numbers read:
>
> 1. **Gemini is not deterministic.** Identical seeds give identical *callers*, not
>    identical answers, so calls outside the change also move in both directions.
> 2. **"Helped" for policy questions is judged by the document the agent cited.** Gemini
>    cites a `[K]` tag in only **29%** of answered turns (206/711). Its "helped" is therefore
>    a lower bound, and its wrong-answer rate an upper bound. Neither is comparable with the
>    offline agent's, which always cites.

## 1. What was this run for

To run the week-4 feedback cycle with a real LLM as the agent, and to see whether the
loop still works:
- does the gap report still find the gaps?
- does the change still help, and do the guardrails hold?

## 2. What it showed

**The gap report found the same gaps.** The planted gaps rank **#2, #4 and #5**, identical to
the offline-agent run (`../2026-10-15-gap-report-cycle-before-vertex-gemini/`).
- **Why it held:** diagnoses come from retrieval signals, which do not depend on the model.
- **What did change:** #1 is now `weak_coverage → 04-delivery-mobile-and-willcall.md` (26
  calls), where the offline agent's #1 was the transfers document.

**The change** was the same: ingest `11-weather-and-outdoor-events.md` (action #2; #1 needs a
human author).

**The change works where it should:**

| paired, same caller and seed | contained | helped (hidden label) |
|---|---|---|
| **weather_outdoor** (the target) | **+16 / −0**, McNemar p = 3.1 × 10⁻⁵ | **+14 / −0**, p = 1.2 × 10⁻⁴ |
| every other intent | +14 / −22, p = 0.24 | +21 / −25, p = 0.66 |

weather_outdoor went from **0/20 to 14/20** helped. Outside the target, calls moved both ways
with no significant drift. That spread is the run-to-run noise of a sampled model.

**Headline rates.** Denominators exclude calls that ended in a provider error (8 before, 12
after).

| metric | before | after | Δ |
|---|---|---|---|
| containment | 0.678 [0.62, 0.73] (198/292) | 0.715 [0.66, 0.76] (206/288) | +0.037 |
| handoff rate | 0.312 (91/292) | 0.264 (76/288) | −0.048 |
| wrong-answer rate *(guardrail; upper bound, see above)* | 0.232 (46/198) | 0.214 (44/206) | −0.019 |
| resolved rate | 0.520 (152/292) | 0.562 (162/288) | +0.042 |
| duplicate side effects | 0 | 0 | — |

The guardrails held: the wrong-answer rate did not rise, and handoffs fell.

**Where the Gemini agent loses calls:**
- **Handoffs:** `retrieval_very_weak` caused 70 of 91 baseline handoffs. That is the same
  deterministic rule as for the offline agent.
- **Provider errors:** 20 calls died on them, 18 read timeouts (> 10 s) and 2 Vertex 504s;
  none were quota. The 2.5 s turn budget leaves no time to retry, so the caller hears the
  fixed fallback line.
- **Tool loops:** `iteration_cap_reached` grew from 3 to 6 handoffs, from Gemini looping
  tool calls.

**Gemini against the offline agent** (same protocol and seed; not like-for-like, see caveat 2):
- Containment: Gemini 0.678 → 0.715, offline 0.737 → 0.793.
- Gemini's lower baseline comes mostly from more `retrieval_very_weak` handoffs: it asks
  clarifying questions ("which venue?"), and those follow-up turns retrieve weakly.

## 3. What changed as a result

Fixed before this run, found by Gemini smoke tests:
- **The simulator judged the offline agent's wording.** It now judges lookups by whether the
  reply states a true fact from the order record, and answers when asked for the
  recipient's email.
- **Gemini offered to look orders up instead of doing it.** The prompt now says to do it, and
  says policy questions need no order details.

Next:
1. **Raise the citation rate**, or judge answers with the LLM judge (`eval/judge.py`)
   instead of citations. Until then, Gemini's wrong-answer rate cannot be compared with
   another agent's.
2. **Give the turn budget room for one retry** on a provider timeout, or use a regional
   endpoint with a lower tail. Provider errors cost 3.3% of calls.
3. **Claude on Vertex**, once the project has quota, through the same protocol.
