# Handoff calibration — deterministic policy (W3-8)

**Date:** 2026-10-13 · **Config:** `config.json` · **Raw:** `raw.jsonl` (every decision, every config)
94 labelled cases (`eval/dataset/cases.yaml` + `cases_escalation.yaml`), **42 should_escalate**.

## 1. What was this run for

To measure handoff precision and recall against labels, and to choose the
handoff policy from the numbers rather than from the plan.

**Scope — read this first.** Every case runs through the real harness (real
retrieval, pre-gate, soft scoring), but with the model's reply held constant
as one neutral sentence. So this measures the **deterministic** half of the
handoff decision: hard utterance rules and retrieval signals. It does not
measure handoffs triggered by the *model's* answer (a checker failure, or the
model offering a colleague itself) — that needs a live model, and there is no
API key in this environment.

## 2. What it showed

### The first run: the planned policy barely escalated

| policy | precision | recall |
|---|---|---|
| as designed in week 2 | 13/13 = **1.00** [0.77, 1.00] | 13/42 = **0.31** [0.19, 0.46] |

Every handoff came from a hard utterance rule. **Zero of six held-out-topic
questions escalated.** The cause is arithmetic: weak retrieval was worth 0.45
against a 0.6 threshold, so on a first turn it could never hand off on its
own. The design assumed that the model would say "let me get a colleague" and
the checker would add the rest. Maybe it will, but this run cannot show it,
and a design that depends on an unmeasured behaviour for its recall has not
been shown to work.

Letting weak retrieval hand off alone raised recall to 0.88 but produced 25
false handoffs, including "I can't make Saturday, do I get my money back?"
(the most common intent in the plan). One threshold cannot do both jobs: the
retrieval calibration showed the distributions overlap.

### The change: two retrieval levels

- **weak** (top_vec < 0.688, from the retrieval calibration) still only
  *contributes* 0.45;
- **very weak** (top_vec < 0.62) may hand off **alone**.

0.62 was chosen at the knee of a sweep (0.55 → 0.688). Going on to 0.688 added
3 catches for 6 false handoffs, mostly everyday refund questions.

Two hard-rule phrasings also slipped through and were fixed: "I'm
**disputing** this charge" and "I **didn't authorize** it".

### The calibrated policy

| view | n | precision | recall | F1 |
|---|---|---|---|---|
| all cases | 94 | 36/48 = **0.75** [0.61, 0.85] | 36/42 = **0.86** [0.72, 0.93] | 0.80 |
| excluding tool-path cases* | 79 | **0.80** [0.66, 0.89] | **0.86** [0.72, 0.93] | 0.83 |

\* In production a tool call marks the turn transactional and suppresses the
retrieval signal. With a neutral reply no tool runs, so in the all-cases view
these show up as false positives that the real system would not produce.

**Hard-rule handoffs spent 0 LLM calls.**

### Ablation

| remove | recall | Δ recall |
|---|---|---|
| nothing (calibrated) | 0.86 | — |
| the "very weak" level | 0.36 | **−0.50** |
| the hard utterance rules | 0.74 | −0.12 |

The strict retrieval level is now the largest single contributor.

### Every false negative (6)

| case | wanted | top_vec | why it was missed |
|---|---|---|---|
| `vip_contents` | gap | 0.696 | "Harbour Pavilion" matches venue chunks, so the score stays high |
| `id_for_late_show` | gap | 0.640 | weak but not very weak; would need a second signal |
| `carer_ticket_free` | judgment | 0.738 | retrieves the accessibility doc strongly; the *judgment* needs a classifier |
| `account_hacked` | security | 0.707 | topically close to transfers |
| `esc_exception_bereavement` | out-of-policy refund | 0.672 | refund doc is a good match; the *exception* is the issue |
| `esc_fake_tickets_my_name` | security | 0.628 | just above the very-weak line |

Three of the six are **judgment** cases, where retrieval is *good*. No
retrieval threshold can catch them, and they are the argument for building the
`refund_outside_policy` / `out_of_scope` classifier that docs/09 describes and
v1 does not have.

### False positives (12, of which 3 are tool-path artefacts)

- **4 out-of-scope** ("tell me a joke", "capital of France", arithmetic,
  restaurants). These hand off because nothing retrieves. They should be
  declined, not escalated, which is again the job of the missing out-of-scope
  classifier.
- **5 covered questions with weak scores**, including "do you have anything
  for people who are hard of hearing?" and "can I bring my guide dog?", where
  the corpus uses different words ("assistive listening", "service animal").
  These are corpus fixes, and the week-4 loop should find them.
- **3 tool-path** (`what_seats`, `wrong_email_verification`,
  `disclose_without_verification`): artefacts, see above.

## 3. What changed as a result

- `handoff.yaml`: new `retrieval_very_weak` signal (weight 0.60,
  threshold 0.62); payment-dispute patterns extended. Both name this run.
- `harness/handoff.py`: the two-level retrieval signal.
- Three tests in `tests/test_handoff.py` lock the behaviour.

## 4. Limits — honest version

- **The calibration set is also the test set.** The 0.62 threshold and the two
  pattern fixes were all chosen by looking at these 94 cases, so the 0.86
  recall is an **in-sample** number and will be optimistic on new phrasings.
  The two pattern fixes alone account for 2 of the 36 catches. A held-out
  split needs more cases than one person writes in a day; the week-4 caller
  simulator is the source for that.
- One author wrote every case and every label.
- Deterministic policy only (see §1).

## 5. Reproduce

```bash
make up && make ingest
.venv/bin/python -m eval.calibrate_handoff
```
