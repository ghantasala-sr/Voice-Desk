# 09 — Handoff

Handoff is a measurable decision with labelled ground truth, not a vibe. The metric is
precision and recall against the `should_escalate` label on the eval set, reported with a
confidence interval ([00-decisions.md](00-decisions.md) D13).

The asymmetry that drives every design choice here: **a missed handoff produces a confident
wrong answer to a customer; an unnecessary handoff produces a mildly annoyed customer and a
human doing a minute of work.** The operating point is tuned for recall.

---

## 1. `handoff.yaml`

```yaml
# domains/ticketing/handoff.yaml
version: 1

hard_rules:
  - id: explicit_human_request
    match: utterance_pattern
    patterns: ["talk to (a |an )?(human|person|agent|representative|someone)",
               "real person", "get me (a |an )?(manager|supervisor)",
               "transfer me", "stop talking to me"]

  - id: verification_failed
    match: state
    condition: "verify_caller failed twice in this call"

  - id: payment_dispute
    match: intent_or_pattern
    patterns: ["chargeback", "dispute(d)? (the )?charge", "fraud",
               "my card was charged", "unauthorized"]

  - id: refund_outside_policy
    match: classifier
    condition: "refund requested AND policy does not permit AND caller insists"

  - id: accessibility_judgment
    match: intent
    intents: [accessibility_accommodation]
    note: >
      Factual ADA information is answered from the corpus. Anything requiring a
      judgment call about an individual's accommodation is a human decision.

  - id: tool_state_unknown
    match: state
    condition: "any tool_execution in status 'unknown' after reconciliation"

  - id: retrieval_unavailable
    match: state
    condition: "retrieval raised an error"

  - id: distress
    match: classifier
    condition: "caller indicates medical emergency, threat, or acute distress"

soft_signals:
  retrieval_insufficient:
    weight: 0.45
    # PROVISIONAL until eval/calibrate_retrieval.py runs. See docs/05 §6.
    threshold_top_score: 0.0240     # PROVISIONAL
    threshold_margin:    0.0040     # PROVISIONAL

  entity_confidence_low:
    weight: 0.30
    threshold: 0.60                 # min word confidence on a key entity
    entity_types: [order_id, email, quantity, date, event_name]

  rephrase_repeat:
    weight: 0.35
    trigger_count: 2                # same question asked twice
    similarity: 0.82                # cosine on utterance embeddings

  checker_failure:
    weight: 0.40
    note: "two in one call is a hard handoff"

  tool_failure:
    weight: 0.50

  out_of_scope:
    weight: 0.55
    note: "cheap classifier; see §3"

  budget_exceeded_repeat:
    weight: 0.30
    trigger_count: 2

  iteration_cap_reached:
    weight: 0.45

  negative_sentiment:
    weight: 0.20
    note: "contributes, never decides. Frustration alone is not a handoff."

threshold: 0.60       # PROVISIONAL — calibrated in week 3
decay_per_turn: 0.5   # signals from older turns count less
```

### Why weighted scoring rather than "any signal fires"

Any-signal-fires over-escalates badly. Real calls routinely have one mediocre retrieval or
one low-confidence word, and escalating on each would push containment toward zero. But two
mediocre signals together — a weak retrieval *and* a caller repeating themselves — is a
genuinely bad turn. The weights make that composition expressible.

`decay_per_turn = 0.5` means a signal from two turns ago contributes a quarter of its
weight. Without decay, a long successful call eventually accumulates enough noise to
escalate for no reason.

---

## 2. Evaluation order

```python
# harness/handoff.py
def evaluate(call: CallState, turn: TurnState) -> HandoffDecision:
```

1. **Hard rules, pre-model.** Pattern and state rules run on the raw transcript before
   retrieval or any LLM call. Cheapest exit wins, and "get me a person" should cost nothing.
2. **Hard rules, post-state.** `tool_state_unknown` and `retrieval_unavailable` are checked
   after those stages run.
3. **Soft scoring.** Sum `weight × decay^age` over active signals across the call. Compare
   to `threshold`.
4. **Classifier rules.** `refund_outside_policy`, `out_of_scope`, and `distress` need a
   model call. They run as **one** combined classification, not three, and only when the
   deterministic signals leave the decision near the threshold (within ±0.15). Three
   separate classifier calls per turn would be the single largest latency regression in the
   system.

```python
@dataclass(frozen=True)
class HandoffDecision:
    should_handoff: bool
    rule_type: Literal["hard","soft"] | None
    reason: str                  # dominant reason, for the human agent
    signals: list[dict]          # [{signal, weight, value, turn_age}]
    score: float
```

---

## 3. Signal implementation notes

**`entity_confidence_low`** uses `Transcript.word_confidences`, not utterance-mean
confidence ([04-providers.md](04-providers.md) §2). An order number heard at 0.4 inside an
otherwise clean sentence is exactly the failure that matters, and a mean hides it. Key
entities are located by the same extractors the checker uses
([08-checker.md](08-checker.md) §2), so the two subsystems agree on what an entity is.

**`rephrase_repeat`** compares the current utterance embedding against previous *user*
utterances in the call at cosine ≥ 0.82. It is embedding-based rather than lexical because
a caller rephrasing deliberately uses different words — that is what rephrasing is. A
caller saying the same thing twice because the agent did not understand is the strongest
cheap signal that the agent is failing.

**`out_of_scope`** is one short classification over the utterance plus the domain's topic
list, invoked only near the threshold. Prompted to return a single label, capped at 64
output tokens, and run with the same provider as the main loop so it shares the connection
pool.

**Combined classifier call.** When invoked, one request returns a small JSON object:

```json
{"out_of_scope": false, "refund_outside_policy": true, "distress": false}
```

One round trip, roughly 200–300 ms, and only on borderline turns. Its cost appears in the
latency table as its own stage so the trade-off is visible.

---

## 4. The handoff summary

What the human agent actually receives. This is the deliverable of the whole handoff path,
and the thing a buyer evaluates the product on.

```python
@dataclass(frozen=True)
class HandoffSummary:
    call_id: str
    created_at: str
    # who
    caller_verified: bool
    order_id: str | None
    caller_email: str | None
    # why they called
    intent: str
    intent_confidence: float
    asked: list[str]               # the caller's questions, verbatim, in order
    # what happened
    attempted: list[str]           # human-readable: "looked up order TK-88213"
    tools_committed: list[dict]    # name + args + result, side effects only
    tools_failed: list[dict]
    answered: list[str]            # what the agent actually told them
    # why we're here
    handoff_reason: str
    rule_type: str
    signals: list[dict]
    # what the human needs to do
    suggested_next_step: str
    policy_refs: list[str]         # heading paths of relevant chunks
    # raw
    transcript: list[dict]         # [{role, text, ts}]
    duration_s: int
    turn_count: int
```

Rendered for a human as:

```
HANDOFF — call 01JQ8F…  ·  2m 14s  ·  3 turns
Reason: refund requested outside policy (hard rule), caller insisted twice

CALLER     verified ✓   order TK-88213   sam@example.com
INTENT     refund_request (0.91)

ASKED
  1. "Can I get a refund if I can't make it anymore?"
  2. "But I bought these four months ago, that's not fair"

AGENT DID
  ✓ verify_caller            verified
  ✓ lookup_order TK-88213    2× GA, Harbour Pavilion, Sat Mar 14, mobile delivery
  ✗ no refund filed          (agent may not approve or promise money decisions)

AGENT SAID
  "Tickets for this event are non-refundable unless the event is cancelled or
   postponed." [Refunds and exchanges > Eligibility]
  "I want to be sure I get that right — let me bring in a colleague."

RELEVANT POLICY
  Refunds and exchanges > Eligibility
  Refunds and exchanges > Event cancelled by the organizer

SUGGESTED NEXT STEP
  Caller is outside the refund window and the event is not cancelled. If a
  goodwill exception is appropriate, file via request_refund with reason_code
  = cannot_attend. The agent has NOT promised anything.
```

The last line is load-bearing. The human needs to know what was and was not promised before
they open their mouth, and "the agent has not promised anything" is the single most useful
sentence on the page.

### Generation

Structured fields come from `CallState` directly — no model involved, so they cannot be
wrong. Only `intent`, `suggested_next_step`, and the `asked` paraphrase are LLM-generated,
in one call, off the hot path (the caller is already being transferred, so 2 s is fine).
If that call fails, the summary still renders with the structured fields and a note that
the narrative is unavailable. The human never gets nothing.

---

## 5. Measurement

`should_escalate` is a boolean label on every eval case. The subset is grown to **≥40
positives** using caller-simulator output as candidates, hand-labelled
([00-decisions.md](00-decisions.md) D13).

Reported, always together:

```
Handoff, run 2026-xx-xx-xxxx   (n=130 cases, 42 positive)
  precision  0.xx  [0.xx, 0.xx]   Wilson 95%
  recall     0.xx  [0.xx, 0.xx]
  F1         0.xx
  false negatives: n  ← the expensive errors; enumerate every one
```

**Every false negative is enumerated individually in the run README**, with the case id and
the signals that did not fire. A false negative is a confident wrong answer to a customer,
and there will be few enough to list. Listing them is what turns the metric into
engineering rather than a scoreboard.

### Threshold selection

Sweep `threshold` from 0.3 to 0.9 and plot precision/recall. Choose the lowest threshold
achieving **recall ≥ 0.90**, then report the precision that costs. Save the curve to
`results/<run>/handoff_pr_curve.png` and record the chosen value in `handoff.yaml` with the
run directory in a comment.

Signal weights are tuned by ablation: drop one signal, rerun, record the ΔF1. The ablation
table goes in the run README and is more informative than the weights themselves — it shows
which signals are actually doing work and which are decoration.
