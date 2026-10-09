# 13 — Caller simulator

The feedback loop needs call volume. Hand-testing produces maybe thirty calls, and
clustering thirty calls by intent yields clusters of one. Without this component Week 4 has
no deliverable ([00-decisions.md](00-decisions.md) D8).

**Build it first in Week 4, before the gap report.**

---

## 1. What it is

An LLM plays a caller with a persona, a goal, and a behaviour profile. It is driven turn by
turn against the harness through the text driver, with no audio and no STT. Each simulated
call writes a real row to `calls`, real `turns`, real `call_events` — the analytics job
cannot tell the difference, and should not have to.

```
personas.yaml ──┐
intents.yaml  ──┼──► simulate.py ──► text_driver ──► Harness ──► Postgres
seed          ──┘                                                   │
                                                                    └──► gap_report.py
```

---

## 2. Personas

```yaml
# eval/caller_sim/personas.yaml
personas:
  - id: direct
    weight: 0.30
    style: >
      States the problem in one clear sentence. Answers questions directly.
      Provides the order number when asked.
    interrupts: 0.05
    rephrases_when_unsatisfied: 0.2

  - id: vague
    weight: 0.20
    style: >
      Describes the problem indirectly: "my tickets aren't working", "the thing
      won't let me in". Needs two or three turns before the real issue surfaces.
      Does not volunteer the order number.
    interrupts: 0.10
    rephrases_when_unsatisfied: 0.6

  - id: rushed
    weight: 0.15
    style: >
      Short sentences, impatient, interrupts mid-answer. Asks for the bottom line.
      Often starts speaking before the agent finishes.
    interrupts: 0.45
    rephrases_when_unsatisfied: 0.4

  - id: frustrated
    weight: 0.15
    style: >
      Already annoyed. Repeats the complaint. Escalates in tone if the first
      answer is not what they wanted. Never abusive.
    interrupts: 0.25
    rephrases_when_unsatisfied: 0.8
    demands_human_after_turns: 4

  - id: detailed
    weight: 0.12
    style: >
      Gives lots of context, much of it irrelevant. Long first turn. Asks
      follow-up questions about edge cases.
    interrupts: 0.05
    rephrases_when_unsatisfied: 0.3

  - id: confused_elderly
    weight: 0.08
    style: >
      Unfamiliar with mobile tickets and app terminology. Asks the agent to
      repeat. Mixes up the order number digits once.
    interrupts: 0.02
    rephrases_when_unsatisfied: 0.7
    misstates_entity_once: true
```

Personas are **behavioural**, never demographic beyond what affects the conversation.
`confused_elderly` exists because unfamiliarity with mobile ticketing is a real and common
support pattern that stresses the agent differently — not as a stereotype, and the style
text describes behaviour, not a person.

### Why personas matter for the result

A population of only `direct` callers produces a containment number that will not survive
contact with reality, and no interruptions, so the barge-in path is never exercised at
scale. The `interrupts` probability is what generates the interruption volume that makes
the duplicate-side-effect result statistically meaningful.

---

## 3. Intent distribution

```yaml
# eval/caller_sim/intents.yaml
intents:
  - {id: where_are_my_tickets,  weight: 0.18, covered: true}
  - {id: refund_request,        weight: 0.15, covered: true}
  - {id: ticket_transfer,       weight: 0.12, covered: true}
  - {id: event_status,          weight: 0.10, covered: true}
  - {id: delivery_problem,      weight: 0.09, covered: true}
  - {id: venue_logistics,       weight: 0.08, covered: true}   # bag, parking
  - {id: payment_question,      weight: 0.06, covered: true}
  - {id: resale_question,       weight: 0.05, covered: true}
  - {id: accessibility,         weight: 0.04, covered: partial}
  # --- held-out topics: no indexed document covers these (docs/00 D4) ---
  - {id: group_booking_vip,     weight: 0.06, covered: false}
  - {id: weather_outdoor,       weight: 0.04, covered: false}
  - {id: age_restriction_id,    weight: 0.03, covered: false}
```

Roughly a 13% share of uncovered topics, concentrated in `group_booking_vip`. That
concentration is deliberate and is the heart of the Week-4 result:

> The gap report should rank `group_booking_vip` first because it is both frequent **and**
> failing. Whether it actually does is a genuine test of the clustering, not a foregone
> conclusion — a report that surfaces the rare `age_restriction_id` instead means the
> ranking is weighting failure rate over volume and needs fixing.

The distribution is weighted, not uniform, because real call distributions are long-tailed
and a uniform mix would make every cluster the same size and the ranking meaningless.

---

## 4. Running a simulated call

```python
# eval/caller_sim/simulate.py
async def simulate_call(persona, intent, order, *, seed: int) -> str:
```

1. Pick persona × intent by weight; pick an order from the mock dataset consistent with the
   intent (a `delivery_problem` gets an order whose delivery actually failed).
2. Build the caller system prompt: persona style + the concrete goal + the facts this
   caller knows (order number, email, event) + **the rule that they only reveal facts when
   asked**.
3. Loop, to `max_turns` (8):
   - Caller model produces one utterance.
   - With probability `interrupts`, schedule a barge-in at a random offset into the agent's
     reply.
   - Feed to the harness; capture the reply.
   - Caller model decides: satisfied / rephrase / escalate / hang up.
4. Terminate on satisfaction, handoff, hang-up, or `max_turns`.
5. `calls.source = 'simulation'`; everything else is written exactly as a real call.

### Determinism

Every call is seeded from `hash(persona, intent, order, run_seed)`. A run is reproducible
from its seed, which matters because the gap report's conclusions need to be reproducible
to be trustworthy.

### Cost

300 calls × ~5 turns × 2 model calls per turn (caller + agent) ≈ 3000 model calls. With a
fast model for the caller side and prompt caching on the agent side, this is a few dollars
and about 20 minutes. Record the actual cost in the run README — "the feedback cycle costs
$X to run" is a genuinely useful operational number.

---

## 5. Labelling simulated calls

Each simulated call stores its ground truth in `call_events` as a `simulation_label` event:

```json
{"persona": "vague", "intent": "group_booking_vip", "covered": false,
 "order_id": "TK-88213", "seed": 918273,
 "expected_outcome": "handoff"}
```

This enables two things the gap report cannot do on its own:

1. **Measuring the gap report itself.** Does it find the gaps you planted? A clustering job
   nobody validated is a clustering job you cannot trust.
2. **Clean containment accounting.** A handoff on an uncovered topic is *correct*
   behaviour, not a failure. Without the label, every correct handoff looks like a loss and
   the containment metric punishes the system for being safe.

The gap report itself must **not** read these labels — it clusters from utterances and
outcomes only, exactly as it would on real traffic. The labels are the answer key, held
back and used only for scoring.

---

## 6. Honesty about what this is

Simulated calls are not real traffic. Every claim derived from them says so, in the report
and on the resume bullet. What the simulation legitimately demonstrates:

- the analytics mechanism finds genuine knowledge-base gaps from call data;
- the system behaves sensibly across a varied caller population, not just the happy path;
- interruption and idempotency behaviour holds at volume rather than in five hand-run tests.

What it does **not** demonstrate: a real containment rate for a real support team. The
distribution was chosen by the author, and an author-chosen distribution is a hypothesis,
not an observation. Say that plainly once, in the report, and the honest version of the
result is stronger than an overclaimed one.
