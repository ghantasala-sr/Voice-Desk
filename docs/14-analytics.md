# 14 — Analytics and the feedback loop

The loop: calls → events → gap report → one fix → rerun → measured delta.

The deliverable is not the dashboard. It is **one completed cycle with a before and after
number**, and a mechanism that found the gap rather than being told about it.

---

## 1. Event writing

`analytics/events.py` is the single writer to `call_events`
([11-observability.md](11-observability.md) §4). Buffered per turn, flushed in a `finally`,
validated against the closed `event_type` vocabulary.

One rule worth stating: **the harness never writes to Postgres directly.** It emits events
and state; a writer persists them. This keeps the harness testable without a database,
which is what lets the text driver run in milliseconds.

---

## 2. Intent labelling

Clustering needs intents, and simulated calls carry a hidden label the gap report is not
allowed to read ([13-caller-sim.md](13-caller-sim.md) §5). So intents are derived:

1. **First-turn utterance embedding** (same model as retrieval).
2. **HDBSCAN** over the embeddings of all calls in the window. Not k-means: the number of
   intents is unknown, cluster sizes are wildly unequal, and k-means would force the
   long tail into whichever large cluster is nearest.
3. **One LLM call per cluster** to name it from 5 sampled utterances
   (`"group bookings and VIP package contents"`). One call per cluster, not per call —
   the cost difference across 300 calls is two orders of magnitude.
4. Noise points (HDBSCAN label `-1`) become a `miscellaneous` bucket, reported with its
   size. A large noise bucket is itself a finding: it means callers are asking things that
   do not resemble each other, which is a different problem from a coverage gap.

---

## 3. Gap report

`make gap-report` → `analytics/gap_report.py`.

### Inputs

All calls in the window (default: last 30 days, or `--run <id>` for a simulation run),
their turns, retrieval scores, outcomes, checker verdicts, and handoff reasons.

### Per cluster

| Field | Meaning |
|---|---|
| `label` | LLM-generated name |
| `n_calls` | cluster size |
| `share` | fraction of window volume |
| `failure_rate` | handoff or wrong-answer ÷ cluster size |
| `median_top_score` | median `retrieval_top_score` |
| `median_margin` | median `retrieval_margin` |
| `dominant_handoff_reason` | modal reason |
| `covering_docs` | ingested docs retrieved for this cluster, by frequency |
| `held_out_match` | an un-ingested document whose `topics` match this cluster |
| `example_utterances` | 5 verbatim samples |
| `diagnosis` | one of the five categories below |

### Diagnosis

The categories are what make the report actionable instead of a list of numbers. Each maps
to a different fix:

| Diagnosis | Signature | Fix |
|---|---|---|
| `no_coverage` | low top score, low margin, no doc consistently retrieved | Write a document |
| `held_out_coverage` | as above, **and** an un-ingested doc's topics match | Ingest the existing doc |
| `weak_coverage` | a doc is retrieved consistently but failure rate is high | Rewrite/expand that section |
| `threshold_too_strict` | **high** top score and margin, but handoffs for `retrieval_insufficient` | Lower the threshold |
| `tool_failure` | failures dominated by `tool_failure` | Fix the tool, not the corpus |

`threshold_too_strict` is the category people forget to build, and it is the one that
prevents the classic feedback-loop failure mode: every cycle adds a document, nobody ever
checks whether the threshold was the problem, and the corpus grows while containment does
not.

### Ranking

```
priority = n_calls × failure_rate × confidence
```

Volume × failure rate, because fixing a rare total failure moves no business metric and
fixing a frequent mild annoyance moves little either. `confidence` damps clusters under 5
calls, which would otherwise top the ranking on a 100% failure rate with n=2.

### Output

`gap_reports` row plus:

```
GAP REPORT  ·  window 2026-xx-xx → 2026-xx-xx  ·  312 calls  ·  18 clusters

#1  group bookings and VIP package contents            priority 11.4
    19 calls (6.1%)   failure rate 94.7%
    retrieval: median top score 0.0141, median margin 0.0007
    dominant handoff reason: retrieval_insufficient
    diagnosis: HELD_OUT_COVERAGE
      → domains/ticketing/corpus/10-group-and-vip-packages.md exists but is
        not ingested (topics: group_booking, vip, hospitality)
    examples:
      "what's actually included in the VIP package"
      "we're booking for fourteen people, is there a group rate"
      ...
    SUGGESTED ACTION: ingest document 10 and re-run the eval set.

#2  weather policy for outdoor shows                    priority 4.8
    ...
```

### Validating the report

Because simulated calls carry hidden labels, the report can be scored:

```
Gap report accuracy (run 2026-xx-xx-xxxx)
  planted gaps: 3
  surfaced in top 3: n/3
  rank of each: group_booking_vip #1, weather_outdoor #2, age_restriction_id #7
  false positives in top 3: n
```

Report this. "The clustering found the gaps I planted, at these ranks" is a far stronger
claim than "I built a clustering job," and it is the difference between a mechanism and a
script.

---

## 4. Static report

`make report` → `analytics/report.py` renders one self-contained HTML file
([00-decisions.md](00-decisions.md) D15). No server, no framework, inline CSS, charts as
inline SVG.

Sections:

1. **Headline** — containment, wrong-answer rate, handoff P/R, latency p50/p95, each with
   `n` and the run date.
2. **Latency breakdown** — stacked bar of stages [A]–[H], with endpointing visibly
   separated ([00-decisions.md](00-decisions.md) D11).
3. **STT comparison** — entity accuracy by provider × noise × audio quality.
4. **Checker** — catch rate by hallucination class, with the known gap row and the
   false-positive rate.
5. **Safety** — duplicate side effects (0), context invariant violations (0), improvised
   answers after fault (0), with the denominators.
6. **Gap report** — top clusters with diagnoses.
7. **Feedback cycle** — before/after table.
8. **Provenance** — git sha, corpus hash, config, run directories for every number above.

Section 8 is not optional. A report without provenance is a slide, not a result.

---

## 5. The feedback cycle protocol

Run exactly once for v1, and follow the protocol precisely — the discipline is what makes
the delta mean anything.

**Step 0 — freeze.** Tag the eval set. No case edits from here until the cycle completes.
A changed test set invalidates the comparison, and it is very tempting to "just fix a
label" mid-cycle.

**Step 1 — baseline.** `make simulate N=300` then `make eval`. Record:

```
corpus_hash: <hash of the 9 ingested docs>
containment: xx.x%  [CI]   n=300
wrong-answer rate (of contained): x.x%  [CI]
handoff P/R: 0.xx / 0.xx
```

**Step 2 — gap report.** `make gap-report`. Take the **#1 ranked cluster only.**

**Step 3 — one change.** Exactly one, and write it down before making it:

- ingest a held-out document, **or**
- rewrite one corpus section, **or**
- adjust one threshold.

One change, because with two you cannot attribute the delta. This is the whole reason the
protocol exists.

**Step 4 — re-measure.** Re-ingest, recompute the corpus hash, rerun the **identical**
simulation seeds and the frozen eval set.

**Step 5 — report both numbers and both guardrails.**

```
FEEDBACK CYCLE 1
  Change: ingested 10-group-and-vip-packages.md (gap report #1, HELD_OUT_COVERAGE)
  Corpus: a1b2c3… → d4e5f6…
  Seeds:  identical

                              before            after           delta
  containment                 xx.x% [CI]        xx.x% [CI]      +x.x pts
  wrong-answer rate           x.x%  [CI]        x.x%  [CI]      +x.x pts   ← guardrail
  handoff recall              0.xx              0.xx            -0.0x      ← guardrail
  group_booking_vip calls     19                19
    of which contained        1 (5%)            xx (xx%)
  eval-set accuracy           xx/50             xx/50
```

**The guardrails are the point.** Containment always goes up if you lower the handoff
threshold; the claim only means something if the wrong-answer rate did not rise with it.
A cycle that improves containment while degrading the wrong-answer rate is a failed cycle
and must be reported as one.

### Negative results are reported

If containment does not move, say so and diagnose why. "Ingesting the document did not
improve containment because the retrieval threshold was still rejecting the new chunks"
is a genuinely interesting finding about threshold calibration — more interesting, in an
interview, than a number that went up for reasons you did not investigate.

---

## 6. Scheduling

v1 runs the job on demand via `make gap-report`. Not a cron, not a scheduler, not Kafka.

The stretch path (Kafka streaming for call events, scheduled clustering) is a scaling
story, and scaling a thing that processes 300 calls on demand is a story about
infrastructure rather than about the problem. Build it if the rest is finished and there is
time, not instead of the cycle above.
