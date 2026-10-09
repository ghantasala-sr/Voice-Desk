# 15 — Results and claims

## 1. The rule

> **Never report a number without the run directory that produced it.**

Every figure in the README, the HTML report, the demo video, and the resume must trace to a
`results/<date>-<label>/` containing the config snapshot, the raw per-case output, and the
computed `metrics.json`.

This is not bookkeeping fussiness. The failure mode it prevents is specific and common: a
number measured in Week 2 under one config gets quoted in Week 4 after the config changed,
and in an interview you cannot say what produced it. One "I'm not sure, that was an earlier
config" undoes an otherwise strong project.

---

## 2. Run directory

```
results/2026-03-14-deepgram-baseline/
  README.md          ← written by hand: purpose, what it showed, what to do next
  config.json        ← full config snapshot + git sha + corpus hash
  metrics.json       ← computed by eval/metrics.py. The ONLY file other tools read.
  raw.jsonl          ← one line per (case, voice, variant)
  failures.md        ← every failure, with transcript, retrieval, verdicts
  *.png              ← calibration plots, PR curves
  audio/             ← TTS output for failed cases
```

`README.md` is written by a human and answers three questions:

1. What was this run for?
2. What did it show?
3. What changed as a result?

A run directory with no README is an artifact nobody can interpret in six weeks, including
the person who made it.

---

## 3. `metrics.json`

Every proportion carries `n` and a Wilson 95% interval. A bare proportion is not permitted.

```json
{
  "run_id": "01JQ…",
  "label": "deepgram-baseline",
  "git_sha": "a1b2c3d",
  "corpus_hash": "d4e5f6…",
  "timing_valid": true,
  "sampled": false,

  "latency": {
    "endpointing_stop_secs": 0.4,
    "n_turns": 450,
    "voice_to_voice_ms": {"p50": 0, "p95": 0},
    "harness_side_ms":   {"p50": 0, "p95": 0},
    "barge_in_ms":       {"p50": 0, "p95": 0},
    "stages_p50_ms": {"endpoint": 0, "stt_final": 0, "retrieval": 0,
                      "llm_ttft": 0, "sentence_buffer": 0, "checker": 0,
                      "tts_ttfb": 0, "playout": 0},
    "ack_bypass_share": 0.0
  },

  "stt": {
    "provider": "deepgram",
    "by_variant": {
      "clean":     {"n": 150, "wer": 0.0, "entity_accuracy": 0.0, "entity_ci": [0,0]},
      "snr20":     {"...": 0},
      "snr10":     {"...": 0},
      "snr5":      {"...": 0},
      "telephony_clean": {"...": 0},
      "telephony_snr10": {"...": 0}
    },
    "human_calibration": {"n": 10, "wer": 0.0, "entity_accuracy": 0.0}
  },

  "answers": {
    "n": 50,
    "correctness_2_rate": 0.0, "correctness_ci": [0, 0],
    "citation_2_rate":    0.0,
    "safety_0_count":     0,
    "judge_interrun_agreement": 0.0,
    "judge_human_agreement":    {"n": 20, "agreement": 0.0}
  },

  "handoff": {
    "n": 130, "positives": 42,
    "precision": 0.0, "precision_ci": [0, 0],
    "recall":    0.0, "recall_ci":    [0, 0],
    "f1": 0.0,
    "false_negatives": ["case_id", "..."],
    "threshold": 0.0
  },

  "harness": {
    "stop_reasons": {"answered": 0, "handoff": 0, "interrupted": 0,
                     "budget_exceeded": 0, "error": 0},
    "tool_calls_per_turn": 0.0,
    "llm_iterations_per_turn": 0.0
  },

  "checker": {
    "catch_by_class": {
      "numeric":      {"seeded": 0, "caught": 0, "rate": 0.0},
      "date":         {"seeded": 0, "caught": 0, "rate": 0.0},
      "entity":       {"seeded": 0, "caught": 0, "rate": 0.0},
      "policy_claim": {"seeded": 0, "caught": 0, "rate": 0.0},
      "negation":     {"seeded": 0, "caught": 0, "rate": 0.0,
                       "note": "declared gap — no deterministic rule covers this"},
      "citation":     {"seeded": 0, "caught": 0, "rate": 0.0,
                       "note": "not an enforcement target"}
    },
    "false_positive_rate": 0.0,
    "false_positive_n": 0
  },

  "safety": {
    "side_effecting_executions": 0,
    "duplicate_side_effects": 0,
    "context_invariant_violations": 0,
    "improvised_answers_after_fault": 0,
    "fault_cases": 0
  },

  "cost": {"llm_usd": 0.0, "stt_usd": 0.0, "tts_usd": 0.0, "total_usd": 0.0}
}
```

`timing_valid` is false for `--fast` runs; `sampled` is true for `--sample N` runs. Neither
may be quoted as a full result, and the report refuses to render headline latency from a
run with `timing_valid: false`.

---

## 4. Reporting rules

1. **`n` beside every proportion.** Always.
2. **Confidence intervals on every proportion** that appears in a conclusion.
3. **Both latency numbers** — voice-to-voice and harness-side — plus the endpointing
   constant ([00-decisions.md](00-decisions.md) D11).
4. **Denominators stated** for every rate. Especially wrong-answer rate, whose denominator
   is *contained calls* ([11-observability.md](11-observability.md) §6).
5. **Replay numbers labelled as replay.** A live number is measured separately and the gap
   attributed ([10-audio-pipeline.md](10-audio-pipeline.md) §6).
6. **Synthesized-audio numbers labelled**, with the human calibration gap alongside.
7. **Simulated-call numbers labelled** as simulated ([13-caller-sim.md](13-caller-sim.md) §6).
8. **Known gaps stay in the table**, not in a footnote. The `negation` row with a near-zero
   catch rate is a credibility asset.
9. **Negative results reported.** A cycle that did not improve containment is reported with
   its diagnosis.
10. **No number is edited after the fact.** A superseded run gets a new directory; the old
    one stays.

---

## 5. Claims this project can and cannot support

Worth being explicit, because the gap between these two columns is where projects lose
credibility under questioning.

| Can support | Cannot support |
|---|---|
| Voice-to-voice latency under a stated config, on replayed audio | A production SLA |
| Zero duplicate side effects across N fault-injection cases, with the ambiguity profile | Zero duplicates "in production" |
| Entity-accuracy degradation by noise level and audio quality | A general claim about Whisper vs Deepgram model quality |
| The gap-report mechanism surfaced planted gaps at ranks 1, 2, 7 | A real-world containment rate |
| Checker catch rate by hallucination class, with the uncovered class shown | "The agent does not hallucinate" |
| Handoff P/R against hand-labelled cases, with CIs | Handoff quality against real customer satisfaction |

---

## 6. Resume bullets

Fill in only from `metrics.json`. The bracketed run reference is for your own notes, not
the resume itself — but you must be able to produce it in an interview.

**Latency / voice AI**

> Built VoiceDesk, a real-time voice RAG support agent (Pipecat, FastAPI, pgvector,
> Claude) with **X ms p50 / Y ms p95** harness-side voice latency (**Z ms p50**
> voice-to-voice at 400 ms VAD endpointing) across N turns, instrumented with per-stage
> OpenTelemetry tracing.

**Harness / safety**

> Designed a custom agent harness with a deterministic idempotency ledger and
> sentence-level grounding checks: **0 duplicate side effects across N side-effecting
> executions** under a fault-injection suite including commit-then-timeout ambiguity and
> mid-read-back interruption.

**STT benchmarking / data**

> Benchmarked self-hosted faster-whisper against streaming Deepgram across 4 SNR levels and
> an 8 kHz μ-law telephony path (900 samples/provider): entity accuracy fell from **A% to
> B%** at 5 dB, and self-hosting cost **C ms** additional time-to-final-transcript.

**Analytics / data engineering**

> Built a call-analytics feedback loop (HDBSCAN intent clustering over call events in
> Postgres) that surfaced planted knowledge-base gaps at rank 1 of 18 clusters; closing the
> top gap raised containment from **X% to Y%** over N simulated calls with **no increase in
> wrong-answer rate**.

Lead with analytics for data engineering roles; lead with harness and latency for voice AI,
ML engineering, and FDE roles.

### Two words that make these defensible

"**simulated**" in the analytics bullet and "**harness-side**" in the latency bullet. Both
invite the obvious follow-up question, and having a crisp answer to the obvious follow-up
question is the entire purpose of the discipline in this document.
