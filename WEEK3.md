# Week 3 — status

**Evaluation.** Every number below has a run directory. Nothing in this week
called a paid API: there is no key in this environment, so everything was
done with the real harness, real retrieval, and local tools.

```
307 tests passing · ruff clean
```

## Run directories

| Run | What it measured |
|---|---|
| `results/2026-10-13-retrieval-calibration/` | which retrieval signal to threshold, and where |
| `results/2026-10-13-handoff-calibration/` | handoff precision/recall + ablation, deterministic policy |
| `results/2026-10-13-checker-catch-rate/` | checker catch rate by hallucination class + false positives |
| `results/2026-10-13-stt-faster-whisper-smallen/` | STT accuracy and decode cost under noise and phone audio |

---

## What each task delivered

| Task | State | Evidence |
|---|---|---|
| **W3-1** case set | done | 50 cases in the planned composition + 44 handoff cases → **94 labelled, 42 escalation positives**; strict loader rejects unknown keys |
| **W3-2** audio synthesis | done, **deviation** | 150 files, 3 voices, macOS `say` (Cartesia needs a key). Human calibration subset **not recorded** — needs you |
| **W3-3** noise + telephony | done, **deviation** | 900 files. Babble from 6 other voices; pink/brown noise as **synthetic stand-ins** for café/street. SNR labels hold within **0.64 dB worst case** under an independent re-measurement |
| **W3-4** replay runner | **not built** | full-pipeline audio replay needs the Pipecat stack + keys; file-level STT eval used instead |
| **W3-5** metrics | done | Wilson CI, percentiles matching Postgres, WER with symmetric spoken-ID folding, per-entity accuracy |
| **W3-6** judge | written, **not run** | rubric, 3× shuffled runs, median, agreement, `must_not_contain` bypass, server-side refusal fallback; 7 tests with a fake client |
| **W3-7** retrieval calibration | done | dense cosine AUC **0.91** vs fused margin **0.53**; threshold **0.688** (recall 0.93 on uncovered, 36% false alarm) |
| **W3-8** handoff calibration | done (deterministic half) | recall **0.31 → 0.86**, precision **0.80**; ablation; every false negative listed |
| **W3-9** checker measurement | done | numeric/date/entity **32/32**, policy claims **7/8**, false positives **2/46**; gaps published |
| **W3-10** STT comparison | **half** | faster-whisper measured; Deepgram needs a key |
| **W3-11** remaining faults | done | STT + retrieval fault groups; **33** fault tests total |
| **W3-12** write-up | done | a hand-written README in every run directory |

---

## STT (faster-whisper small.en, CPU)

| variant | WER | key-entity accuracy |
|---|---|---|
| clean | 0.025 | 0.83 (55/66) |
| 5 dB | 0.052 | **0.62** (41/66) |
| phone, 10 dB | 0.053 | 0.71 (47/66) |

Order ids hold up (27/27 clean, 21/27 at 5 dB). **Spoken emails do not: 20/30
on clean audio, 13/30 at 5 dB.** Decode costs ~0.3 s per second of audio at
p50 on CPU — enough on its own to break the latency target.

## What the measurements changed

This is why the week mattered. Each item below is a change made **because a
number said so**, and each run README says what it was before.

1. **Retrieval accuracy was flattered.** Top-1 fell from 12/12 to **9/14** on
   caller phrasing; top-k is **12/14**. The week-2 number is retired.
2. **The handoff policy as designed caught 31% of cases needing a human.**
   Weak retrieval could never hand off alone. A second, stricter level
   (`retrieval_very_weak`, top_vec < 0.62) took recall to **0.86**.
3. **A production bug in the checker:** ISO dates from tool results were
   parsed as number ranges, so a looked-up event date could never be spoken.
4. **Rule 2 caught 1 in 8 fabricated claims.** Raising its overlap threshold
   and stopping venue names from "anchoring" a claim took it to **7/8** with no
   added false positives.
5. **Two payment-dispute phrasings slipped the hard rules** ("disputing",
   "didn't authorize"). Fixed — and labelled in-sample in the README.
6. **Spoken emails are the weak entity** (67% on clean audio). Verification
   should read the email back before calling a mismatch a failure — noted as
   the next harness change.
7. **The WER scorer was penalising formatting.** "T K 8 8 2 1 3" vs
   "TK-88213" counted as five errors. Both sides are now folded identically;
   genuine mishearings still count.

## Honest limits

- **Calibration set = test set.** Thresholds and pattern fixes were chosen on
  these 94 cases; the headline recall is in-sample. The week-4 caller
  simulator is what produces a held-out split.
- **One author** wrote every case and every label.
- **Synthesized audio** is cleaner than people. The human calibration subset
  (10 recordings) is the fix and needs a microphone.
- **No live model.** Handoffs triggered by the model's own answer, answer
  correctness, and the judge are all unmeasured until a key is added.

## Next

Week 4: the caller simulator first — it is also what fixes the
"calibration set = test set" problem above.
