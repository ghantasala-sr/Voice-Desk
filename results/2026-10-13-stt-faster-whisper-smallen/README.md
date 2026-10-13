# STT under noise and phone audio — faster-whisper (W3-10, half)

**Date:** 2026-10-13 · **Model:** faster-whisper `small.en`, int8, laptop CPU, beam 1
**Audio:** 900 files = 50 cases × 3 voices × 6 variants, synthesized locally (`eval/synth.py`)
**Files:** `metrics.json`, `entity_by_type.json`, `raw.jsonl` (every transcript)

## 1. What was this run for

To measure what self-hosted ASR costs on a real-time path (docs/00 D14):
accuracy on the words that matter, and decode time. The streaming Deepgram
half of the comparison needs an API key and was not run.

## 2. What it showed

| variant | WER | key-entity accuracy | decode time / audio second (p50 / p95) |
|---|---|---|---|
| clean | 0.025 | 0.83 [0.73, 0.90] (55/66) | 0.29 / 0.51 |
| 20 dB | 0.024 | 0.82 [0.71, 0.89] | 0.33 / 0.54 |
| 10 dB | 0.037 | 0.77 [0.66, 0.86] | 0.32 / 0.53 |
| **5 dB** | 0.052 | **0.62** [0.50, 0.73] (41/66) | 0.32 / 0.53 |
| phone, clean | 0.033 | 0.79 [0.67, 0.87] | 0.34 / 0.54 |
| phone, 10 dB | 0.053 | 0.71 [0.59, 0.81] | 0.32 / 0.51 |

**WER hides the failure.** 2.5% WER on clean audio sounds solved; one in six
key entities is still wrong. Per entity type:

| entity | clean | 10 dB | 5 dB | phone 10 dB |
|---|---|---|---|---|
| order id | **27/27** | 26/27 | 21/27 | 24/27 |
| **email** | **20/30 (0.67)** | 17/30 | **13/30 (0.43)** | 15/30 |
| quantity | 6/6 | 6/6 | 6/6 | 6/6 |

**Spoken emails are the failure mode**, even on clean audio: "sam at example
dot com" → `samatexample.com`; "jo" → "joe"; "lee plus tickets" → "LEA Plus
tickets"; "alex dot kim" → `alex.com`. Order ids survive because they are
spelled one character at a time.

Babble is the hardest noise (WER 0.056 vs 0.022 for brown), as expected for
speech-shaped noise. The Indian-English voice has roughly double the WER of
the other two (0.056 vs 0.025/0.031).

**Cost:** decoding takes about 0.3 s per second of audio at p50 and 0.5 s at
p95 on CPU — and a batch model only starts after the caller stops speaking,
so that time lands directly on stage [B] of the latency budget. A 4-second
utterance adds about 1.2-2 s, which on its own breaks the 1600 ms harness-side
p95 target. That is the headline cost of self-hosting on this hardware.

## 3. What changed as a result

- The WER scorer now folds spoken identifiers identically on both sides; it
  initially charged "T K 8 8 2 1 3" vs "TK-88213" as five errors.
- **Action for the harness (not yet made):** verification takes the email as
  heard. Given 67% email accuracy on clean audio, `verify_caller` should
  confirm the email back to the caller — or ask for it to be spelled — before
  counting a mismatch as a failed verification. Otherwise mishearings will
  show up as `verification_failed` handoffs.

## 4. Limits

- Synthesized voices (macOS `say`) are cleaner than people; absolute
  accuracy is inflated. The 10-recording human calibration set is not made.
- Pink and brown noise are synthetic stand-ins, not café or street recordings.
- 66 entity instances per variant; CIs are wide.
- File-level transcription, not streamed through the live pipeline.
- One model size on one machine.

## 5. Reproduce

```bash
.venv/bin/pip install faster-whisper
.venv/bin/python -m eval.synth && .venv/bin/python -m eval.noise_mix && .venv/bin/python -m eval.telephony
.venv/bin/python -m eval.stt_eval
```
