# 10 — Audio pipeline

Everything in this document lives in `app/`. Nothing here knows what a policy is, and
nothing in `harness/` knows what a sample rate is
([01-architecture.md](01-architecture.md) §1).

---

## 1. Pipecat wiring

```python
# app/pipeline.py — the only module importing pipecat
pipeline = Pipeline([
    transport.input(),              # 16 kHz PCM from the browser
    stt_service,                    # STTProvider adapter
    harness_bridge,                 # app/bridge.py — the seam
    tts_service,                    # TTSProvider adapter
    transport.output(),
])

task = PipelineTask(
    pipeline,
    params=PipelineParams(
        allow_interruptions=True,
        enable_metrics=True,
        enable_usage_metrics=True,
    ),
)
```

`app/bridge.py` is the adapter and is kept deliberately thin — it translates and nothing
else. All decisions happen on the harness side of it.

| Pipecat frame | Harness event |
|---|---|
| `TranscriptionFrame` (final) | `InEvent.Transcript(text, confidence, word_confidences)` |
| `InterimTranscriptionFrame` | *(dropped; used only for the UI transcript)* |
| `UserStartedSpeakingFrame` | `InEvent.BargeIn()` if the agent is speaking |
| `UserStoppedSpeakingFrame` | timestamp for the `endpoint_ms` stage |
| `BotStoppedSpeakingFrame` | playout completion for `spoken_text` finalization |
| — | `OutCommand.Speak(text, segment_id)` → `TTSSpeakFrame` |
| — | `OutCommand.CancelSpeech()` → cancel TTS context, read offset |
| — | `OutCommand.EndCall(reason)` → `EndFrame` |

The bridge owns one piece of real logic: **mapping TTS context offsets back to characters**
so `spoken_text` can be reconstructed. That is audio-domain knowledge and belongs here, not
in the harness.

---

## 2. Turn-taking — the most important tuning in the project

Endpointing is the single largest term in voice-to-voice latency and the dominant driver of
whether the agent feels natural or rude. It gets dedicated time (W1-7) and a written log,
not a config guess ([00-decisions.md](00-decisions.md) D18).

### Parameters

```python
VADParams(
    confidence = 0.70,   # VD_VAD_CONFIDENCE — Silero speech probability
    start_secs = 0.20,   # VD_VAD_START_SECS — speech this long = speech started
    stop_secs  = 0.40,   # VD_VAD_STOP_SECS  — silence this long = turn ended
    min_volume = 0.60,
)
```

### The trade-off, stated plainly

`stop_secs` is a direct dial between latency and interruption quality:

| `stop_secs` | Effect |
|---|---|
| 0.2 s | Fast, but the agent cuts off anyone who pauses to think mid-sentence |
| 0.4 s | The default. Natural for most speakers |
| 0.8 s | Feels patient but laggy; voice-to-voice p50 rises by 400 ms for nothing |

Lowering `stop_secs` improves the headline latency number while making the product worse.
This is exactly why [00-decisions.md](00-decisions.md) D11 requires reporting the
endpointing constant alongside every latency figure — otherwise the number can be bought
rather than earned.

### Tuning protocol (W1-7)

1. Record 10 natural utterances including three with deliberate mid-sentence pauses
   ("I need to… uh… move two tickets").
2. Replay at `stop_secs` ∈ {0.2, 0.3, 0.4, 0.6, 0.8}.
3. Record per setting: premature-endpoint count (agent interrupted a thinking pause), mean
   `endpoint_ms`, and a subjective 1–5 naturalness score.
4. Choose the lowest value with **zero premature endpoints** on the pause utterances.
5. Write the table to `results/<date>-endpointing/README.md`. This table is a result and
   belongs in the final report.

### Deepgram interaction

When using Deepgram, both Silero VAD and Deepgram's own endpointing are active. Configure
Deepgram's `endpointing` to roughly match `stop_secs × 1000` and let Silero drive
interruption detection (it runs locally with no network latency, which is what barge-in
responsiveness needs). Do not let the two fight; record both in the config snapshot.

---

## 3. Barge-in

Target: **under 150 ms from caller speech onset to agent audio stopping.** This is reported
separately from voice-to-voice latency and is one of the numbers in the metrics table.

Sequence, with the budget each step has:

```
 t+0    Silero crosses confidence for start_secs (200ms)
 t+200  UserStartedSpeakingFrame
 t+205  bridge → harness BargeIn
 t+210  harness: CancelSpeech
 t+215  transport output buffer flushed      ← caller stops hearing audio HERE
 t+220  TTS context cancelled, offset returned
 t+225  LLM stream cancelled (aclose)
 t+230  context hygiene (docs/06 §5)
 t+235  pending tool executions cancelled; dispatched left alone (docs/07 §3.2)
```

**Flush the transport buffer first, cancel upstream second.** The caller's experience is
determined entirely by when audio stops coming out of their speaker; cleaning up the LLM
stream can take another 50 ms without anyone noticing. Getting this order backwards is the
common implementation bug and it is audible.

Measured by injecting a known-timestamp speech onset in the replay harness and recording
the last non-silent output sample.

---

## 4. WebSocket protocol

`/ws` — binary audio frames plus JSON control messages on the same connection.

**Client → server:** raw 16 kHz mono PCM16 binary frames (20 ms, 640 bytes), plus:

```json
{"type": "start", "domain": "ticketing", "source": "browser"}
{"type": "stop"}
```

**Server → client:** binary PCM16 at the TTS sample rate, plus:

```json
{"type":"call_started","call_id":"01JQ…"}
{"type":"transcript","role":"user","text":"…","is_final":false}
{"type":"transcript","role":"agent","text":"…","segment_id":3}
{"type":"state","verified":true,"order_id":"TK-88213"}
{"type":"tool","name":"lookup_order","status":"dispatched"}
{"type":"readback","text":"Transferring 2 tickets…"}
{"type":"checker","segment_id":4,"verdict":"fail_ungrounded_entity"}
{"type":"handoff","reason":"refund outside policy","summary_url":"/handoff/01JQ…"}
{"type":"metrics","voice_to_voice_ms":1180,"stages":{...}}
{"type":"call_ended","outcome":"handoff"}
```

The `checker`, `tool`, and `metrics` messages exist for the demo. Showing a checker verdict
and a live latency breakdown on screen while the agent talks is the single most effective
thing in the demo video — it makes an invisible system legible in three seconds.

### Reconnection

The client reconnects with `{"type":"start","resume_call_id":"…"}`. The server reloads
`CallState` from Postgres (which is why state is flushed per turn, not at call end) and
continues. A resume that arrives more than 60 s after the drop starts a new call instead.

---

## 5. Browser client

`app/static/index.html` + `client.js`. Vanilla, no framework, no build step.

- `getUserMedia` → `AudioWorklet` → downsample to 16 kHz → PCM16 → WebSocket.
  `AudioWorklet`, not `ScriptProcessor` (deprecated, and it glitches under load).
- Playback via `AudioWorklet` with a small jitter buffer (3 × 20 ms). Playback must be
  **flushable instantly** on a `CancelSpeech` — a long buffer is extra barge-in latency the
  caller hears.
- Echo cancellation on: `{echoCancellation: true, noiseSuppression: true, autoGainControl: true}`.
  Without AEC the agent hears itself and barge-in fires constantly on laptop speakers.
- UI, three panes:
  1. live transcript, user and agent;
  2. a trace pane — retrieval scores, tool calls and statuses, checker verdicts;
  3. a latency strip showing the per-stage breakdown for the last turn.

Pane 2 and 3 are the demo. Build them in Week 1 while the plumbing is fresh, not in Week 4
when time is gone.

---

## 6. Replay transport

`eval/replay.py` substitutes a file-reading transport for the browser one. Everything
downstream — VAD, STT, harness, checker, TTS — is identical, which is what makes replay
results comparable to live ones.

- Audio is fed at **real time by default** (VAD behaviour depends on wall-clock pacing;
  feeding faster invalidates endpointing measurements). `--fast` feeds as quickly as
  possible for correctness-only runs and marks the run `timing_invalid: true` in
  `metrics.json` so the numbers can never be quoted as latency.
- TTS output is written to a wav per turn rather than played, so demo failures can be
  listened to.
- Injectable events: `--interrupt-at <turn>:<ms>` fires a synthetic barge-in at an exact
  offset, which is how the interruption suite gets deterministic timing.

### Replay vs live latency

They differ, and the difference is reported rather than elided:

| Term | Replay | Live |
|---|---|---|
| Mic capture + browser encode | absent | ~20–40 ms |
| Network to server | absent (localhost) | 10–60 ms |
| Jitter buffer | absent | ~60 ms |
| Everything else | identical | identical |

Replay is the reproducible number and the one used for before/after comparisons. A live
number is measured once, on the demo machine, and reported alongside with the gap
attributed. Quoting a replay number as a live number is the kind of thing that falls apart
in an interview.

---

## 7. Telephony (stretch)

Twilio Media Streams over the same WebSocket. The changes are contained:

- 8 kHz μ-law in and out instead of 16 kHz PCM16 — one codec adapter in `bridge.py`.
- No browser AEC; the carrier provides it.
- DTMF frames become a new `InEvent.DTMF` (useful for order-number entry, which is far more
  reliable than ASR on a noisy line).

The eval set already includes an 8 kHz μ-law variant
([12-eval.md](12-eval.md) §4), so the accuracy consequences of the phone path are measured
before any phone number is bought. That ordering is deliberate: the measurement is the
valuable part, the phone number is a demo nicety.
