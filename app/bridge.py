"""The seam between the audio domain and the harness.

Deliberately Pipecat-free. The one piece of real logic on this boundary is
playout accounting: working out how much of what we handed to TTS the caller
had actually heard when they interrupted. That is audio-domain knowledge, it
is the input to `spoken_text`, and it is the thing most worth testing — so it
lives here as plain Python rather than inside the Pipecat shim.

`app/pipeline.py` converts Pipecat frames into calls on this module.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

# Rough speaking rate, matching harness/context.py. Used to convert delivered
# audio bytes into an approximate character offset.
CHARS_PER_SECOND = 14.0


@dataclass
class PlayoutTracker:
    """Tracks how far into a turn's speech the transport has actually got.

    The harness needs ONE number on a barge-in: how many characters of this
    turn's approved text the caller heard. Everything here exists to produce
    that number honestly.

    It is an estimate, and it is biased low on purpose. See
    providers/tts/base.py and harness/context.py — erring toward "the caller
    heard less" means the agent may repeat itself, which is awkward, instead
    of believing it said something it never did, which is the failure mode.
    """

    sample_rate: int = 24_000
    #: segment_id -> text handed to TTS
    segments: dict[int, str] = field(default_factory=dict)
    #: segment ids whose audio has fully drained to the transport
    completed: set[int] = field(default_factory=set)
    #: segment currently being played, and bytes delivered for it
    active_id: int | None = None
    active_bytes: int = 0

    def reset(self) -> None:
        self.segments.clear()
        self.completed.clear()
        self.active_id = None
        self.active_bytes = 0

    def segment_started(self, segment_id: int, text: str) -> None:
        self.segments[segment_id] = text
        self.active_id = segment_id
        self.active_bytes = 0

    def audio_delivered(self, num_bytes: int) -> None:
        self.active_bytes += num_bytes

    def segment_completed(self, segment_id: int) -> None:
        self.completed.add(segment_id)
        if self.active_id == segment_id:
            self.active_id = None
            self.active_bytes = 0

    def played_chars(self) -> int:
        """Characters of the turn's speech delivered so far.

        Counts fully-drained segments in full, plus a byte-proportional
        estimate of the segment still playing.
        """
        total = sum(len(self.segments[i]) for i in sorted(self.completed) if i in self.segments)
        if self.active_id is not None:
            text = self.segments.get(self.active_id, "")
            seconds = self.active_bytes / (self.sample_rate * 2)  # PCM16
            total += min(len(text), int(seconds * CHARS_PER_SECOND))
        return total


class AudioTimeline:
    """Per-turn audio timestamps, kept until the turn's FIRST AUDIO exists.

    Replaces writing TurnTimings at TurnCompleted. The harness completes a turn
    when generation ends, which is usually BEFORE its audio has played; the
    timings were then reset and the previous turn's still-playing audio was
    credited to the next turn (one live turn recorded voice-to-voice = -9151 ms,
    and first_audio_ms held an absolute clock reading).

    Every value is a DURATION in ms:
      tts_ttfb_ms        first Speak of the turn -> its first audio byte out
      first_audio_ms     transcript in -> first audio byte out
      voice_to_voice_ms  caller stopped (VAD) -> first audio byte out
    A turn's timings are ready once BOTH the turn is saved and its audio began.
    """

    #: A VAD stop older than this is not the end of THIS utterance.
    MAX_STOP_TO_TRANSCRIPT_MS = 10_000

    def __init__(self) -> None:
        self._last_stop: int | None = None
        self._last_start: int | None = None
        self._pending: tuple[int, int | None] | None = None   # (transcript_at, stopped_at)
        #: The transcript (pending, or already a turn) still waiting for the
        #: VAD stop of its own utterance.
        self._awaiting_stop: str | None = None
        self._turns: dict[str, dict[str, Any]] = {}
        self._current: str | None = None

    def user_started(self, now: int) -> None:
        self._last_start = now

    def user_stopped(self, now: int) -> None:
        self._last_stop = now
        # Deepgram's final transcript often lands BEFORE VAD declares the end
        # of speech (VAD waits stop_secs of silence). That stop belongs to the
        # transcript that is waiting for it — not to the previous utterance.
        if self._awaiting_stop == "pending" and self._pending is not None:
            self._pending = (self._pending[0], now)
        elif self._awaiting_stop in self._turns:
            self._turns[self._awaiting_stop]["stopped_at"] = now
        self._awaiting_stop = None

    def transcript(self, now: int) -> None:
        stop = self._last_stop
        still_speaking = self._last_start is not None and (stop is None or self._last_start > stop)
        if still_speaking or (stop is not None and not 0 <= now - stop <= self.MAX_STOP_TO_TRANSCRIPT_MS):
            stop = None
        self._awaiting_stop = "pending" if still_speaking else None
        self._pending = (now, stop)

    def speak(self, turn_id: str, now: int) -> None:
        if turn_id in self._turns:
            return
        transcript_at, stopped_at = self._pending or (None, None)
        self._pending = None
        self._turns[turn_id] = {"transcript_at": transcript_at, "stopped_at": stopped_at,
                                "first_speak_at": now, "first_audio_at": None,
                                "completed": False, "written": False}
        self._current = turn_id
        if self._awaiting_stop == "pending":
            self._awaiting_stop = turn_id

    def audio(self, now: int) -> str | None:
        """First audio byte of the current turn? Returns its id if so."""
        turn = self._turns.get(self._current or "")
        if turn is None or turn["first_audio_at"] is not None:
            return None
        turn["first_audio_at"] = now
        return self._current

    def completed(self, turn_id: str) -> None:
        if turn_id in self._turns:
            self._turns[turn_id]["completed"] = True

    def take_ready(self, turn_id: str) -> dict[str, int | None] | None:
        """The turn's timings, once, when saved AND audible; else None."""
        t = self._turns.get(turn_id)
        if t is None or t["written"] or not t["completed"] or t["first_audio_at"] is None:
            return None
        t["written"] = True
        audio = t["first_audio_at"]

        def since(start: int | None) -> int | None:
            return audio - start if start is not None and audio >= start else None
        # Not ready while this turn still waits for its own VAD stop — unless
        # the audio is long past, in which case the stop is simply unknown.

        return {"tts_ttfb_ms": since(t["first_speak_at"]), "first_audio_ms": since(t["transcript_at"]),
                "voice_to_voice_ms": since(t["stopped_at"]), "playout_ms": None}


# ---------------------------------------------------------------------------
# Browser protocol (docs/10 §4)
# ---------------------------------------------------------------------------
def ws_message(kind: str, **fields: Any) -> str:
    """Serialize a control message for the browser client.

    Audio travels as binary frames on the same socket; these JSON messages
    carry transcript, state, and the trace/latency panes that make the demo
    legible.
    """
    return json.dumps({"type": kind, **fields})


def transcript_message(role: str, text: str, *, is_final: bool = True, segment_id: int | None = None) -> str:
    return ws_message(
        "transcript", role=role, text=text, is_final=is_final, segment_id=segment_id
    )
