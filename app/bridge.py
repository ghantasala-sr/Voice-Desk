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


@dataclass
class TurnTimings:
    """Audio-side stages the harness cannot see.

    The harness measures up to the moment it hands a segment to TTS. Stages
    [G] (TTS time-to-first-byte) and [H] (playout start) are measured here and
    written onto the turn row afterwards, rather than guessed at by the
    harness. See analytics/store.py::update_turn_audio_timing.
    """

    user_stopped_ms: int | None = None
    tts_ttfb_ms: int | None = None
    playout_ms: int | None = None
    first_audio_ms: int | None = None

    def voice_to_voice_ms(self) -> int | None:
        if self.user_stopped_ms is None or self.first_audio_ms is None:
            return None
        return self.first_audio_ms - self.user_stopped_ms


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
