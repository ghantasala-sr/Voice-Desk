"""W1-7/W1-8 — playout accounting, the one piece of real logic on the
audio/harness boundary.

This number becomes `played_chars` on a BargeIn, which becomes `spoken_text`,
which becomes what the model believes it told the caller. Getting it wrong is
how an agent ends up referring back to a sentence nobody heard.
"""

from __future__ import annotations

import json

from app.bridge import PlayoutTracker, TurnTimings, transcript_message, ws_message

SAMPLE_RATE = 24_000
BYTES_PER_SECOND = SAMPLE_RATE * 2  # PCM16


def test_nothing_played_yet() -> None:
    assert PlayoutTracker().played_chars() == 0


def test_completed_segments_count_in_full() -> None:
    tracker = PlayoutTracker(sample_rate=SAMPLE_RATE)
    tracker.segment_started(0, "Refunds take five days.")
    tracker.segment_completed(0)
    assert tracker.played_chars() == len("Refunds take five days.")


def test_active_segment_counts_proportionally() -> None:
    tracker = PlayoutTracker(sample_rate=SAMPLE_RATE)
    tracker.segment_started(0, "Refunds are available within thirty days of purchase.")
    tracker.audio_delivered(BYTES_PER_SECOND)  # one second of audio

    played = tracker.played_chars()
    assert 0 < played < 52
    assert played == 14  # CHARS_PER_SECOND


def test_active_segment_never_exceeds_its_text() -> None:
    tracker = PlayoutTracker(sample_rate=SAMPLE_RATE)
    tracker.segment_started(0, "Short.")
    tracker.audio_delivered(BYTES_PER_SECOND * 30)
    assert tracker.played_chars() == len("Short.")


def test_mixed_completed_and_active() -> None:
    tracker = PlayoutTracker(sample_rate=SAMPLE_RATE)
    tracker.segment_started(0, "First sentence.")
    tracker.segment_completed(0)
    tracker.segment_started(1, "Second sentence still playing.")
    tracker.audio_delivered(BYTES_PER_SECOND // 2)

    assert tracker.played_chars() == len("First sentence.") + 7


def test_reset_clears_everything() -> None:
    tracker = PlayoutTracker()
    tracker.segment_started(0, "text")
    tracker.audio_delivered(1000)
    tracker.reset()
    assert tracker.played_chars() == 0


def test_voice_to_voice_needs_both_ends() -> None:
    assert TurnTimings().voice_to_voice_ms() is None
    assert TurnTimings(user_stopped_ms=100).voice_to_voice_ms() is None
    assert TurnTimings(user_stopped_ms=100, first_audio_ms=1250).voice_to_voice_ms() == 1150


def test_control_messages_are_json() -> None:
    assert json.loads(ws_message("call_started", call_id="abc")) == {
        "type": "call_started",
        "call_id": "abc",
    }
    message = json.loads(transcript_message("agent", "hello", segment_id=2))
    assert message["role"] == "agent"
    assert message["segment_id"] == 2
