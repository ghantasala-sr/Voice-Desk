"""W1-7/W1-8 — playout accounting, the one piece of real logic on the
audio/harness boundary.

This number becomes `played_chars` on a BargeIn, which becomes `spoken_text`,
which becomes what the model believes it told the caller. Getting it wrong is
how an agent ends up referring back to a sentence nobody heard.
"""

from __future__ import annotations

import json

from app.bridge import AudioTimeline, PlayoutTracker, transcript_message, ws_message

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


def test_audio_timings_are_durations_measured_per_turn() -> None:
    t = AudioTimeline()
    t.user_stopped(1_000)
    t.transcript(1_400)            # VAD stop -> transcript: endpointing + STT final
    t.speak("turn-a", 1_900)
    t.completed("turn-a")          # generation ended before any audio played
    assert t.take_ready("turn-a") is None, "not audible yet — nothing to write"
    assert t.audio(2_000) == "turn-a"
    assert t.take_ready("turn-a") == {"tts_ttfb_ms": 100, "first_audio_ms": 600,
                                      "voice_to_voice_ms": 1_000, "playout_ms": None}
    assert t.take_ready("turn-a") is None, "written once"


def test_late_audio_is_not_credited_to_the_next_turn() -> None:
    """The bug: turn A's audio still playing when turn B's timings were read
    gave voice-to-voice = -9151 ms."""
    t = AudioTimeline()
    t.transcript(0)
    t.speak("turn-a", 100)
    t.audio(300)
    t.completed("turn-a")
    t.audio(5_000)                 # more of turn A's audio — not a new first byte
    t.user_stopped(10_000)
    t.transcript(10_400)
    t.speak("turn-b", 10_900)
    t.completed("turn-b")
    assert t.audio(11_100) == "turn-b"
    assert t.take_ready("turn-b")["voice_to_voice_ms"] == 1_100


def test_a_stale_vad_stop_is_not_used() -> None:
    t = AudioTimeline()
    t.user_stopped(0)
    t.transcript(60_000)           # a minute later: not this utterance's stop
    t.speak("x", 60_100)
    t.completed("x")
    t.audio(60_200)
    assert t.take_ready("x")["voice_to_voice_ms"] is None


def test_control_messages_are_json() -> None:
    assert json.loads(ws_message("call_started", call_id="abc")) == {
        "type": "call_started",
        "call_id": "abc",
    }
    message = json.loads(transcript_message("agent", "hello", segment_id=2))
    assert message["role"] == "agent"
    assert message["segment_id"] == 2


def test_a_vad_stop_after_the_transcript_still_belongs_to_it() -> None:
    """Live: Deepgram's final arrived before VAD declared the stop, so the
    previous utterance's stop was used and voice-to-voice read 10,372 ms."""
    t = AudioTimeline()
    t.user_started(0)
    t.user_stopped(800)            # the greeting
    t.user_started(9_000)          # the question begins
    t.transcript(10_900)           # final transcript BEFORE the VAD stop
    t.user_stopped(11_000)
    t.speak("q", 11_100)
    t.completed("q")
    t.audio(11_500)
    assert t.take_ready("q")["voice_to_voice_ms"] == 500
