"""The voice adapter's TTS queue (app/pipeline.py) — no network, no audio.

Verified live against Deepgram + Cartesia + Gemini on 2026-10-15; this pins the
two behaviours that live run exposed:
  * segments reach TTS one at a time (the Cartesia plan allows 2 concurrent
    requests; four queued at once made the extras fail silently), and
  * a cancel drops everything queued and interrupts the pipeline.
"""

from __future__ import annotations

import asyncio
import importlib.util
from typing import Any

import pytest

from app.pipeline import _TTSAdapter

# find_spec, not importorskip: importing pipecat at collection would break
# tests/test_app.py's check that pipecat stays a lazy import.
pytestmark = pytest.mark.skipif(importlib.util.find_spec("pipecat") is None,
                                reason="audio extra not installed")


class FakeProcessor:
    def __init__(self) -> None:
        self.pushed: list[str] = []
        self.interrupted = 0

    async def push_frame(self, frame: Any, direction: Any) -> None:
        self.pushed.append(frame.text)

    async def broadcast_interruption(self) -> None:
        self.interrupted += 1


async def test_segments_go_to_tts_one_at_a_time() -> None:
    processor = FakeProcessor()
    tts = _TTSAdapter(processor)
    for text in ("one.", "two.", "three."):
        await tts.speak(text, context_id="t1")
    await asyncio.sleep(0.05)
    assert processor.pushed == ["one."], "the next segment waits for the previous one"

    tts.segment_done()
    await asyncio.sleep(0.05)
    assert processor.pushed == ["one.", "two."]

    tts.segment_done()
    await asyncio.sleep(0.05)
    tts.segment_done()
    await tts.wait_idle(1.0)
    assert processor.pushed == ["one.", "two.", "three."]
    tts.close()


async def test_cancel_drops_queued_speech_and_interrupts() -> None:
    processor = FakeProcessor()
    tts = _TTSAdapter(processor)
    for text in ("one.", "two.", "three."):
        await tts.speak(text, context_id="t1")
    await asyncio.sleep(0.05)
    await tts.cancel("t1")
    await asyncio.sleep(0.05)
    assert processor.pushed == ["one."], "nothing queued is spoken after a barge-in"
    assert processor.interrupted == 1
    await tts.wait_idle(0.5)
    tts.close()


async def test_a_lost_stop_frame_cannot_wedge_the_call(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(_TTSAdapter, "SEGMENT_TIMEOUT_S", 0.05)
    processor = FakeProcessor()
    tts = _TTSAdapter(processor)
    await tts.speak("one.", context_id="t1")
    await tts.speak("two.", context_id="t1")
    await asyncio.sleep(0.3)
    assert processor.pushed == ["one.", "two."]
    tts.close()
