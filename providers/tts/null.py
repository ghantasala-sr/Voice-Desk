"""A TTS provider that produces silence and records what it was asked to say.

Used by the text driver and by harness tests. `spoken` and `cancelled` make it
a test double as well as a stub: a test can assert exactly which segments
reached TTS and where a barge-in cut the audio.
"""

from __future__ import annotations

from collections.abc import AsyncIterator

from providers.tts.base import AudioChunk, TTSProvider


class NullTTS(TTSProvider):
    name = "null"
    sample_rate = 16_000

    def __init__(self) -> None:
        self.spoken: list[str] = []
        self.cancelled: list[str] = []
        # Set by a test to simulate "the caller heard only this many characters
        # of the current segment before interrupting".
        self.played_chars_on_cancel: int | None = None
        self._in_flight: dict[str, str] = {}

    async def synthesize(self, text: str, *, context_id: str) -> AsyncIterator[AudioChunk]:
        self.spoken.append(text)
        self._in_flight[context_id] = text
        yield AudioChunk(pcm=b"", sample_rate=self.sample_rate, is_final=True, char_offset=len(text))

    async def cancel(self, context_id: str) -> int:
        self.cancelled.append(context_id)
        if self.played_chars_on_cancel is not None:
            return self.played_chars_on_cancel
        # Nothing was really played, so report the full text as delivered.
        return len(self._in_flight.get(context_id, ""))
