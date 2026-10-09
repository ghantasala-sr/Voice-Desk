"""An STT provider that never produces anything.

Used by the text driver, where transcripts are typed rather than spoken.
Having a real object here rather than `None` means the harness has exactly one
code path, so a text-mode test exercises the same loop a phone call does.
"""

from __future__ import annotations

from collections.abc import AsyncIterator

from providers.stt.base import STTProvider, Transcript


class NullSTT(STTProvider):
    name = "null"
    streaming = False

    async def stream(self, audio: AsyncIterator[bytes]) -> AsyncIterator[Transcript]:
        # Drain the input so an upstream producer is not left blocked, and
        # yield nothing.
        async for _ in audio:
            pass
        return
        yield  # pragma: no cover - makes this an async generator

    async def transcribe_file(self, path: str) -> Transcript:
        raise NotImplementedError("NullSTT cannot transcribe audio; use a real provider")
