"""Cartesia Sonic.

Chosen for time-to-first-byte, which sits directly inside the voice-to-voice
latency we report (docs/00 D2).

As with Deepgram, there are two surfaces:

  * `synthesize()` — HTTP byte stream. Used by week-3 eval audio synthesis and
    by any non-Pipecat path.
  * The live browser pipeline uses Pipecat's own Cartesia service, configured
    from these same settings.

`cancel()` reports the character offset we believe was played. The harness
treats that as approximate and trims further — see providers/tts/base.py.

Needs a real key to exercise; not covered by `make test`.
"""

from __future__ import annotations

from collections.abc import AsyncIterator

from core.settings import Settings
from providers.tts.base import AudioChunk, TTSProvider

_URL = "https://api.cartesia.ai/tts/bytes"
# Pin the API version explicitly: Cartesia dates its API, and an unpinned
# client silently changes behaviour under you.
_API_VERSION = "2024-06-10"


class CartesiaProvider(TTSProvider):
    name = "cartesia"
    sample_rate = 24_000

    def __init__(self, settings: Settings, *, model_id: str = "sonic-2") -> None:
        self._api_key = settings.require("cartesia_api_key")
        self._voice_id = settings.cartesia_voice_id
        if not self._voice_id:
            raise RuntimeError("CARTESIA_VOICE_ID is not set. Add it to .env.")
        self._model_id = model_id
        # context_id -> (text, bytes delivered)
        self._in_flight: dict[str, tuple[str, int]] = {}

    async def synthesize(self, text: str, *, context_id: str) -> AsyncIterator[AudioChunk]:
        import httpx

        self._in_flight[context_id] = (text, 0)
        body = {
            "model_id": self._model_id,
            "transcript": text,
            "voice": {"mode": "id", "id": self._voice_id},
            "output_format": {
                "container": "raw",
                "encoding": "pcm_s16le",
                "sample_rate": self.sample_rate,
            },
            "language": "en",
        }
        headers = {
            "X-API-Key": self._api_key,
            "Cartesia-Version": _API_VERSION,
            "Content-Type": "application/json",
        }

        delivered = 0
        async with httpx.AsyncClient(timeout=30) as client, client.stream(
            "POST", _URL, json=body, headers=headers
        ) as response:
            response.raise_for_status()
            async for chunk in response.aiter_bytes():
                if not chunk:
                    continue
                delivered += len(chunk)
                self._in_flight[context_id] = (text, delivered)
                yield AudioChunk(
                    pcm=chunk,
                    sample_rate=self.sample_rate,
                    char_offset=_chars_delivered(text, delivered, self.sample_rate),
                )

        yield AudioChunk(
            pcm=b"", sample_rate=self.sample_rate, is_final=True, char_offset=len(text)
        )

    async def cancel(self, context_id: str) -> int:
        text, delivered = self._in_flight.pop(context_id, ("", 0))
        return _chars_delivered(text, delivered, self.sample_rate)


def _chars_delivered(text: str, audio_bytes: int, sample_rate: int) -> int:
    """Estimate how far into `text` the delivered audio reaches.

    Linear interpolation against an assumed speaking rate. This is an estimate,
    and the harness compounds it with VD_PLAYOUT_TRIM_MS and a word-boundary
    truncation before trusting it (docs/04 §3). We deliberately do not pretend
    to character-accurate alignment.
    """
    if not text:
        return 0
    seconds = audio_bytes / (sample_rate * 2)  # 2 bytes per PCM16 sample
    chars_per_second = 14.0  # ~165 wpm at ~5 characters per word
    return min(len(text), int(seconds * chars_per_second))
