"""Deepgram — a true streaming ASR model.

Two surfaces, both configured from the same Settings object so the config
snapshot describes whichever one produced a number:

  * `transcribe_file()` — batch REST. Used by WER measurement in week 3.
  * `stream()` — live WebSocket. Used by non-Pipecat paths and by tests.

In the live browser pipeline (W1-7) audio is carried by Pipecat's own Deepgram
service; this class is the measurement surface. Both read the same settings,
so `stt.provider` in the snapshot is accurate either way.

`numerals=true` matters more than it looks: it makes Deepgram surface
"eighty-eight two thirteen" as digits, which is what the week-2 entity
extractor and the order-id lexicon need in order to match.

Request shapes here follow Deepgram's documented API. They are exercised by
`make dev` and by week-3 eval runs — both need a real key, so they are not
covered by `make test`.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from pathlib import Path
from urllib.parse import urlencode

from core.settings import Settings
from providers.stt.base import STTProvider, Transcript

_REST_URL = "https://api.deepgram.com/v1/listen"
_WS_URL = "wss://api.deepgram.com/v1/listen"

_COMMON_PARAMS = {
    "model": "nova-3",
    "language": "en",
    "smart_format": "true",
    "numerals": "true",
    "punctuate": "true",
}


class DeepgramProvider(STTProvider):
    name = "deepgram"
    streaming = True

    def __init__(self, settings: Settings, *, sample_rate: int = 16_000) -> None:
        self._api_key = settings.require("deepgram_api_key")
        self._sample_rate = sample_rate
        # Keep Deepgram's own endpointing roughly aligned with our VAD so the
        # two do not fight. Both values go into the config snapshot.
        self._endpointing_ms = int(settings.vad_stop_secs * 1000)

    # -- batch --------------------------------------------------------------

    async def transcribe_file(self, path: str) -> Transcript:
        import asyncio

        import httpx

        # Read off the event loop: a 6s wav is small, but the replay harness
        # runs hundreds of these and a blocking read stalls every other call.
        audio = await asyncio.to_thread(Path(path).read_bytes)
        params = {**_COMMON_PARAMS}

        async with httpx.AsyncClient(timeout=60) as client:
            response = await client.post(
                f"{_REST_URL}?{urlencode(params)}",
                content=audio,
                headers={
                    "Authorization": f"Token {self._api_key}",
                    "Content-Type": "audio/wav",
                },
            )
            response.raise_for_status()
            return _parse_alternative(response.json(), provider=self.name, is_final=True)

    # -- live ---------------------------------------------------------------

    async def stream(self, audio: AsyncIterator[bytes]) -> AsyncIterator[Transcript]:
        import asyncio

        import websockets

        params = {
            **_COMMON_PARAMS,
            "encoding": "linear16",
            "sample_rate": str(self._sample_rate),
            "channels": "1",
            "interim_results": "true",
            "endpointing": str(self._endpointing_ms),
        }
        url = f"{_WS_URL}?{urlencode(params)}"

        async with websockets.connect(
            url, additional_headers={"Authorization": f"Token {self._api_key}"}
        ) as socket:

            async def pump() -> None:
                try:
                    async for frame in audio:
                        await socket.send(frame)
                finally:
                    # Tells Deepgram to flush and finalize rather than hanging.
                    await socket.send(json.dumps({"type": "CloseStream"}))

            pump_task = asyncio.create_task(pump())
            try:
                async for raw in socket:
                    message = json.loads(raw)
                    if message.get("type") != "Results":
                        continue
                    transcript = _parse_alternative(
                        message, provider=self.name, is_final=bool(message.get("is_final"))
                    )
                    if transcript.text:
                        yield transcript
            finally:
                pump_task.cancel()


def _parse_alternative(payload: dict, *, provider: str, is_final: bool) -> Transcript:
    """Pull the top alternative out of either response shape.

    The batch response nests under `results`; the live one does not. Everything
    below that point is the same.
    """
    root = payload.get("results", payload)
    channels = root.get("channels") or [root.get("channel", {})]
    alternatives = (channels[0] or {}).get("alternatives") or [{}]
    best = alternatives[0]

    words = [(w["word"], float(w.get("confidence", 1.0))) for w in best.get("words", [])]
    return Transcript(
        text=best.get("transcript", "").strip(),
        is_final=is_final,
        confidence=float(best.get("confidence", 1.0)),
        word_confidences=words,
        provider=provider,
    )
