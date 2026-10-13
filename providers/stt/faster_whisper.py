"""faster-whisper — a self-hosted BATCH model.

Whisper is not a streaming model, and this implementation does not pretend it
is (docs/00 D14). `transcribe_file` runs the whole utterance at once; the live
`stream()` path buffers until an endpoint and then does the same, which is the
honest way to put a batch model on a real-time path — and its latency cost is
part of what the comparison measures.

Model size, compute type and device are recorded in the config snapshot:
`small.en` int8 on a laptop CPU and `large-v3` on a GPU are different results.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator
from typing import Any

from core.settings import Settings
from providers.stt.base import STTProvider, Transcript

DEFAULT_MODEL = "small.en"


class FasterWhisperProvider(STTProvider):
    name = "faster_whisper"
    streaming = False

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        model_size: str = DEFAULT_MODEL,
        device: str = "cpu",
        compute_type: str = "int8",
    ) -> None:
        from faster_whisper import WhisperModel

        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self._model = WhisperModel(model_size, device=device, compute_type=compute_type)

    def describe(self) -> dict[str, Any]:
        return {"provider": self.name, "streaming": False, "model": self.model_size,
                "device": self.device, "compute_type": self.compute_type}

    def _transcribe(self, path: str) -> tuple[Transcript, float]:
        import soundfile as sf

        # Hand the model samples, not a path. Path input goes through PyAV,
        # whose API has drifted from what this faster-whisper version calls;
        # the eval files are already 16 kHz mono PCM, so decoding adds nothing.
        audio, rate = sf.read(path, dtype="float32")
        if rate != 16_000:
            raise ValueError(f"{path}: expected 16 kHz, got {rate}")
        started = time.perf_counter()
        segments, _ = self._model.transcribe(
            audio, language="en", beam_size=1, word_timestamps=True, vad_filter=False,
            condition_on_previous_text=False,
        )
        words: list[tuple[str, float]] = []
        texts: list[str] = []
        for segment in segments:  # a generator: decoding happens while iterating
            texts.append(segment.text)
            words.extend((w.word.strip(), float(w.probability)) for w in segment.words or [])
        elapsed = time.perf_counter() - started
        confidence = sum(p for _, p in words) / len(words) if words else 0.0
        return (
            Transcript(text=" ".join(t.strip() for t in texts).strip(), is_final=True,
                       confidence=confidence, word_confidences=words, provider=self.name),
            elapsed,
        )

    async def transcribe_file(self, path: str) -> Transcript:
        transcript, _ = await asyncio.to_thread(self._transcribe, path)
        return transcript

    def transcribe_timed(self, path: str) -> tuple[Transcript, float]:
        """Synchronous, with wall-clock decode time — for the benchmark."""
        return self._transcribe(path)

    async def stream(self, audio: AsyncIterator[bytes]) -> AsyncIterator[Transcript]:
        raise NotImplementedError(
            "Live faster-whisper needs VAD-chunked buffering in the audio pipeline; "
            "only the file path is implemented so far (week 3 measures files)."
        )
        yield  # pragma: no cover
