"""The STT provider interface.

`word_confidences` is load-bearing and is why this interface is not just
`-> str`. The handoff signal is *low confidence on a key entity*, not low
average confidence over the utterance: an order number heard at 0.4 inside an
otherwise clean sentence must be catchable, and a sentence mean hides it
(docs/09 §3).

`streaming` says whether the provider is a true streaming model (Deepgram) or
a batch model being chunked (Whisper). It is recorded in the config snapshot
because the two are not comparable on latency, and pretending otherwise is the
methodological trap in docs/00 D14.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Transcript:
    text: str
    is_final: bool
    confidence: float = 1.0
    word_confidences: list[tuple[str, float]] = field(default_factory=list)
    started_at_ms: int = 0
    ended_at_ms: int = 0
    provider: str = ""

    def min_confidence(self) -> float:
        """Lowest word confidence — the number the handoff signal actually wants."""
        if not self.word_confidences:
            return self.confidence
        return min(score for _, score in self.word_confidences)


class STTProvider(ABC):
    name: str
    streaming: bool

    @abstractmethod
    def stream(self, audio: AsyncIterator[bytes]) -> AsyncIterator[Transcript]:
        """Consume PCM16 frames, yield interim and final transcripts."""

    @abstractmethod
    async def transcribe_file(self, path: str) -> Transcript:
        """Batch transcription. Used by the replay harness and WER measurement."""

    async def aclose(self) -> None:
        return None
