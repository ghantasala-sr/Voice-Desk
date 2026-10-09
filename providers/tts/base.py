"""The TTS provider interface.

`cancel()` returning a character offset is how `spoken_text` gets reconstructed
after a barge-in. It is **approximate**, and saying so is part of the design
(docs/04 §3):

  * The offset reflects audio handed to the transport, not audio that reached
    the caller's ear. Jitter buffer and playout latency mean the caller heard
    slightly less — typically a word or two.
  * The harness therefore subtracts VD_PLAYOUT_TRIM_MS and truncates to the
    nearest word boundary BEFORE the computed offset. It errs toward assuming
    the caller heard less, which is the safe direction: the agent may repeat
    itself, but it will never assume the caller heard something they did not.

Claiming exact knowledge of what was heard is a claim the system cannot
support, and an interviewer will ask.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from dataclasses import dataclass


@dataclass(frozen=True)
class AudioChunk:
    pcm: bytes
    sample_rate: int
    is_final: bool = False
    char_offset: int = 0
    """How far into the submitted text this chunk corresponds to."""


class TTSProvider(ABC):
    name: str
    sample_rate: int

    @abstractmethod
    def synthesize(self, text: str, *, context_id: str) -> AsyncIterator[AudioChunk]:
        """Stream audio for `text`.

        Consecutive segments sharing a `context_id` should join into one
        prosodic stream rather than sounding like separate utterances.
        """

    @abstractmethod
    async def cancel(self, context_id: str) -> int:
        """Stop synthesis and playout. Returns characters actually played."""

    async def aclose(self) -> None:
        return None
