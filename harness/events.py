"""Events in, commands out.

The harness's entire interface with the outside world. It receives text and
events; it returns text and commands. It never touches an audio device, and
nothing here mentions a sample rate.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# In — produced by app/bridge.py (live) or eval/text_driver.py (text mode)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class TranscriptIn:
    """A final transcript. The ordinary trigger for a turn."""

    text: str
    confidence: float = 1.0
    word_confidences: list[tuple[str, float]] = field(default_factory=list)
    provider: str = ""
    # Stage [A]: how long VAD actually waited in silence before deciding the
    # caller had stopped. Reported separately from everything else, always.
    endpoint_ms: int | None = None
    # Stage [B]: transcript finalization.
    stt_final_ms: int | None = None


@dataclass(frozen=True)
class BargeIn:
    """The caller started speaking while the agent was speaking.

    `played_chars` counts characters of THIS TURN'S approved text that the
    transport had delivered when the interruption landed. The bridge computes
    it as (fully-played segments) + (offset into the current segment).

    It is an estimate — see providers/tts/base.py. The harness trims it
    further before trusting it.
    """

    played_chars: int | None = None


@dataclass(frozen=True)
class SilenceTimeout:
    """No speech for VD_SILENCE_TIMEOUT_MS."""


@dataclass(frozen=True)
class Shutdown:
    """Stop the run loop. The transport closed, or the caller hung up."""

    reason: str = "closed"


InEvent = TranscriptIn | BargeIn | SilenceTimeout | Shutdown


# ---------------------------------------------------------------------------
# Out — consumed by app/bridge.py or the text driver
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Speak:
    """Send this text to TTS.

    Segments sharing a `context_id` join into one prosodic stream, so the
    caller hears a continuous sentence rather than stitched fragments.
    """

    text: str
    segment_id: int
    context_id: str


@dataclass(frozen=True)
class CancelSpeech:
    """Stop audio NOW.

    Issued first on barge-in, before any upstream cleanup: the caller's
    experience is determined entirely by when sound stops coming out of their
    speaker (docs/10 §3).
    """

    context_id: str


@dataclass(frozen=True)
class EndCall:
    reason: str


@dataclass(frozen=True)
class Emit:
    """A structured event for the event writer and the browser trace pane."""

    event_type: str
    payload: dict[str, Any] = field(default_factory=dict)
    turn_id: str | None = None


@dataclass(frozen=True)
class TurnCompleted:
    """A finished turn, for the runner to persist.

    The harness never writes to Postgres itself — it emits state and a writer
    persists it. That is what keeps the harness testable without a database.
    """

    turn: Any  # harness.state.TurnState; untyped here to avoid a cycle


OutCommand = Speak | CancelSpeech | EndCall | Emit | TurnCompleted


# ---------------------------------------------------------------------------
# EventStream — the harness's inbox
# ---------------------------------------------------------------------------
class EventStream:
    """An inbox that supports pushback.

    While a turn is running, a watcher task drains this looking for a BargeIn.
    Anything else it pulls off is held and pushed back when the turn ends, so
    no event is lost and ordering is preserved.
    """

    def __init__(self) -> None:
        import asyncio
        from collections import deque

        self._queue: asyncio.Queue[InEvent] = asyncio.Queue()
        self._pushed: deque[InEvent] = deque()

    def put(self, event: InEvent) -> None:
        self._queue.put_nowait(event)

    def pushback(self, events: list[InEvent]) -> None:
        for event in reversed(events):
            self._pushed.appendleft(event)

    async def get(self) -> InEvent:
        if self._pushed:
            return self._pushed.popleft()
        return await self._queue.get()
