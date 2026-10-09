"""Time, injected rather than imported.

Every latency number in this project comes from a clock. If the harness called
`time.monotonic()` directly, latency behaviour could only be tested by actually
waiting, and the barge-in tests could not assert "interrupt 150 ms into segment
two" deterministically.

So the harness takes a Clock. Production passes SystemClock; tests pass
ManualClock and advance it by hand.
"""

from __future__ import annotations

import time
from datetime import UTC, datetime
from typing import Protocol


class Clock(Protocol):
    def now_ms(self) -> int:
        """Monotonic milliseconds. For measuring durations."""

    def wall(self) -> datetime:
        """Wall-clock UTC. For timestamps written to the database."""


class SystemClock:
    """Real time."""

    def now_ms(self) -> int:
        return int(time.monotonic() * 1000)

    def wall(self) -> datetime:
        return datetime.now(UTC)


class ManualClock:
    """Test clock. Time only moves when you move it."""

    def __init__(self, start_ms: int = 0) -> None:
        self._ms = start_ms
        self._wall = datetime(2026, 1, 1, tzinfo=UTC)

    def now_ms(self) -> int:
        return self._ms

    def wall(self) -> datetime:
        return self._wall

    def advance(self, ms: int) -> None:
        self._ms += ms


class Stopwatch:
    """Accumulates the per-stage latency breakdown for one turn.

    Usage mirrors the stage table in docs/01-architecture.md §5:

        sw = Stopwatch(clock)
        with sw.stage("retrieval"):
            ...
        sw.stages            # {"retrieval": 12, ...}
        sw.elapsed_ms()      # since construction
    """

    def __init__(self, clock: Clock) -> None:
        self._clock = clock
        self._start = clock.now_ms()
        self.stages: dict[str, int] = {}

    def elapsed_ms(self) -> int:
        return self._clock.now_ms() - self._start

    def mark(self, name: str) -> int:
        """Record a stage as "everything since the turn began". Returns the value."""
        value = self.elapsed_ms()
        self.stages[name] = value
        return value

    def stage(self, name: str) -> _StageTimer:
        return _StageTimer(self, name)

    def _record(self, name: str, duration_ms: int) -> None:
        self.stages[name] = self.stages.get(name, 0) + duration_ms


class _StageTimer:
    def __init__(self, parent: Stopwatch, name: str) -> None:
        self._parent = parent
        self._name = name
        self._started = 0

    def __enter__(self) -> _StageTimer:
        self._started = self._parent._clock.now_ms()
        return self

    def __exit__(self, *exc: object) -> None:
        self._parent._record(self._name, self._parent._clock.now_ms() - self._started)
