"""The call event writer — the only thing that writes to `call_events`.

Three rules (docs/11 §4):

  * The event vocabulary is CLOSED. An unknown type raises in tests and logs a
    warning in production. An open vocabulary becomes unqueryable within a
    week, and the gap report depends on being able to query it.
  * Writes are buffered and flushed per turn, not per event — one round trip
    per turn instead of fifteen.
  * Flush happens in a `finally`. A crashed call must still have its events.
"""

from __future__ import annotations

import json
import os
from typing import Any, Protocol

from core.logging import get_logger

log = get_logger(__name__)

# The closed vocabulary. Adding an event type means adding it here first.
EVENT_TYPES: frozenset[str] = frozenset(
    {
        # call lifecycle
        "call_started", "call_ended", "error",
        # turn lifecycle
        "turn_started", "turn_completed", "budget_exceeded",
        # speech recognition
        "stt_final", "silence_reprompt",
        # retrieval (week 2)
        "retrieval_completed",
        # generation
        "llm_iteration_started", "llm_tool_call", "llm_text_completed",
        # checker (week 2)
        "checker_verdict",
        # read-back (week 2)
        "readback_spoken", "readback_confirmed", "readback_rejected",
        # tools (week 2)
        "tool_dispatched", "tool_committed", "tool_failed",
        "tool_cancelled", "tool_unknown", "tool_reconciled", "filler_spoken",
        # interruption
        "barge_in", "context_trimmed",
        # handoff (week 2)
        "handoff_signal_raised", "handoff_triggered",
        # analytics (week 4)
        "simulation_label",
    }
)

#: Flushed immediately rather than at the end of the turn, so the live console
#: shows a barge-in, a tool changing state, or a handoff as it happens.
URGENT_EVENTS: frozenset[str] = frozenset(
    {
        "call_started", "call_ended", "turn_started", "barge_in", "checker_verdict",
        "handoff_signal_raised", "handoff_triggered", "readback_spoken",
        "tool_dispatched", "tool_committed", "tool_failed", "tool_cancelled",
        "tool_unknown", "tool_reconciled", "error",
    }
)

def strict() -> bool:
    """In tests, an unknown event type should fail loudly rather than drift.

    Read at call time: pytest sets PYTEST_CURRENT_TEST per test, AFTER modules
    are imported. As an import-time constant this was always False, so for
    three weeks no test could ever have failed on an unknown type (found by
    the W4-2 audit)."""
    return os.environ.get("PYTEST_CURRENT_TEST") is not None


class EventWriter(Protocol):
    async def emit(
        self, call_id: str, event_type: str, payload: dict[str, Any], turn_id: str | None = None
    ) -> None: ...

    async def flush(self) -> None: ...


def validate(event_type: str) -> None:
    if event_type in EVENT_TYPES:
        return
    message = f"unknown event_type {event_type!r}; add it to analytics.events.EVENT_TYPES"
    if strict():
        raise ValueError(message)
    log.warning("unknown_event_type", event_type=event_type)


class NullEventWriter:
    """Discards events, but still validates them. Used by harness unit tests."""

    def __init__(self) -> None:
        self.events: list[tuple[str, dict[str, Any]]] = []

    async def emit(
        self, call_id: str, event_type: str, payload: dict[str, Any], turn_id: str | None = None
    ) -> None:
        validate(event_type)
        self.events.append((event_type, payload))

    async def flush(self) -> None:
        return None

    def types(self) -> list[str]:
        return [event_type for event_type, _ in self.events]


class PostgresEventWriter:
    """Buffers events and writes them in one statement per flush."""

    def __init__(self, *, flush_every: int = 64) -> None:
        self._buffer: list[tuple[str, str | None, int, str, str]] = []
        self._seq = 0
        self._flush_every = flush_every

    async def emit(
        self, call_id: str, event_type: str, payload: dict[str, Any], turn_id: str | None = None
    ) -> None:
        validate(event_type)
        self._buffer.append((call_id, turn_id, self._seq, event_type, json.dumps(payload)))
        self._seq += 1
        if event_type in URGENT_EVENTS or len(self._buffer) >= self._flush_every:
            await self.flush()

    async def flush(self) -> None:
        if not self._buffer:
            return
        rows, self._buffer = self._buffer, []

        from core.db import pool

        async with pool().connection() as conn, conn.cursor() as cur:
            await cur.executemany(
                """
                INSERT INTO call_events (call_id, turn_id, seq, event_type, payload)
                VALUES (%s, %s, %s, %s, %s::jsonb)
                """,
                rows,
            )
