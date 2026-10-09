"""A deterministic LLM provider that replays a recorded event sequence.

This is what makes the entire harness test suite fast, free and repeatable.
The turn loop's logic — tool sequences, segmentation, checker verdicts,
mid-stream interruption — is exercised with no network call.

Fixture format (YAML list, in order):

    - text: "I can help with that. "
    - text: "Let me pull up the order."
      delay_ms: 120                     # wait before yielding this event
    - tool_call:
        name: lookup_order
        arguments: {order_id: "TK-88213"}
    - end:
        stop_reason: tool_use

`delay_ms` is what makes interruption timing deterministic: a test can assert
that barge-in 150 ms into the second text delta trims context exactly where
expected.

A fixture may hold SEVERAL responses separated by an `end` event. Each call to
`stream()` consumes the next one, so one fixture describes a whole multi-turn
conversation.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import yaml

from providers.llm.base import (
    LLMEvent,
    LLMProvider,
    LLMRequest,
    StreamEnd,
    TextDelta,
    ToolCallDelta,
)


class ReplayProvider(LLMProvider):
    name = "replay"

    def __init__(
        self,
        *,
        fixture: str | Path | None = None,
        events: list[dict[str, Any]] | None = None,
        model: str = "replay",
    ) -> None:
        if (fixture is None) == (events is None):
            raise ValueError("pass exactly one of `fixture` or `events`")

        self.model = model
        raw = (
            yaml.safe_load(Path(fixture).read_text(encoding="utf-8"))
            if fixture is not None
            else events
        )
        self._responses = _split_into_responses(raw or [])
        self._index = 0

        # Inspectable by tests: how the harness called us.
        self.requests: list[LLMRequest] = []

    @property
    def call_count(self) -> int:
        return len(self.requests)

    async def stream(self, request: LLMRequest) -> AsyncIterator[LLMEvent]:
        self.requests.append(request)

        if self._index >= len(self._responses):
            raise AssertionError(
                f"replay fixture exhausted: the harness made {self._index + 1} LLM "
                f"calls but the fixture defines {len(self._responses)}"
            )
        response = self._responses[self._index]
        self._index += 1

        for step in response:
            delay = step.get("delay_ms", 0)
            if delay:
                # A real await, so cancellation lands here exactly as it would
                # in a network stream.
                await asyncio.sleep(delay / 1000)
            yield _to_event(step)


def _split_into_responses(raw: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    """Split a flat fixture into one list per `stream()` call, on `end` events."""
    responses: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []

    for step in raw:
        current.append(step)
        if "end" in step:
            responses.append(current)
            current = []

    if current:
        # Tolerate a fixture whose last response forgot its `end` marker.
        current.append({"end": {"stop_reason": "end_turn"}})
        responses.append(current)
    return responses


def _to_event(step: dict[str, Any]) -> LLMEvent:
    if "text" in step:
        return TextDelta(text=step["text"])

    if "tool_call" in step:
        call = step["tool_call"]
        return ToolCallDelta(
            call_id=call.get("call_id", f"replay_{call['name']}"),
            name=call["name"],
            arguments=call.get("arguments", {}),
        )

    if "end" in step:
        end = step["end"] or {}
        return StreamEnd(
            stop_reason=end.get("stop_reason", "end_turn"),
            input_tokens=end.get("input_tokens", 0),
            output_tokens=end.get("output_tokens", 0),
            cache_read_tokens=end.get("cache_read_tokens", 0),
            error=end.get("error"),
        )

    raise ValueError(f"replay fixture step has no known key: {step!r}")
