"""W1-4 — the replay provider, which the whole fast test suite rests on."""

from __future__ import annotations

import asyncio

import pytest

from providers.llm.base import LLMRequest, StreamEnd, TextDelta, ToolCallDelta
from providers.llm.replay import ReplayProvider

REQUEST = LLMRequest(system_static="s", messages=[])


async def _drain(provider: ReplayProvider) -> list[object]:
    return [event async for event in provider.stream(REQUEST)]


async def test_emits_events_in_order() -> None:
    provider = ReplayProvider(
        events=[
            {"text": "Hello. "},
            {"tool_call": {"name": "lookup_order", "arguments": {"order_id": "TK-88213"}}},
            {"end": {"stop_reason": "tool_use", "input_tokens": 10, "output_tokens": 4}},
        ]
    )
    events = await _drain(provider)

    assert isinstance(events[0], TextDelta)
    assert events[0].text == "Hello. "

    assert isinstance(events[1], ToolCallDelta)
    assert events[1].name == "lookup_order"
    assert events[1].arguments == {"order_id": "TK-88213"}

    assert isinstance(events[2], StreamEnd)
    assert events[2].stop_reason == "tool_use"
    assert events[2].input_tokens == 10


async def test_one_fixture_serves_several_turns() -> None:
    """A fixture split on `end` markers describes a whole conversation."""
    provider = ReplayProvider(
        events=[
            {"text": "First."},
            {"end": {}},
            {"text": "Second."},
            {"end": {}},
        ]
    )
    first = await _drain(provider)
    second = await _drain(provider)

    assert first[0].text == "First."
    assert second[0].text == "Second."
    assert provider.call_count == 2


async def test_exhausted_fixture_fails_loudly() -> None:
    provider = ReplayProvider(events=[{"text": "only one"}, {"end": {}}])
    await _drain(provider)

    with pytest.raises(AssertionError, match="fixture exhausted"):
        await _drain(provider)


async def test_missing_end_marker_is_tolerated() -> None:
    provider = ReplayProvider(events=[{"text": "no end marker"}])
    events = await _drain(provider)
    assert isinstance(events[-1], StreamEnd)
    assert events[-1].stop_reason == "end_turn"


async def test_delay_is_real_and_cancellable() -> None:
    """`delay_ms` must be a real await, so cancellation lands in the same place
    it would in a network stream. Barge-in correctness depends on this."""
    provider = ReplayProvider(
        events=[{"text": "slow", "delay_ms": 5000}, {"end": {}}]
    )
    stream = provider.stream(REQUEST)

    task = asyncio.ensure_future(stream.__anext__())
    await asyncio.sleep(0.02)
    assert not task.done(), "the delay was not awaited"

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    await stream.aclose()


async def test_rejects_ambiguous_construction() -> None:
    with pytest.raises(ValueError, match="exactly one"):
        ReplayProvider()


def test_openai_translation_of_tool_blocks() -> None:
    """The second provider must handle tool history, or it is not an interface."""
    from providers.llm.openai import _to_openai_messages

    neutral = [
        {"role": "user", "content": [{"type": "text", "text": "where are my tickets"}]},
        {"role": "assistant", "content": [
            {"type": "text", "text": "Checking."},
            {"type": "tool_use", "id": "tu1", "name": "lookup_order", "input": {"order_id": "TK-88213"}},
        ]},
        {"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "tu1", "content": '{"quantity": 4}'},
            {"type": "text", "text": "thanks"},
        ]},
    ]
    out = _to_openai_messages(neutral)
    assert [m["role"] for m in out] == ["user", "assistant", "tool", "user"]
    assert out[1]["tool_calls"][0]["function"]["name"] == "lookup_order"
    assert out[2]["tool_call_id"] == "tu1"
