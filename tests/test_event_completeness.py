"""W4-2 — event completeness audit.

The gap report, the console and every analytics query read `call_events`. An
event type that is in the vocabulary but never emitted is a query that silently
returns nothing, so this file asserts two things:

  1. each call PATH emits the events expected on that path, and
  2. the union of the paths covers the whole closed vocabulary.

The audit found three types that nothing emitted (llm_iteration_started,
llm_text_completed, stt_final — all added to the loop) and one that nothing
COULD emit (stt_partial: partials never leave Pipecat — removed).
"""

from __future__ import annotations

import pytest

from analytics.events import EVENT_TYPES, NullEventWriter, validate
from eval.fault_injection.helpers import TRANSFER_ARGS, end, say, tool_call, tool_driver
from harness.events import SilenceTimeout
from harness.retriever import FakeRetriever

LIFECYCLE = {"call_started", "turn_started", "stt_final", "retrieval_completed",
             "llm_iteration_started", "turn_completed", "call_ended"}
TRANSFER = [tool_call("transfer_tickets", TRANSFER_ARGS), end("tool_use")]
seen: set[str] = set()


def types(driver) -> set[str]:
    writer: NullEventWriter = driver.writer
    found = set(writer.types())
    seen.update(found)
    return found


async def test_answered_path(settings) -> None:
    driver, _, _ = tool_driver(settings, [say("Transfers are free. "), end()],
                               checker_enabled=True,
                               retriever=FakeRetriever.with_chunks(["Transfers are free."]))
    async with driver:
        await driver.say("do transfers cost anything?")
    assert LIFECYCLE | {"checker_verdict", "llm_text_completed"} <= types(driver)


async def test_transfer_path(settings) -> None:
    driver, _, _ = tool_driver(settings, [*TRANSFER, say("Done. "), end()],
                               retriever=FakeRetriever.with_chunks(["Transfers are free."]))
    async with driver:
        await driver.say("send two tickets to friend@example.com")
        await driver.say("yes")
    assert LIFECYCLE | {"llm_tool_call", "readback_spoken", "readback_confirmed",
                        "tool_dispatched", "tool_committed"} <= types(driver)


async def test_declined_transfer_path(settings) -> None:
    driver, _, _ = tool_driver(settings, [*TRANSFER, say("Okay. "), end()])
    async with driver:
        await driver.say("send two tickets to friend@example.com")
        await driver.say("no")
    assert {"readback_rejected", "tool_cancelled"} <= types(driver)


async def test_unknown_outcome_path(settings) -> None:
    driver, _, _ = tool_driver(settings, TRANSFER, profile="blackout", handoff_enabled=True)
    async with driver:
        await driver.say("send two tickets to friend@example.com")
        await driver.say("yes")
    assert {"tool_unknown", "handoff_triggered"} <= types(driver)


async def test_reconciled_path(settings) -> None:
    """Commit-then-timeout with reconciliation up: the ledger finds the commit."""
    driver, _, _ = tool_driver(settings, [*TRANSFER, say("Done. "), end()], profile="ambiguous")
    async with driver:
        await driver.say("send two tickets to friend@example.com")
        await driver.say("yes")
    assert "tool_reconciled" in types(driver)


async def test_failed_tool_and_slow_tool_paths(settings) -> None:
    events = [tool_call("lookup_order", {"order_id": "TK-88213"}), end("tool_use"),
              say("Found it. "), end()]
    driver, _, _ = tool_driver(settings, events, profile="slow", tool_filler_ms=200)
    async with driver:
        await driver.say("where are my tickets")
    assert "filler_spoken" in types(driver)

    events = [tool_call("lookup_order", {"order_id": "TK-88213"}), end("tool_use"),
              say("I need your details first. "), end()]
    driver, _, _ = tool_driver(settings, events, verified=False)
    async with driver:
        await driver.say("where are my tickets")
    assert "tool_failed" in types(driver)


async def test_handoff_signal_path(settings) -> None:
    lookup = [tool_call("lookup_order", {"order_id": "TK-88213"}), end("tool_use")]
    driver, _, _ = tool_driver(settings, lookup * 4, handoff_enabled=True, max_tool_iterations=3)
    async with driver:
        await driver.say("where are my tickets")
    assert {"handoff_signal_raised", "handoff_triggered"} <= types(driver)


async def test_barge_in_path(make_driver) -> None:
    events = [{"text": "Refunds are available within thirty days of purchase. ", "delay_ms": 10},
              {"text": "Transfers are free of charge. ", "delay_ms": 400},
              {"end": {"stop_reason": "end_turn"}}]
    async with make_driver(events) as driver:
        await driver.say("can I get a refund?", interrupt_at_ms=120, played_chars=25)
    assert {"barge_in", "context_trimmed"} <= types(driver)


async def test_budget_and_error_paths(make_driver) -> None:
    slow = [{"text": "Too slow. ", "delay_ms": 300}, {"end": {"stop_reason": "end_turn"}}]
    async with make_driver(slow, turn_budget_ms=60) as driver:
        await driver.say("hello")
    assert "budget_exceeded" in types(driver)

    failing = [{"end": {"stop_reason": "error", "error": "503"}}] * 2
    async with make_driver(failing) as driver:
        await driver.say("hello")
    assert "error" in types(driver)


async def test_silence_path(make_driver) -> None:
    async with make_driver([]) as driver:
        await driver.send(SilenceTimeout())
    assert "silence_reprompt" in types(driver)


def test_unknown_types_raise() -> None:
    with pytest.raises(ValueError, match="unknown event_type"):
        validate("stt_partial")


def test_every_type_in_the_vocabulary_is_emitted_on_some_path() -> None:
    """Runs last in this file (pytest keeps definition order). The simulator's
    answer key is emitted by eval/caller_sim, not the harness."""
    missing = EVENT_TYPES - seen - {"simulation_label"}
    assert not missing, f"in the vocabulary but never emitted: {sorted(missing)}"
