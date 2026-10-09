"""docs/12 §7.2 — interruption.

The scenario the whole idempotency design exists for: the caller cuts in while
an action is being read back, then asks for the same thing again. Exactly one
side effect, ever.
"""

from __future__ import annotations

import asyncio

from eval.fault_injection.helpers import TRANSFER_ARGS, end, say, tool_call, tool_driver

TRANSFER = [tool_call("transfer_tickets", TRANSFER_ARGS), end("tool_use")]


async def test_interrupt_mid_readback_commits_nothing(settings) -> None:
    driver, backend, ledger = tool_driver(settings, TRANSFER)
    async with driver:
        readback = await driver.say("send two tickets to friend@example.com")
        assert "readback_spoken" in readback.events
        assert readback.spoken[0].startswith("Transferring two tickets")

        # The caller talks over the read-back while it is still playing.
        cut = await driver.barge_in(played_chars=15)

    assert "readback_rejected" in cut.events
    assert backend.count_effects() == 0
    assert [e.status for e in ledger.rows.values()] == ["cancelled"]
    assert driver.call.pending_readback is None


async def test_interrupt_then_same_request_commits_exactly_once(settings) -> None:
    """THE demo moment."""
    events = TRANSFER + TRANSFER + [say("Done — they're on their way to your friend. "), end()]
    driver, backend, ledger = tool_driver(settings, events)
    async with driver:
        await driver.say("send two tickets to friend@example.com")
        await driver.barge_in(played_chars=15)          # no — cancelled
        await driver.say("sorry, yes, send those two tickets to friend@example.com")
        confirmed = await driver.say("yes")

    assert "readback_confirmed" in confirmed.events
    assert backend.count_effects("transfer_tickets") == 1
    committed = ledger.committed("transfer_tickets")
    assert len(committed) == 1
    assert committed[0].attempt == 2, "the cancelled claim was re-used, not duplicated"
    driver.assert_invariant()


async def test_interrupt_mid_tool_call_never_cancels_a_dispatched_call(settings) -> None:
    """The caller cuts in while a lookup is in flight. The call is NOT
    cancelled — it resolves in the background — and its result is dropped
    from context so no tool_result can be orphaned."""
    events = [tool_call("lookup_order", {"order_id": "TK-88213"}), end("tool_use"),
              say("Which order did you mean? "), end()]
    driver, _backend, ledger = tool_driver(settings, events, profile="slow")
    async with driver:
        result = await driver.say("where are my tickets", interrupt_at_ms=150, played_chars=0)
        assert result.stop_reason == "interrupted"
        driver.assert_invariant()
        assert not any(
            block.get("type") in ("tool_use", "tool_result")
            for m in driver.call.messages for block in m["content"]
        )

        await asyncio.sleep(1.4)  # the slow profile answers in 0.8-1.2s
        statuses = [e.status for e in ledger.rows.values()]
        assert statuses == ["committed"], "dispatched work resolves; it is never cancelled"

        after = await driver.say("the other one")
        assert after.stop_reason == "answered"
        driver.assert_invariant()


async def test_barge_in_during_playout_rewrites_history(settings) -> None:
    """The COMMON live case: generation finished, TTS had not."""
    events = [say("Refunds take five to seven business days. "),
              say("They go back to your original card. "), end()]
    driver, _, _ = tool_driver(settings, events)
    async with driver:
        answered = await driver.say("how long do refunds take?")
        assert answered.stop_reason == "answered"
        full = answered.turn.approved_text

        cut = await driver.barge_in(played_chars=30)

    turn = driver.call.turns[-1]
    assert "context_trimmed" in cut.events
    assert turn.stop_reason == "interrupted"
    assert len(turn.spoken_text) < len(full)
    assert driver.call.messages[-1]["content"][0]["text"] == turn.spoken_text
    # The stored row is re-written: it said "answered" with the full text.
    assert driver.store.turns[-1].stop_reason == "interrupted"
    driver.assert_invariant()


async def test_barge_in_after_playout_finished_changes_nothing(settings) -> None:
    events = [say("Refunds take five to seven business days. "), end()]
    driver, _, _ = tool_driver(settings, events)
    async with driver:
        answered = await driver.say("how long?")
        cut = await driver.barge_in(played_chars=len(answered.turn.approved_text) + 10)

    assert "context_trimmed" not in cut.events
    assert driver.call.turns[-1].stop_reason == "answered"


async def test_double_interrupt_keeps_history_valid(settings) -> None:
    events = [say("Refunds are available within thirty days. ", delay_ms=5),
              say("After that the order is final. ", delay_ms=400), end(),
              say("Which order do you mean? ", delay_ms=5),
              say("I can look it up. ", delay_ms=400), end(),
              say("Got it. "), end()]
    driver, _, _ = tool_driver(settings, events)
    async with driver:
        await driver.say("refund?", interrupt_at_ms=120, played_chars=20)
        await driver.say("no wait", interrupt_at_ms=120, played_chars=0)
        last = await driver.say("the TK-88213 one")
        assert last.stop_reason == "answered"
        driver.assert_invariant()


async def test_report_line(settings) -> None:
    """The summary the fault-injection report prints (docs/12 §7.5)."""
    events = TRANSFER * 3 + [say("Done. "), end()]
    driver, backend, ledger = tool_driver(settings, events, profile="ambiguous")
    async with driver:
        await driver.say("send two tickets to friend@example.com")
        await driver.barge_in(played_chars=10)
        await driver.say("send two tickets to friend@example.com")
        await driver.say("yes")
        await driver.say("send them again to friend@example.com")

    side_effecting = [e for e in ledger.rows.values() if e.side_effecting]
    print(f"\n  duplicate side effects observed: "
          f"{backend.count_effects() - len({e.idempotency_key for e in side_effecting if e.status == 'committed'})}"
          f" / {len(side_effecting)} side-effecting executions (profile=ambiguous)")
    assert backend.count_effects("transfer_tickets") == 1
