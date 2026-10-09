"""W2-9 / W2-11 — tools and read-back inside the turn loop."""

from __future__ import annotations

from eval.fault_injection.helpers import TRANSFER_ARGS, end, say, tool_call, tool_driver
from harness.retriever import FakeRetriever

TRANSFER = [tool_call("transfer_tickets", TRANSFER_ARGS), end("tool_use")]


# ---------------------------------------------------------------------------
# Read-only tool loop
# ---------------------------------------------------------------------------
async def test_verify_then_lookup_then_answer(settings) -> None:
    """The realistic worst case for one turn: three LLM calls."""
    events = [
        tool_call("verify_caller", {"order_id": "TK-88213", "email": "sam@example.com"}), end("tool_use"),
        tool_call("lookup_order", {"order_id": "TK-88213"}), end("tool_use"),
        say("You have 4 tickets in section 112. "), end(),
    ]
    # Checker ON: "4 tickets" and "section 112" are grounded by the lookup
    # result, not by any chunk.
    driver, _, _ledger = tool_driver(settings, events, verified=False, checker_enabled=True,
                                    retriever=FakeRetriever.with_chunks(["Mobile tickets."]))
    async with driver:
        result = await driver.say("I'm order TK-88213, sam@example.com — where are my tickets?")

    turn = result.turn
    assert turn.llm_iterations == 3
    assert driver.call.caller_verified
    assert driver.call.last_order["section"] == "112"
    assert result.spoken == ["You have four tickets in section one hundred twelve."]
    assert [v.verdict for v in turn.checker_verdicts] == ["pass"]
    driver.assert_invariant()
    kinds = [b["type"] for m in driver.call.messages for b in m["content"]]
    assert kinds.count("tool_use") == 2 and kinds.count("tool_result") == 2


async def test_unverified_lookup_goes_back_to_the_model(settings) -> None:
    events = [tool_call("lookup_order", {"order_id": "TK-88213"}), end("tool_use"),
              say("Could I get your order number and email first? "), end()]
    driver, backend, _ = tool_driver(settings, events, verified=False)
    async with driver:
        result = await driver.say("where are my tickets")

    assert "tool_failed" in result.events
    assert result.spoken == ["Could I get your order number and email first?"]
    result_blocks = [b for m in driver.call.messages for b in m["content"]
                     if b["type"] == "tool_result"]
    assert "unverified" in result_blocks[0]["content"]
    assert backend.count_effects() == 0


async def test_iteration_cap_stops_calling_the_model(settings) -> None:
    lookup = [tool_call("lookup_order", {"order_id": "TK-88213"}), end("tool_use")]
    driver, _, _ = tool_driver(settings, lookup * 4, handoff_enabled=True,
                               max_tool_iterations=3)
    async with driver:
        result = await driver.say("where are my tickets")

    assert driver.harness.llm.call_count == 3
    assert "iteration_cap_reached" in result.turn.handoff_signals
    assert result.stop_reason == "handoff"


async def test_slow_tool_speaks_one_filler(settings) -> None:
    events = [tool_call("lookup_order", {"order_id": "TK-88213"}), end("tool_use"),
              say("Found it. "), end()]
    driver, _, _ = tool_driver(settings, events, profile="slow", tool_filler_ms=200)
    async with driver:
        result = await driver.say("where are my tickets")

    assert "filler_spoken" in result.events
    assert result.spoken[0] == "Still pulling that up."
    assert result.events.count("filler_spoken") == 1


# ---------------------------------------------------------------------------
# Read-back
# ---------------------------------------------------------------------------
async def test_readback_then_yes_commits_once(settings) -> None:
    events = [*TRANSFER, say("Done — they're on their way. "), end()]
    driver, backend, _ = tool_driver(settings, events)
    async with driver:
        first = await driver.say("send two tickets to friend@example.com")
        assert backend.count_effects() == 0, "nothing happens before the yes"
        assert first.spoken == [
            "Transferring two tickets for your event to friend at example dot com. "
            "Should I go ahead?"
        ]
        second = await driver.say("yes please")

    assert backend.count_effects("transfer_tickets") == 1
    assert second.spoken == ["Done — they're on their way."]
    note = driver.harness.llm.requests[-1].system_dynamic
    assert "The caller confirmed transfer_tickets" in note


async def test_readback_is_rendered_from_arguments_not_from_the_model(settings) -> None:
    """If the model's own words disagree with its arguments, the caller hears
    the ARGUMENTS — the values that will actually be dispatched."""
    events = [say("Sure, I'll send all four tickets. "),
              tool_call("transfer_tickets", TRANSFER_ARGS), end("tool_use")]
    driver, _, _ = tool_driver(settings, events)
    async with driver:
        result = await driver.say("send them to friend@example.com")

    assert result.spoken[-1].startswith("Transferring two tickets")


async def test_readback_uses_the_event_name_from_a_lookup(settings) -> None:
    events = [tool_call("lookup_order", {"order_id": "TK-88213"}), end("tool_use"),
              *TRANSFER]
    driver, _, _ = tool_driver(settings, events)
    async with driver:
        result = await driver.say("send two of my Saturday tickets to friend@example.com")

    assert "the Saturday show at Harbour Pavilion" in result.spoken[-1]


async def test_no_cancels_the_action(settings) -> None:
    events = [*TRANSFER, say("No problem, I won't send them. "), end()]
    driver, backend, ledger = tool_driver(settings, events)
    async with driver:
        await driver.say("send two tickets to friend@example.com")
        result = await driver.say("no, three tickets")

    assert "readback_rejected" in result.events
    assert backend.count_effects() == 0
    assert [e.status for e in ledger.rows.values()] == ["cancelled"]
    assert "NOT confirm" in driver.harness.llm.requests[-1].system_dynamic


async def test_unclear_answer_re_asks_once_then_hands_off(settings) -> None:
    driver, backend, _ = tool_driver(settings, TRANSFER, handoff_enabled=True)
    async with driver:
        await driver.say("send two tickets to friend@example.com")
        again = await driver.say("hmm, what was that")
        assert again.spoken[0] == "Sorry, I didn't catch that. Should I go ahead — yes or no?"
        assert again.spoken[1].startswith("Transferring two tickets")
        final = await driver.say("I dunno")

    assert final.stop_reason == "handoff"
    assert driver.call.handoff_reason == "readback_unclear"
    assert backend.count_effects() == 0, "never executed on an ambiguous answer"


async def test_repeat_of_a_committed_action_is_not_repeated(settings) -> None:
    events = [*TRANSFER, say("Done. "), end(), *TRANSFER]
    driver, backend, _ = tool_driver(settings, events)
    async with driver:
        await driver.say("send two tickets to friend@example.com")
        await driver.say("yes")
        again = await driver.say("send two tickets to friend@example.com")

    assert again.spoken == ["That's already gone through — it was completed earlier in this call."]
    assert backend.count_effects("transfer_tickets") == 1


async def test_tool_outcome_unknown_is_a_hard_handoff(settings) -> None:
    """Committed upstream, timed out, reconciliation down: we cannot tell the
    caller whether it happened. Only a human should."""
    driver, backend, _ = tool_driver(settings, TRANSFER, profile="blackout",
                                     handoff_enabled=True)
    async with driver:
        await driver.say("send two tickets to friend@example.com")
        result = await driver.say("yes")

    assert result.stop_reason == "handoff"
    assert driver.call.handoff_reason == "tool_state_unknown"
    assert "may or may not" in driver.call.handoff["summary"]["suggested_next_step"]
    assert backend.count_effects() == 1  # it did happen upstream
