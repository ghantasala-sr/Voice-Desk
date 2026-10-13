"""docs/12 §7.3 — speech recognition faults.

The agent cannot know it misheard. These tests are about RECOVERY: the
read-back must expose a mishearing before anything happens, and a key entity
heard with low confidence must be visible to the handoff policy.
"""

from __future__ import annotations

from eval.fault_injection.helpers import end, say, tool_call, tool_driver

MISHEARD = {"order_id": "TK-88213", "quantity": 40, "recipient_email": "friend@example.com"}


async def test_misheard_quantity_is_exposed_by_the_readback(settings) -> None:
    """The caller said "fourteen"; ASR heard "forty". The schema caps transfers
    at 10, so the misheard value never even reaches a read-back — the model
    gets the validation error and asks again."""
    events = [tool_call("transfer_tickets", MISHEARD), end("tool_use"),
              say("How many tickets did you want to send? "), end()]
    driver, backend, _ = tool_driver(settings, events)
    async with driver:
        result = await driver.say("send forty tickets to friend@example.com")

    assert "tool_failed" in result.events
    assert result.spoken == ["How many tickets did you want to send?"]
    assert backend.count_effects() == 0


async def test_misheard_but_valid_quantity_is_caught_by_the_caller(settings) -> None:
    """Heard "three" for "two": valid, so it IS read back — and the caller's
    correction is a deny, never an execution."""
    wrong = {"order_id": "TK-88213", "quantity": 3, "recipient_email": "friend@example.com"}
    right = {**wrong, "quantity": 2}
    events = [tool_call("transfer_tickets", wrong), end("tool_use"),
              tool_call("transfer_tickets", right, call_id="tu_2"), end("tool_use"),
              say("Done. "), end()]
    driver, backend, _ = tool_driver(settings, events)
    async with driver:
        first = await driver.say("send two tickets to friend@example.com")
        assert first.spoken[0].startswith("Transferring three tickets")
        correction = await driver.say("no, two tickets")
        assert "readback_rejected" in correction.events
        assert correction.spoken[0].startswith("Transferring two tickets")
        await driver.say("yes")

    assert backend.count_effects("transfer_tickets") == 1
    assert backend.effects[0].args["quantity"] == 2


async def test_low_confidence_on_a_key_entity_raises_a_signal(settings) -> None:
    """A mean confidence of 0.92 hides one order-number token at 0.41."""
    driver, _, _ = tool_driver(settings, [say("Could you repeat the order number? "), end()],
                               handoff_enabled=True)
    async with driver:
        result = await driver.say(
            "my order is TK-88213", confidence=0.92,
            word_confidences=[("my", 0.99), ("order", 0.98), ("is", 0.99), ("TK-88213", 0.41)],
        )

    assert "entity_confidence_low" in result.turn.handoff_signals
    assert result.turn.entity_confidences == {"TK-88213": 0.41}
    assert result.stop_reason == "answered", "one soft signal alone does not hand off"


async def test_low_confidence_on_a_filler_word_is_ignored(settings) -> None:
    driver, _, _ = tool_driver(settings, [say("Sure. "), end()], handoff_enabled=True)
    async with driver:
        result = await driver.say("um where are my tickets",
                                  word_confidences=[("um", 0.2), ("tickets", 0.95)])

    assert "entity_confidence_low" not in result.turn.handoff_signals
