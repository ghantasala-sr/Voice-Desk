"""docs/12 §7.4 — retrieval and verification faults.

The invariant: the system never answers without grounding. An empty or
failed retrieval becomes a handoff, not an improvisation.
"""

from __future__ import annotations

from eval.fault_injection.helpers import end, say, tool_driver
from harness.retriever import FakeRetriever, empty_result


async def test_empty_retrieval_hands_off(settings) -> None:
    """Zero chunks scores 0.0 — far below the very-weak line."""
    driver, _, _ = tool_driver(settings, [say("Let me check. "), end()],
                               handoff_enabled=True, retriever=FakeRetriever(empty_result()))
    async with driver:
        result = await driver.say("what is the policy on drones")

    assert result.stop_reason == "handoff"
    assert "retrieval_very_weak" in result.turn.handoff_signals


async def test_empty_retrieval_never_lets_an_invented_fact_through(settings) -> None:
    """Even with handoff off, the checker has nothing to ground against."""
    driver, _, _ = tool_driver(settings, [say("Drones are allowed for $20 extra. "), end()],
                               checker_enabled=True, retriever=FakeRetriever(empty_result()))
    async with driver:
        result = await driver.say("can I fly a drone at the show")

    assert "twenty" not in " ".join(result.spoken)
    assert result.turn.checker_verdicts[0].verdict == "fail_ungrounded_entity"


async def test_database_down_mid_call_is_a_hard_handoff(settings) -> None:
    class Down:
        async def fetch(self, utterance, *, previous=None):
            raise OSError("connection refused")

    driver, _, _ = tool_driver(settings, [say("x. "), end()], handoff_enabled=True, retriever=Down())
    async with driver:
        result = await driver.say("what's the bag policy")

    assert driver.call.handoff_reason == "retrieval_unavailable"
    assert driver.harness.llm.call_count == 0
    assert result.spoken == [
        "I'm going to connect you with a colleague who can help. I'll pass along everything we've covered."
    ]
