"""W2-12 — handoff. Recall is the priority: a missed handoff is a confident
wrong answer to a customer."""

from __future__ import annotations

import pytest

from eval.fault_injection.helpers import end, say, tool_call, tool_driver
from harness.retriever import FakeRetriever

ANSWER = [say("Let me help with that. "), end()]


@pytest.mark.parametrize(
    "utterance,rule",
    [
        ("can I talk to a real person", "explicit_human_request"),
        ("just get me a manager", "explicit_human_request"),
        ("That's ridiculous. Let me talk to a manager.", "explicit_human_request"),
        ("can I speak with your supervisor", "explicit_human_request"),
        ("I want a human", "explicit_human_request"),
        ("put me through to someone", "explicit_human_request"),
        ("I'm going to dispute the charge with my bank", "payment_dispute"),
        ("this looks like fraud", "payment_dispute"),
    ],
)
async def test_hard_utterance_rules_cost_no_model_call(settings, utterance, rule) -> None:
    driver, _, _ = tool_driver(settings, ANSWER, handoff_enabled=True)
    from harness.events import EndCall

    async with driver:
        result = await driver.say(utterance)
        # The turn completes first; the call ends right after it.
        after = [await driver._out.get(), await driver._out.get()]

    assert driver.harness.llm.call_count == 0, "the cheapest exit wins"
    assert result.stop_reason == "handoff"
    assert driver.call.outcome == "handoff"
    assert driver.call.handoff_reason == rule
    assert isinstance(after[-1], EndCall)


async def test_summary_says_what_was_and_was_not_promised(settings) -> None:
    driver, _, _ = tool_driver(settings, ANSWER, handoff_enabled=True)
    async with driver:
        await driver.say("let me speak to someone")

    handoff = driver.call.handoff
    assert "has NOT promised any refund" in handoff["summary"]["money_statement"]
    assert "has NOT promised" in handoff["summary_text"]
    assert handoff["summary"]["asked"] == ["let me speak to someone"]


async def test_weak_retrieval_alone_does_not_hand_off(settings) -> None:
    """One mediocre signal is normal. Escalating on it would push containment
    toward zero."""
    # 0.65: under the 0.688 "weak" line, above the 0.62 "very weak" one.
    driver, _, _ = tool_driver(settings, ANSWER, handoff_enabled=True,
                               retriever=FakeRetriever.with_chunks(["x"], top_vec_score=0.65))
    async with driver:
        result = await driver.say("what's in the VIP package")

    assert "retrieval_insufficient" in result.turn.handoff_signals
    assert result.stop_reason == "answered"


async def test_weak_retrieval_plus_a_checker_failure_hands_off(settings) -> None:
    events = [say("The VIP package costs $200. "), end()]
    driver, _, _ = tool_driver(
        settings, events, handoff_enabled=True, checker_enabled=True,
        retriever=FakeRetriever.with_chunks(["Parking is 25 dollars."], top_vec_score=0.50),
    )
    async with driver:
        result = await driver.say("how much is the VIP package")

    assert result.stop_reason == "handoff"
    assert driver.call.handoff["rule_type"] == "soft"
    assert driver.call.handoff["score"] >= 0.6
    assert "two hundred" not in " ".join(result.spoken)


async def test_old_signals_decay(settings) -> None:
    """A weak retrieval two turns ago counts a quarter of its weight."""
    weak = FakeRetriever.with_chunks(["x"], top_vec_score=0.65)
    driver, _, _ = tool_driver(settings, ANSWER * 3, handoff_enabled=True, retriever=weak)
    async with driver:
        await driver.say("one")
        driver.harness.retriever = FakeRetriever.with_chunks(["x"], top_vec_score=0.9)
        await driver.say("two")
        await driver.say("three")

    decision = driver.harness.policy.score(driver.call)
    assert not decision.should_handoff
    assert decision.score == pytest.approx(0.45 * 0.25, abs=0.01)


async def test_two_checker_failures_are_a_hard_handoff(settings) -> None:
    bad = [say("Parking is $99. "), end()]
    driver, _, _ = tool_driver(
        settings, bad * 2, handoff_enabled=True, checker_enabled=True,
        retriever=FakeRetriever.with_chunks(["Parking is 25 dollars."], top_vec_score=0.9),
    )
    async with driver:
        first = await driver.say("parking?")
        assert first.stop_reason == "answered"
        second = await driver.say("parking at the other venue?")

    assert second.stop_reason == "handoff"
    assert driver.call.handoff_reason == "checker_failed_twice"


async def test_failed_verification_twice_is_a_hard_handoff(settings) -> None:
    wrong = [tool_call("verify_caller", {"order_id": "TK-88213", "email": "wrong@example.com"}),
             end("tool_use"), say("That didn't match. "), end()]
    driver, _, _ = tool_driver(settings, wrong * 2, verified=False, handoff_enabled=True)
    async with driver:
        await driver.say("TK-88213, wrong@example.com")
        result = await driver.say("TK-88213, wrong@example.com again")

    assert result.stop_reason == "handoff"
    assert driver.call.handoff_reason == "verification_failed"
    assert not driver.call.caller_verified


async def test_rephrase_signal_uses_similarity(settings) -> None:
    driver, _, _ = tool_driver(settings, ANSWER * 2, handoff_enabled=True)
    driver.harness.similarity = lambda a, b: 0.95
    async with driver:
        await driver.say("how do I get my tickets")
        result = await driver.say("where can I find my tickets")

    assert "rephrase_repeat" in result.turn.handoff_signals
    assert driver.call.rephrase_count == 1


async def test_retrieval_failure_is_a_hard_handoff(settings) -> None:
    class Broken:
        async def fetch(self, utterance, *, previous=None):
            raise ConnectionError("database down")

    driver, _, _ = tool_driver(settings, ANSWER, handoff_enabled=True, retriever=Broken())
    async with driver:
        result = await driver.say("what is the bag policy")

    assert result.stop_reason == "handoff"
    assert driver.call.handoff_reason == "retrieval_unavailable"
    assert driver.harness.llm.call_count == 0, "never answer without grounding"


async def test_transactional_turns_do_not_accumulate_retrieval_signals(settings) -> None:
    """A transfer conversation retrieves weakly on every turn — "yes" matches
    no policy — and must not drift into a handoff because of it."""
    from eval.fault_injection.helpers import TRANSFER_ARGS

    transfer = [tool_call("transfer_tickets", TRANSFER_ARGS), end("tool_use")]
    weak = FakeRetriever.with_chunks(["Bag policy."], top_vec_score=0.40)
    driver, backend, _ = tool_driver(
        settings, [*transfer, say("Done. "), end()], handoff_enabled=True, retriever=weak
    )
    async with driver:
        await driver.say("send two tickets to friend@example.com")
        result = await driver.say("yes")

    assert result.stop_reason == "answered"
    assert driver.call.outcome != "handoff"
    assert backend.count_effects("transfer_tickets") == 1
    assert not any("retrieval_insufficient" in t.handoff_signals for t in driver.call.turns)


async def test_repeating_after_cutting_the_agent_off_is_not_a_rephrase(settings) -> None:
    """Asking again after an interruption is re-issuing, not a sign of failure."""
    events = [say("Refunds take five days. ", delay_ms=5), say("More detail. ", delay_ms=400), end(),
              say("Refunds take five days. "), end()]
    driver, _, _ = tool_driver(settings, events, handoff_enabled=True)
    driver.harness.similarity = lambda a, b: 0.99
    async with driver:
        await driver.say("how long do refunds take", interrupt_at_ms=120, played_chars=10)
        result = await driver.say("how long do refunds take")

    assert "rephrase_repeat" not in result.turn.handoff_signals


async def test_summary_carries_policy_from_earlier_turns(settings) -> None:
    driver, _, _ = tool_driver(
        settings, ANSWER, handoff_enabled=True,
        retriever=FakeRetriever.with_chunks(["Refunds..."], heading="Refunds > Eligibility"),
    )
    async with driver:
        await driver.say("can I get a refund")
        await driver.say("let me talk to a manager")

    assert driver.call.handoff["summary"]["policy_refs"] == ["Refunds > Eligibility"]


@pytest.mark.parametrize("utterance", [
    "I'm disputing this charge with my card company.",
    "My card was charged and I didn't authorize it.",
])
async def test_payment_dispute_phrasings_found_by_the_week3_eval(settings, utterance) -> None:
    """Both slipped past the week-2 rules (results/2026-10-13-handoff-calibration/)."""
    driver, _, _ = tool_driver(settings, ANSWER, handoff_enabled=True)
    async with driver:
        await driver.say(utterance)
    assert driver.call.handoff_reason == "payment_dispute"


async def test_very_weak_retrieval_hands_off_alone(settings) -> None:
    very_weak = FakeRetriever.with_chunks(["Bag policy."], top_vec_score=0.40)
    driver, _, _ = tool_driver(settings, ANSWER, handoff_enabled=True, retriever=very_weak)
    async with driver:
        result = await driver.say("what's in the VIP package")
    assert "retrieval_very_weak" in result.turn.handoff_signals
    assert result.stop_reason == "handoff"


async def test_merely_weak_retrieval_still_only_contributes(settings) -> None:
    weak = FakeRetriever.with_chunks(["Refunds..."], top_vec_score=0.65)
    driver, _, _ = tool_driver(settings, ANSWER, handoff_enabled=True, retriever=weak)
    async with driver:
        result = await driver.say("can I get my money back")
    assert "retrieval_insufficient" in result.turn.handoff_signals
    assert "retrieval_very_weak" not in result.turn.handoff_signals
    assert result.stop_reason == "answered"


async def test_a_greeting_is_not_handed_off_on_weak_retrieval(settings) -> None:
    """Found live: "Hi there." scored 0.478 against the knowledge base, tripped
    retrieval_very_weak, and the caller was passed to a human after saying hi —
    preceded by "Let me check that for you."."""
    slow_hello = [{"text": "Hi, how can I help you today? ", "delay_ms": 500},
                  {"end": {"stop_reason": "end_turn"}}]
    driver, _, _ = tool_driver(settings, slow_hello, handoff_enabled=True, ack_after_ms=50,
                               retriever=FakeRetriever.with_chunks(["x"], top_vec_score=0.45))
    async with driver:
        result = await driver.say("Hi there.")

    assert result.stop_reason == "answered"
    assert result.turn.social
    assert "retrieval_very_weak" not in result.turn.handoff_signals
    assert result.spoken == ["Hi, how can I help you today?"], "no acknowledgment for a greeting"


async def test_a_real_question_with_the_same_score_still_hands_off(settings) -> None:
    driver, _, _ = tool_driver(settings, ANSWER, handoff_enabled=True,
                               retriever=FakeRetriever.with_chunks(["x"], top_vec_score=0.45))
    async with driver:
        result = await driver.say("Is there a group discount for fourteen people?")
    assert result.stop_reason == "handoff"


@pytest.mark.parametrize("text, social", [
    ("Hi there.", True), ("hello", True), ("Thanks!", True), ("Hey, how are you doing today?", True),
    ("Tell me a joke", False), ("What's the capital of France?", False),
    ("hi, where are my tickets", False), ("yes", False),
])
def test_social_utterances_are_a_closed_set(text: str, social: bool) -> None:
    from harness.policy import is_social

    assert is_social(text) is social
