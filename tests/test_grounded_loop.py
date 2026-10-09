"""W2-5 / W2-6 / W2-8 — the checker, normalization and acknowledgments,
running inside the real turn loop with the checker ON, as in production.
"""

from __future__ import annotations

from eval.text_driver import TextDriver
from harness.retriever import FakeRetriever
from providers.llm.replay import ReplayProvider

REFUND_CHUNK = (
    "Approved refunds are processed within 5 to 7 business days. The money returns "
    "to the original payment method."
)
PARKING_CHUNK = "The Harbour Pavilion garage costs 25 dollars and opens 2 hours before doors."


def driver(settings, events, *, chunks, top_vec_score=0.80, **overrides):
    local = settings.model_copy(update={"checker_enabled": True, **overrides})
    return TextDriver(
        llm=ReplayProvider(events=events),
        settings=local,
        retriever=FakeRetriever.with_chunks(chunks, top_vec_score=top_vec_score),
    )


async def test_a_grounded_answer_is_spoken_in_spoken_form(settings) -> None:
    events = [{"text": "Parking at Harbour Pavilion is $25 [K1]. "}, {"end": {}}]
    async with driver(settings, events, chunks=[PARKING_CHUNK]) as d:
        result = await d.say("is there parking?")

    assert result.spoken == ["Parking at Harbour Pavilion is twenty-five dollars."]
    turn = result.turn
    assert [v.verdict for v in turn.checker_verdicts] == ["pass"]
    assert turn.cited_chunk_ids == ["c0"], "the [K1] tag is recorded, never spoken"
    assert "checker" in turn.stages_ms or turn.checker_verdicts[0].checker_ms == 0


async def test_an_invented_price_is_never_spoken(settings) -> None:
    events = [
        {"text": "Parking is $25 at the garage. "},
        {"text": "Valet parking is $60 if you prefer. "},
        {"text": "You can also prepay online. "},
        {"end": {}},
    ]
    async with driver(settings, events, chunks=[PARKING_CHUNK]) as d:
        result = await d.say("parking?")

    turn = result.turn
    # Segment 1 is grounded and spoken; segment 2 invents $60 and is replaced.
    assert result.spoken[0] == "Parking is twenty-five dollars at the garage."
    assert result.spoken[1] == "Let me double-check that detail before I give you a number."
    # Generation stopped: the rest of an answer just caught inventing a number
    # is not worth trusting.
    assert len(result.spoken) == 2
    assert "sixty" not in turn.approved_text
    assert "$60" in turn.generated_text, "generated_text keeps the evidence"
    assert "checker_failure" in turn.handoff_signals
    assert [v.verdict for v in turn.checker_verdicts] == ["pass", "fail_ungrounded_entity"]
    assert [v.action_taken for v in turn.checker_verdicts] == ["released", "replaced"]


async def test_unsupported_claim_with_weak_retrieval_is_replaced(settings) -> None:
    events = [{"text": "You can transfer tickets any time before the event. "}, {"end": {}}]
    async with driver(settings, events, chunks=[PARKING_CHUNK], top_vec_score=0.50) as d:
        result = await d.say("can I transfer my tickets?")

    assert result.spoken == [
        "I want to be sure I get that right — let me bring in a colleague."
    ]
    assert result.turn.checker_verdicts[0].verdict == "fail_unsupported_claim"


async def test_the_callers_own_order_number_is_grounded(settings) -> None:
    events = [{"text": "I can see order TK-88213 now. "}, {"end": {}}]
    async with driver(settings, events, chunks=[REFUND_CHUNK]) as d:
        result = await d.say("my order is TK-88213")

    assert result.spoken == ["I can see order T K, eight eight two one three now."]


async def test_acknowledgment_covers_a_slow_first_token(settings) -> None:
    events = [
        {"text": "Refunds take 5 to 7 business days. ", "delay_ms": 250},
        {"end": {}},
    ]
    async with driver(settings, events, chunks=[REFUND_CHUNK], ack_after_ms=60) as d:
        result = await d.say("how long do refunds take?")

    turn = result.turn
    assert result.spoken[0] == "Let me check that for you."
    assert turn.ack_bypass_used
    # The acknowledgment is audio but not content: it never enters history.
    assert "check that" not in turn.approved_text
    # Two first-audio numbers. first_content is the honest headline.
    assert turn.stages_ms["first_speak"] < turn.stages_ms["first_content"]


async def test_no_acknowledgment_when_the_answer_is_fast(settings) -> None:
    events = [{"text": "Refunds take 5 to 7 business days. "}, {"end": {}}]
    async with driver(settings, events, chunks=[REFUND_CHUNK], ack_after_ms=200) as d:
        result = await d.say("how long?")

    assert not result.turn.ack_bypass_used
    assert result.spoken == ["Refunds take five to seven business days."]


async def test_acknowledgments_do_not_repeat_back_to_back(settings) -> None:
    slow = [{"text": "Refunds take 5 to 7 business days. ", "delay_ms": 150}, {"end": {}}]
    async with driver(settings, slow * 3, chunks=[REFUND_CHUNK], ack_after_ms=30) as d:
        acks = [(await d.say(f"q{i}")).spoken[0] for i in range(3)]

    assert len(set(acks)) == 3


async def test_spoken_text_after_barge_in_uses_the_spoken_form(settings) -> None:
    """played_chars counts characters of what TTS RECEIVED, so approved_text
    must hold the spoken form for the offset to mean anything."""
    events = [
        {"text": "Parking is $25 at the garage. ", "delay_ms": 5},
        {"text": "It opens 2 hours before doors. ", "delay_ms": 400},
        {"end": {}},
    ]
    async with driver(settings, events, chunks=[PARKING_CHUNK]) as d:
        result = await d.say("parking?", interrupt_at_ms=150, played_chars=40)

    turn = result.turn
    assert turn.stop_reason == "interrupted"
    assert turn.approved_text.startswith("Parking is twenty-five dollars")
    assert turn.approved_text.startswith(turn.spoken_text)
