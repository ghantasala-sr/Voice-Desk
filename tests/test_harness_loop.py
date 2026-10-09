"""W1-5 / W1-8 — the turn loop end to end, with no network and no audio.

Every test here runs in milliseconds because the harness cannot import
pipecat. That is the architectural contract paying for itself.
"""

from __future__ import annotations

from harness.context import assert_invariant

TWO_SENTENCES = [
    {"text": "Refunds take five to seven business days. "},
    {"text": "Transfers are free of charge. "},
    {"end": {"stop_reason": "end_turn", "input_tokens": 120, "output_tokens": 18}},
]


# ---------------------------------------------------------------------------
# The happy path
# ---------------------------------------------------------------------------
async def test_turn_answers_and_records_state(make_driver) -> None:
    async with make_driver(TWO_SENTENCES) as driver:
        result = await driver.say("how long do refunds take?")

    assert result.stop_reason == "answered"
    assert result.spoken == [
        "Refunds take five to seven business days.",
        "Transfers are free of charge.",
    ]

    turn = result.turn
    assert turn.seq == 0
    assert turn.trigger == "transcript"
    assert turn.llm_iterations == 1
    # generated ⊇ approved ⊇ spoken, and with no interruption the last two match.
    assert turn.approved_text in turn.generated_text
    assert turn.spoken_text == turn.approved_text


async def test_history_alternates_and_holds_the_invariant(make_driver) -> None:
    async with make_driver(TWO_SENTENCES + TWO_SENTENCES) as driver:
        await driver.say("how long do refunds take?")
        await driver.say("and transfers?")
        driver.assert_invariant()
        assert [m["role"] for m in driver.call.messages] == [
            "user", "assistant", "user", "assistant",
        ]


async def test_latency_stages_are_recorded(make_driver) -> None:
    async with make_driver(TWO_SENTENCES) as driver:
        result = await driver.say("hello", endpoint_ms=400)

    stages = result.turn.stages_ms
    # [A] arrives on the event — the harness cannot measure VAD itself.
    assert stages["endpoint"] == 400
    # [D] and [E] are measured here. [E] is the cost the checker's existence
    # imposes, so it is reported on its own rather than folded into [D].
    assert "llm_ttft" in stages
    assert "sentence_buffer" in stages
    assert "first_audio" in stages


async def test_blank_transcript_does_not_start_a_turn(make_driver) -> None:
    """VAD occasionally finalizes on a cough. That must not cost an LLM call."""
    from harness.events import TranscriptIn

    async with make_driver(TWO_SENTENCES) as driver:
        driver._stream.put(TranscriptIn(text="   ", provider="text"))
        result = await driver.say("hello")

    assert result.stop_reason == "answered"
    assert len(driver.store.turns) == 1
    assert driver.harness.llm.call_count == 1


# ---------------------------------------------------------------------------
# Barge-in
# ---------------------------------------------------------------------------
INTERRUPTIBLE = [
    {"text": "Refunds are available within thirty days of purchase. ", "delay_ms": 10},
    {"text": "Transfers are free of charge. ", "delay_ms": 400},
    {"end": {"stop_reason": "end_turn"}},
]


async def test_barge_in_stops_the_turn_and_trims_context(make_driver) -> None:
    async with make_driver(INTERRUPTIBLE) as driver:
        result = await driver.say(
            "can I get a refund?", interrupt_at_ms=120, played_chars=25
        )

        assert result.stop_reason == "interrupted"
        assert "barge_in" in result.events
        assert "context_trimmed" in result.events

        turn = result.turn
        # spoken ⊂ approved: the caller heard less than we handed to TTS.
        assert turn.spoken_text
        assert turn.spoken_text in turn.approved_text
        assert len(turn.spoken_text) < len(turn.approved_text)

        # History reflects what was HEARD, not what was generated.
        assert driver.call.messages[-1]["content"][0]["text"] == turn.spoken_text
        driver.assert_invariant()


async def test_cancel_speech_is_issued_before_bookkeeping(make_driver) -> None:
    """The caller's experience is decided by when audio stops, so CancelSpeech
    must precede every trim event (docs/10 §3)."""
    from harness.events import CancelSpeech, Emit

    async with make_driver(INTERRUPTIBLE) as driver:
        result = await driver.say("hi", interrupt_at_ms=120, played_chars=25)

    kinds = [type(c) for c in result.commands]
    cancel_at = kinds.index(CancelSpeech)
    trim_at = next(
        i for i, c in enumerate(result.commands)
        if isinstance(c, Emit) and c.event_type == "context_trimmed"
    )
    assert cancel_at < trim_at


async def test_interruption_before_any_audio_removes_the_message(make_driver) -> None:
    slow = [
        {"text": "Refunds are available. ", "delay_ms": 500},
        {"end": {"stop_reason": "end_turn"}},
    ]
    async with make_driver(slow) as driver:
        result = await driver.say("hi", interrupt_at_ms=30, played_chars=0)

        assert result.stop_reason == "interrupted"
        assert result.turn.spoken_text == ""
        # No assistant message survives — an empty one is an API error.
        assert all(m["role"] == "user" for m in driver.call.messages)
        driver.assert_invariant()


async def test_conversation_continues_cleanly_after_an_interruption(make_driver) -> None:
    events = [
        {"text": "Refunds are available. ", "delay_ms": 400},
        {"end": {"stop_reason": "end_turn"}},
        {"text": "That order ships to the mobile app. "},
        {"end": {"stop_reason": "end_turn"}},
    ]
    async with make_driver(events) as driver:
        first = await driver.say("refund?", interrupt_at_ms=30, played_chars=0)
        second = await driver.say("no, the other order")

        assert first.stop_reason == "interrupted"
        assert second.stop_reason == "answered"
        driver.assert_invariant()


# ---------------------------------------------------------------------------
# Budget and failure paths
# ---------------------------------------------------------------------------
async def test_budget_exceeded_speaks_a_fixed_fallback(make_driver) -> None:
    slow = [
        {"text": "This answer took far too long to start. ", "delay_ms": 300},
        {"end": {"stop_reason": "end_turn"}},
    ]
    async with make_driver(slow, turn_budget_ms=60) as driver:
        result = await driver.say("hello")

    assert result.stop_reason == "budget_exceeded"
    assert "budget_exceeded" in result.events
    assert result.spoken[0] == "Sorry — give me just a second."
    # The fallback is spoken but is NOT part of approved_text, so it never
    # enters conversation history as something the agent "said".
    assert "Sorry" not in result.turn.approved_text


async def test_provider_error_retries_once_then_falls_back(make_driver) -> None:
    events = [
        {"end": {"stop_reason": "error", "error": "upstream 503"}},
        {"end": {"stop_reason": "error", "error": "upstream 503"}},
    ]
    async with make_driver(events) as driver:
        result = await driver.say("hello")

    assert result.turn.llm_iterations == 2, "the harness must retry exactly once"
    assert result.stop_reason == "error"
    assert "llm_failure" in result.turn.handoff_signals
    # A failure becomes a handoff signal and a FIXED line — never an
    # improvised answer.
    assert result.spoken == ["I'm having trouble on my end. Let me get a colleague for you."]


async def test_retry_succeeds_on_the_second_attempt(make_driver) -> None:
    events = [
        {"end": {"stop_reason": "error", "error": "transient"}},
        {"text": "Refunds take five business days. "},
        {"end": {"stop_reason": "end_turn"}},
    ]
    async with make_driver(events) as driver:
        result = await driver.say("how long?")

    assert result.turn.llm_iterations == 2
    assert result.stop_reason == "answered"
    assert result.spoken == ["Refunds take five business days."]


# ---------------------------------------------------------------------------
# Silence
# ---------------------------------------------------------------------------
async def test_silence_reprompts_once_then_ends_the_call(make_driver) -> None:
    """Silence produces no turn, so this drains commands rather than waiting
    for a TurnCompleted that will never arrive."""
    from harness.events import EndCall, SilenceTimeout

    async with make_driver(TWO_SENTENCES) as driver:
        first = await driver.send(SilenceTimeout())
        assert "silence_reprompt" in first.events
        assert first.spoken == ["Are you still there?"]
        assert not any(isinstance(c, EndCall) for c in first.commands)

        second = await driver.send(SilenceTimeout())
        assert any(isinstance(c, EndCall) for c in second.commands)
        assert driver.call.outcome == "dropped"
        assert driver.harness.llm.call_count == 0, "silence must not cost an LLM call"


# ---------------------------------------------------------------------------
# Invariants that must hold for every turn
# ---------------------------------------------------------------------------
async def test_every_turn_ends_with_exactly_one_stop_reason(make_driver) -> None:
    async with make_driver(TWO_SENTENCES + TWO_SENTENCES) as driver:
        await driver.say("one")
        await driver.say("two")

    assert len(driver.store.turns) == 2
    for turn in driver.store.turns:
        assert turn.stop_reason is not None
    assert_invariant(driver.call.messages)


# ---------------------------------------------------------------------------
# W2-4 — retrieval reaches the prompt
# ---------------------------------------------------------------------------
async def test_retrieved_chunks_reach_the_model_and_the_turn(settings) -> None:
    """The knowledge block is assembled per turn, after the cache breakpoint."""
    from eval.text_driver import TextDriver
    from harness.retriever import FakeRetriever
    from providers.llm.replay import ReplayProvider

    retriever = FakeRetriever.with_chunks(
        ["Refunds are processed within 5 to 7 business days."],
        heading="Refunds and exchanges > How long a refund takes",
    )
    async with TextDriver(
        llm=ReplayProvider(events=TWO_SENTENCES), settings=settings, retriever=retriever
    ) as driver:
        result = await driver.say("how long does a refund take?")

    request = driver.harness.llm.requests[0]
    assert "<knowledge>" in request.system_dynamic
    assert "[K1]" in request.system_dynamic
    assert "5 to 7 business days" in request.system_dynamic

    # The static half REFERENCES <knowledge> in its grounding rule, which is
    # fine — what must never appear there is retrieved content, because that
    # changes every turn and would break the prompt cache.
    assert "5 to 7 business days" not in request.system_static
    assert "[K1]" not in request.system_static

    turn = result.turn
    assert turn.retrieved_chunk_ids == ["c0"]
    assert turn.retrieval_top_vec_score == 0.80
    assert "retrieval" in turn.stages_ms
    assert "retrieval_completed" in result.events


async def test_follow_up_turns_pass_the_previous_utterance(settings) -> None:
    """A bare follow-up inherits its subject from the previous turn."""
    from eval.text_driver import TextDriver
    from harness.retriever import FakeRetriever
    from providers.llm.replay import ReplayProvider

    retriever = FakeRetriever.with_chunks(["Parking at Harbour Pavilion is 25 dollars."])
    async with TextDriver(
        llm=ReplayProvider(events=TWO_SENTENCES + TWO_SENTENCES),
        settings=settings,
        retriever=retriever,
    ) as driver:
        await driver.say("is there parking at Harbour Pavilion?")
        await driver.say("what about at Kestrel?")

    assert retriever.calls[0] == ("is there parking at Harbour Pavilion?", None)
    assert retriever.calls[1] == (
        "what about at Kestrel?",
        "is there parking at Harbour Pavilion?",
    )


async def test_static_prompt_is_byte_stable_across_turns(settings) -> None:
    """If this drifts, every turn is a cache miss and time-to-first-token
    silently regresses."""
    from eval.text_driver import TextDriver
    from harness.retriever import FakeRetriever
    from providers.llm.replay import ReplayProvider

    async with TextDriver(
        llm=ReplayProvider(events=TWO_SENTENCES + TWO_SENTENCES),
        settings=settings,
        retriever=FakeRetriever.with_chunks(["a"]),
    ) as driver:
        await driver.say("one")
        await driver.say("two")

    statics = {r.system_static for r in driver.harness.llm.requests}
    assert len(statics) == 1


async def test_a_provider_that_raises_does_not_crash_the_call(make_driver) -> None:
    """Same path as a reported error: retry once, fixed line, handoff signal."""
    from providers.llm.base import LLMProvider

    class Exploding(LLMProvider):
        name, model = "exploding", "x"

        async def stream(self, request):
            raise RuntimeError("provider bug")
            yield  # pragma: no cover

    driver = make_driver(TWO_SENTENCES)
    driver.harness.llm = Exploding()
    async with driver:
        result = await driver.say("hello")

    assert result.stop_reason == "error"
    assert result.turn.llm_iterations == 2
    assert result.spoken == ["I'm having trouble on my end. Let me get a colleague for you."]
