"""W1-8 — context hygiene after a barge-in.

The bug this prevents: trim history to what was *approved* instead of what was
*heard*, and the model believes it told the caller something it never said,
then refers back to it.
"""

from __future__ import annotations

import pytest

from harness.context import (
    ContextInvariantError,
    append_user,
    assert_invariant,
    assistant_text,
    drop_orphan_tool_results,
    estimate_spoken_text,
    trim_on_interrupt,
    user_text,
)
from harness.state import CallState, ToolExecution, TurnState

APPROVED = "Refunds are available within thirty days of purchase."


def make_call() -> CallState:
    return CallState(call_id="c1", domain_config="ticketing", source="text")


def make_turn(approved: str = APPROVED) -> TurnState:
    turn = TurnState(turn_id="t1", seq=0, trigger="transcript", user_text="hi")
    turn.approved_text = approved
    return turn


# ---------------------------------------------------------------------------
# estimate_spoken_text
# ---------------------------------------------------------------------------
def test_estimate_errs_toward_the_caller_hearing_less() -> None:
    spoken = estimate_spoken_text(APPROVED, played_chars=30, trim_ms=120)
    assert spoken in APPROVED
    assert len(spoken) < 30, "the playout trim was not applied"


def test_estimate_never_cuts_mid_word() -> None:
    # APPROVED[:20] lands inside "available", so the partial word is dropped.
    spoken = estimate_spoken_text(APPROVED, played_chars=20, trim_ms=0)
    assert APPROVED.startswith(spoken)
    assert spoken == "Refunds are"


def test_unknown_offset_assumes_nothing_was_heard() -> None:
    """If the transport cannot tell us, assume the caller heard nothing.

    The agent may repeat itself, which is awkward. The alternative — assuming
    the caller heard something they did not — is the failure mode.
    """
    assert estimate_spoken_text(APPROVED, played_chars=None, trim_ms=120) == ""


def test_full_playback_is_the_whole_text() -> None:
    assert estimate_spoken_text(APPROVED, played_chars=len(APPROVED) * 2, trim_ms=0) == APPROVED


# ---------------------------------------------------------------------------
# trim_on_interrupt
# ---------------------------------------------------------------------------
def test_assistant_message_is_rewritten_to_what_was_heard() -> None:
    call = make_call()
    turn = make_turn()
    call.messages = [user_text("can I get a refund?"), assistant_text(APPROVED)]

    trim_on_interrupt(call, turn, played_chars=30, playout_trim_ms=120)

    assert call.messages[-1]["role"] == "assistant"
    rewritten = call.messages[-1]["content"][0]["text"]
    assert rewritten == turn.spoken_text
    assert rewritten != APPROVED
    assert_invariant(call.messages)


def test_assistant_message_is_removed_when_nothing_was_heard() -> None:
    call = make_call()
    turn = make_turn()
    call.messages = [user_text("can I get a refund?"), assistant_text(APPROVED)]

    report = trim_on_interrupt(call, turn, played_chars=0, playout_trim_ms=120)

    assert report.assistant_message_removed
    assert len(call.messages) == 1
    assert_invariant(call.messages)


def test_next_utterance_merges_rather_than_doubling_user_turns() -> None:
    """After a removed assistant message, history ends on a user message.

    Appending the caller's next utterance naively would produce two
    consecutive user messages, which every provider rejects.
    """
    call = make_call()
    turn = make_turn()
    call.messages = [user_text("can I get a refund?"), assistant_text(APPROVED)]
    trim_on_interrupt(call, turn, played_chars=0, playout_trim_ms=120)

    append_user(call.messages, "no, the other order")

    assert len(call.messages) == 1
    assert len(call.messages[0]["content"]) == 2
    assert_invariant(call.messages)


def test_pending_executions_cancel_but_dispatched_ones_do_not() -> None:
    """The rule that prevents duplicate side effects (docs/07 §3.2).

    Once a request has left the process it is resolved or reconciled — never
    cancelled. Marking a dispatched call 'cancelled' on barge-in is exactly
    how a duplicate gets created.
    """
    call = make_call()
    turn = make_turn()
    call.messages = [user_text("transfer two tickets"), assistant_text(APPROVED)]
    pending = ToolExecution("e1", "transfer_tickets", True, "k1", {}, status="pending")
    dispatched = ToolExecution("e2", "request_refund", True, "k2", {}, status="dispatched")
    call.tool_executions = [pending, dispatched]

    trim_on_interrupt(call, turn, played_chars=10, playout_trim_ms=0)

    assert pending.status == "cancelled"
    assert dispatched.status == "dispatched"


# ---------------------------------------------------------------------------
# orphan tool results
# ---------------------------------------------------------------------------
def test_orphan_tool_results_are_dropped() -> None:
    messages = [
        user_text("where are my tickets"),
        {"role": "assistant", "content": [{"type": "text", "text": "checking"}]},
        {
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": "gone", "content": "{}"}],
        },
    ]
    dropped = drop_orphan_tool_results(messages)

    assert dropped == 1
    assert len(messages) == 2, "a message left with no blocks must not survive as empty"


def test_matched_tool_pairs_survive() -> None:
    messages = [
        user_text("where are my tickets"),
        {
            "role": "assistant",
            "content": [
                {"type": "tool_use", "id": "u1", "name": "lookup_order", "input": {}}
            ],
        },
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "u1", "content": "{}"}]},
    ]
    assert drop_orphan_tool_results(messages) == 0
    assert_invariant(messages)


# ---------------------------------------------------------------------------
# the invariant itself
# ---------------------------------------------------------------------------
def test_invariant_rejects_consecutive_same_role() -> None:
    with pytest.raises(ContextInvariantError, match="consecutive"):
        assert_invariant([user_text("a"), user_text("b")])


def test_invariant_rejects_empty_content() -> None:
    with pytest.raises(ContextInvariantError, match="empty content"):
        assert_invariant([{"role": "user", "content": []}])


def test_invariant_rejects_unmatched_tool_use() -> None:
    messages = [
        user_text("a"),
        {
            "role": "assistant",
            "content": [{"type": "tool_use", "id": "u1", "name": "x", "input": {}}],
        },
    ]
    with pytest.raises(ContextInvariantError, match="no tool_result"):
        assert_invariant(messages)
