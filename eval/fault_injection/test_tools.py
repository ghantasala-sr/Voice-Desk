"""docs/12 §7.1 — tool faults.

The headline assertion in every test is the same: count the side effects the
BACKEND actually performed. The mock never deduplicates, so a duplicate here
is a real duplicate, caused by the harness.
"""

from __future__ import annotations

import pytest

from eval.fault_injection.conftest import TRANSFER
from harness.tools.execution import ToolError


async def test_benign_transfer_commits_once(executor_for, verified_call, backend) -> None:
    outcome = await executor_for("benign").run(verified_call, "t1", "transfer_tickets", TRANSFER)
    assert outcome.ok
    assert backend.count_effects("transfer_tickets") == 1


async def test_tool_timeout_before_commit(executor_for, verified_call, backend, ledger) -> None:
    """Times out BEFORE committing: reconciliation finds nothing -> failed."""
    from dataclasses import replace

    from harness.tools.mocks.faults import PROFILES

    executor = executor_for("benign")
    executor.backend.profile = replace(PROFILES["benign"], timeout_rate=1.0)

    outcome = await executor.run(verified_call, "t1", "transfer_tickets", TRANSFER)

    assert outcome.status == "failed"
    assert outcome.reconciled
    assert backend.count_effects() == 0


async def test_commit_then_timeout_reconciles_to_committed(
    executor_for, verified_call, backend, ledger
) -> None:
    """THE case. The write happened, then the call timed out.

    A naive retry here duplicates the transfer. Reconciliation must discover
    the commit and report it truthfully instead.
    """
    executor = executor_for("ambiguous")
    outcome = await executor.run(verified_call, "t1", "transfer_tickets", TRANSFER)

    assert outcome.status == "committed"
    assert outcome.reconciled
    assert backend.count_effects("transfer_tickets") == 1

    # The model asks for the same transfer again. It must NOT happen twice.
    again = await executor.run(verified_call, "t2", "transfer_tickets", TRANSFER)
    assert again.duplicate_suppressed
    assert backend.count_effects("transfer_tickets") == 1


async def test_reconciliation_unavailable_is_a_hard_handoff(
    executor_for, verified_call, backend
) -> None:
    """Committed, timed out, AND the reconciliation read is down. We genuinely
    cannot know. The only honest move is a human."""
    outcome = await executor_for("blackout").run(verified_call, "t1", "transfer_tickets", TRANSFER)

    assert outcome.status == "unknown"
    assert outcome.needs_hard_handoff
    assert backend.count_effects("transfer_tickets") == 1  # it DID happen upstream


async def test_garbage_response_from_a_read_is_a_typed_failure(
    executor_for, verified_call
) -> None:
    from dataclasses import replace

    from harness.tools.mocks.faults import PROFILES

    executor = executor_for("benign")
    executor.backend.profile = replace(PROFILES["benign"], garbage_rate=1.0)

    outcome = await executor.run(verified_call, "t1", "lookup_order", {"order_id": "TK-88213"})
    assert outcome.status == "failed"
    assert outcome.error_type == "malformed"


async def test_garbage_response_from_a_write_reconciles(
    executor_for, verified_call, backend
) -> None:
    """Nonsense back from a WRITE is ambiguous, not a failure — reconcile."""
    from dataclasses import replace

    from harness.tools.mocks.faults import PROFILES

    executor = executor_for("benign")
    executor.backend.profile = replace(PROFILES["benign"], garbage_rate=1.0)

    outcome = await executor.run(verified_call, "t1", "transfer_tickets", TRANSFER)
    assert outcome.status == "committed" and outcome.reconciled
    assert backend.count_effects("transfer_tickets") == 1


async def test_duplicate_delivery_is_suppressed(executor_for, verified_call, backend, ledger) -> None:
    """The model emits the same tool call five times. One effect."""
    executor = executor_for("benign")
    outcomes = [
        await executor.run(verified_call, f"t{i}", "transfer_tickets", TRANSFER) for i in range(5)
    ]
    assert all(o.ok for o in outcomes)
    assert sum(o.duplicate_suppressed for o in outcomes) == 4
    assert backend.count_effects("transfer_tickets") == 1
    assert len(ledger.committed("transfer_tickets")) == 1


async def test_concurrent_duplicates_are_suppressed(executor_for, verified_call, backend) -> None:
    """Two identical calls racing. The ledger, not timing, decides."""
    import asyncio

    executor = executor_for("benign")
    results = await asyncio.gather(
        *(executor.run(verified_call, "t1", "transfer_tickets", TRANSFER) for _ in range(3))
    )
    assert all(r.ok for r in results)
    assert backend.count_effects("transfer_tickets") == 1


async def test_rephrased_free_text_does_not_slip_past_the_key(
    executor_for, verified_call, backend
) -> None:
    """reason_text is excluded from the key, so a rephrase still collides."""
    executor = executor_for("benign")
    a = {"order_id": "TK-88213", "reason_code": "cannot_attend", "reason_text": "I'm sick"}
    b = {"order_id": "TK-88213", "reason_code": "cannot_attend", "reason_text": "feeling unwell"}
    await executor.run(verified_call, "t1", "request_refund", a)
    second = await executor.run(verified_call, "t2", "request_refund", b)
    assert second.duplicate_suppressed
    assert backend.count_effects("request_refund") == 1


async def test_different_callers_are_different_actions(executor_for, backend) -> None:
    from harness.state import CallState

    executor = executor_for("benign")
    for call_id in ("caller-a", "caller-b"):
        call = CallState(call_id=call_id, domain_config="ticketing", source="text")
        call.caller_verified = True
        await executor.run(call, "t1", "transfer_tickets",
                           {**TRANSFER, "quantity": 1})
    assert backend.count_effects("transfer_tickets") == 2


async def test_cancel_only_from_pending(executor_for, verified_call) -> None:
    """The one rule. Once dispatched, never cancelled."""
    executor = executor_for("benign")
    prepared = await executor.prepare(verified_call, "t1", "transfer_tickets", TRANSFER)
    assert await executor.cancel(verified_call, prepared)
    assert prepared.execution.status == "cancelled"

    committed = await executor.run(verified_call, "t2", "transfer_tickets", TRANSFER)
    assert committed.ok and committed.execution.attempt == 2
    assert not await executor.cancel(verified_call, type(prepared)(
        prepared.tool, committed.execution, TRANSFER))


async def test_verification_is_enforced_by_the_executor(executor_for) -> None:
    """Not just the prompt. A prompt-only disclosure rule is not a control."""
    from harness.state import CallState

    call = CallState(call_id="unverified", domain_config="ticketing", source="text")
    with pytest.raises(ToolError, match="unverified"):
        await executor_for("benign").run(call, "t1", "lookup_order", {"order_id": "TK-88213"})


async def test_invalid_arguments_never_reach_the_backend(executor_for, verified_call, backend) -> None:
    with pytest.raises(ToolError, match="invalid_arguments"):
        await executor_for("benign").run(
            verified_call, "t1", "transfer_tickets",
            {"order_id": "88213", "quantity": 0, "recipient_email": "x"},
        )
    assert backend.count_effects() == 0


async def test_refund_schema_has_nowhere_to_put_an_amount(executor_for, verified_call) -> None:
    """The design rule — the agent never approves money — enforced by schema."""
    with pytest.raises(ToolError, match="unknown argument 'amount'"):
        await executor_for("benign").run(
            verified_call, "t1", "request_refund",
            {"order_id": "TK-88213", "reason_code": "cannot_attend", "amount": 120},
        )


async def test_sweep_reconciles_rows_left_by_a_crash(executor_for, verified_call, backend, ledger) -> None:
    from dataclasses import replace

    executor = executor_for("benign")
    prepared = await executor.prepare(verified_call, "t1", "transfer_tickets", TRANSFER)
    # Simulate: the dispatch was written, the call went out and committed,
    # then the process died before recording the result.
    await backend.call("transfer_tickets", TRANSFER, idempotency_key=prepared.execution.idempotency_key)
    await ledger.update(replace(prepared.execution, status="dispatched"))

    outcomes = await executor.sweep(verified_call)
    assert [o.status for o in outcomes] == ["committed"]
    assert backend.count_effects("transfer_tickets") == 1


async def test_ambiguity_only_applies_to_writes(executor_for, verified_call) -> None:
    """A read under the `ambiguous` profile is unaffected: commit-then-timeout
    is a property of writes."""
    outcome = await executor_for("ambiguous").run(
        verified_call, "t1", "lookup_order", {"order_id": "TK-88213"}
    )
    assert outcome.ok
