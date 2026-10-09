"""The idempotency claim, against the REAL enforcement point.

The in-memory ledger mirrors the Postgres semantics; this proves the Postgres
side actually behaves that way — including under concurrency, where only the
partial unique index can be the arbiter.
"""

from __future__ import annotations

import asyncio
import json

import pytest

from core import db
from core.ids import new_id
from core.settings import Settings
from eval.fault_injection.conftest import TRANSFER
from harness.state import CallState
from harness.tools.execution import PostgresLedger, ToolExecutor
from harness.tools.mocks.faults import PROFILES, FaultyBackend
from harness.tools.mocks.ticketing import MockTicketing
from harness.tools.registry import load_registry

settings = Settings()
pytestmark = [
    pytest.mark.db,
    pytest.mark.skipif(not db.is_reachable(settings.database_url), reason="run `make up`"),
]


@pytest.fixture
async def pg_call():
    await db.open_pool(settings)
    call = CallState(call_id=new_id(), domain_config="ticketing", source="text")
    call.caller_verified = True
    await db.execute(
        "INSERT INTO calls (call_id, domain_config, source, config_snapshot)"
        " VALUES (%(id)s, 'ticketing', 'text', %(snap)s::jsonb)",
        {"id": call.call_id, "snap": json.dumps({"test": "postgres_ledger"})},
    )
    try:
        yield call
    finally:
        await db.execute("DELETE FROM calls WHERE call_id = %(id)s", {"id": call.call_id})
        await db.close_pool()


def executor(backend: MockTicketing, profile: str = "benign") -> ToolExecutor:
    return ToolExecutor(
        registry=load_registry("ticketing"),
        backend=FaultyBackend(backend, PROFILES[profile]),
        ledger=PostgresLedger(),
    )


async def _committed_rows(call_id: str) -> int:
    row = await db.fetch_one(
        "SELECT count(*) AS n FROM tool_executions"
        " WHERE call_id = %(id)s AND side_effecting AND status = 'committed'",
        {"id": call_id},
    )
    return row["n"]


async def test_concurrent_duplicates_lose_to_the_index(pg_call) -> None:
    backend = MockTicketing()
    ex = executor(backend)
    results = await asyncio.gather(
        *(ex.run(pg_call, None, "transfer_tickets", TRANSFER) for _ in range(5))
    )
    assert all(r.ok for r in results)
    assert backend.count_effects("transfer_tickets") == 1
    assert await _committed_rows(pg_call.call_id) == 1


async def test_commit_then_timeout_against_postgres(pg_call) -> None:
    backend = MockTicketing()
    outcome = await executor(backend, "ambiguous").run(pg_call, None, "transfer_tickets", TRANSFER)
    assert outcome.status == "committed" and outcome.reconciled

    row = await db.fetch_one(
        "SELECT status, dispatched_at IS NOT NULL AS dispatched_recorded"
        " FROM tool_executions WHERE call_id = %(id)s", {"id": pg_call.call_id},
    )
    assert row["status"] == "committed"
    # The dispatch was durably recorded before the outbound call.
    assert row["dispatched_recorded"]


async def test_the_duplicate_query_stays_empty(pg_call) -> None:
    """The exact query from docs/03 §5 that must always return zero rows."""
    backend = MockTicketing()
    ex = executor(backend)
    for _ in range(3):
        await ex.run(pg_call, None, "transfer_tickets", TRANSFER)
    rows = await db.fetch_all(
        "SELECT idempotency_key, count(*) FROM tool_executions"
        " WHERE side_effecting AND status = 'committed' GROUP BY 1 HAVING count(*) > 1"
    )
    assert rows == []
