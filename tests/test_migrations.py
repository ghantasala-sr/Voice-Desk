"""W1-3 — the schema applies cleanly and serves the queries it exists for.

Marked `db`: skipped unless Postgres is reachable, so `make test` stays green
on a laptop with Docker stopped. Run `make up` first to exercise it.

The queries below are the ones from docs/03-data-model.md §5. If one of them
ever needs JSONB extraction, a column is missing from the schema.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from core.settings import Settings

ROOT = Path(__file__).resolve().parent.parent
MIGRATIONS = sorted((ROOT / "migrations").glob("*.sql"))


def _reachable() -> bool:
    from core.db import is_reachable

    return is_reachable(Settings().database_url)


pytestmark = [
    pytest.mark.db,
    pytest.mark.skipif(not _reachable(), reason="Postgres not reachable; run `make up`"),
]


async def _connect():
    import psycopg
    from psycopg.rows import dict_row

    return await psycopg.AsyncConnection.connect(
        Settings().database_url, row_factory=dict_row, autocommit=True
    )


async def test_migrations_are_idempotent() -> None:
    """`make migrate` must be safe to re-run — it is how a new migration
    reaches an existing database."""
    conn = await _connect()
    try:
        for _ in range(2):
            for path in MIGRATIONS:
                await conn.execute(path.read_text(encoding="utf-8"))
    finally:
        await conn.close()


async def test_every_expected_table_exists() -> None:
    expected = {
        "calls", "turns", "call_events", "tool_executions", "handoffs",
        "checker_verdicts", "documents", "chunks", "eval_runs",
        "eval_results", "gap_reports",
    }
    conn = await _connect()
    try:
        rows = await (
            await conn.execute(
                "SELECT tablename FROM pg_tables WHERE schemaname = 'public'"
            )
        ).fetchall()
    finally:
        await conn.close()

    assert expected <= {row["tablename"] for row in rows}


async def test_metric_queries_run_against_an_empty_schema() -> None:
    """Every query the schema was designed for, checked for syntax and shape."""
    queries = [
        # latency percentiles with sample size
        """SELECT count(*) AS n,
                  percentile_cont(0.50) WITHIN GROUP (ORDER BY voice_to_voice_ms) AS p50,
                  percentile_cont(0.95) WITHIN GROUP (ORDER BY voice_to_voice_ms) AS p95
             FROM turns t JOIN calls c USING (call_id)
            WHERE c.source = 'replay'""",
        # duplicate side effects — must always be empty
        """SELECT idempotency_key, count(*) FROM tool_executions
            WHERE side_effecting AND status = 'committed'
            GROUP BY 1 HAVING count(*) > 1""",
        # checker catch rate by hallucination class
        """SELECT seeded_class, count(*) AS seeded,
                  count(*) FILTER (WHERE verdict LIKE 'fail%') AS caught
             FROM checker_verdicts WHERE seeded_class IS NOT NULL GROUP BY 1""",
        # stop-reason distribution
        """SELECT stop_reason, count(*),
                  round(100.0 * count(*) / NULLIF(sum(count(*)) OVER (), 0), 1) AS pct
             FROM turns GROUP BY 1""",
    ]
    conn = await _connect()
    try:
        for sql in queries:
            await (await conn.execute(sql)).fetchall()
    finally:
        await conn.close()


async def test_idempotency_index_blocks_a_duplicate_side_effect() -> None:
    """The constraint behind 'zero duplicate side effects'.

    This is a DATABASE guarantee, not application logic — a duplicate loses to
    the index deterministically, however it arrives.
    """
    import json

    import psycopg

    from core.ids import new_id

    call_id = new_id()
    key = f"test-{new_id()}"
    conn = await _connect()
    try:
        await conn.execute(
            "INSERT INTO calls (call_id, domain_config, source, config_snapshot)"
            " VALUES (%s, 'ticketing', 'text', %s::jsonb)",
            (call_id, json.dumps({})),
        )
        insert = (
            "INSERT INTO tool_executions"
            " (execution_id, call_id, tool_name, side_effecting,"
            "  idempotency_key, args_canonical, status)"
            " VALUES (%s, %s, 'transfer_tickets', TRUE, %s, '{}'::jsonb, 'committed')"
        )
        await conn.execute(insert, (new_id(), call_id, key))

        with pytest.raises(psycopg.errors.UniqueViolation):
            await conn.execute(insert, (new_id(), call_id, key))

        # A non-side-effecting tool may repeat freely — the partial index
        # deliberately does not constrain reads.
        read_insert = insert.replace("TRUE", "FALSE").replace(
            "transfer_tickets", "lookup_order"
        )
        await conn.execute(read_insert, (new_id(), call_id, key))
        await conn.execute(read_insert, (new_id(), call_id, key))
    finally:
        await conn.execute("DELETE FROM calls WHERE call_id = %s", (call_id,))
        await conn.close()
