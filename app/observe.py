"""Read-only observability API and live event stream for the console.

The console (`/console`) is a live monitor, not a report: it shows what the
system is doing right now, and its numbers carry no run directory. Anything
that needs to be quoted belongs in `results/` (docs/15).

Two design choices worth knowing:

  * Everything is read from Postgres, never from in-process state. The browser
    app, the text driver, and the batch simulator are different processes;
    tailing `call_events` by its BIGSERIAL `event_id` shows all of them through
    one stream, and `event_id` doubles as the SSE `id:` so a reconnecting
    EventSource resumes exactly where it stopped.
  * Every SELECT names its columns. Migrations add columns to these tables; a
    `SELECT *` would silently change the response shape under the console.

The database is optional for the app (see `app.main.lifespan`), so every
endpoint answers 503 `{"error": "database unavailable"}` rather than a 500
when the pool is not open.
"""

from __future__ import annotations

import asyncio
import functools
import json
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from datetime import datetime
from typing import Any

from fastapi import APIRouter, Query, Request
from fastapi.responses import JSONResponse, StreamingResponse

from core import db
from core.logging import get_logger

log = get_logger(__name__)

router = APIRouter()

# -- stream tuning ------------------------------------------------------------
STREAM_POLL_S = 0.4
STREAM_HEARTBEAT_S = 15.0
STREAM_BATCH = 200
# Events replayed to a client that connects with no cursor, so the feed is not
# blank until the next call happens.
STREAM_BACKFILL = 50
# event_id is assigned at INSERT, but rows become visible at COMMIT, so two
# concurrent writers can commit ids out of order. A gap in the ids may be an
# insert that is still in flight; hold the cursor behind it for this long
# before treating it as permanent (a rollback or a deleted call).
STREAM_GAP_GRACE_S = 1.0

# The latency stages, in pipeline order. `endpoint_ms` is deliberately listed
# apart: it is the VAD silence setting, not harness work, and must never be
# summed into the harness number (docs/01 §5, docs/10 §2).
HARNESS_STAGES = (
    "stt_final_ms", "retrieval_ms", "llm_ttft_ms", "sentence_buffer_ms",
    "checker_ms", "tts_ttfb_ms", "playout_ms",
)

_CALL_COLUMNS = """
    call_id, source, domain_config, outcome, handoff_reason, caller_verified,
    order_id, rephrase_count, turn_count, started_at, ended_at, duration_ms
"""

_TURN_COLUMNS = """
    turn_id, seq, trigger, user_text, stt_confidence, stt_provider,
    retrieved_chunk_ids, retrieval_top_score, retrieval_margin,
    retrieval_top_vec_score, retrieval_top_lex_score, retrieval_vec_margin,
    llm_iterations, generated_text, approved_text, spoken_text, stop_reason,
    handoff_signals, handoff_score,
    endpoint_ms, stt_final_ms, retrieval_ms, llm_ttft_ms, sentence_buffer_ms,
    checker_ms, tts_ttfb_ms, playout_ms, first_audio_ms, voice_to_voice_ms,
    ack_bypass_used, trace_id, created_at
"""

_EVENT_COLUMNS = "event_id, call_id, turn_id, seq, event_type, payload, ts"


# -- database guard -----------------------------------------------------------
def _unavailable(detail: str = "database unavailable") -> JSONResponse:
    return JSONResponse({"error": detail}, status_code=503)


def _db_open(request: Request) -> bool:
    if not getattr(request.app.state, "db_ready", False):
        return False
    try:
        db.pool()
    except RuntimeError:
        return False
    return True


def _connection_errors() -> tuple[type[BaseException], ...]:
    # psycopg is imported lazily, as in core.db: importing app.main must not
    # require it.
    import psycopg
    import psycopg_pool

    return (RuntimeError, psycopg.OperationalError, psycopg_pool.PoolTimeout)


def _schema_errors() -> tuple[type[BaseException], ...]:
    import psycopg

    return (psycopg.errors.UndefinedColumn, psycopg.errors.UndefinedTable)


Handler = Callable[..., Awaitable[Any]]


def _needs_db(handler: Handler) -> Handler:
    """Turn 'no database' into a 503 instead of a 500.

    A schema older than the code is also a 503 with a hint, because the fix is
    operational (`make migrate`), not a bug in the request.
    """

    @functools.wraps(handler)
    async def wrapper(request: Request, *args: Any, **kwargs: Any) -> Any:
        if not _db_open(request):
            return _unavailable()
        try:
            return await handler(request, *args, **kwargs)
        except _schema_errors() as exc:
            log.warning("observe_schema_mismatch", detail=str(exc))
            return _unavailable("database schema out of date — run `make migrate`")
        except _connection_errors() as exc:
            log.warning("observe_db_error", detail=str(exc))
            return _unavailable()

    return wrapper


# -- calls --------------------------------------------------------------------
@router.get("/api/calls")
@_needs_db
async def list_calls(
    request: Request,
    limit: int = Query(50, ge=1, le=500),
    source: str | None = Query(None),
) -> dict[str, Any]:
    rows = await db.fetch_all(
        f"""
        SELECT {_CALL_COLUMNS}
          FROM calls
         WHERE (%(source)s::text IS NULL OR source = %(source)s)
         ORDER BY started_at DESC, call_id DESC
         LIMIT %(limit)s
        """,
        {"source": source or None, "limit": limit},
    )
    return {"calls": [_call_summary(row) for row in rows]}


def _call_summary(row: dict[str, Any]) -> dict[str, Any]:
    out = dict(row)
    out["domain"] = out.pop("domain_config")
    return out


@router.get("/api/calls/{call_id}")
@_needs_db
async def call_detail(request: Request, call_id: str) -> Any:
    call = await db.fetch_one(
        f"SELECT {_CALL_COLUMNS}, config_snapshot FROM calls WHERE call_id = %(id)s",
        {"id": call_id},
    )
    if call is None:
        return JSONResponse({"error": "call not found", "call_id": call_id}, status_code=404)

    params = {"id": call_id}
    turns = await db.fetch_all(
        f"SELECT {_TURN_COLUMNS} FROM turns WHERE call_id = %(id)s ORDER BY seq", params
    )
    events = await db.fetch_all(
        f"SELECT {_EVENT_COLUMNS} FROM call_events WHERE call_id = %(id)s"
        " ORDER BY event_id LIMIT 5000",
        params,
    )
    tools = await db.fetch_all(
        """
        SELECT execution_id, turn_id, tool_name, side_effecting, idempotency_key,
               args_canonical, status, attempt, error_type, error_detail,
               created_at, dispatched_at, resolved_at, duration_ms
          FROM tool_executions WHERE call_id = %(id)s ORDER BY created_at, execution_id
        """,
        params,
    )
    verdicts = await db.fetch_all(
        """
        SELECT verdict_id, turn_id, segment_index, segment_text, verdict,
               checkable_spans, action_taken, checker_ms, seeded_class, created_at
          FROM checker_verdicts WHERE call_id = %(id)s ORDER BY turn_id, segment_index
        """,
        params,
    )
    handoffs = await db.fetch_all(
        """
        SELECT handoff_id, turn_id, reason, rule_type, signals, score,
               summary_text, created_at
          FROM handoffs WHERE call_id = %(id)s ORDER BY created_at
        """,
        params,
    )

    chunk_ids = sorted({cid for turn in turns for cid in (turn["retrieved_chunk_ids"] or [])})
    chunks: dict[str, dict[str, Any]] = {}
    if chunk_ids:
        rows = await db.fetch_all(
            """
            SELECT chunk_id, heading_path, left(text, 320) AS snippet
              FROM chunks WHERE chunk_id = ANY(%(ids)s)
            """,
            {"ids": chunk_ids},
        )
        chunks = {row["chunk_id"]: row for row in rows}

    for turn in turns:
        # Keep retrieval rank order, and keep ids whose chunk was since
        # re-ingested away: a missing heading is itself worth seeing.
        turn["retrieved_chunks"] = [
            chunks.get(cid, {"chunk_id": cid, "heading_path": None, "snippet": None})
            for cid in (turn["retrieved_chunk_ids"] or [])
        ]

    return {
        "call": _call_summary(call),
        "turns": turns,
        "events": events,
        "tool_executions": tools,
        "checker_verdicts": verdicts,
        "handoffs": handoffs,
    }


# -- stats --------------------------------------------------------------------
@router.get("/api/stats")
@_needs_db
async def stats(
    request: Request, window_minutes: int = Query(60, ge=1, le=7 * 24 * 60)
) -> dict[str, Any]:
    window = {"w": window_minutes}
    since = "now() - make_interval(mins => %(w)s)"

    calls = await db.fetch_one(
        f"""
        SELECT count(*) AS n, count(*) FILTER (WHERE ended_at IS NULL) AS active
          FROM calls WHERE started_at > {since}
        """,
        window,
    )
    stop_rows = await db.fetch_all(
        f"""
        SELECT coalesce(stop_reason, 'none') AS key, count(*) AS n
          FROM turns WHERE created_at > {since} GROUP BY 1 ORDER BY 2 DESC
        """,
        window,
    )
    latency: dict[str, dict[str, Any]] = {}
    # Each column gets its own n: a turn interrupted before audio has a
    # first_audio_ms of NULL, and pooling the counts would overstate one of them.
    for column in ("first_audio_ms", "voice_to_voice_ms", "endpoint_ms"):
        row = await db.fetch_one(
            f"""
            SELECT count({column}) AS n,
                   percentile_cont(0.50) WITHIN GROUP (ORDER BY {column}) AS p50,
                   percentile_cont(0.95) WITHIN GROUP (ORDER BY {column}) AS p95
              FROM turns WHERE created_at > {since} AND {column} IS NOT NULL
            """,
            window,
        )
        latency[column] = _percentiles(row)

    # Not windowed. This is an invariant, not a rate: one duplicate ever is a
    # bug, and a window would let it scroll out of view.
    dup_rows = await db.fetch_all(
        """
        SELECT idempotency_key, count(*) AS n
          FROM tool_executions
         WHERE side_effecting AND status = 'committed'
         GROUP BY idempotency_key HAVING count(*) > 1
         ORDER BY 2 DESC LIMIT 20
        """
    )
    dup_count = await db.fetch_one(
        """
        SELECT count(*) AS n FROM (
          SELECT 1 FROM tool_executions
           WHERE side_effecting AND status = 'committed'
           GROUP BY idempotency_key HAVING count(*) > 1) d
        """
    )
    verdict_rows = await db.fetch_all(
        f"""
        SELECT verdict AS key, count(*) AS n FROM checker_verdicts
         WHERE created_at > {since} GROUP BY 1 ORDER BY 2 DESC
        """,
        window,
    )
    tool_rows = await db.fetch_all(
        f"""
        SELECT status AS key, count(*) AS n FROM tool_executions
         WHERE created_at > {since} GROUP BY 1 ORDER BY 2 DESC
        """,
        window,
    )
    handoff_rows = await db.fetch_all(
        f"""
        SELECT reason AS key, count(*) AS n FROM handoffs
         WHERE created_at > {since} GROUP BY 1 ORDER BY 2 DESC
        """,
        window,
    )
    chunk_row = await db.fetch_one("SELECT count(*) AS n FROM chunks")

    return {
        "window_minutes": window_minutes,
        "calls": {"n": calls["n"] if calls else 0, "active": calls["active"] if calls else 0},
        "stop_reasons": _distribution(stop_rows),
        "latency": latency,
        "duplicate_side_effects": {
            "count": dup_count["n"] if dup_count else 0,
            "keys": dup_rows,
            "scope": "all time",
        },
        "checker_verdicts": _distribution(verdict_rows),
        "tool_status": _distribution(tool_rows),
        "handoffs": _distribution(handoff_rows),
        "chunks_ingested": chunk_row["n"] if chunk_row else 0,
    }


def _distribution(rows: list[dict[str, Any]]) -> dict[str, Any]:
    counts = {row["key"]: row["n"] for row in rows}
    return {"n": sum(counts.values()), "counts": counts}


def _percentiles(row: dict[str, Any] | None) -> dict[str, Any]:
    if not row or not row["n"]:
        return {"n": 0, "p50": None, "p95": None}
    return {"n": row["n"], "p50": round(row["p50"]), "p95": round(row["p95"])}


# -- live stream --------------------------------------------------------------
@router.get("/api/stream")
async def stream(request: Request, after: int | None = Query(None, ge=0)) -> Any:
    """Server-Sent Events tail of `call_events`.

    Cursor precedence: the `Last-Event-ID` header (an EventSource reconnecting
    on its own), then `?after=`, then the last STREAM_BACKFILL events.
    """
    if not _db_open(request):
        return _unavailable()

    cursor = _last_event_id(request)
    if cursor is None:
        cursor = after
    if cursor is None:
        try:
            cursor = await _backfill_cursor()
        except _connection_errors():
            return _unavailable()

    return StreamingResponse(
        event_stream(request, cursor),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


def _last_event_id(request: Request) -> int | None:
    raw = request.headers.get("last-event-id")
    if raw is None:
        return None
    try:
        return max(int(raw), 0)
    except ValueError:
        return None


async def _backfill_cursor() -> int:
    row = await db.fetch_one(
        """
        SELECT coalesce(min(event_id) - 1, 0) AS cursor FROM (
          SELECT event_id FROM call_events ORDER BY event_id DESC LIMIT %(n)s) recent
        """,
        {"n": STREAM_BACKFILL},
    )
    return int(row["cursor"]) if row else 0


async def event_stream(
    request: Request,
    cursor: int,
    *,
    poll_s: float = STREAM_POLL_S,
    heartbeat_s: float = STREAM_HEARTBEAT_S,
) -> AsyncIterator[str]:
    """Yield SSE frames for every event after `cursor`, until the client leaves.

    Ids are delivered strictly in order, so the browser's Last-Event-ID is
    always a safe resume point (see STREAM_GAP_GRACE_S).
    """
    # Tell EventSource to retry quickly; the default is ~3 s.
    yield "retry: 2000\n\n"
    last_sent = time.monotonic()
    while not await request.is_disconnected():
        try:
            rows = await db.fetch_all(
                f"""
                SELECT {_EVENT_COLUMNS},
                       extract(epoch FROM clock_timestamp() - ts)::float8 AS age_s
                  FROM call_events WHERE event_id > %(cursor)s
                 ORDER BY event_id LIMIT %(limit)s
                """,
                {"cursor": cursor, "limit": STREAM_BATCH},
            )
        except _connection_errors() as exc:
            # Keep the connection: the console shows the gap, and the next poll
            # picks up where this one failed once Postgres is back.
            log.warning("observe_stream_db_error", detail=str(exc))
            rows = []

        sent = 0
        for row in rows:
            age_s = row.pop("age_s")
            if row["event_id"] != cursor + 1 and age_s < STREAM_GAP_GRACE_S:
                break  # an earlier id may still be committing; wait for it
            cursor = row["event_id"]
            yield _sse_frame(row)
            sent += 1
            # Stop mid-batch, not at the next poll: a client that left must
            # not be sent the rest of a batch fetched before it went.
            if await request.is_disconnected():
                return

        now = time.monotonic()
        if sent:
            last_sent = now
        elif now - last_sent >= heartbeat_s:
            yield ": heartbeat\n\n"
            last_sent = now

        if sent < STREAM_BATCH:
            await asyncio.sleep(poll_s)


def _sse_frame(row: dict[str, Any]) -> str:
    return f"id: {row['event_id']}\ndata: {json.dumps(row, default=_json_default)}\n\n"


def _json_default(value: Any) -> Any:
    if isinstance(value, datetime):
        return value.isoformat()
    return str(value)
