"""The console's read-only API and live stream (app/observe.py, docs/16).

DB-backed tests seed their own rows under fresh ULIDs and delete them in a
`finally` (deleting the call cascades), so they are safe against a database
that other work is writing to at the same time. Assertions only ever look at
the seeded rows, never at global counts.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import socket
import threading
import time
from collections.abc import Iterator
from typing import Any

import pytest
from fastapi.testclient import TestClient

from app.main import app
from core.ids import new_id
from core.settings import Settings


def _reachable() -> bool:
    from core.db import is_reachable

    return is_reachable(Settings().database_url)


def needs_db(test: Any) -> Any:
    """Mark and skip a test that needs Postgres (as tests/test_migrations.py does)."""
    skip = pytest.mark.skipif(not _reachable(), reason="Postgres not reachable; run `make up`")
    return pytest.mark.db(skip(test))


# -- seeding ------------------------------------------------------------------
@contextlib.contextmanager
def seeded_call() -> Iterator[dict[str, Any]]:
    """One call with two turns (the second interrupted), events, a committed
    side-effecting tool execution, a checker verdict, a handoff, and a chunk."""
    import psycopg

    ids = {
        "call_id": new_id(), "turn0": new_id(), "turn1": new_id(),
        "doc_id": new_id(), "chunk_id": new_id(), "exec_id": new_id(),
        "handoff_id": new_id(), "idem": f"test-observe-{new_id()}",
    }
    zero_vec = "[" + ",".join(["0"] * 384) + "]"
    conn = psycopg.connect(Settings().database_url, autocommit=True)
    try:
        conn.execute(
            "INSERT INTO documents (doc_id, domain_config, path, title, content_hash)"
            " VALUES (%s, 'ticketing', %s, 'Test doc', 'x')",
            (ids["doc_id"], f"test/{ids['doc_id']}.md"),
        )
        conn.execute(
            "INSERT INTO chunks (chunk_id, doc_id, domain_config, seq, heading_path,"
            " text, embed_text, word_count, embedding)"
            " VALUES (%s, %s, 'ticketing', 0, 'Refunds > Event cancelled', %s, 'e', 4, %s::vector)",
            (ids["chunk_id"], ids["doc_id"], "Full refunds within 30 days.", zero_vec),
        )
        conn.execute(
            "INSERT INTO calls (call_id, domain_config, source, config_snapshot, turn_count)"
            " VALUES (%s, 'ticketing', 'text', '{}'::jsonb, 2)",
            (ids["call_id"],),
        )
        conn.execute(
            "INSERT INTO turns (turn_id, call_id, seq, trigger, user_text,"
            " retrieved_chunk_ids, retrieval_top_vec_score, generated_text, approved_text,"
            " spoken_text, stop_reason, endpoint_ms, stt_final_ms, llm_ttft_ms,"
            " first_audio_ms, voice_to_voice_ms, created_at)"
            " VALUES (%s, %s, 0, 'transcript', 'can I get a refund', %s, 0.81,"
            " 'Yes, within 30 days.', 'Yes, within 30 days.', 'Yes, within 30 days.',"
            " 'answered', 400, 120, 300, 700, 1100, now())",
            (ids["turn0"], ids["call_id"], [ids["chunk_id"]]),
        )
        conn.execute(
            "INSERT INTO turns (turn_id, call_id, seq, trigger, user_text, generated_text,"
            " approved_text, spoken_text, stop_reason, endpoint_ms, created_at)"
            " VALUES (%s, %s, 1, 'transcript', 'and the fee', 'The fee is waived for you.',"
            " 'The fee is waived', 'The fee', 'interrupted', 380, now())",
            (ids["turn1"], ids["call_id"]),
        )
        conn.execute(
            "INSERT INTO tool_executions (execution_id, call_id, turn_id, tool_name,"
            " side_effecting, idempotency_key, args_canonical, status)"
            " VALUES (%s, %s, %s, 'transfer_tickets', TRUE, %s, '{}'::jsonb, 'committed')",
            (ids["exec_id"], ids["call_id"], ids["turn0"], ids["idem"]),
        )
        conn.execute(
            "INSERT INTO checker_verdicts (call_id, turn_id, segment_index, segment_text,"
            " verdict, action_taken) VALUES (%s, %s, 0, 'Yes, within 30 days.', 'pass', 'released')",
            (ids["call_id"], ids["turn0"]),
        )
        conn.execute(
            "INSERT INTO handoffs (handoff_id, call_id, turn_id, reason, rule_type, signals,"
            " summary, summary_text) VALUES (%s, %s, %s, 'caller_request', 'hard',"
            " '[]'::jsonb, '{}'::jsonb, 'Caller asked for a human.')",
            (ids["handoff_id"], ids["call_id"], ids["turn1"]),
        )
        rows = conn.execute(
            "INSERT INTO call_events (call_id, turn_id, seq, event_type, payload) VALUES"
            " (%s, NULL, 0, 'call_started', '{\"source\": \"text\"}'::jsonb),"
            " (%s, %s, 1, 'barge_in', '{\"played_chars\": 7}'::jsonb)"
            " RETURNING event_id",
            (ids["call_id"], ids["call_id"], ids["turn1"]),
        ).fetchall()
        ids["event_ids"] = [row[0] for row in rows]
        yield ids
    finally:
        conn.execute("DELETE FROM calls WHERE call_id = %s", (ids["call_id"],))
        conn.execute("DELETE FROM documents WHERE doc_id = %s", (ids["doc_id"],))
        conn.close()


# -- page ---------------------------------------------------------------------
def test_console_page_is_served() -> None:
    with TestClient(app) as client:
        page = client.get("/console")
        assert page.status_code == 200
        assert "VoiceDesk" in page.text
        assert "live — not a result" in page.text
        for asset in ("console.js", "console.css"):
            assert client.get(f"/static/console/{asset}").status_code == 200


# -- 503 path -----------------------------------------------------------------
UNAVAILABLE_PATHS = ["/api/calls", "/api/calls/anything", "/api/stats", "/api/stream"]


def test_endpoints_answer_503_when_the_database_is_not_ready(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with TestClient(app) as client:
        monkeypatch.setattr(app.state, "db_ready", False)
        for path in UNAVAILABLE_PATHS:
            res = client.get(path)
            assert res.status_code == 503, path
            assert res.json() == {"error": "database unavailable"}, path


def test_endpoints_answer_503_when_the_pool_is_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    """db_ready can be stale (pool closed after startup); that is still a 503."""
    from core import db

    with TestClient(app) as client:
        monkeypatch.setattr(app.state, "db_ready", True)
        # Patch the accessor, not `db._pool`: nulling the global would stop
        # the lifespan from closing the real pool and leak it into later tests.
        def closed() -> None:
            raise RuntimeError("database pool is not open")

        monkeypatch.setattr(db, "pool", closed)
        for path in UNAVAILABLE_PATHS:
            res = client.get(path)
            assert res.status_code == 503, path
            assert res.json()["error"] == "database unavailable", path


# -- calls --------------------------------------------------------------------
@needs_db
def test_calls_list_includes_a_seeded_call() -> None:
    with seeded_call() as ids, TestClient(app) as client:
        body = client.get("/api/calls", params={"limit": 500, "source": "text"}).json()
        match = [c for c in body["calls"] if c["call_id"] == ids["call_id"]]
        assert len(match) == 1
        assert set(match[0]) >= {
            "call_id", "source", "domain", "outcome", "turn_count", "started_at", "duration_ms",
        }
        assert match[0]["domain"] == "ticketing"
        assert match[0]["turn_count"] == 2

        other = client.get("/api/calls", params={"limit": 500, "source": "browser"}).json()
        assert all(c["call_id"] != ids["call_id"] for c in other["calls"])


@needs_db
def test_call_detail_returns_everything_for_the_call() -> None:
    with seeded_call() as ids, TestClient(app) as client:
        res = client.get(f"/api/calls/{ids['call_id']}")
        assert res.status_code == 200
        body = res.json()

        assert body["call"]["call_id"] == ids["call_id"]
        turns = body["turns"]
        assert [t["seq"] for t in turns] == [0, 1]
        assert turns[0]["retrieved_chunks"][0]["heading_path"] == "Refunds > Event cancelled"
        assert turns[0]["retrieved_chunks"][0]["snippet"].startswith("Full refunds")
        # The barge-in turn keeps all three texts distinct.
        assert turns[1]["generated_text"] != turns[1]["approved_text"] != turns[1]["spoken_text"]
        assert turns[1]["stop_reason"] == "interrupted"
        assert turns[1]["first_audio_ms"] is None  # stage did not run: NULL, not 0

        assert [e["event_type"] for e in body["events"]] == ["call_started", "barge_in"]
        assert body["tool_executions"][0]["status"] == "committed"
        assert body["checker_verdicts"][0]["verdict"] == "pass"
        assert body["handoffs"][0]["reason"] == "caller_request"


@needs_db
def test_call_detail_404s_for_an_unknown_call() -> None:
    with TestClient(app) as client:
        res = client.get(f"/api/calls/{new_id()}")
        assert res.status_code == 404
        assert res.json()["error"] == "call not found"


# -- stats --------------------------------------------------------------------
@needs_db
def test_stats_shape_and_the_duplicate_side_effect_invariant() -> None:
    with seeded_call(), TestClient(app) as client:
        body = client.get("/api/stats", params={"window_minutes": 60}).json()

        assert body["window_minutes"] == 60
        for key in ("stop_reasons", "checker_verdicts", "tool_status", "handoffs"):
            assert set(body[key]) == {"n", "counts"}, key
            assert body[key]["n"] == sum(body[key]["counts"].values()), key
        # Seeded rows are inside the window, so these are at least 1.
        assert body["stop_reasons"]["counts"].get("interrupted", 0) >= 1
        assert body["tool_status"]["counts"].get("committed", 0) >= 1
        assert body["handoffs"]["counts"].get("caller_request", 0) >= 1

        for column in ("first_audio_ms", "voice_to_voice_ms", "endpoint_ms"):
            lat = body["latency"][column]
            assert set(lat) == {"n", "p50", "p95"}
            assert lat["n"] >= 1

        dup = body["duplicate_side_effects"]
        # The unique index makes a committed duplicate impossible to seed; the
        # point of the tile is that it stays at exactly zero.
        assert dup["count"] == 0
        assert dup["keys"] == []
        assert isinstance(body["chunks_ingested"], int) and body["chunks_ingested"] >= 1


def test_stats_rejects_a_nonsense_window() -> None:
    with TestClient(app) as client:
        assert client.get("/api/stats", params={"window_minutes": 0}).status_code == 422


# -- stream -------------------------------------------------------------------
def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


@contextlib.contextmanager
def running_server() -> Iterator[str]:
    """A real uvicorn in a thread. TestClient buffers the whole response body,
    so an endless SSE response would hang it; a socket does not."""
    import uvicorn

    port = _free_port()
    server = uvicorn.Server(
        uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning",
                       timeout_graceful_shutdown=2)
    )
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.monotonic() + 10
    while not server.started:
        assert time.monotonic() < deadline, "uvicorn did not start"
        time.sleep(0.05)
    try:
        yield f"http://127.0.0.1:{port}"
    finally:
        server.should_exit = True
        thread.join(timeout=10)


@needs_db
def test_stream_emits_a_seeded_event_with_its_event_id() -> None:
    import httpx

    with seeded_call() as ids, running_server() as base:
        first, second = ids["event_ids"]
        frames: list[tuple[int, dict[str, Any]]] = []
        current_id: int | None = None
        with httpx.stream(
            "GET", f"{base}/api/stream", params={"after": first - 1}, timeout=10
        ) as res:
            assert res.status_code == 200
            assert res.headers["content-type"].startswith("text/event-stream")
            for i, line in enumerate(res.iter_lines()):
                assert i < 200, "stream never delivered the seeded events"
                if line.startswith("id: "):
                    current_id = int(line[4:])
                elif line.startswith("data: "):
                    data = json.loads(line[6:])
                    assert data["event_id"] == current_id, "SSE id must equal event_id"
                    if data["call_id"] == ids["call_id"]:
                        frames.append((current_id, data))
                    if len(frames) == 2:
                        break

        assert [f[0] for f in frames] == [first, second]
        assert frames[1][1]["event_type"] == "barge_in"
        assert frames[1][1]["payload"] == {"played_chars": 7}


@needs_db
def test_stream_resumes_from_last_event_id() -> None:
    """A reconnecting EventSource sends Last-Event-ID; it beats `?after=`."""
    import httpx

    with seeded_call() as ids, running_server() as base:
        first, second = ids["event_ids"]
        with httpx.stream(
            "GET", f"{base}/api/stream", params={"after": 0},
            headers={"Last-Event-ID": str(first)}, timeout=10,
        ) as res:
            for i, line in enumerate(res.iter_lines()):
                assert i < 200
                if line.startswith("id: "):
                    assert int(line[4:]) == second
                    break


class _FakeRequest:
    def __init__(self) -> None:
        self.gone = False

    async def is_disconnected(self) -> bool:
        return self.gone


@needs_db
async def test_stream_holds_behind_a_fresh_gap_then_stops_on_disconnect() -> None:
    """An id gap may be an insert still committing; the cursor waits briefly
    rather than skipping it forever. And a departed client ends the loop."""
    import psycopg

    from app.observe import STREAM_GAP_GRACE_S, event_stream
    from core import db

    settings = Settings()
    conn = await psycopg.AsyncConnection.connect(settings.database_url, autocommit=True)
    call_id = new_id()
    await db.open_pool(settings)
    try:
        await conn.execute(
            "INSERT INTO calls (call_id, domain_config, source, config_snapshot)"
            " VALUES (%s, 'ticketing', 'text', '{}'::jsonb)",
            (call_id,),
        )
        burned = (
            await (await conn.execute("SELECT nextval('call_events_event_id_seq')")).fetchone()
        )[0]
        await conn.execute(
            "INSERT INTO call_events (call_id, seq, event_type) VALUES (%s, 0, 'call_started')",
            (call_id,),
        )

        request = _FakeRequest()
        stream = event_stream(request, burned - 1, poll_s=0.05)  # type: ignore[arg-type]
        started = time.monotonic()
        got: dict[str, Any] | None = None
        async with asyncio.timeout(10):
            async for frame in stream:
                if frame.startswith("id: "):
                    data = json.loads(frame.split("data: ", 1)[1])
                    if data["call_id"] == call_id:
                        got = data
                        break
        elapsed = time.monotonic() - started
        assert got is not None and got["event_id"] > burned
        assert elapsed >= STREAM_GAP_GRACE_S * 0.5, "a fresh gap must not be skipped at once"

        request.gone = True
        async with asyncio.timeout(5):
            remaining = [frame async for frame in stream]
        assert all(not f.startswith("id: ") for f in remaining)
    finally:
        await conn.execute("DELETE FROM calls WHERE call_id = %s", (call_id,))
        await conn.close()
        await db.close_pool()
