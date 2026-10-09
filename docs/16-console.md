# 16 — The live console

`http://localhost:8088/console` (any port the app runs on). A live monitor over Postgres:
calls arriving, each turn's retrieval, latency stages, generated vs. approved vs. spoken
text, checker verdicts, tool executions, handoffs.

**It is not a results source.** Its numbers have no run directory, so they are never quoted
(docs/15). The health strip says "live — not a result" for that reason.

Code: `app/observe.py` (API), `app/static/console/` (vanilla HTML/CSS/JS, no build step).

---

## 1. Panels

| Panel | Shows |
|---|---|
| **Health strip** (top, every 5 s) | Duplicate side effects (all time; green at 0, flashing red otherwise), calls in window and how many have no `ended_at`, stop-reason distribution, `first_audio_ms` p50/p95 (harness side, B..H), `endpoint_ms` p50/p95 (the VAD setting), `voice_to_voice_ms` p50/p95, checker verdicts, tool status, handoffs by reason, ingested chunks. Every figure carries its `n`. |
| **Calls** (left) | Newest first, with an outcome badge (`live` until the call has an outcome). Updates when `call_started` / `call_ended` / `turn_completed` arrive, or when an event names an unknown call. Auto-follows the newest call until you click one; "follow newest call" turns it back on. Esc returns to the all-calls feed. |
| **Timeline** (center) | One card per turn. Caller text; agent text as *heard* (normal), *approved but not heard* (struck through, amber — the barge-in cut), and *generated but not approved* (struck, red — what the checker withheld). Stop-reason badge. A latency waterfall with **two tracks**: `endpoint · VAD` alone, and `harness` with the stages stacked and a tick at `first_audio_ms`; voice-to-voice is printed as text, labelled as including endpoint. Retrieval signals (top vec score highlighted as the primary signal; fused score and rank margin dimmed, see docs/05 §6) and the retrieved chunks' heading paths — click one for a snippet. Checker verdicts per segment, tool executions with their position in pending → dispatched → committed (or failed/cancelled/unknown), handoff signals and handoff summaries. |
| **Live events** (right) | `call_events` for the selected call (history + live), or all calls when none is selected. Colour by type: `barge_in`/`context_trimmed` amber, `error`/`budget_exceeded` red, `tool_*` blue, `checker_verdict` purple, `handoff_*` orange. Click a row for the full payload. `stt_partial` hidden by default. |

Endpointing is drawn apart on purpose: it is a VAD setting, and folding it into the harness
bar would make a VAD change look like a harness speed-up (docs/10 §2).

All data reaches the DOM via `textContent`; transcripts and payloads are caller-controlled.

## 2. Endpoints

All read-only. All return `503 {"error": "database unavailable"}` when the pool is not open
(and `503` with a `make migrate` hint if the schema is older than the code), never a 500.

| Endpoint | Returns |
|---|---|
| `GET /api/calls?limit=50&source=` | `{"calls": [{call_id, source, domain, outcome, handoff_reason, caller_verified, order_id, rephrase_count, turn_count, started_at, ended_at, duration_ms}]}` newest first. `limit` 1–500. |
| `GET /api/calls/{call_id}` | `{call, turns, events, tool_executions, checker_verdicts, handoffs}`. Turns in `seq` order with every latency stage, the three texts, retrieval signals, and `retrieved_chunks: [{chunk_id, heading_path, snippet}]` in rank order. 404 `{"error": "call not found"}`. |
| `GET /api/stats?window_minutes=60` | `{window_minutes, calls:{n,active}, stop_reasons:{n,counts}, latency:{first_audio_ms|voice_to_voice_ms|endpoint_ms: {n,p50,p95}}, duplicate_side_effects:{count, keys, scope:"all time"}, checker_verdicts:{n,counts}, tool_status:{n,counts}, handoffs:{n,counts}, chunks_ingested}`. Window 1 min – 7 days. |
| `GET /api/stream?after=<event_id>` | Server-Sent Events, below. |

Duplicate side effects use the docs/03 query (side-effecting, committed, grouped by
idempotency key, `HAVING count(*) > 1`) and are **not windowed**: it is an invariant, and a
window would let a violation scroll out of view.

## 3. The live stream

The stream tails `call_events` by its `BIGSERIAL event_id`: every ~400 ms,
`WHERE event_id > cursor ORDER BY event_id LIMIT 200` (no sleep while a full batch comes
back). Reading Postgres rather than in-process state means the browser app, the text driver
and the batch simulator all show up through one stream, whichever process wrote them.

- Each event is one frame: `id: <event_id>` and `data: <event as JSON>`. The first frame is
  `retry: 2000`.
- Cursor: `Last-Event-ID` header (a reconnecting EventSource) → `?after=` → the last 50
  events, so a fresh console is not blank.
- `: heartbeat` comment after 15 s without events, so proxies keep the connection.
- The loop ends when `request.is_disconnected()`; a Postgres hiccup mid-stream is logged and
  retried on the next poll rather than closing the connection.
- **Ordered delivery across a gap.** An `event_id` is assigned at INSERT but visible at
  COMMIT, so concurrent writers can commit out of order. When the next row's id is not
  `cursor + 1` and that row is younger than 1 s, the stream waits rather than skip what may
  be an in-flight insert; older gaps (rollbacks, deleted calls) are skipped. Ids therefore
  arrive strictly increasing and Last-Event-ID is always a safe resume point. Cost: up to
  1 s of extra lag right after a gap.

Granularity is whatever the writer flushes: `PostgresEventWriter` buffers per turn
(docs/11 §4), so events land in bursts at turn boundaries, not one by one.

## 4. Tests

`tests/test_observe.py`. DB tests (`-m db`) seed their own rows under fresh ULIDs and delete
them. The SSE tests run a real uvicorn on a free port, because `TestClient` buffers the whole
body and would hang on an endless stream.
