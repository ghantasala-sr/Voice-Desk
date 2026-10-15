"""Text chat with the full production agent, over a WebSocket.

The browser's Talk view. The point is that it is NOT a demo mode: the same
`Harness`, the same `build_stack` (hybrid retrieval, tools against the mock
backend, read-back, grounding checker, handoff policy) that the voice pipeline
and the simulator use. Only the transport differs — typed text instead of a
transcript — so what a reviewer sees here is what a caller would get, minus
the audio stages.

Protocol (docs/17 §3). Client -> server, JSON text frames:

    {"type": "user", "text": "..."}      one caller utterance
    {"type": "end"}                      hang up

Server -> client:

    {"type": "status", "state": "connecting", "detail": ...}
    {"type": "ready", "call_id", "agent": {...}, "stack": [...], "recorded": bool}
    {"type": "turn_started", "turn_id", "user_text"}
    {"type": "say", "turn_id", "segment_id", "text"}
    {"type": "event", "turn_id", "name", "payload"}
    {"type": "turn", "turn_id", "details": {...}}       after each finished turn
    {"type": "handoff", "summary_text", "reason", "summary"}
    {"type": "ended", "outcome", "reason"}
    {"type": "error", "code", "message"}

Two choices worth knowing:

  * The read-back is answered by the caller's own typed words. There is no
    confirm button on purpose: `harness.tools.readback.classify` must see
    what the person actually said, or the confirmation the system is built
    around is not being exercised.
  * The per-turn details are built from `TurnState` (what was persisted), not
    reconstructed from the event stream, so the panel shows exactly what the
    console will show for the same turn.

The stack factory, settings and persistence choice are FastAPI dependencies so
tests can drive the whole socket without Postgres or the embedding model.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Awaitable, Callable
from typing import Annotated, Any

from fastapi import APIRouter, Depends, WebSocket, WebSocketDisconnect

from analytics.events import NullEventWriter, PostgresEventWriter
from analytics.store import NullStore, PostgresStore
from core import db
from core.clock import SystemClock
from core.ids import new_id
from core.logging import get_logger
from core.settings import Settings, get_settings
from harness.assembly import Stack, build_stack
from harness.events import (
    Emit,
    EndCall,
    EventStream,
    Shutdown,
    Speak,
    TranscriptIn,
    TurnCompleted,
)
from harness.loop import Harness
from harness.state import CallState, TurnState
from providers import build_llm, config_snapshot, effective_model

log = get_logger(__name__)

router = APIRouter()

# A typed message longer than this is not a caller utterance; refuse it rather
# than feed a page of text to retrieval and the model.
MAX_USER_CHARS = 2_000

OFFLINE_NOTICE = (
    "Offline agent — answers are quoted from the knowledge base; "
    "set ANTHROPIC_API_KEY for Claude"
)

# Tool lifecycle events -> the status shown next to the tool in the details
# panel. The latest event for a tool wins.
_TOOL_STATUS = {
    "llm_tool_call": "proposed",
    "tool_dispatched": "dispatched",
    "tool_committed": "committed",
    "tool_failed": "failed",
    "tool_unknown": "unknown",
    "tool_cancelled": "cancelled",
    "tool_reconciled": "reconciled",
    "readback_spoken": "awaiting confirmation",
    "readback_confirmed": "confirmed",
    "readback_rejected": "declined",
}


# ---------------------------------------------------------------------------
# Which agent is talking
# ---------------------------------------------------------------------------
def _has_secret(value: Any) -> bool:
    return bool(value is not None and value.get_secret_value())


def agent_settings(base: Settings) -> tuple[Settings, dict[str, Any]]:
    """The settings the chat agent actually runs with, and a description.

    No key means the offline extractive provider, said plainly in the banner.
    Failing every message with a 401 would hide the rest of the system, which
    works without a model; silently pretending to be Claude would be worse.
    """
    configured = base.llm_provider
    settings = base
    offline = False
    notice = ""
    if configured == "anthropic" and not _has_secret(base.anthropic_api_key):
        offline, notice = True, OFFLINE_NOTICE
    elif configured == "openai" and not _has_secret(base.openai_api_key):
        offline = True
        notice = OFFLINE_NOTICE.replace("ANTHROPIC_API_KEY for Claude", "OPENAI_API_KEY")
    if offline:
        settings = base.model_copy(update={"llm_provider": "extractive"})
    elif configured == "extractive":
        offline, notice = True, OFFLINE_NOTICE
    model = effective_model(settings)
    return settings, {
        "configured_provider": configured,
        "provider": settings.llm_provider,
        "model": model,
        "offline": offline,
        "notice": notice,
    }


# ---------------------------------------------------------------------------
# Dependencies — overridden in tests
# ---------------------------------------------------------------------------
StackFactory = Callable[[Settings, bool], Awaitable[Stack]]


def get_chat_settings() -> Settings:
    return get_settings()


async def _default_stack(settings: Settings, persist: bool) -> Stack:
    return await build_stack(
        settings, domain=settings.domain, retrieval="real", tools=True,
        persist_ledger=persist,
    )


def get_stack_factory() -> StackFactory:
    return _default_stack


def get_persist(websocket: WebSocket) -> bool:
    """Record to Postgres only when the app's pool is actually open."""
    if not getattr(websocket.app.state, "db_ready", False):
        return False
    try:
        db.pool()
    except RuntimeError:
        return False
    return db.is_reachable(get_settings().database_url)


# ---------------------------------------------------------------------------
# The socket
# ---------------------------------------------------------------------------
@router.websocket("/ws/chat")
async def chat_socket(
    websocket: WebSocket,
    base_settings: Annotated[Settings, Depends(get_chat_settings)],
    stack_factory: Annotated[StackFactory, Depends(get_stack_factory)],
    persist: Annotated[bool, Depends(get_persist)],
) -> None:
    await websocket.accept()
    session = _ChatSession(websocket)
    try:
        await session.run(base_settings, stack_factory, persist)
    except WebSocketDisconnect:
        log.info("chat_client_disconnected")
    except Exception:
        # Logged in full here; the browser gets a typed message, never a trace.
        log.exception("chat_failed")
        await session.send_error("internal", "The agent hit an internal error. Start a new call.")
    finally:
        await session.close()


class _ChatSession:
    """One browser chat: one CallState, one Harness, one stream."""

    def __init__(self, websocket: WebSocket) -> None:
        self.websocket = websocket
        self.stream = EventStream()
        self.clock = SystemClock()
        self.call: CallState | None = None
        self.store: Any = NullStore()
        self.writer: Any = NullEventWriter()
        self._started_ms = self.clock.now_ms()
        self._open = True
        self._finished = False
        # Per-turn accumulators fed by Emit, read when the turn completes.
        self._tools: dict[str, dict[str, dict[str, Any]]] = {}
        self._checker_reasons: dict[str, dict[int, str]] = {}

    # -- sending ------------------------------------------------------------
    async def send(self, kind: str, **fields: Any) -> None:
        if not self._open:
            return
        try:
            await self.websocket.send_text(json.dumps({"type": kind, **fields}, default=str))
        except (WebSocketDisconnect, RuntimeError):
            self._open = False

    async def send_error(self, code: str, message: str) -> None:
        await self.send("error", code=code, message=message)

    # -- lifecycle ----------------------------------------------------------
    async def run(self, base: Settings, stack_factory: StackFactory, persist: bool) -> None:
        settings, agent = agent_settings(base)
        await self.send("status", state="connecting",
                        detail="Loading the knowledge base and embedding model…")

        self.call = CallState(
            call_id=new_id(),
            domain_config=settings.domain,
            source="text",
            config_snapshot=config_snapshot(settings),
        )
        recorded = False
        if persist:
            try:
                self.store, self.writer = PostgresStore(), PostgresEventWriter()
                await self.store.create_call(self.call)
                recorded = True
            except Exception as exc:
                log.warning("chat_not_recorded", detail=str(exc))
                self.store, self.writer = NullStore(), NullEventWriter()

        stack = await stack_factory(settings, recorded)
        harness = Harness(
            llm=build_llm(settings), settings=settings, clock=self.clock,
            retriever=stack.retriever, executor=stack.executor, similarity=stack.similarity,
        )
        await self.send(
            "ready", call_id=self.call.call_id, agent=agent, stack=list(stack.banner),
            recorded=recorded, domain=settings.domain,
        )

        reader = asyncio.create_task(self._read())
        try:
            async for command in harness.run(self.call, self.stream):
                await self._handle(command)
        finally:
            reader.cancel()
            await asyncio.gather(reader, return_exceptions=True)

    async def close(self) -> None:
        if self._finished:
            return
        self._finished = True
        if self.call is not None:
            try:
                await self.writer.flush()
                await self.store.finish_call(self.call, duration_ms=self.clock.now_ms() - self._started_ms)
            except Exception as exc:
                log.warning("chat_finish_not_recorded", detail=str(exc))
        if self._open:
            self._open = False
            try:
                await self.websocket.close()
            except RuntimeError:
                pass

    # -- inbound ------------------------------------------------------------
    async def _read(self) -> None:
        """Browser -> harness. Any way the socket ends becomes a Shutdown."""
        reason = "closed"
        try:
            while True:
                raw = await self.websocket.receive_text()
                try:
                    message = json.loads(raw)
                except ValueError:
                    await self.send_error("bad_message", "Messages must be JSON.")
                    continue
                if not isinstance(message, dict):
                    await self.send_error("bad_message", "Messages must be JSON objects.")
                    continue
                kind = message.get("type")
                if kind == "end":
                    reason = "caller_hung_up"
                    return
                if kind != "user":
                    await self.send_error("bad_message", f"Unknown message type: {kind!r}.")
                    continue
                text = message.get("text")
                if not isinstance(text, str) or not text.strip():
                    await self.send_error("empty", "Type something to send.")
                    continue
                if len(text) > MAX_USER_CHARS:
                    await self.send_error("too_long", f"Keep a message under {MAX_USER_CHARS} characters.")
                    continue
                self.stream.put(TranscriptIn(text=text.strip(), provider="text"))
        except WebSocketDisconnect:
            self._open = False
            reason = "transport_closed"
        except RuntimeError:
            self._open = False
            reason = "transport_closed"
        finally:
            self.stream.put(Shutdown(reason))

    # -- outbound -----------------------------------------------------------
    async def _handle(self, command: Any) -> None:
        assert self.call is not None
        if isinstance(command, Speak):
            await self.send("say", turn_id=command.context_id, segment_id=command.segment_id,
                            text=command.text)

        elif isinstance(command, Emit):
            await self.writer.emit(self.call.call_id, command.event_type, command.payload,
                                   command.turn_id)
            self._absorb(command)
            if command.event_type == "turn_started":
                await self.send("turn_started", turn_id=command.turn_id,
                                user_text=command.payload.get("user_text", ""))
            await self.send("event", turn_id=command.turn_id, name=command.event_type,
                            payload=command.payload)

        elif isinstance(command, TurnCompleted):
            turn: TurnState = command.turn
            await self.store.save_turn(self.call, turn)
            # Flush per turn so the console's live feed shows this chat as it
            # happens, not when the tab closes.
            await self.writer.flush()
            await self.send("turn", turn_id=turn.turn_id, details=self._details(turn))

        elif isinstance(command, EndCall):
            if self.call.handoff:
                handoff = self.call.handoff
                await self.send(
                    "handoff",
                    summary_text=handoff.get("summary_text", ""),
                    reason=handoff.get("reason"),
                    rule_type=handoff.get("rule_type"),
                    summary=handoff.get("summary"),
                )
            await self.send("ended", outcome=self.call.outcome or command.reason,
                            reason=command.reason)

    def _absorb(self, command: Emit) -> None:
        """Collect tool and checker events for the turn's details panel."""
        turn_id = command.turn_id or ""
        payload = command.payload
        if command.event_type in _TOOL_STATUS:
            name = payload.get("tool") or payload.get("name")
            if not name:
                return
            tools = self._tools.setdefault(turn_id, {})
            entry = tools.setdefault(name, {"tool": name})
            entry["status"] = _TOOL_STATUS[command.event_type]
            for key in ("args", "arguments"):
                if isinstance(payload.get(key), dict):
                    entry["args"] = payload[key]
            for key in ("error", "attempt", "duplicate_suppressed", "reason"):
                if payload.get(key) not in (None, False):
                    entry[key] = payload[key]
        elif command.event_type == "checker_verdict" and payload.get("reason"):
            self._checker_reasons.setdefault(turn_id, {})[payload.get("segment_index", -1)] = (
                str(payload["reason"])
            )

    def _details(self, turn: TurnState) -> dict[str, Any]:
        assert self.call is not None
        cited = set(turn.cited_chunk_ids)
        headings = turn.retrieved_headings
        chunks = [
            {"rank": index + 1, "chunk_id": chunk_id,
             "heading": headings[index] if index < len(headings) else "",
             "cited": chunk_id in cited}
            for index, chunk_id in enumerate(turn.retrieved_chunk_ids)
        ]
        reasons = self._checker_reasons.pop(turn.turn_id, {})
        checker = [
            {"segment_index": v.segment_index, "text": v.segment_text, "verdict": v.verdict,
             "action": v.action_taken, "spans": len(v.spans), "checker_ms": v.checker_ms,
             "reason": reasons.get(v.segment_index, "")}
            for v in turn.checker_verdicts
        ]
        # Anything still pending after the turn is what the caller's next words
        # will be classified against — a fresh read-back or a re-ask.
        pending = self.call.pending_readback
        confirmation = (
            {"tool": pending.prepared.tool.name, "text": pending.text}
            if pending is not None else None
        )
        return {
            "stop_reason": turn.stop_reason,
            # Fixed fillers (acknowledgments, safe lines) are spoken but not
            # approved; the browser greys out segments missing from this.
            "approved_text": turn.approved_text,
            "llm_iterations": turn.llm_iterations,
            "retrieval": {
                "top_score": turn.retrieval_top_score,
                "top_vec_score": turn.retrieval_top_vec_score,
                "top_lex_score": turn.retrieval_top_lex_score,
                "chunks": chunks,
            },
            "checker": checker,
            "tools": list(self._tools.pop(turn.turn_id, {}).values()),
            "signals": list(turn.handoff_signals),
            "stages_ms": dict(turn.stages_ms),
            "ack_bypass_used": turn.ack_bypass_used,
            "budget_exceeded": turn.budget_exceeded,
            "awaiting_confirmation": confirmation,
        }
