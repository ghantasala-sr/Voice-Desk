"""FastAPI app: the browser client, the WebSocket endpoint, and health.

Run with `make dev`, then open http://localhost:$(APP_PORT) (8088 here): `/` is the
web app, `/voice` the original browser voice client, `/console` the live console.

The WebSocket handler is thin on purpose. It opens a CallState, starts the
Pipecat pipeline, and forwards harness commands to the browser. All decisions
happen in `harness/`.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.chat import router as chat_router
from app.insights import router as insights_router
from app.observe import router as observe_router
from core import db
from core.logging import configure_logging, get_logger
from core.settings import get_settings
from core.tracing import configure_tracing

STATIC = Path(__file__).parent / "static"
log = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> Any:
    settings = get_settings()
    configure_logging(level=settings.log_level, json_output=settings.log_json)
    configure_tracing(settings)

    # The database is optional for a demo: a call still works without it, it
    # just is not recorded. Failing to boot because Postgres is down would
    # make the demo more fragile than it needs to be.
    #
    # Probe before opening the pool. Opening it against a dead port blocks for
    # the full pool timeout, which turns `make dev` into a ten-second wait and
    # makes every app test slow.
    app.state.db_ready = False
    if db.is_reachable(settings.database_url):
        try:
            await db.open_pool(settings)
            app.state.db_ready = True
        except Exception as exc:
            log.warning("database_unavailable", detail=str(exc))
    else:
        log.warning("database_unavailable", detail="not reachable; calls will not be recorded")

    app.state.settings = settings
    yield
    await db.close_pool()


app = FastAPI(title="VoiceDesk", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=STATIC), name="static")
app.include_router(observe_router)
app.include_router(chat_router)
app.include_router(insights_router)


@app.api_route("/health", methods=["GET", "HEAD"])
async def health() -> dict[str, Any]:
    settings = get_settings()
    return {
        "status": "ok",
        "domain": settings.domain,
        "llm": f"{settings.llm_provider}:{settings.llm_model}",
        "stt": settings.stt_provider,
        "tts": settings.tts_provider,
        "database": getattr(app.state, "db_ready", False),
        "vad_stop_secs": settings.vad_stop_secs,
    }


# HEAD too: preview tools and load balancers health-check with HEAD.
@app.api_route("/", methods=["GET", "HEAD"])
async def index() -> FileResponse:
    return FileResponse(STATIC / "app" / "index.html")


@app.api_route("/voice", methods=["GET", "HEAD"])
async def voice() -> FileResponse:
    return FileResponse(STATIC / "index.html")


@app.get("/console")
async def console() -> FileResponse:
    return FileResponse(STATIC / "console" / "index.html")


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    """Audio in, audio out, plus JSON control messages (docs/10 §4)."""
    await websocket.accept()

    # Imported here, not at module scope: pipecat is an optional extra, and
    # /health must work without it so `make dev` can diagnose a bad install.
    from app.pipeline import run_call

    try:
        await run_call(websocket, settings=get_settings())
    except WebSocketDisconnect:
        log.info("client_disconnected")
    except Exception:
        log.exception("call_failed")
        await websocket.close(code=1011)
