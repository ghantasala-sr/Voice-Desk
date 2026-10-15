"""The web app: shell routes, the read-only APIs, and the text-chat socket.

The chat tests drive `/ws/chat` end to end through the real Harness with the
offline extractive provider and scripted chunks, so they need neither
Postgres, the embedding model, nor an API key. The stack factory and the
persistence choice are FastAPI dependencies precisely so these tests can
swap them (see app/chat.py).
"""

from __future__ import annotations

import json
from typing import Any, get_args

import pytest
from fastapi.testclient import TestClient

from app import chat
from app.insights import RESULTS, headline_numbers, readme_outline
from app.main import app
from core.settings import LLMProviderName, Settings
from harness.assembly import build_stack

REFUND_CHUNK = (
    "If you can't make the event, you can get a refund up to seven days before it. "
    "Within seven days of the event, tickets can be transferred but not refunded."
)


def _db_reachable() -> bool:
    from core.db import is_reachable

    return is_reachable(Settings().database_url)


db = pytest.mark.skipif(not _db_reachable(), reason="Postgres not reachable; run `make up`")


# ---------------------------------------------------------------------------
# Shell routes
# ---------------------------------------------------------------------------
@pytest.fixture
def client() -> Any:
    with TestClient(app) as test_client:
        yield test_client


def test_root_serves_the_app_shell(client: TestClient) -> None:
    body = client.get("/").text
    assert "VoiceDesk" in body
    assert "/static/app/app.js" in body
    assert client.head("/").status_code == 200


def test_voice_client_moved_to_voice(client: TestClient) -> None:
    response = client.get("/voice")
    assert response.status_code == 200
    assert "/static/client.js" in response.text
    assert client.head("/voice").status_code == 200


def test_console_and_health_still_work(client: TestClient) -> None:
    assert "console" in client.get("/console").text
    assert client.get("/health").json()["status"] == "ok"


@pytest.mark.parametrize(
    "path", ["app/app.js", "app/app.css", "app/markdown.js", "app/charts.js",
             "client.js", "console/console.js"],
)
def test_static_assets_load(client: TestClient, path: str) -> None:
    assert client.get(f"/static/{path}").status_code == 200


def test_app_never_assigns_inner_html() -> None:
    """Transcripts, chunk text and READMEs are untrusted: DOM APIs only."""
    from app.main import STATIC

    for name in ("app.js", "markdown.js", "charts.js"):
        source = (STATIC / "app" / name).read_text()
        assert "innerHTML" not in source.replace("Nothing here assigns innerHTML", "")
        assert "insertAdjacentHTML" not in source
        assert "outerHTML" not in source


# ---------------------------------------------------------------------------
# /api/status
# ---------------------------------------------------------------------------
def test_status_shape(client: TestClient) -> None:
    body = client.get("/api/status").json()
    assert set(body) >= {"agent", "keys", "voice", "database", "chunks", "domain"}
    assert set(body["keys"]) >= {"anthropic", "deepgram", "cartesia"}
    assert all(isinstance(v, bool) for v in body["keys"].values())
    assert isinstance(body["voice"]["ready"], bool)
    assert body["agent"]["provider"] in get_args(LLMProviderName)


def test_status_never_contains_key_material(monkeypatch: pytest.MonkeyPatch) -> None:
    secrets = {
        "ANTHROPIC_API_KEY": "sk-ant-test-SENTINEL-1111",
        "DEEPGRAM_API_KEY": "dg-SENTINEL-2222",
        "CARTESIA_API_KEY": "ca-SENTINEL-3333",
    }
    for name, value in secrets.items():
        monkeypatch.setenv(name, value)
    from core.settings import get_settings

    get_settings.cache_clear()
    try:
        with TestClient(app) as test_client:
            raw = test_client.get("/api/status").text
        body = json.loads(raw)
        assert "SENTINEL" not in raw
        assert body["keys"]["anthropic"] is True
        assert body["agent"]["offline"] is False
    finally:
        get_settings.cache_clear()


def test_offline_agent_when_no_anthropic_key() -> None:
    settings, agent = chat.agent_settings(Settings(VD_LLM_PROVIDER="anthropic", ANTHROPIC_API_KEY=""))
    assert settings.llm_provider == "extractive"
    assert agent["offline"] is True
    assert agent["notice"] == chat.OFFLINE_NOTICE


# ---------------------------------------------------------------------------
# /api/results
# ---------------------------------------------------------------------------
def test_results_lists_real_directories(client: TestClient) -> None:
    runs = client.get("/api/results").json()["runs"]
    names = {r["name"] for r in runs}
    on_disk = {p.name for p in RESULTS.iterdir() if p.is_dir()}
    assert names == on_disk
    for run in runs:
        for number in run["headline"]:
            assert "n" in number, "every headline number carries its n"


def test_result_detail_returns_readme_and_metrics(client: TestClient) -> None:
    run = next(p.name for p in sorted(RESULTS.iterdir()) if (p / "README.md").is_file())
    body = client.get(f"/api/results/{run}").json()
    assert body["readme"].startswith("#")
    assert isinstance(body["pngs"], list)


@pytest.mark.parametrize(
    "name",
    ["..", "%2e%2e", "..%2f..%2fcore", "%2e%2e%2fREADME.md", "README.md",
     "no-such-run", ".hidden", "..%5c..%5cetc"],
)
def test_result_detail_rejects_traversal(client: TestClient, name: str) -> None:
    response = client.get(f"/api/results/{name}")
    assert response.status_code == 404
    assert "error" in response.json() or response.status_code == 404


def test_result_files_are_limited_to_pngs_in_the_run(client: TestClient) -> None:
    with_png = next((p for p in RESULTS.iterdir() if p.is_dir() and list(p.glob("*.png"))), None)
    if with_png is not None:
        png = next(with_png.glob("*.png")).name
        response = client.get(f"/api/results/{with_png.name}/files/{png}")
        assert response.status_code == 200
        assert response.headers["content-type"] == "image/png"
    run = next(p.name for p in RESULTS.iterdir() if p.is_dir())
    for bad in ["metrics.json", "README.md", "..%2f..%2fpyproject.toml", "%2e%2e", "x.png"]:
        assert client.get(f"/api/results/{run}/files/{bad}").status_code == 404
    assert client.get("/api/results/../files/x.png").status_code == 404


def test_readme_outline_and_headline_numbers() -> None:
    title, summary = readme_outline(
        "# My run\n\nintro\n\n## 1. What was this run for\n\nwhy\n\n"
        "## 2. What it showed\n\n| a | b |\n|---|---|\n| 1 | 2 |\n\n**Recall** is `0.9`.\n\n## 3. Next\n"
    )
    assert title == "My run"
    assert summary == "Recall is 0.9."
    numbers = headline_numbers(
        {"run": "x", "deep": {"deeper": {"r": {"value": 0.5, "n": 4, "successes": 2}}},
         "top": {"value": 0.9, "n": 10, "successes": 9, "ci95": [0.6, 0.98]}}
    )
    assert [n["value"] for n in numbers] == [0.9, 0.5], "shallowest first"
    assert numbers[0]["n"] == 10


def test_feedback_cycle_latest_is_404_or_valid(client: TestClient) -> None:
    response = client.get("/api/feedback-cycle/latest")
    assert response.status_code in (200, 404)
    if response.status_code == 200:
        body = response.json()
        assert "-feedback-cycle" in body["run"] and "superseded" not in body["run"]
        assert isinstance(body["metrics"], dict)


# ---------------------------------------------------------------------------
# /api/gap-report/latest
# ---------------------------------------------------------------------------
def test_gap_report_without_database_is_503() -> None:
    with TestClient(app) as test_client:
        app.state.db_ready = False
        response = test_client.get("/api/gap-report/latest")
    assert response.status_code == 503
    assert response.json() == {"error": "database unavailable"}


@db
@pytest.mark.db
def test_gap_report_latest_is_404_or_valid(client: TestClient) -> None:
    response = client.get("/api/gap-report/latest")
    assert response.status_code in (200, 404)
    body = response.json()
    if response.status_code == 404:
        assert "error" in body
        return
    assert set(body) >= {"report_id", "generated_at", "window_start", "window_end",
                         "call_count", "clusters", "actions"}
    assert isinstance(body["clusters"], list)
    assert isinstance(body["actions"], list)


# ---------------------------------------------------------------------------
# /ws/chat
# ---------------------------------------------------------------------------
@pytest.fixture
def chat_client() -> Any:
    """A chat socket on the DB-free path: extractive LLM, scripted chunks,
    tools against the in-memory mock backend, nothing persisted."""
    settings = Settings(
        VD_LLM_PROVIDER="extractive", VD_TTS_PROVIDER="null", VD_STT_PROVIDER="null",
        VD_DOMAIN="ticketing", VD_TURN_BUDGET_MS=5000, VD_TRACING_ENABLED=False,
    )

    async def stack(local: Settings, persist: bool) -> Any:
        assert persist is False
        return await build_stack(local, domain=local.domain, retrieval=[REFUND_CHUNK], tools=True)

    app.dependency_overrides[chat.get_chat_settings] = lambda: settings
    app.dependency_overrides[chat.get_stack_factory] = lambda: stack
    app.dependency_overrides[chat.get_persist] = lambda: False
    try:
        with TestClient(app) as test_client:
            yield test_client
    finally:
        app.dependency_overrides.clear()


def _until(ws: Any, kind: str, limit: int = 200) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    seen: list[dict[str, Any]] = []
    for _ in range(limit):
        message = ws.receive_json()
        seen.append(message)
        if message["type"] == kind:
            return message, seen
        assert message["type"] != "error", message
    raise AssertionError(f"no {kind!r} message in {len(seen)}")


def test_chat_round_trip(chat_client: TestClient) -> None:
    with chat_client.websocket_connect("/ws/chat") as ws:
        ready, _ = _until(ws, "ready")
        assert ready["agent"]["provider"] == "extractive"
        assert ready["recorded"] is False

        ws.send_json({"type": "user", "text": "Can I get a refund if I can't make it?"})
        turn, seen = _until(ws, "turn")

        said = " ".join(m["text"] for m in seen if m["type"] == "say")
        assert "refund" in said.lower()
        assert any(m["type"] == "event" for m in seen)
        details = turn["details"]
        assert details["stop_reason"] == "answered"
        assert details["retrieval"]["chunks"][0]["heading"] == "Script knowledge"
        assert details["checker"], "every model-written segment is checked"
        assert all(v["action"] == "released" for v in details["checker"])
        assert "retrieval" in details["stages_ms"]

        ws.send_json({"type": "end"})
        ended, _ = _until(ws, "ended")
        assert ended["outcome"] == "contained"


def test_chat_transfer_needs_the_callers_own_yes(chat_client: TestClient) -> None:
    with chat_client.websocket_connect("/ws/chat") as ws:
        _until(ws, "ready")
        ws.send_json({"type": "user", "text": "I want to send two tickets to friend@example.com"})
        _until(ws, "turn")

        ws.send_json({"type": "user", "text": "It's order T K 8 8 2 1 3, sam at example dot com"})
        turn, _ = _until(ws, "turn")
        confirmation = turn["details"]["awaiting_confirmation"]
        assert confirmation and confirmation["tool"] == "transfer_tickets"
        assert not any(t["status"] == "committed" and t["tool"] == "transfer_tickets"
                       for t in turn["details"]["tools"])

        ws.send_json({"type": "user", "text": "yes"})
        turn, _ = _until(ws, "turn")
        tools = {t["tool"]: t for t in turn["details"]["tools"]}
        assert tools["transfer_tickets"]["status"] == "committed"
        assert turn["details"]["awaiting_confirmation"] is None


def test_chat_handoff_sends_the_summary_and_ends(chat_client: TestClient) -> None:
    with chat_client.websocket_connect("/ws/chat") as ws:
        _until(ws, "ready")
        ws.send_json({"type": "user", "text": "let me talk to a manager"})
        handoff, _ = _until(ws, "handoff")
        assert handoff["summary_text"].startswith("HANDOFF")
        ended, _ = _until(ws, "ended")
        assert ended["outcome"] == "handoff"


def test_chat_rejects_bad_messages_with_typed_errors(chat_client: TestClient) -> None:
    with chat_client.websocket_connect("/ws/chat") as ws:
        _until(ws, "ready")
        ws.send_text("not json")
        assert _until_error(ws)[0] == {"type": "error", "code": "bad_message",
                                       "message": "Messages must be JSON."}
        ws.send_json({"type": "user", "text": "   "})
        assert _until_error(ws)[0]["code"] == "empty"
        ws.send_json({"type": "user", "text": "x" * (chat.MAX_USER_CHARS + 1)})
        assert _until_error(ws)[0]["code"] == "too_long"


def test_chat_internal_failure_is_a_typed_error_not_a_trace() -> None:
    async def broken(local: Settings, persist: bool) -> Any:
        raise RuntimeError("secret internals /path/to/file.py line 12")

    app.dependency_overrides[chat.get_stack_factory] = lambda: broken
    app.dependency_overrides[chat.get_persist] = lambda: False
    try:
        with TestClient(app) as test_client, test_client.websocket_connect("/ws/chat") as ws:
            error, _ = _until_error(ws)
    finally:
        app.dependency_overrides.clear()
    assert error["code"] == "internal"
    assert "secret internals" not in json.dumps(error)


def _until_error(ws: Any) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    seen = []
    for _ in range(20):
        message = ws.receive_json()
        seen.append(message)
        if message["type"] == "error":
            return message, seen
    raise AssertionError(seen)
