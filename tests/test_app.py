"""W1-7 — the app boots and serves the client without the audio extra.

`make chat` and `/health` must work on a bare install. If importing app.main
required pipecat, a broken audio install would look like a broken app.
"""

from __future__ import annotations

from fastapi.testclient import TestClient

from app.main import app


def test_health_reports_the_active_configuration() -> None:
    with TestClient(app) as client:
        body = client.get("/health").json()

    assert body["status"] == "ok"
    assert body["domain"]
    assert "vad_stop_secs" in body, "the endpointing constant must be visible"


def test_browser_client_is_served() -> None:
    with TestClient(app) as client:
        assert "VoiceDesk" in client.get("/").text
        assert client.get("/static/client.js").status_code == 200
        assert client.get("/static/worklets.js").status_code == 200


def test_app_imports_without_the_audio_extra() -> None:
    import sys

    assert "pipecat" not in sys.modules, "pipecat must stay a lazy import"
