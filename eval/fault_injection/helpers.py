"""Building a TextDriver wired to the mock backend, for loop-level tests."""

from __future__ import annotations

from typing import Any

from core.settings import Settings
from eval.text_driver import TextDriver
from harness.retriever import Retriever
from harness.tools.execution import InMemoryLedger, ToolExecutor
from harness.tools.mocks.faults import PROFILES, FaultyBackend
from harness.tools.mocks.ticketing import MockTicketing
from harness.tools.registry import load_registry
from providers.llm.replay import ReplayProvider

TRANSFER_ARGS = {"order_id": "TK-88213", "quantity": 2, "recipient_email": "friend@example.com"}


def tool_call(name: str, args: dict[str, Any], call_id: str | None = None) -> dict[str, Any]:
    return {"tool_call": {"name": name, "arguments": args, "call_id": call_id or f"tu_{name}"}}


def end(reason: str = "end_turn") -> dict[str, Any]:
    return {"end": {"stop_reason": reason}}


def say(text: str, delay_ms: int = 0) -> dict[str, Any]:
    step: dict[str, Any] = {"text": text}
    if delay_ms:
        step["delay_ms"] = delay_ms
    return step


def tool_driver(
    settings: Settings,
    events: list[dict[str, Any]],
    *,
    profile: str = "benign",
    verified: bool = True,
    retriever: Retriever | None = None,
    **overrides: Any,
) -> tuple[TextDriver, MockTicketing, InMemoryLedger]:
    backend = MockTicketing()
    ledger = InMemoryLedger()
    executor = ToolExecutor(
        registry=load_registry("ticketing"),
        backend=FaultyBackend(backend, PROFILES[profile]),
        ledger=ledger,
    )
    driver = TextDriver(
        llm=ReplayProvider(events=events),
        settings=settings.model_copy(update=overrides),
        executor=executor,
        retriever=retriever,
    )
    if verified:
        driver.call.caller_verified = True
        driver.call.order_id = "TK-88213"
    return driver, backend, ledger
