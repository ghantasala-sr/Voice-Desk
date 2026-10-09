"""Shared fixtures for the fault-injection suite. docs/12 §7.

Every profile is seeded; the seed appears in the profile name in failures, so
a red run reproduces from the report alone.
"""

from __future__ import annotations

import pytest

from harness.state import CallState
from harness.tools.execution import InMemoryLedger, ToolExecutor
from harness.tools.mocks.faults import PROFILES, FaultyBackend
from harness.tools.mocks.ticketing import MockTicketing
from harness.tools.registry import load_registry

TRANSFER = {"order_id": "TK-88213", "quantity": 2, "recipient_email": "friend@example.com"}


@pytest.fixture
def backend() -> MockTicketing:
    return MockTicketing()


@pytest.fixture
def ledger() -> InMemoryLedger:
    return InMemoryLedger()


@pytest.fixture
def verified_call() -> CallState:
    call = CallState(call_id="call-faults", domain_config="ticketing", source="text")
    call.caller_verified = True
    call.order_id = "TK-88213"
    return call


@pytest.fixture
def executor_for(backend: MockTicketing, ledger: InMemoryLedger):
    def _make(profile: str = "benign") -> ToolExecutor:
        return ToolExecutor(
            registry=load_registry("ticketing"),
            backend=FaultyBackend(backend, PROFILES[profile]),
            ledger=ledger,
        )

    return _make
