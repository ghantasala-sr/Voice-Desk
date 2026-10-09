"""Fixtures shared by tests/ and eval/fault_injection/."""

from __future__ import annotations

import pytest

from core.settings import Settings


@pytest.fixture
def settings() -> Settings:
    """Settings for tests: no network, no database, generous budget.

    Constructed directly rather than via get_settings() so a developer's local
    .env cannot change a test outcome.
    """
    return Settings(
        VD_LLM_PROVIDER="replay",
        VD_TTS_PROVIDER="null",
        VD_STT_PROVIDER="null",
        VD_DOMAIN="ticketing",
        VD_TURN_BUDGET_MS=5000,
        VD_TRACING_ENABLED=False,
        # These fixtures exercise LOOP MECHANICS with ungrounded fixture text
        # ("Refunds take five to seven business days" with no chunks behind
        # it), which the checker would correctly reject. Checker behaviour is
        # tested against real corpus chunks in test_checker.py, and its
        # integration with the loop in test_grounded_loop.py — both with the
        # checker ON, as it is in production.
        VD_CHECKER_ENABLED=False,
        VD_HANDOFF_ENABLED=False,
    )
