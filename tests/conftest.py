from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from core.settings import Settings
from eval.text_driver import TextDriver
from providers.llm.replay import ReplayProvider

FIXTURES = Path(__file__).parent / "fixtures" / "llm"


@pytest.fixture
def make_driver(settings: Settings):
    """Build a TextDriver over an inline replay fixture."""

    def _make(events: list[dict[str, Any]], **overrides: Any) -> TextDriver:
        # model_copy takes FIELD names (turn_budget_ms), not the env aliases
        # (VD_TURN_BUDGET_MS) that the constructor accepts. Passing an alias
        # here would silently attach a stray attribute and leave the real
        # field at its default.
        if overrides:
            unknown = set(overrides) - set(type(settings).model_fields)
            assert not unknown, f"not Settings fields: {sorted(unknown)}"
            local = settings.model_copy(update=overrides)
        else:
            local = settings
        return TextDriver(llm=ReplayProvider(events=events), settings=local)

    return _make
