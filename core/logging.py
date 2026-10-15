"""Structured logging.

`call_id` and `turn_id` are bound into a context variable, so every log line
inside a turn carries them without being threaded through every function.

Logs are for development. No number in `results/` may be derived from a log
line — that is what the tables are for (docs/11 §5).
"""

from __future__ import annotations

import logging
import sys
from typing import Any

import structlog


def configure_logging(*, level: str = "INFO", json_output: bool = False) -> None:
    """Call once at process start."""
    logging.basicConfig(format="%(message)s", stream=sys.stdout, level=level.upper())
    # google-genai routes requests through httpx, which logs every request —
    # including HuggingFace's model-cache probes — at INFO. That is noise in a
    # transcript, not signal.
    for noisy in ("httpx", "httpcore", "huggingface_hub", "google_genai"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    processors: list[Any] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]
    processors.append(
        structlog.processors.JSONRenderer()
        if json_output
        else structlog.dev.ConsoleRenderer(colors=sys.stdout.isatty())
    )

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(
            logging.getLevelNamesMapping()[level.upper()]
        ),
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str) -> Any:
    return structlog.get_logger(name)


def bind_call(call_id: str, turn_id: str | None = None) -> None:
    """Bind ids for the current async task. Cleared by `clear_context()`."""
    structlog.contextvars.bind_contextvars(call_id=call_id, turn_id=turn_id)


def clear_context() -> None:
    structlog.contextvars.clear_contextvars()
