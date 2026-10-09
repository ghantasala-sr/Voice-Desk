"""The LLM provider interface.

The hard part is that streaming and tool calling interact: a model may emit
text, then a tool call, then more text, and the harness must be able to cancel
mid-stream on barge-in without leaking the upstream connection.

Four rules every implementation obeys (docs/04 §1):

  1. Partial tool-call JSON never escapes. Providers accumulate the streamed
     argument fragments and emit a ToolCallDelta only once they parse.
  2. Cancellation is real. Closing the iterator terminates the HTTP connection.
     Barge-in correctness depends on it, and a leaked stream shows up as
     mysterious latency three turns later.
  3. Token counts are always reported, including on error — cost per call goes
     in the results table.
  4. No retries inside the provider. Retry policy belongs to the harness,
     because only the harness knows the turn's remaining latency budget.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any, Literal

StopReason = Literal["end_turn", "tool_use", "max_tokens", "cancelled", "error"]


@dataclass(frozen=True)
class TextDelta:
    """A chunk of assistant text. May be a fragment of a word."""

    text: str


@dataclass(frozen=True)
class ToolCallDelta:
    """A COMPLETE tool call.

    Named "delta" because it arrives mid-stream, but the arguments are whole
    and parsed. The harness never sees half a tool call.
    """

    call_id: str
    name: str
    arguments: dict[str, Any]


@dataclass(frozen=True)
class StreamEnd:
    stop_reason: StopReason
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    error: str | None = None


LLMEvent = TextDelta | ToolCallDelta | StreamEnd


@dataclass(frozen=True)
class LLMRequest:
    """A provider-neutral request.

    `system_static` and `system_dynamic` are split at the prompt-cache
    breakpoint. The static half (persona, guardrails, tool descriptions) is
    byte-identical on every turn of every call and is marked cacheable; the
    dynamic half (retrieved chunks, verification state) is not.

    If cache_read_tokens stays zero on turn 2, the breakpoint is misplaced and
    the latency work is being wasted — a bug that is otherwise invisible.
    """

    system_static: str
    system_dynamic: str = ""
    messages: list[dict[str, Any]] = field(default_factory=list)
    tools: list[dict[str, Any]] = field(default_factory=list)
    max_tokens: int = 512
    temperature: float = 0.2
    timeout_ms: int = 10_000


class LLMProvider(ABC):
    name: str
    model: str

    @abstractmethod
    def stream(self, request: LLMRequest) -> AsyncIterator[LLMEvent]:
        """Yield events until StreamEnd.

        Must be cancellable: on `asyncio.CancelledError` the provider closes
        the upstream connection and does not swallow the exception.
        """

    async def aclose(self) -> None:
        """Release any long-lived client. Safe to call more than once."""
        return None
