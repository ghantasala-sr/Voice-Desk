"""Anthropic implementation of LLMProvider.

Two details worth reading:

* **Prompt caching.** The system prompt is sent as two blocks, with
  `cache_control` on the static one. That block is byte-identical on every turn
  of every call, so after the first turn it is a cache read rather than a fresh
  prefill — which is time-to-first-token we do not pay. `cache_read_tokens` is
  reported so the saving is verifiable rather than assumed.

* **Tool-call accumulation.** Tool arguments stream as incremental JSON
  fragments (`input_json_delta`). We buffer them per content-block index and
  emit a ToolCallDelta only when the accumulated string parses. A fragment that
  never parses becomes StreamEnd(stop_reason="error") — the harness never sees
  half a tool call.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

from core.settings import Settings
from providers.llm.base import (
    LLMEvent,
    LLMProvider,
    LLMRequest,
    StreamEnd,
    TextDelta,
    ToolCallDelta,
)


class AnthropicProvider(LLMProvider):
    name = "anthropic"

    def __init__(self, settings: Settings) -> None:
        from anthropic import AsyncAnthropic

        self.model = settings.llm_model
        self._client = AsyncAnthropic(api_key=settings.require("anthropic_api_key"))

    async def aclose(self) -> None:
        await self._client.close()

    async def stream(self, request: LLMRequest) -> AsyncIterator[LLMEvent]:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": request.max_tokens,
            "temperature": request.temperature,
            "system": _system_blocks(request),
            "messages": request.messages,
            "stream": True,
        }
        if request.tools:
            kwargs["tools"] = [_tool_schema(t) for t in request.tools]

        # Per-block accumulators for streamed tool arguments.
        partial_json: dict[int, str] = {}
        tool_meta: dict[int, dict[str, str]] = {}

        input_tokens = output_tokens = cache_read = 0
        stop_reason = "end_turn"

        try:
            stream = await self._client.messages.create(
                timeout=request.timeout_ms / 1000, **kwargs
            )
            async for event in stream:
                etype = event.type

                if etype == "message_start":
                    usage = event.message.usage
                    input_tokens = getattr(usage, "input_tokens", 0) or 0
                    cache_read = getattr(usage, "cache_read_input_tokens", 0) or 0

                elif etype == "content_block_start":
                    block = event.content_block
                    if block.type == "tool_use":
                        partial_json[event.index] = ""
                        tool_meta[event.index] = {"id": block.id, "name": block.name}

                elif etype == "content_block_delta":
                    delta = event.delta
                    if delta.type == "text_delta":
                        yield TextDelta(text=delta.text)
                    elif delta.type == "input_json_delta":
                        partial_json[event.index] = (
                            partial_json.get(event.index, "") + delta.partial_json
                        )

                elif etype == "content_block_stop":
                    if event.index in tool_meta:
                        meta = tool_meta.pop(event.index)
                        raw = partial_json.pop(event.index, "") or "{}"
                        try:
                            arguments = json.loads(raw)
                        except json.JSONDecodeError as exc:
                            yield StreamEnd(
                                stop_reason="error",
                                input_tokens=input_tokens,
                                output_tokens=output_tokens,
                                cache_read_tokens=cache_read,
                                error=f"tool arguments did not parse: {exc}",
                            )
                            return
                        yield ToolCallDelta(
                            call_id=meta["id"], name=meta["name"], arguments=arguments
                        )

                elif etype == "message_delta":
                    stop_reason = event.delta.stop_reason or stop_reason
                    output_tokens = getattr(event.usage, "output_tokens", 0) or output_tokens

        except Exception as exc:  # network, auth, rate limit, timeout
            yield StreamEnd(
                stop_reason="error",
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cache_read_tokens=cache_read,
                error=f"{type(exc).__name__}: {exc}",
            )
            return

        yield StreamEnd(
            stop_reason=_normalize_stop(stop_reason),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cache_read_tokens=cache_read,
        )


def _system_blocks(request: LLMRequest) -> list[dict[str, Any]]:
    """Static block first, marked cacheable. Dynamic block after, never cached."""
    blocks: list[dict[str, Any]] = [
        {
            "type": "text",
            "text": request.system_static,
            "cache_control": {"type": "ephemeral"},
        }
    ]
    if request.system_dynamic:
        blocks.append({"type": "text", "text": request.system_dynamic})
    return blocks


def _tool_schema(tool: dict[str, Any]) -> dict[str, Any]:
    """Neutral tool dict -> Anthropic shape."""
    return {
        "name": tool["name"],
        "description": tool.get("description", ""),
        "input_schema": tool["parameters"],
    }


def _normalize_stop(reason: str) -> str:
    return {"max_tokens": "max_tokens", "tool_use": "tool_use"}.get(reason, "end_turn")
