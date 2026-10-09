"""OpenAI implementation of LLMProvider.

This exists so the interface stays honest. `tests/test_provider_parity.py`
drives one fixed scenario through both providers and asserts the same tool was
called with the same arguments — it does NOT assert identical prose, which
would be testing the model rather than the interface.

Shape differences from Anthropic, all absorbed here:

  * No separate system/cache split — the system prompt is one message, and
    caching is automatic on a long enough prefix. We still concatenate the
    static half first so the prefix is stable.
  * Tool-call fragments arrive indexed inside `delta.tool_calls` and are
    accumulated the same way.
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


class OpenAIProvider(LLMProvider):
    name = "openai"

    def __init__(self, settings: Settings) -> None:
        from openai import AsyncOpenAI

        self.model = settings.llm_model
        self._client = AsyncOpenAI(api_key=settings.require("openai_api_key"))

    async def aclose(self) -> None:
        await self._client.close()

    async def stream(self, request: LLMRequest) -> AsyncIterator[LLMEvent]:
        system = request.system_static
        if request.system_dynamic:
            system = f"{system}\n\n{request.system_dynamic}"

        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": request.max_tokens,
            "temperature": request.temperature,
            "messages": [{"role": "system", "content": system},
                         *_to_openai_messages(request.messages)],
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        if request.tools:
            kwargs["tools"] = [_tool_schema(t) for t in request.tools]
            kwargs["tool_choice"] = "auto"

        pending: dict[int, dict[str, str]] = {}
        input_tokens = output_tokens = 0
        stop_reason = "end_turn"

        try:
            stream = await self._client.chat.completions.create(
                timeout=request.timeout_ms / 1000, **kwargs
            )
            async for chunk in stream:
                if chunk.usage:
                    input_tokens = chunk.usage.prompt_tokens or 0
                    output_tokens = chunk.usage.completion_tokens or 0

                if not chunk.choices:
                    continue
                choice = chunk.choices[0]

                if choice.delta and choice.delta.content:
                    yield TextDelta(text=choice.delta.content)

                for call in choice.delta.tool_calls or []:
                    slot = pending.setdefault(call.index, {"id": "", "name": "", "args": ""})
                    if call.id:
                        slot["id"] = call.id
                    if call.function and call.function.name:
                        slot["name"] = call.function.name
                    if call.function and call.function.arguments:
                        slot["args"] += call.function.arguments

                if choice.finish_reason:
                    stop_reason = (
                        "tool_use"
                        if choice.finish_reason == "tool_calls"
                        else "max_tokens"
                        if choice.finish_reason == "length"
                        else "end_turn"
                    )

        except Exception as exc:
            yield StreamEnd(
                stop_reason="error",
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                error=f"{type(exc).__name__}: {exc}",
            )
            return

        # Tool calls are only complete at the end of an OpenAI stream, so they
        # are emitted here rather than inline.
        for slot in pending.values():
            try:
                arguments = json.loads(slot["args"] or "{}")
            except json.JSONDecodeError as exc:
                yield StreamEnd(
                    stop_reason="error",
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    error=f"tool arguments did not parse: {exc}",
                )
                return
            yield ToolCallDelta(call_id=slot["id"], name=slot["name"], arguments=arguments)

        yield StreamEnd(
            stop_reason=stop_reason, input_tokens=input_tokens, output_tokens=output_tokens
        )


def _tool_schema(tool: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": tool["name"],
            "description": tool.get("description", ""),
            "parameters": tool["parameters"],
        },
    }


def _to_openai_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Neutral block format -> Chat Completions.

    The harness writes Anthropic-shaped blocks (tool_use inside an assistant
    message, tool_result inside a user message). Chat Completions wants
    `tool_calls` on the assistant message and one `role: tool` message per
    result. Without this translation the second provider would only work for
    conversations that never call a tool — which is not an interface.
    """
    out: list[dict[str, Any]] = []
    for message in messages:
        blocks = message.get("content", [])
        if isinstance(blocks, str):
            out.append(message)
            continue
        text = "".join(b.get("text", "") for b in blocks if b.get("type") == "text")
        if message["role"] == "assistant":
            calls = [
                {"id": b["id"], "type": "function",
                 "function": {"name": b["name"], "arguments": json.dumps(b.get("input", {}))}}
                for b in blocks if b.get("type") == "tool_use"
            ]
            entry: dict[str, Any] = {"role": "assistant", "content": text or None}
            if calls:
                entry["tool_calls"] = calls
            out.append(entry)
            continue
        for block in blocks:
            if block.get("type") == "tool_result":
                out.append({"role": "tool", "tool_call_id": block["tool_use_id"],
                            "content": str(block.get("content", ""))})
        if text:
            out.append({"role": "user", "content": text})
    return out
