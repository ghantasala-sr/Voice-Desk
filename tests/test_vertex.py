"""Vertex AI providers, with a fake client — no network, no credentials.

The live path was verified separately against project garden-app-sg26
(gemini-2.5-flash on the global endpoint); see WEEK4.md.
"""

from __future__ import annotations

from typing import Any

import pytest

from core.settings import Settings
from providers import effective_model
from providers.llm.base import LLMRequest, StreamEnd, TextDelta, ToolCallDelta
from providers.llm.vertex import VertexGeminiProvider, to_gemini_contents

types = pytest.importorskip("google.genai.types")

TOOLS = [{"name": "lookup_order", "description": "Look up an order.",
          "parameters": {"type": "object", "properties": {"order_id": {"type": "string"}},
                         "required": ["order_id"]}}]


def chunk(*parts: Any, finish: str | None = None, usage: tuple[int, int] | None = None) -> Any:
    return types.GenerateContentResponse(
        candidates=[types.Candidate(content=types.Content(role="model", parts=list(parts)),
                                    finish_reason=finish)],
        usage_metadata=types.GenerateContentResponseUsageMetadata(
            prompt_token_count=usage[0], candidates_token_count=usage[1]) if usage else None,
    )


class FakeStream:
    def __init__(self, chunks: list[Any]) -> None:
        self._chunks = iter(chunks)
        self.closed = False

    def __aiter__(self) -> FakeStream:
        return self

    async def __anext__(self) -> Any:
        try:
            return next(self._chunks)
        except StopIteration:
            raise StopAsyncIteration from None

    async def aclose(self) -> None:
        self.closed = True


class FakeClient:
    """Mimics `genai.Client(...).aio.models.generate_content_stream`."""

    def __init__(self, *responses: list[Any] | Exception) -> None:
        self.responses = list(responses)
        self.requests: list[dict[str, Any]] = []
        self.streams: list[FakeStream] = []
        self.aio = self
        self.models = self

    async def generate_content_stream(self, **kwargs: Any) -> FakeStream:
        self.requests.append(kwargs)
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        self.streams.append(FakeStream(response))
        return self.streams[-1]


def provider(*responses: list[Any] | Exception) -> tuple[VertexGeminiProvider, FakeClient]:
    client = FakeClient(*responses)
    settings = Settings(VD_LLM_PROVIDER="vertex_gemini")
    return VertexGeminiProvider(settings, client=client), client


async def collect(p: VertexGeminiProvider, request: LLMRequest) -> list[Any]:
    return [e async for e in p.stream(request)]


def test_a_claude_model_id_means_gemini_default() -> None:
    assert effective_model(Settings(VD_LLM_PROVIDER="vertex_gemini")) == "gemini-2.5-flash"
    assert effective_model(Settings(VD_LLM_PROVIDER="vertex_gemini",
                                    VD_LLM_MODEL="gemini-2.5-pro")) == "gemini-2.5-pro"


async def test_text_streams_and_usage_is_reported() -> None:
    p, client = provider([chunk(types.Part(text="Transfers are ")),
                          chunk(types.Part(text="free."), finish="STOP", usage=(120, 6))])
    events = await collect(p, LLMRequest(system_static="S", system_dynamic="D",
                                         messages=[{"role": "user", "content": [{"type": "text", "text": "hi"}]}]))
    assert [e.text for e in events if isinstance(e, TextDelta)] == ["Transfers are ", "free."]
    assert events[-1] == StreamEnd(stop_reason="end_turn", input_tokens=120, output_tokens=6)
    config = client.requests[0]["config"]
    assert config.system_instruction == "S\n\nD", "static half first, for a stable prefix"
    assert config.automatic_function_calling.disable, "the harness owns the tool loop"
    assert config.http_options.timeout == 10_000, "the request deadline reaches the HTTP layer"
    assert client.streams[0].closed


async def test_tool_call_and_signature_round_trip() -> None:
    call = types.Part(function_call=types.FunctionCall(name="lookup_order", args={"order_id": "TK-88213"}),
                      thought_signature=b"sig-123")
    p, client = provider([chunk(call, finish="STOP")], [chunk(types.Part(text="Section 112."), finish="STOP")])

    first = await collect(p, LLMRequest(system_static="S", tools=TOOLS,
                                         messages=[{"role": "user", "content": "where?"}]))
    tool = next(e for e in first if isinstance(e, ToolCallDelta))
    assert tool.name == "lookup_order" and tool.arguments == {"order_id": "TK-88213"}
    assert first[-1].stop_reason == "tool_use"

    history = [
        {"role": "user", "content": "where?"},
        {"role": "assistant", "content": [{"type": "tool_use", "id": tool.call_id,
                                           "name": tool.name, "input": tool.arguments}]},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": tool.call_id,
                                      "content": '{"section": "112"}'}]},
    ]
    await collect(p, LLMRequest(system_static="S", tools=TOOLS, messages=history))
    sent = client.requests[1]["contents"]
    model_part = sent[1].parts[0]
    assert model_part.function_call.name == "lookup_order"
    assert model_part.thought_signature == b"sig-123", "thinking models reject an unsigned call"
    response = sent[2].parts[0].function_response
    assert response.name == "lookup_order", "Gemini needs the NAME, recovered from the tool_use"
    assert response.response == {"result": '{"section": "112"}'}


async def test_quota_error_becomes_an_error_stream_end_not_an_exception() -> None:
    p, _ = provider(RuntimeError("429 RESOURCE_EXHAUSTED"))
    events = await collect(p, LLMRequest(system_static="S", messages=[{"role": "user", "content": "x"}]))
    assert events == [StreamEnd(stop_reason="error", error="RuntimeError: 429 RESOURCE_EXHAUSTED")]


async def test_a_safety_stop_is_an_error_not_an_answer() -> None:
    p, _ = provider([chunk(types.Part(text="Partial"), finish="SAFETY")])
    events = await collect(p, LLMRequest(system_static="S", messages=[{"role": "user", "content": "x"}]))
    assert events[-1].stop_reason == "error" and "SAFETY" in events[-1].error


async def test_thought_parts_are_never_spoken() -> None:
    p, _ = provider([chunk(types.Part(text="reasoning...", thought=True),
                           types.Part(text="Answer."), finish="STOP")])
    events = await collect(p, LLMRequest(system_static="S", messages=[{"role": "user", "content": "x"}]))
    assert [e.text for e in events if isinstance(e, TextDelta)] == ["Answer."]


def test_string_and_text_messages_translate() -> None:
    contents = to_gemini_contents([{"role": "user", "content": "hello"},
                                   {"role": "assistant", "content": [{"type": "text", "text": "hi"}]}])
    assert [(c.role, c.parts[0].text) for c in contents] == [("user", "hello"), ("model", "hi")]
